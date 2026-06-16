"""Orchestrator: coordinates the agent pipeline.

Runs agents in a fixed, known order, accumulates one structured trace,
and wraps every agent so an unexpected exception degrades the pipeline
(recorded as a component failure, confidence lowered, manual review
recommended) instead of crashing the request.

Pipeline: Intake -> Extraction -> CategoryResolution -> DocumentVerification
          -> Derivation -> Consistency -> Adjudication
Extraction runs before the document gate so the gate and the downstream
derivation can use the values detected from the documents (type, readability,
patient name, bill total). A stage may halt the pipeline (ctx.halted) when
continuing would be wrong: intake rejections, the document gate, and a
real-upload claim whose amount/date cannot be derived all do this.
"""

import uuid
from typing import Optional

from app.agents.adjudication import AdjudicationAgent
from app.agents.base import Agent, ClaimContext
from app.agents.category_resolution import CategoryResolutionAgent
from app.agents.consistency import ConsistencyAgent
from app.agents.derivation import ClaimDerivationAgent
from app.agents.document_verification import DocumentVerificationAgent
from app.agents.extraction import DocumentExtractor, ExtractionAgent
from app.agents.intake import IntakeAgent
from app.confidence import score_confidence
from app.config import EngineConfig
from app.models.claim import ClaimSubmission
from app.models.decision import ClaimResult, ComponentFailure, Decision, StepStatus
from app.models.policy import Policy


class ClaimsOrchestrator:
    def __init__(self, policy: Policy,
                 extractor: Optional[DocumentExtractor] = None,
                 config: Optional[EngineConfig] = None):
        self.policy = policy
        self.config = config or EngineConfig()
        self.agents: list[Agent] = [
            IntakeAgent(),
            ExtractionAgent(extractor=extractor, config=self.config),
            CategoryResolutionAgent(),
            DocumentVerificationAgent(),
            ClaimDerivationAgent(),
            ConsistencyAgent(),
            AdjudicationAgent(config=self.config),
        ]

    def process(self, submission: ClaimSubmission) -> ClaimResult:
        category_text = (
            submission.claim_category.value
            if submission.claim_category is not None
            else "TO_BE_DERIVED"
        )
        result = ClaimResult(
            claim_reference=f"CLM-{uuid.uuid4().hex[:10].upper()}",
            member_id=submission.member_id,
            claim_category=category_text,
        )
        ctx = ClaimContext(submission=submission, policy=self.policy, result=result)
        # Preserve the member's stated amount before the derivation agent can
        # overwrite submission.claimed_amount with the bill total (real-upload
        # path), so the claimed-vs-bill cross-check compares the member's number.
        ctx.original_claimed_amount = submission.claimed_amount
        amount_text = (
            f"claimed ₹{submission.claimed_amount:,.0f}"
            if submission.claimed_amount is not None
            else "amount to be derived from the uploaded documents"
        )
        ctx.trace("orchestrator", "start", StepStatus.INFO,
                  f"Processing claim {result.claim_reference}: "
                  f"{category_text if submission.claim_category else 'category to be derived from the documents'} "
                  f"for member {submission.member_id}, {amount_text}.")

        for agent in self.agents:
            if ctx.halted:
                ctx.trace("orchestrator", f"skip:{agent.name}", StepStatus.SKIPPED,
                          f"Stage '{agent.name}' skipped: pipeline halted by an "
                          "earlier stage.")
                continue
            try:
                agent.run(ctx)
            except Exception as e:
                # Graceful degradation: never crash the pipeline.
                ctx.result.component_failures.append(ComponentFailure(
                    component=agent.name,
                    error=str(e),
                    impact=f"Stage '{agent.name}' did not complete; downstream "
                           "stages ran with the data available.",
                ))
                ctx.result.manual_review_recommended = True
                ctx.trace("orchestrator", f"degraded:{agent.name}",
                          StepStatus.DEGRADED,
                          f"Stage '{agent.name}' failed with: {e}. Pipeline "
                          "continued; manual review recommended.")

        # If adjudication never produced a decision and the pipeline was not
        # halted by the document gate, route to manual review rather than
        # returning nothing.
        if not ctx.halted and ctx.result.decision is None:
            ctx.result.decision = Decision.MANUAL_REVIEW
            ctx.result.manual_review_recommended = True
            ctx.result.reasons.append(
                "The pipeline could not reach an automated decision; routed to "
                "manual review."
            )

        # Uniform degradation policy: any component failure anywhere in the
        # pipeline recommends manual review, because processing was incomplete.
        if ctx.result.component_failures:
            ctx.result.manual_review_recommended = True

        if ctx.result.manual_review_recommended and \
                ctx.result.decision != Decision.MANUAL_REVIEW:
            ctx.result.reasons.append(
                "Manual review is recommended because processing was incomplete."
            )

        # Review lifecycle (4a): a held claim awaits a human. status stays
        # DECIDED (the engine produced a terminal routing decision); finality
        # is expressed by review_status.
        if ctx.result.decision == Decision.MANUAL_REVIEW:
            ctx.result.review_status = "PENDING_REVIEW"

        # 4b: carry the extracted record into the result (the reviewer's
        # "what we read" view and the audit evidence).
        ctx.result.extracted_documents = list(ctx.extracted_documents)

        score_confidence(ctx)
        ctx.trace("orchestrator", "complete", StepStatus.INFO,
                  f"Decision: {ctx.result.decision.value if ctx.result.decision else 'NONE (resubmission needed)'}; "
                  f"approved ₹{(ctx.result.approved_amount or 0):,.0f}; "
                  f"confidence {ctx.result.confidence_score:.2f}.")
        return ctx.result
