"""Category resolution agent.

Runs after extraction and before the document gate (the gate's required-
document check is keyed by category, so the category must be settled first).

Contract:
- A provided category is honored byte-for-byte: this agent no-ops entirely.
- No category provided -> derive it from the documents' DECIDE-grade
  (procedural) evidence: line items, treatment, tests. Diagnosis-only and
  hospital-name hits never decide. The derived category is written back to
  the submission as the effective category; downstream stages run unchanged.
  The intake-deferred coverage check re-runs here against the derived value.
- Genuinely ambiguous (no procedural evidence, or two or more specialties)
  -> ask the member, not manual review: status NEEDS_RESUBMISSION with a
  CATEGORY_NEEDED issue and an actionable message, halting before the gate.
  When the ambiguity is caused by extraction failures, the message says the
  files could not be read rather than blaming the documents.
"""

from app.agents.base import Agent, ClaimContext
from app.agents.shared_checks import check_coverage
from app.category_evidence import derive_category
from app.models.claim import ClaimCategory
from app.models.decision import DocumentIssue, StepStatus

STAGE = "category_resolution"


class CategoryResolutionAgent(Agent):
    name = STAGE

    def run(self, ctx: ClaimContext) -> None:
        sub = ctx.submission

        # Provided category: honored byte-for-byte, nothing to resolve.
        if sub.claim_category is not None:
            return

        derived, keyword = derive_category(ctx.extracted_documents)

        if derived is None:
            self._ask_member(ctx)
            return

        # Write back the effective category; downstream stages never know
        # the difference (same pattern as the derived claim amount).
        sub.claim_category = ClaimCategory(derived)
        ctx.result.claim_category = derived
        ctx.category_was_derived = True  # gates per-line categorization (6b)
        ctx.trace(
            STAGE, "derive_category", StepStatus.PASSED,
            f"No category was provided; derived {derived} from the documents' "
            f"procedural evidence (matched on '{keyword}'). The claim is "
            f"adjudicated under this category.",
            {"derived_category": derived, "matched_keyword": keyword},
        )

        # Intake deferred the coverage check because there was no category;
        # it must run now against the derived one. The rule lives in
        # shared_checks.check_coverage (shared with intake.py).
        if check_coverage(ctx, category=derived, stage=STAGE, derived=True):
            return

    # ------------------------------------------------------------------
    def _ask_member(self, ctx: ClaimContext) -> None:
        """Ambiguous documents: ask the member to pick — a fixable input
        gap, not a suspicion, so it does not consume reviewer time."""
        extraction_failed = any(
            cf.component.startswith("extraction:")
            for cf in ctx.result.component_failures
        )
        if extraction_failed:
            message = (
                "We could not read your files well enough to determine the "
                "claim type. Please re-upload clearer copies, or select the "
                "claim type and resubmit. The claim has not been rejected."
            )
        else:
            message = (
                "Your documents do not clearly indicate a claim type "
                "(consultation, diagnostic, pharmacy, dental, vision, or "
                "alternative medicine). Please select the claim type and "
                "resubmit. The claim has not been rejected."
            )
        ctx.result.document_issues.append(DocumentIssue(
            file_id="-",
            issue_code="CATEGORY_NEEDED",
            message=message,
            action_required="Select the claim category and resubmit.",
        ))
        ctx.result.status = "NEEDS_RESUBMISSION"
        ctx.result.decision = None
        ctx.result.reasons.append(message)
        ctx.halted = True
        ctx.trace(STAGE, "derive_category", StepStatus.FAILED, message,
                  {"extraction_failed": extraction_failed})
