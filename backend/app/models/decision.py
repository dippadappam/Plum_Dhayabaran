"""Output contracts: trace, decisions, and the final claim result.

The trace is a first-class object. Every stage appends structured steps,
and the final output must let a reviewer reconstruct the entire decision
without reading code or logs.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


class Decision(str, Enum):
    APPROVED = "APPROVED"
    PARTIAL = "PARTIAL"
    REJECTED = "REJECTED"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class RejectionReason(str, Enum):
    WAITING_PERIOD = "WAITING_PERIOD"
    EXCLUDED_CONDITION = "EXCLUDED_CONDITION"
    PRE_AUTH_MISSING = "PRE_AUTH_MISSING"
    PER_CLAIM_EXCEEDED = "PER_CLAIM_EXCEEDED"
    SUB_LIMIT_EXCEEDED = "SUB_LIMIT_EXCEEDED"
    ANNUAL_LIMIT_EXCEEDED = "ANNUAL_LIMIT_EXCEEDED"
    FAMILY_LIMIT_EXCEEDED = "FAMILY_LIMIT_EXCEEDED"
    MEMBER_NOT_FOUND = "MEMBER_NOT_FOUND"
    POLICY_MISMATCH = "POLICY_MISMATCH"
    NOT_COVERED = "NOT_COVERED"
    BELOW_MINIMUM_AMOUNT = "BELOW_MINIMUM_AMOUNT"
    SUBMISSION_DEADLINE_PASSED = "SUBMISSION_DEADLINE_PASSED"


class StepStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"
    DEGRADED = "DEGRADED"  # component failed; pipeline continued without it
    INFO = "INFO"


class TraceStep(BaseModel):
    """One audited step. `detail` says what was checked and why it resolved
    the way it did, in language an ops reviewer can read directly."""

    stage: str
    check: str
    status: StepStatus
    detail: str
    data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentIssue(BaseModel):
    """A specific, actionable document problem surfaced to the member."""

    file_id: str
    file_name: Optional[str] = None
    issue_code: str  # WRONG_TYPE | UNREADABLE | PATIENT_MISMATCH | MISSING_REQUIRED
    message: str  # names the uploaded type and the required type / specific names found
    action_required: str  # exactly what the member must do


class LineItemDecision(BaseModel):
    """Per-line adjudication for itemized bills (drives PARTIAL decisions)."""

    description: str
    claimed_amount: float
    approved_amount: float
    status: str  # APPROVED | REJECTED | REVIEW
    reason: str
    # Pharmacy branded/generic co-pay (Fix 5): the extractor's per-line drug type
    # ("BRANDED" | "GENERIC") and its confidence, carried through so the pharmacy
    # money math can apply per-line co-pay. Null/None for non-pharmacy lines and
    # whenever the document did not make the drug type clear.
    drug_type: Optional[str] = None
    drug_type_confidence: Optional[float] = None


class CategoryBreakdown(BaseModel):
    """Per-category money math within a claim (Batch 6b per-line
    categorization). For a single-category claim this carries the one group and
    mirrors the top-level AmountBreakdown fields; for a multi-category claim
    each group is one entry and the top-level fields are the aggregates."""

    category: str
    eligible: float
    network_discount_percent: float = 0
    network_discount_amount: float = 0
    amount_after_discount: float = 0
    copay_percent: float = 0
    copay_amount: float = 0
    approved: float = 0


class AmountBreakdown(BaseModel):
    """Deterministic money math, shown step by step (TC010 requirement).

    Top-level fields are the claim total: for a single-category claim they are
    that category's own numbers; for a multi-category claim they are the
    aggregates across category_breakdowns. category_breakdowns is additive —
    pre-6b consumers (and every existing assertion) read only the top-level
    fields, so a single-category claim stays byte-identical."""

    claimed_amount: float
    eligible_amount: float  # after line-item exclusions
    network_discount_percent: float = 0
    network_discount_amount: float = 0
    amount_after_discount: float = 0
    copay_percent: float = 0
    copay_amount: float = 0
    approved_amount: float = 0
    category_breakdowns: list[CategoryBreakdown] = Field(default_factory=list)


class ComponentFailure(BaseModel):
    component: str
    error: str
    impact: str


class ClaimResult(BaseModel):
    """Root output contract: the decision plus everything needed to audit it.

    `status` separates pipeline outcome from claim decision: a document
    problem yields status NEEDS_RESUBMISSION with decision null (no claim
    decision was made), matching the TC001-TC003 contract.
    """

    claim_reference: str
    member_id: str
    claim_category: str
    status: str = "DECIDED"  # DECIDED | NEEDS_RESUBMISSION
    decision: Optional[Decision] = None
    approved_amount: Optional[float] = None
    rejection_reasons: list[RejectionReason] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)  # human-readable explanations
    line_items: list[LineItemDecision] = Field(default_factory=list)
    amount_breakdown: Optional[AmountBreakdown] = None
    document_issues: list[DocumentIssue] = Field(default_factory=list)
    fraud_signals: list[str] = Field(default_factory=list)
    # Transparent weighted fraud score in [0, 1]; a secondary graduated view on
    # top of the binary same-day/monthly/high-value signals. The per-signal
    # contributions that produced it are recorded in the trace.
    fraud_score: float = 0.0
    component_failures: list[ComponentFailure] = Field(default_factory=list)
    manual_review_recommended: bool = False
    # Reviewer triage for manual-review routing: "high" (patient not on the
    # roster at all) | "normal" (e.g. a different covered person). Null when
    # no review tier applies.
    review_priority: Optional[str] = None
    # Review lifecycle (4a). null = no human review needed (auto-final);
    # "PENDING_REVIEW" = held, awaiting a reviewer; "RESOLVED" = a reviewer
    # acted. `status` stays DECIDED throughout (finality is review_status, not
    # status). resolved_at is an audit timestamp, never a decision input.
    review_status: Optional[str] = None
    resolved_by: Optional[str] = None
    resolved_at: Optional[str] = None
    resolution: Optional[str] = None          # APPROVED | REJECTED | CLOSED
    resolution_reason: Optional[str] = None
    # The category the documents themselves indicate (deterministic keyword
    # evidence). Display/transparency only: adjudication still runs under the
    # filed category, and a clear mismatch routes to manual review. Null when
    # the documents carry no category evidence.
    derived_category: Optional[str] = None
    # The extracted record per document (file_id, type, fields) — the
    # reviewer's "what we read" view and the audit evidence. Rides in
    # result_json; no separate extraction_json column.
    extracted_documents: list[dict[str, Any]] = Field(default_factory=list)
    confidence_score: float = 0.0
    confidence_factors: list[str] = Field(default_factory=list)
    # Per-claim LLM token usage, summed across the extraction calls this claim
    # made (input/output tokens + paid-call count). Zero on the structured/test
    # path (no live extractor). Audit/cost visibility only — never a decision input.
    token_usage: dict[str, int] = Field(
        default_factory=lambda: {"input_tokens": 0, "output_tokens": 0, "calls": 0})
    trace: list[TraceStep] = Field(default_factory=list)


# --- Lifecycle predicates (explicit over status, decision, review_status) ---

def is_final(result: "ClaimResult") -> bool:
    """A claim is final when nothing further (automated or human) will change
    its decision: an auto-decided claim (DECIDED, review_status None) or a
    reviewer-resolved one (review_status RESOLVED). A held claim
    (PENDING_REVIEW) is NOT final, and NEEDS_RESUBMISSION is NOT final (the
    member must still act)."""
    if result.status == "NEEDS_RESUBMISSION":
        return False
    return result.review_status in (None, "RESOLVED")


def is_reviewer_resolvable(result: "ClaimResult") -> bool:
    """Only a held claim — DECIDED, decision MANUAL_REVIEW, review_status
    PENDING_REVIEW — can be resolved by a reviewer. Auto-final DECIDED claims,
    already-RESOLVED claims, and NEEDS_RESUBMISSION are not resolvable."""
    return (result.status == "DECIDED"
            and result.decision == Decision.MANUAL_REVIEW
            and result.review_status == "PENDING_REVIEW")
