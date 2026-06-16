"""Input contracts: claim submission and documents.

These models define the exact shape of what enters the system.
Validation failures here are caught at the Intake stage with specific errors.
"""

from datetime import date
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator

# Input caps: enforced at the validation layer so oversized or junk
# submissions are rejected with a free 422 before any paid extraction call.
MAX_DOCUMENTS_PER_CLAIM = 10
MAX_FILE_DATA_CHARS = 15_000_000  # ~11 MB binary as base64
ALLOWED_MEDIA_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/gif", "application/pdf",
}


class DocumentType(str, Enum):
    PRESCRIPTION = "PRESCRIPTION"
    HOSPITAL_BILL = "HOSPITAL_BILL"
    PHARMACY_BILL = "PHARMACY_BILL"
    LAB_REPORT = "LAB_REPORT"
    DIAGNOSTIC_REPORT = "DIAGNOSTIC_REPORT"
    DISCHARGE_SUMMARY = "DISCHARGE_SUMMARY"
    DENTAL_REPORT = "DENTAL_REPORT"
    UNKNOWN = "UNKNOWN"


class ClaimCategory(str, Enum):
    CONSULTATION = "CONSULTATION"
    DIAGNOSTIC = "DIAGNOSTIC"
    PHARMACY = "PHARMACY"
    DENTAL = "DENTAL"
    VISION = "VISION"
    ALTERNATIVE_MEDICINE = "ALTERNATIVE_MEDICINE"


class DocumentQuality(str, Enum):
    GOOD = "GOOD"
    UNREADABLE = "UNREADABLE"


class SubmittedDocument(BaseModel):
    """A single document attached to a claim.

    Exactly one of `content` (pre-extracted structured data, used by the
    test harness) or `file_data` (base64 image/PDF bytes for live vision
    extraction) is expected. `actual_type`, `quality`, and
    `patient_name_on_doc` are test-harness hints that stand in for what
    the extraction stage would otherwise produce from the raw file.
    """

    file_id: str
    file_name: Optional[str] = None
    actual_type: Optional[DocumentType] = None
    quality: Optional[DocumentQuality] = None
    patient_name_on_doc: Optional[str] = None
    content: Optional[dict[str, Any]] = None
    file_data: Optional[str] = Field(
        default=None, max_length=MAX_FILE_DATA_CHARS,
        description="Base64-encoded image/PDF for live extraction",
    )
    media_type: Optional[str] = Field(
        default=None, description="MIME type of file_data, e.g. image/jpeg"
    )

    @field_validator("media_type")
    @classmethod
    def _media_type_allowed(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ALLOWED_MEDIA_TYPES:
            # Clear, member-facing message (iPhones default to HEIC). Accepting
            # and converting HEIC is Phase 2; this batch only fixes the message.
            raise ValueError(
                "This photo format is not supported. Please upload a JPG, "
                "PNG, or PDF."
            )
        return v


class PriorClaim(BaseModel):
    """One entry of the member's claim history, used by fraud checks."""

    claim_id: str
    date: date
    amount: float
    provider: Optional[str] = None


class ClaimSubmission(BaseModel):
    """The complete claim as submitted. Root input contract of the system."""

    member_id: str
    policy_id: str
    # claim_category is optional: when provided it is honored byte-for-byte;
    # when absent (real-upload path) the CategoryResolutionAgent derives it
    # from the documents' procedural evidence and writes it back, or asks the
    # member to pick when the documents are genuinely ambiguous.
    claim_category: Optional[ClaimCategory] = None
    # treatment_date and claimed_amount are optional: on the real-upload path
    # they are not typed by the member but derived from the extracted bill
    # (see ClaimDerivationAgent). On the test/structured path they are provided.
    treatment_date: Optional[date] = None
    # When the claim was submitted. Stamped by the API if not provided; used by
    # the deterministic future-date sanity check (a treatment date after the
    # submission date is impossible and routes to review).
    submission_date: Optional[date] = None
    # The genuine date intake RECEIVED the claim. Production intake supplies it;
    # it is NEVER auto-stamped. The 30-day submission deadline is measured
    # against this, not against the auto-stamped submission_date — so the
    # deadline cannot retroactively fail historical or demo data filed with a
    # server "today". Absent it, the deadline check is skipped.
    received_date: Optional[date] = None
    claimed_amount: Optional[float] = Field(default=None, gt=0,
                                            allow_inf_nan=False)
    hospital_name: Optional[str] = None
    ytd_claims_amount: float = Field(default=0, allow_inf_nan=False)
    # Family-floater year-to-date: the total already approved this policy year
    # across the whole family (employee + dependents). Computed from storage and
    # injected by the API (the engine stays a pure function of its inputs and
    # never reads storage), the same way ytd_claims_amount is passed in. Defaults
    # to 0, so a claim with no family accumulation (the 12 official cases) is
    # never affected by the family-limit check.
    family_ytd_amount: float = Field(default=0, allow_inf_nan=False)
    claims_history: list[PriorClaim] = Field(default_factory=list)
    # Set by the API when the uploaded document set is byte-identical to an
    # already-decided claim (H3 duplicate detection); the fraud stage turns it
    # into a duplicate signal that routes a payable claim to manual review.
    duplicate_of: Optional[str] = None
    # Explicit resubmission link (4b): the original claim's reference, passed by
    # the UI when resubmitting after a NEEDS_RESUBMISSION. Never inferred.
    parent_claim_reference: Optional[str] = None
    documents: list[SubmittedDocument] = Field(
        min_length=1, max_length=MAX_DOCUMENTS_PER_CLAIM)
    simulate_component_failure: bool = False
