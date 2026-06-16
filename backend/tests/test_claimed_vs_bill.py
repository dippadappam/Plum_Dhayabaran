"""Claimed-amount vs bill-total cross-check.

When the member's stated amount and the documented bill total diverge beyond the
tolerance band, the claim routes to manual review (never auto-reject) with both
numbers in the trace; when they agree or differ only within tolerance, it does
not. On the real-upload derive path the member states no amount, so there is
nothing to compare and the rule stays silent (it reads the member's ORIGINAL
stated amount, not the bill against itself). The 12 official cases have
claimed == bill, so the rule is silent for them (golden-trace test guards that).
"""

from app.models.claim import ClaimSubmission, SubmittedDocument
from app.models.decision import Decision
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


def _structured_claim(claimed, bill_total):
    """A consultation with structured content: the bill's line item equals its
    total (so intra-bill reconciliation passes) while `claimed` is set
    independently, to isolate the claimed-vs-bill check."""
    return ClaimSubmission(
        member_id="EMP001", policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION", treatment_date="2024-11-01",
        claimed_amount=claimed,
        documents=[
            SubmittedDocument(file_id="F1", actual_type="PRESCRIPTION",
                              content={"patient_name": "Rajesh Kumar",
                                       "diagnosis": "Viral Fever",
                                       "doctor_registration": "KA/45678/2015"}),
            SubmittedDocument(file_id="F2", actual_type="HOSPITAL_BILL",
                              content={"patient_name": "Rajesh Kumar",
                                       "line_items": [{"description": "Consultation Fee",
                                                       "amount": bill_total}],
                                       "total": bill_total}),
        ])


def test_stated_amount_far_from_bill_routes_to_review():
    result = ClaimsOrchestrator(policy=load_policy()).process(
        _structured_claim(claimed=5000, bill_total=1000))
    assert result.decision == Decision.MANUAL_REVIEW
    steps = [s for s in result.trace if s.check == "claimed_amount_mismatch"]
    assert len(steps) == 1
    assert steps[0].status.value == "FAILED"
    detail = steps[0].detail
    assert "5,000" in detail and "1,000" in detail   # both numbers named
    assert steps[0].data["claimed_amount"] == 5000
    assert steps[0].data["bill_total"] == 1000
    assert any("differs materially" in r for r in result.reasons)


def test_amounts_agree_no_mismatch():
    result = ClaimsOrchestrator(policy=load_policy()).process(
        _structured_claim(claimed=1000, bill_total=1000))
    assert not any(s.check == "claimed_amount_mismatch" for s in result.trace)
    assert result.decision == Decision.APPROVED


def test_within_tolerance_no_mismatch():
    # claimed 1000 vs bill 1100 -> 9% < 20% band -> not flagged.
    result = ClaimsOrchestrator(policy=load_policy()).process(
        _structured_claim(claimed=1000, bill_total=1100))
    assert not any(s.check == "claimed_amount_mismatch" for s in result.trace)
    assert result.decision == Decision.APPROVED


class _BillExtractor:
    """Mock extractor returning a bill whose total is fixed at `bill_total`."""

    def __init__(self, bill_total):
        self.bill_total = bill_total

    def extract(self, document):
        dtype = document.actual_type.value if document.actual_type else "UNKNOWN"
        f = {"document_type": dtype, "readability": "GOOD",
             "_extraction_confidence": 0.9, "patient_name": "Rajesh Kumar"}
        if dtype == "HOSPITAL_BILL":
            f["total"] = self.bill_total
            f["line_items"] = [{"description": "Consultation Fee",
                                "amount": self.bill_total}]
        else:
            f["diagnosis"] = "Viral Fever"
            f["date"] = "2024-11-01"
        return f


def _live_docs():
    return [
        SubmittedDocument(file_id="A", actual_type="PRESCRIPTION",
                          file_data="b25l", media_type="image/png"),
        SubmittedDocument(file_id="B", actual_type="HOSPITAL_BILL",
                          file_data="dHdv", media_type="image/png"),
    ]


def test_live_stated_amount_diverges_from_extracted_bill_routes_to_review():
    """Real-upload path WITH a stated amount: the member typed ₹5,000 but the
    uploaded bill extracts to ₹1,000. The original stated amount (preserved
    before derivation) is compared against the extracted bill total → review."""
    orch = ClaimsOrchestrator(policy=load_policy(),
                              extractor=_BillExtractor(bill_total=1000))
    sub = ClaimSubmission(
        member_id="EMP001", policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION", treatment_date="2024-11-01",
        claimed_amount=5000, documents=_live_docs())
    result = orch.process(sub)
    assert result.decision == Decision.MANUAL_REVIEW
    steps = [s for s in result.trace if s.check == "claimed_amount_mismatch"]
    assert len(steps) == 1
    assert steps[0].data["claimed_amount"] == 5000   # the member's number, not the bill
    assert steps[0].data["bill_total"] == 1000


def test_derive_path_states_no_amount_no_self_compare():
    """Real-upload path with NO stated amount: claimed is derived from the bill,
    so there is nothing to cross-check — the rule must not fire (no comparing the
    bill against itself)."""
    orch = ClaimsOrchestrator(policy=load_policy(),
                              extractor=_BillExtractor(bill_total=2000))
    sub = ClaimSubmission(
        member_id="EMP001", policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION", treatment_date="2024-11-01",
        documents=_live_docs())  # no claimed_amount
    result = orch.process(sub)
    assert not any(s.check == "claimed_amount_mismatch" for s in result.trace)
