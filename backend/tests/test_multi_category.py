"""Multi-category (per-line money math) guard tests — Batch 6b.

The multi-group path runs only when a claim is on the DERIVED path (no
claim_category provided) AND its lines span two or more categories. These tests
reach it via structured content with no category provided (no LLM, no network)
and pin the behaviors the single non-network multi_service bundle does not
cover: per-category network discount, the multi-group per-claim cap, and the
consultation per-line cap inside a group.
"""

import pytest

from app.models.claim import ClaimSubmission
from app.models.decision import Decision, RejectionReason
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


@pytest.fixture(scope="module")
def orchestrator():
    return ClaimsOrchestrator(policy=load_policy())


def _derived_bill_claim(line_items, hospital, member_id="EMP001",
                        patient="Rajesh Kumar"):
    """A derived-path claim (no claim_category) with one hospital bill, so the
    category is derived and per-line categorization runs in adjudication."""
    total = sum(i["amount"] for i in line_items)
    return ClaimSubmission.model_validate({
        "member_id": member_id, "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-11-01", "claimed_amount": total,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content": {
                "patient_name": patient, "hospital_name": hospital,
                "date": "2024-11-01", "line_items": line_items, "total": total}},
        ]})


def _by_category(result):
    return {c.category: c for c in result.amount_breakdown.category_breakdowns}


def test_multigroup_network_discount_is_per_category(orchestrator):
    """Consult + dental at Apollo (network). The 20% network discount applies to
    the CONSULTATION group only, never claim-wide: consultation 1500 -> 20% ->
    1200 -> 10% co-pay -> 1080; dental 2000 -> no discount, 0% co-pay -> 2000;
    total 3080. Guards against re-introducing a claim-wide discount."""
    r = orchestrator.process(_derived_bill_claim(
        [{"description": "Consultation Fee", "amount": 1500},
         {"description": "Root Canal Treatment", "amount": 2000}],
        hospital="Apollo Hospitals"))
    assert r.decision == Decision.APPROVED
    assert r.approved_amount == 3080
    groups = _by_category(r)
    assert set(groups) == {"CONSULTATION", "DENTAL"}  # the multi-group path ran
    consult = groups["CONSULTATION"]
    assert consult.network_discount_percent == 20
    assert consult.network_discount_amount == 300
    assert consult.amount_after_discount == 1200
    assert consult.copay_percent == 10
    assert consult.copay_amount == 120
    assert consult.approved == 1080
    dental = groups["DENTAL"]
    assert dental.network_discount_amount == 0  # dental has no network discount
    assert dental.copay_amount == 0             # and no co-pay
    assert dental.approved == 2000
    # Top-level fields hold the aggregates.
    assert r.amount_breakdown.network_discount_amount == 300
    assert r.amount_breakdown.copay_amount == 120
    assert r.amount_breakdown.approved_amount == 3080


def test_multigroup_per_claim_cap_rejects_over_aggregate(orchestrator):
    """Multi-group per-claim cap = max(per_claim 5000, max present sub_limit).
    Consult + dental -> cap 10000. A pre-discount aggregate of 11000 exceeds it
    -> REJECTED PER_CLAIM_EXCEEDED, guarding the multi-group cap path."""
    r = orchestrator.process(_derived_bill_claim(
        [{"description": "Consultation Fee", "amount": 2000},
         {"description": "Root Canal Treatment", "amount": 9000}],
        hospital="City Medical Centre, Bengaluru"))
    assert r.decision == Decision.REJECTED
    assert RejectionReason.PER_CLAIM_EXCEEDED in r.rejection_reasons
    # Confirm it rejected on the MULTI-group cap path, not the single path.
    assert any(s.check == "per_claim_limit" and s.status.value == "FAILED"
               and "Aggregate" in s.detail for s in r.trace)


def test_multigroup_consultation_per_line_cap_fires_in_group(orchestrator):
    """The consultation per-line sub-limit (2000) caps a consultation line
    inside its group even in a multi-category claim: consult 2500 -> capped to
    2000 -> 10% co-pay -> 1800; dental 2000 -> 2000; total 3800."""
    r = orchestrator.process(_derived_bill_claim(
        [{"description": "Consultation Fee", "amount": 2500},
         {"description": "Root Canal Treatment", "amount": 2000}],
        hospital="City Medical Centre, Bengaluru"))
    assert r.decision == Decision.APPROVED
    assert r.approved_amount == 3800
    consult_line = next(li for li in r.line_items
                        if li.description == "Consultation Fee")
    assert consult_line.approved_amount == 2000  # capped from 2500
    assert any(s.check == "consultation_fee_sub_limit" for s in r.trace)
    groups = _by_category(r)
    assert groups["CONSULTATION"].eligible == 2000   # post-cap
    assert groups["CONSULTATION"].approved == 1800
    assert groups["DENTAL"].approved == 2000
