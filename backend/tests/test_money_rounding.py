"""Pins the round-half-up currency rounding (assumptions.md A12).

Currency amounts use round-half-up (round_money), not Python's banker's round().
These tests lock that at both the helper level and through the engine's co-pay
math, so a regression back to banker's rounding (or to round()) fails the suite.
A half-paisa (x.xx5) must round UP, the value banker's rounding would round down.
"""

import pytest

from app.agents.rules import round_money
from app.models.claim import ClaimSubmission
from app.models.decision import Decision
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


def test_round_money_rounds_half_up_where_round_is_bankers():
    """Exact half-paisa values round UP under round_money; Python's round() is
    banker's (round-half-to-even) and goes the other way at the boundary."""
    assert round_money(0.125) == 0.13 and round(0.125, 2) == 0.12
    assert round_money(250.125) == 250.13 and round(250.125, 2) == 250.12
    # 2.675's nearest double is just below 2.675, so round() yields 2.67;
    # round_money uses the decimal value and yields 2.68.
    assert round_money(2.675) == 2.68
    # Non-boundary amounts are unchanged — identical to round() (so the 12
    # official cases, none of which hit a half-paisa, stay byte-identical).
    for v in (0.0, 360.0, 900.0, 1350.0, 73.62, 2251.12):
        assert round_money(v) == round(v, 2) == v


@pytest.fixture(scope="module")
def orchestrator():
    return ClaimsOrchestrator(policy=load_policy())


def test_copay_rounds_half_up_at_half_paisa_boundary(orchestrator):
    """Through the engine's co-pay math: a 10% co-pay on ₹2,501.25 is exactly
    ₹250.125, which rounds UP to ₹250.13 under round-half-up — banker's rounding
    would round to even (₹250.12). The bill line avoids the word 'consultation'
    so the consultation sub-limit does not cap it, and no network hospital is
    given so only the co-pay applies."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 2501.25,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Day Care Procedure", "amount": 2501.25}],
              "total": 2501.25}},
        ]}))
    assert result.decision == Decision.APPROVED
    b = result.amount_breakdown
    assert b.copay_percent == 10
    # 2,501.25 x 10% = 250.125 -> 250.13 (half-up). Banker's would give 250.12:
    assert round(250.125, 2) == 250.12          # the value the old rule produced
    assert b.copay_amount == 250.13             # the value the new rule produces
    # approved = 2,501.25 - 250.13 = 2,251.12
    assert b.approved_amount == 2251.12
    assert result.approved_amount == 2251.12
