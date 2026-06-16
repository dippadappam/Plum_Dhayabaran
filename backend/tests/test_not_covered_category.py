"""Coverage for the NOT_COVERED rejection in shared_checks.check_coverage.

Every category in the shipped policy is covered, so this branch is never hit by
the official suite or the live tests — a future policy edit flipping a category
to not-covered would land untested. These tests exercise it on BOTH paths, using
a NON-OFFICIAL, in-memory copy of the policy with exactly one category (dental)
flipped to covered=false:

- provided-category path  -> IntakeAgent's coverage check (derived=False)
- derived-category path   -> CategoryResolutionAgent's coverage check (derived=True)

The official policy_terms.json is read (via load_policy) but never modified.
"""

import pytest

from app.models.claim import ClaimSubmission
from app.models.decision import Decision, RejectionReason
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


@pytest.fixture(scope="module")
def dental_not_covered_policy():
    """A non-official, in-memory copy of the policy with dental.covered=false.
    Built by deep-copying the loaded official policy and flipping exactly one
    flag; the policy_terms.json file on disk is never touched."""
    policy = load_policy().model_copy(deep=True)
    policy.opd_categories["dental"].covered = False
    return policy


@pytest.fixture(scope="module")
def orchestrator(dental_not_covered_policy):
    return ClaimsOrchestrator(policy=dental_not_covered_policy)


def test_provided_not_covered_category_rejected_at_intake(orchestrator):
    """Provided-category path: a filed DENTAL claim is rejected NOT_COVERED at
    the intake coverage check, with the provided-path message and trace."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP002", "policy_id": "PLUM_GHI_2024",
        "claim_category": "DENTAL", "treatment_date": "2024-10-15",
        "claimed_amount": 5000,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Priya Singh",
              "line_items": [{"description": "Root Canal Treatment", "amount": 5000}],
              "total": 5000}},
        ]}))
    assert result.decision == Decision.REJECTED
    assert RejectionReason.NOT_COVERED in result.rejection_reasons
    assert "Category 'DENTAL' is not covered under this policy." in result.reasons
    steps = [s for s in result.trace
             if s.stage == "intake" and s.check == "category_covered"]
    assert len(steps) == 1
    assert steps[0].status.value == "FAILED"
    assert steps[0].detail == "Category 'DENTAL' is not covered."
    # Rejected at intake (stage 1): adjudication and category-resolution never ran.
    assert not any(s.stage == "adjudication" for s in result.trace)
    assert not any(s.stage == "category_resolution" for s in result.trace)


def test_derived_not_covered_category_rejected_at_category_resolution(orchestrator):
    """Derived-category path: a no-category claim whose bill evidence derives
    DENTAL is rejected NOT_COVERED at the category-resolution coverage check,
    with the derived-path message and trace."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP002", "policy_id": "PLUM_GHI_2024",
        # No claim_category -> derived from the bill's procedural evidence.
        "treatment_date": "2024-10-15", "claimed_amount": 5000,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Priya Singh",
              "line_items": [{"description": "Root Canal Treatment", "amount": 5000}],
              "total": 5000}},
        ]}))
    assert result.decision == Decision.REJECTED
    assert RejectionReason.NOT_COVERED in result.rejection_reasons
    assert ("The derived category 'DENTAL' is not covered under this policy."
            in result.reasons)
    assert result.claim_category == "DENTAL"  # derived before the coverage check
    steps = [s for s in result.trace
             if s.stage == "category_resolution" and s.check == "category_covered"]
    assert len(steps) == 1
    assert steps[0].status.value == "FAILED"
    assert steps[0].detail == (
        "Derived category 'DENTAL' is not covered (check was deferred at intake).")
    assert not any(s.stage == "adjudication" for s in result.trace)


def test_official_policy_dental_is_still_covered():
    """Guard: the non-official fixture did not mutate the real policy — a freshly
    loaded official policy still has dental covered."""
    assert load_policy().opd_categories["dental"].covered is True
