"""Tests for the Consistency and Derivation agents.

Formalizes the verification matrix for the two newest pipeline stages:
category/patient cross-checks (consistency) and document-derived claim
values (derivation). No LLM, no network: structured content or a mock
extractor throughout.
"""

import pytest

from app.models.claim import ClaimSubmission
from app.models.decision import Decision, RejectionReason
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


@pytest.fixture(scope="module")
def policy():
    return load_policy()


@pytest.fixture(scope="module")
def orchestrator(policy):
    return ClaimsOrchestrator(policy=policy)


class PerDocExtractor:
    """Mock DocumentExtractor returning canned fields keyed by file_id."""

    def __init__(self, by_id):
        self.by_id = by_id
        self.calls = 0

    def extract(self, document):
        self.calls += 1
        return dict(self.by_id[document.file_id])


def test_category_mismatch_routes_to_manual_review(orchestrator):
    """Filed VISION, documents describe a general consultation: no vision
    evidence, clear consultation evidence -> MANUAL_REVIEW, never a silent
    approval under the wrong category's rules."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP010", "policy_id": "PLUM_GHI_2024",
        "claim_category": "VISION", "treatment_date": "2024-11-03",
        "claimed_amount": 4500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Deepak Shah", "diagnosis": "Acute Bronchitis",
              "medicines": ["Amoxicillin 500mg"]}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Deepak Shah",
              "line_items": [{"description": "Consultation Fee", "amount": 1500},
                             {"description": "Medicines", "amount": 3000}],
              "total": 4500}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    text = " ".join(result.reasons)
    assert "VISION" in text and "CONSULTATION" in text
    assert result.derived_category == "CONSULTATION"
    # Adjudication never ran: no claim money was computed.
    assert not any(s.stage == "adjudication" for s in result.trace)


def test_different_covered_person_routes_to_manual_review(orchestrator):
    """Filed as the child (DEP002), documents are for the father (EMP001):
    manual review with both names, normal priority, no auto-reject."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "DEP002", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Consultation Fee", "amount": 1500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert result.review_priority == "normal"
    text = " ".join(result.reasons)
    assert "Rajesh Kumar" in text and "Arjun Kumar" in text
    assert RejectionReason.EXCLUDED_CONDITION not in result.rejection_reasons
    assert not result.rejection_reasons, "Must not auto-reject"


def test_patient_not_on_roster_is_high_priority(orchestrator):
    """Documents for a person on nobody's roster: manual review, high
    priority for the reviewer."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Suresh Gupta", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Suresh Gupta",
              "line_items": [{"description": "Consultation Fee", "amount": 1500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert result.review_priority == "high"
    assert "not on the policy member roster" in " ".join(result.reasons)


def test_derived_amount_below_minimum_rejected(policy):
    """Real-upload path: the bill totals 300, below the 500 minimum. The
    deferred intake check must fire after derivation."""
    extractor = PerDocExtractor({
        "U1": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "date": "2024-11-01", "readability": "GOOD",
               "_extraction_confidence": 0.95},
        "U2": {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
               "hospital_name": "City Clinic", "date": "2024-11-01",
               "line_items": [{"description": "Consultation Fee", "amount": 300}],
               "total": 300, "readability": "GOOD",
               "_extraction_confidence": 0.95},
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "U1", "file_data": "ZmFrZS1yeA==",
             "media_type": "image/jpeg"},
            {"file_id": "U2", "file_data": "ZmFrZS1iaWxs",
             "media_type": "image/jpeg"},
        ]}))
    assert result.decision == Decision.REJECTED
    assert result.rejection_reasons == [RejectionReason.BELOW_MINIMUM_AMOUNT]


def test_no_category_evidence_is_not_flagged(orchestrator):
    """Lenient rule: a diagnosis with no category keywords anywhere must not
    flag; the filed category stands and the claim approves."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Migraine"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "total": 1500}},
        ]}))
    assert result.decision == Decision.APPROVED


def test_generic_consultation_line_does_not_flag_specialty(orchestrator):
    """TC011's trap: an ayurveda bill with a generic 'Consultation' line.
    The supported filed category must never be overridden by generic
    consultation evidence."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP006", "policy_id": "PLUM_GHI_2024",
        "claim_category": "ALTERNATIVE_MEDICINE",
        "treatment_date": "2024-10-28", "claimed_amount": 4000,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"doctor_registration": "AYUR/KL/2345/2019",
              "diagnosis": "Chronic Joint Pain",
              "treatment": "Panchakarma Therapy"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"hospital_name": "Ayur Wellness Centre", "total": 4000,
              "line_items": [
                  {"description": "Panchakarma Therapy (5 sessions)",
                   "amount": 3000},
                  {"description": "Consultation", "amount": 1000}]}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.derived_category == "ALTERNATIVE_MEDICINE"


# ---------------------------------------------------------------------------
# Identity batch: directional dependents + name-not-extracted relaxation.
# ---------------------------------------------------------------------------

def test_dependent_of_filer_passes_cleanly(orchestrator):
    """The most common family claim: the employee (EMP001) files for their own
    child (DEP002, Arjun Kumar, a listed dependent). The patient is a dependent
    OF the filer, so identity passes cleanly and the claim is decided — not
    held."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Arjun Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Arjun Kumar",
              "line_items": [{"description": "Consultation Fee", "amount": 1000},
                             {"description": "CBC Test", "amount": 500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 1350  # 10% consultation co-pay, unchanged
    assert result.review_priority is None
    assert not result.rejection_reasons
    assert any(s.stage == "consistency" and s.check == "patient_identity"
               and s.status.value == "PASSED" for s in result.trace)


def test_different_employee_as_patient_still_holds(orchestrator):
    """Fail closed stays: EMP001 files with documents for a *different
    employee* (Priya Singh, EMP002) — a covered person but not a dependent of
    the filer. Must still hold for review, naming both."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Priya Singh", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Priya Singh",
              "line_items": [{"description": "Consultation Fee", "amount": 1500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert result.review_priority == "normal"
    text = " ".join(result.reasons)
    assert "Priya Singh" in text and "Rajesh Kumar" in text
    assert not result.rejection_reasons


def test_off_roster_patient_still_holds_high(orchestrator):
    """Fail closed stays: EMP001 files with documents for a person on nobody's
    roster — must hold for review at high priority."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Suresh Gupta", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Suresh Gupta",
              "line_items": [{"description": "Consultation Fee", "amount": 1500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert result.review_priority == "high"
    assert "not on the policy member roster" in " ".join(result.reasons)


def test_names_match_tolerates_honorifics_suffixes_initials():
    """names_match: honorifics, punctuation, S/o suffixes, and a middle initial
    match the same person; a different surname or an unrelated name does not."""
    from app.condition_mapping import names_match
    assert names_match("Mr. Rajesh Kumar", "Rajesh Kumar")
    assert names_match("RAJESH KUMAR", "Rajesh Kumar")
    assert names_match("Rajesh Kumar S/o Mohan Kumar", "Rajesh Kumar")
    assert names_match("Rajesh K Kumar", "Rajesh Kumar")
    assert not names_match("Rajesh Kumar", "Suresh Gupta")
    assert not names_match("Rajesh Kumar", "Rajesh Sharma")  # different surname


def test_honorific_patient_name_matches_filing_member(orchestrator):
    """A patient name with an honorific ('Mr. Rajesh Kumar') matches the filing
    member after normalization and is not flagged off-roster (it was, under the
    old exact-string match)."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Mr. Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Mr. Rajesh Kumar",
              "line_items": [{"description": "Consultation Fee", "amount": 1500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.review_priority is None


def test_no_extracted_name_is_not_routed_to_review(orchestrator):
    """Name-not-extracted relaxation: a payable claim where no document carried
    a readable patient name, but the filing member and policy are otherwise
    consistent, is NOT routed to review on that basis alone — only an advisory
    trace note is recorded."""
    result = orchestrator.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"line_items": [{"description": "Consultation Fee", "amount": 1000},
                             {"description": "CBC Test", "amount": 500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 1350
    assert result.manual_review_recommended is False  # not routed on missing name
    assert not result.component_failures
    # The missing-name situation is recorded as an advisory only.
    assert any(s.check == "identity_advisory" for s in result.trace)


def test_live_no_name_high_value_holds(policy):
    """Live upload, no readable patient name, amount >= the high-value
    threshold: held for review (identity unverified on a high-value claim)."""
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "diagnosis": "Viral Fever",
               "date": "2024-11-01", "readability": "GOOD",
               "_extraction_confidence": 0.95},
        "bill": {"document_type": "HOSPITAL_BILL", "hospital_name": "City Clinic",
                 "date": "2024-11-01",
                 "line_items": [{"description": "Consultation Fee", "amount": 1500},
                                {"description": "Medicines", "amount": 3000}],
                 "total": 4500, "readability": "GOOD",
                 "_extraction_confidence": 0.95},
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "rx", "file_data": "ZmFrZS1yeA==", "media_type": "image/jpeg"},
            {"file_id": "bill", "file_data": "ZmFrZS1iaWxs", "media_type": "image/jpeg"},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert result.manual_review_recommended is True
    assert "high-value" in " ".join(result.reasons).lower()


def test_live_no_name_low_value_still_passes(policy):
    """Live upload, no readable patient name, amount below the high-value
    threshold: passes with an advisory note, not held on the missing-name
    basis."""
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "diagnosis": "Viral Fever",
               "date": "2024-11-01", "readability": "GOOD",
               "_extraction_confidence": 0.95},
        "bill": {"document_type": "HOSPITAL_BILL", "hospital_name": "City Clinic",
                 "date": "2024-11-01",
                 "line_items": [{"description": "Consultation Fee", "amount": 800},
                                {"description": "Medicines", "amount": 700}],
                 "total": 1500, "readability": "GOOD",
                 "_extraction_confidence": 0.95},
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "rx", "file_data": "ZmFrZS1yeA==", "media_type": "image/jpeg"},
            {"file_id": "bill", "file_data": "ZmFrZS1iaWxs", "media_type": "image/jpeg"},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.manual_review_recommended is False
    assert any(s.check == "identity_advisory" for s in result.trace)
