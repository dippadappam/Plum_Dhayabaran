"""Tests for the CategoryResolutionAgent.

The rule: a provided category is honored byte-for-byte; when absent, the
category is derived from DECIDE-grade (procedural) evidence only — line
items, treatment, tests. Diagnosis-only and hospital-name hits never decide.
Ambiguous documents ask the member (CATEGORY_NEEDED), never manual review.
"""

import pytest

from app.models.claim import ClaimSubmission
from app.models.decision import Decision
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


@pytest.fixture(scope="module")
def policy():
    return load_policy()


@pytest.fixture(scope="module")
def orchestrator(policy):
    return ClaimsOrchestrator(policy=policy)


def run(orchestrator, payload):
    return orchestrator.process(ClaimSubmission.model_validate(payload))


def test_no_category_derives_consultation_and_adjudicates(orchestrator):
    """A consultation bill with no category provided derives CONSULTATION
    and lands on the identical money math as the provided path."""
    r = run(orchestrator, {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-11-01", "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "hospital_name": "City Clinic",
              "line_items": [{"description": "Consultation Fee", "amount": 1000},
                             {"description": "CBC Test", "amount": 500}],
              "total": 1500}},
        ]})
    assert r.claim_category == "CONSULTATION"
    assert r.decision == Decision.APPROVED
    assert r.approved_amount == 1350  # 10% consultation co-pay, unchanged math
    assert any(s.stage == "category_resolution" and s.status.value == "PASSED"
               for s in r.trace)


def test_no_category_derives_dental_specialty(orchestrator):
    """Specialty procedural evidence (root canal) derives DENTAL."""
    r = run(orchestrator, {
        "member_id": "EMP002", "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-10-15", "claimed_amount": 8000,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Priya Singh",
              "line_items": [{"description": "Root Canal Treatment",
                              "amount": 8000}],
              "total": 8000}},
        ]})
    assert r.claim_category == "DENTAL"
    assert r.decision == Decision.APPROVED
    assert r.approved_amount == 8000  # dental has no co-pay


def test_no_procedural_evidence_asks_the_member(orchestrator):
    """Migraine-style claim: no procedural evidence at all. The member is
    asked to pick (CATEGORY_NEEDED); the claim is not rejected and never
    reaches a reviewer queue."""
    r = run(orchestrator, {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-11-01", "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Migraine"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "total": 1500}},
        ]})
    assert r.status == "NEEDS_RESUBMISSION"
    assert r.decision is None
    issues = [i for i in r.document_issues if i.issue_code == "CATEGORY_NEEDED"]
    assert issues and "select the claim type" in issues[0].message.lower()
    assert not any(s.stage == "adjudication" for s in r.trace)


def test_diagnosis_only_evidence_never_decides(orchestrator):
    """'Eye pain' as a diagnosis with a generic bill must NOT derive VISION
    (0% co-pay would overpay); decide-grade evidence is procedural only.

    The bill line uses a deliberately category-neutral wording ("Service
    charges"). Note: "Doctor Fee" was the original neutral wording here, but
    Batch 2 broadened the consultation vocabulary to recognize "Doctor Fee"
    as a consultation signal (so real consultation bills stop bouncing), so
    this test was re-pointed at a wording that genuinely matches no category."""
    r = run(orchestrator, {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-11-01", "claimed_amount": 900,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar",
              "diagnosis": "Eye pain and irritation"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Service charges", "amount": 900}],
              "total": 900}},
        ]})
    assert r.claim_category == "TO_BE_DERIVED"  # nothing decided
    assert r.status == "NEEDS_RESUBMISSION"
    assert any(i.issue_code == "CATEGORY_NEEDED" for i in r.document_issues)


def test_doctor_fee_wording_derives_consultation(orchestrator):
    """Batch 2 vocabulary: a bill worded 'Doctor Fee' now derives CONSULTATION
    instead of bouncing the member to pick a category."""
    r = run(orchestrator, {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-11-01", "claimed_amount": 1000,
        "documents": [
            {"file_id": "F0", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Doctor Fee", "amount": 1000}],
              "total": 1000}},
        ]})
    assert r.claim_category == "CONSULTATION"
    assert r.decision == Decision.APPROVED
    assert r.approved_amount == 900  # 10% consultation co-pay on 1000


def test_two_specialties_is_ambiguous(orchestrator):
    """Root canal + eye examination on one bill: two specialties in
    evidence is genuinely ambiguous; ask, do not guess."""
    r = run(orchestrator, {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-11-01", "claimed_amount": 5000,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [
                  {"description": "Root Canal Treatment", "amount": 4000},
                  {"description": "Eye Examination", "amount": 1000}],
              "total": 5000}},
        ]})
    assert r.status == "NEEDS_RESUBMISSION"
    assert any(i.issue_code == "CATEGORY_NEEDED" for i in r.document_issues)


def test_extraction_failure_says_could_not_read(policy):
    """When ambiguity is caused by extraction failures, the message blames
    the files ('could not read'), not the documents' content."""
    class FailingExtractor:
        def extract(self, document):
            raise RuntimeError("simulated outage")

    orch = ClaimsOrchestrator(policy=policy, extractor=FailingExtractor())
    r = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "documents": [
            {"file_id": "U1", "file_data": "ZmFrZS1yeA==",
             "media_type": "image/jpeg"},
            {"file_id": "U2", "file_data": "ZmFrZS1iaWxs",
             "media_type": "image/jpeg"},
        ]}))
    assert r.status == "NEEDS_RESUBMISSION"
    issues = [i for i in r.document_issues if i.issue_code == "CATEGORY_NEEDED"]
    assert issues
    assert "could not read" in issues[0].message.lower()
    assert "do not clearly indicate" not in issues[0].message.lower()


def test_pharmacy_line_on_hospital_bill_does_not_derive_pharmacy(orchestrator):
    """'Medicines (Pharmacy)' is a routine sub-line on hospital bills; it
    must not derive a PHARMACY claim. Consultation evidence wins."""
    r = run(orchestrator, {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-11-15", "claimed_amount": 3000,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Acute Bronchitis"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "hospital_name": "Apollo Hospitals",
              "line_items": [
                  {"description": "Consultation Fee - Dr. Arun Sharma",
                   "amount": 1500},
                  {"description": "Medicines (Pharmacy)", "amount": 1500}],
              "total": 3000}},
        ]})
    assert r.claim_category == "CONSULTATION"
    assert r.decision == Decision.APPROVED
    # Apollo is a network hospital: 3000 -> -20% = 2400 -> -10% = 2160.
    assert r.approved_amount == 2160


def test_pharmacy_bill_document_type_derives_pharmacy(orchestrator):
    """An actual PHARMACY_BILL among the documents is the evidence that
    derives PHARMACY."""
    r = run(orchestrator, {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "treatment_date": "2024-11-01", "claimed_amount": 800,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever",
              "medicines": ["Paracetamol 650mg"]}},
            {"file_id": "F2", "actual_type": "PHARMACY_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Paracetamol 650mg", "amount": 800,
                              "drug_type": "GENERIC",
                              "drug_type_confidence": 0.95}],
              "total": 800}},
        ]})
    assert r.claim_category == "PHARMACY"
    assert r.decision == Decision.APPROVED


def test_provided_category_is_never_rederived(orchestrator):
    """Provided category is honored byte-for-byte: a filed VISION claim with
    consultation documents takes the existing consistency path (manual
    review), not silent re-derivation."""
    r = run(orchestrator, {
        "member_id": "EMP010", "policy_id": "PLUM_GHI_2024",
        "claim_category": "VISION", "treatment_date": "2024-11-03",
        "claimed_amount": 4500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Deepak Shah", "diagnosis": "Acute Bronchitis"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Deepak Shah",
              "line_items": [{"description": "Consultation Fee", "amount": 1500},
                             {"description": "Medicines", "amount": 3000}],
              "total": 4500}},
        ]})
    assert r.claim_category == "VISION"  # never silently rewritten
    assert r.decision == Decision.MANUAL_REVIEW
    assert not any(s.stage == "category_resolution" and "derived" in s.detail
                   for s in r.trace)
