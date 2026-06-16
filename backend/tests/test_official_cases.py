"""The 12 official test cases, run as the acceptance suite.

Each test loads its input straight from test_cases.json (the file is the
spec; nothing is re-typed), runs it through the real orchestrator with no
LLM involved (all cases carry structured content), and asserts the expected
contract: decision, amounts, reason codes, and the system_must behaviors.
"""

import json
from pathlib import Path

import pytest

from app.models.claim import ClaimSubmission
from app.models.decision import Decision, RejectionReason
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator

CASES_PATH = Path(__file__).parent / "test_cases.json"
with open(CASES_PATH, "r", encoding="utf-8") as f:
    _CASES = {c["case_id"]: c for c in json.load(f)["test_cases"]}


@pytest.fixture(scope="module")
def orchestrator():
    return ClaimsOrchestrator(policy=load_policy())


def run_case(orchestrator, case_id: str):
    case = _CASES[case_id]
    submission = ClaimSubmission.model_validate(case["input"])
    return orchestrator.process(submission), case


# ---------------------------------------------------------------------------
# TC001-TC003: document gate. No claim decision may be made.
# ---------------------------------------------------------------------------

def test_tc001_wrong_document_type(orchestrator):
    result, _ = run_case(orchestrator, "TC001")
    assert result.decision is None, "Must stop before any claim decision"
    assert result.status == "NEEDS_RESUBMISSION"
    assert result.document_issues, "Must surface a document issue"
    messages = " ".join(i.message for i in result.document_issues)
    # Message must name the uploaded type and the required type.
    assert "PRESCRIPTION" in messages
    assert "HOSPITAL_BILL" in messages
    # Adjudication must never have run.
    assert not any(s.stage == "adjudication" for s in result.trace)


def test_tc002_unreadable_document(orchestrator):
    result, _ = run_case(orchestrator, "TC002")
    assert result.decision is None
    assert result.status == "NEEDS_RESUBMISSION"
    issues = [i for i in result.document_issues if i.issue_code == "UNREADABLE"]
    assert issues, "Must identify the unreadable document"
    issue = issues[0]
    # Must name the specific document and ask for re-upload, not reject.
    assert issue.file_id == "F004"
    assert "re-upload" in issue.message.lower() or "re-upload" in issue.action_required.lower()
    assert "not been rejected" in issue.message.lower()
    assert "PHARMACY_BILL" in issue.message


def test_tc003_different_patients(orchestrator):
    result, _ = run_case(orchestrator, "TC003")
    assert result.decision is None
    assert result.status == "NEEDS_RESUBMISSION"
    issues = [i for i in result.document_issues if i.issue_code == "PATIENT_MISMATCH"]
    assert issues, "Must detect documents belonging to different people"
    # Must surface the specific names found on each document.
    assert "Rajesh Kumar" in issues[0].message
    assert "Arjun Mehta" in issues[0].message
    assert not any(s.stage == "adjudication" for s in result.trace)


# ---------------------------------------------------------------------------
# TC004: clean consultation, full approval with 10% co-pay.
# ---------------------------------------------------------------------------

def test_tc004_clean_consultation_full_approval(orchestrator):
    result, _ = run_case(orchestrator, "TC004")
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 1350
    assert result.confidence_score > 0.85
    b = result.amount_breakdown
    assert b is not None
    assert b.copay_percent == 10
    assert b.copay_amount == 150


# ---------------------------------------------------------------------------
# TC005: diabetes within 90-day waiting period. Must state eligibility date.
# ---------------------------------------------------------------------------

def test_tc005_waiting_period_diabetes(orchestrator):
    result, _ = run_case(orchestrator, "TC005")
    assert result.decision == Decision.REJECTED
    assert RejectionReason.WAITING_PERIOD in result.rejection_reasons
    # Join 2024-09-01 + 90 days = 2024-11-30. Must be stated.
    all_text = " ".join(result.reasons) + " ".join(s.detail for s in result.trace)
    assert "2024-11-30" in all_text, "Must state the date the member becomes eligible"


# ---------------------------------------------------------------------------
# TC006: dental partial. Root canal approved, whitening rejected, per-item reasons.
# ---------------------------------------------------------------------------

def test_tc006_dental_partial(orchestrator):
    result, _ = run_case(orchestrator, "TC006")
    assert result.decision == Decision.PARTIAL
    assert result.approved_amount == 8000
    assert len(result.line_items) == 2
    by_desc = {li.description: li for li in result.line_items}
    rc = by_desc["Root Canal Treatment"]
    tw = by_desc["Teeth Whitening"]
    assert rc.status == "APPROVED" and rc.approved_amount == 8000
    assert tw.status == "REJECTED" and tw.approved_amount == 0
    assert tw.reason, "Rejected line item must carry its own reason"


# ---------------------------------------------------------------------------
# TC007: MRI 15,000 without pre-auth. Decisive reason PRE_AUTH_MISSING only.
# ---------------------------------------------------------------------------

def test_tc007_mri_without_pre_auth(orchestrator):
    result, _ = run_case(orchestrator, "TC007")
    assert result.decision == Decision.REJECTED
    assert result.rejection_reasons == [RejectionReason.PRE_AUTH_MISSING]
    text = " ".join(result.reasons)
    assert "pre-authorization" in text.lower()
    # Must tell the member what to do to resubmit.
    assert "resubmit" in text.lower()


# ---------------------------------------------------------------------------
# TC008: consultation 7,500 over the 5,000 per-claim limit.
# ---------------------------------------------------------------------------

def test_tc008_per_claim_limit(orchestrator):
    result, _ = run_case(orchestrator, "TC008")
    assert result.decision == Decision.REJECTED
    assert RejectionReason.PER_CLAIM_EXCEEDED in result.rejection_reasons
    text = " ".join(result.reasons)
    # Must state both the limit and the claimed amount.
    assert "5,000" in text and "7,500" in text


# ---------------------------------------------------------------------------
# TC009: 4th same-day claim. Manual review with specific signals, no auto-reject.
# ---------------------------------------------------------------------------

def test_tc009_same_day_fraud_signal(orchestrator):
    result, _ = run_case(orchestrator, "TC009")
    assert result.decision == Decision.MANUAL_REVIEW
    assert result.fraud_signals, "Must include the specific triggering signals"
    signals = " ".join(result.fraud_signals)
    assert "4" in signals and "same-day" in signals.lower()


# ---------------------------------------------------------------------------
# TC010: network discount before co-pay. 4500 -> 3600 -> 3240, breakdown shown.
# ---------------------------------------------------------------------------

def test_tc010_network_discount_order(orchestrator):
    result, _ = run_case(orchestrator, "TC010")
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 3240
    b = result.amount_breakdown
    assert b.network_discount_percent == 20
    assert b.network_discount_amount == 900
    assert b.amount_after_discount == 3600
    assert b.copay_percent == 10
    assert b.copay_amount == 360
    assert b.approved_amount == 3240


# ---------------------------------------------------------------------------
# TC011: simulated component failure. No crash, visible failure, lower
# confidence, manual review recommended, still APPROVED.
# ---------------------------------------------------------------------------

def test_tc011_graceful_degradation(orchestrator):
    result, _ = run_case(orchestrator, "TC011")
    assert result.decision == Decision.APPROVED
    assert result.component_failures, "Failure must be visible in the output"
    assert result.manual_review_recommended is True
    text = " ".join(result.reasons).lower()
    assert "manual review" in text
    # Confidence must be lower than a normal full-pipeline approval (TC004).
    clean, _ = run_case(orchestrator, "TC004")
    assert result.confidence_score < clean.confidence_score


# ---------------------------------------------------------------------------
# TC012: excluded obesity treatment. Confident rejection.
# ---------------------------------------------------------------------------

def test_tc012_excluded_treatment(orchestrator):
    result, _ = run_case(orchestrator, "TC012")
    assert result.decision == Decision.REJECTED
    assert RejectionReason.EXCLUDED_CONDITION in result.rejection_reasons
    assert result.confidence_score > 0.90


# ---------------------------------------------------------------------------
# PART 3 byte-identical guard: the new weighted fraud score must never be what
# reroutes an official case. None of the 12 may reach the review threshold, so
# the score only ever adds a trace line for them — TC009 still routes via the
# binary same-day signal (its score is elevated but below threshold).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("case_id", list(_CASES.keys()))
def test_no_official_case_crosses_fraud_score_threshold(orchestrator, case_id):
    threshold = load_policy().fraud_thresholds.fraud_score_manual_review_threshold
    result, _ = run_case(orchestrator, case_id)
    assert result.fraud_score < threshold, (
        f"{case_id} fraud_score {result.fraud_score} reached the review "
        f"threshold {threshold}; the score must not reroute an official case.")
