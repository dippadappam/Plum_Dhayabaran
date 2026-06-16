"""Component unit tests beyond the 12 official cases.

The LLM is never called: a mock implements the DocumentExtractor protocol.
Covers the live-extraction path, extractor failure degradation, the
confidence formula, the condition mapper, and policy loader integrity.
"""

import threading

import pytest

from app.condition_mapping import map_text_to_conditions, match_in_list
from app.models.claim import ClaimSubmission
from app.models.decision import Decision, RejectionReason
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


@pytest.fixture(scope="module")
def policy():
    return load_policy()


class FakeExtractor:
    """Mock DocumentExtractor: returns canned fields, no network. The call
    counter is lock-guarded because extraction now runs documents in parallel
    threads."""

    def __init__(self, fields=None, fail=False):
        self.fields = fields or {}
        self.fail = fail
        self.calls = 0
        self._lock = threading.Lock()

    def extract(self, document):
        with self._lock:
            self.calls += 1
        if self.fail:
            raise RuntimeError("simulated extractor outage")
        return dict(self.fields)


def _base_claim(**overrides):
    base = {
        "member_id": "EMP001",
        "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "treatment_date": "2024-11-01",
        "claimed_amount": 1500,
        "documents": [
            # Distinct bytes per document: extraction dedupes by content
            # hash, and two different documents have different content.
            {"file_id": "L1", "actual_type": "PRESCRIPTION",
             "file_data": "ZmFrZS1yeA==", "media_type": "image/jpeg"},
            {"file_id": "L2", "actual_type": "HOSPITAL_BILL",
             "file_data": "ZmFrZS1iaWxs", "media_type": "image/jpeg"},
        ],
    }
    base.update(overrides)
    return ClaimSubmission.model_validate(base)


def test_live_extraction_path_feeds_adjudication(policy):
    extractor = FakeExtractor(fields={
        "patient_name": "Rajesh Kumar",
        "diagnosis": "Viral Fever",
        "line_items": [{"description": "Consultation Fee", "amount": 1000},
                       {"description": "CBC Test", "amount": 500}],
        "total": 1500,
        "_extraction_confidence": 0.9,
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    result = orch.process(_base_claim())
    assert extractor.calls == 2
    assert result.decision == Decision.APPROVED
    # 10% consultation co-pay on 1500.
    assert result.approved_amount == 1350
    # Extraction confidence (0.9) propagates: 0.95 * 0.9 = 0.855 -> 0.85/0.86.
    assert result.confidence_score < 0.95


def test_extractor_outage_degrades_not_crashes(policy):
    orch = ClaimsOrchestrator(policy=policy, extractor=FakeExtractor(fail=True))
    result = orch.process(_base_claim())
    # Must not raise; failure visible, manual review recommended.
    assert result.component_failures
    assert result.manual_review_recommended is True
    assert result.confidence_score < 0.70


def test_no_extractor_configured_is_skipped_not_fatal(policy):
    orch = ClaimsOrchestrator(policy=policy, extractor=None)
    result = orch.process(_base_claim())
    assert result.decision is not None  # pipeline still concludes


def test_condition_mapping_word_boundaries():
    # 'herniation' must not match the 'hernia' waiting condition.
    m = map_text_to_conditions("Suspected Lumbar Disc Herniation")
    assert "hernia" not in m.matched_waiting_conditions
    m2 = map_text_to_conditions("Inguinal hernia repair")
    assert "hernia" in m2.matched_waiting_conditions


def test_comorbidity_line_item_does_not_trigger_waiting_period(policy):
    """A comorbidity mentioned in a LINE ITEM (not the primary diagnosis) must
    not trigger a waiting-period rejection of an unrelated claim. EMP005 is
    inside the diabetes window, but this claim is for a viral fever — a
    'diabetic' line item must not reject it. (TC005, where diabetes IS the
    primary diagnosis, still rejects — covered by the official suite.)"""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP005", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-10-15",
        "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Vikram Joshi", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Vikram Joshi",
              "line_items": [{"description": "Consultation Fee", "amount": 1000},
                             {"description": "Diabetic foot screening", "amount": 500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 1350
    assert not any(s.check.startswith("waiting_period:") for s in result.trace)


def test_condition_mapping_diabetes_and_exclusions():
    m = map_text_to_conditions("Type 2 Diabetes Mellitus")
    assert m.matched_waiting_conditions == ["diabetes"]
    m2 = map_text_to_conditions("Bariatric Consultation and Customised Diet Plan")
    assert "Bariatric surgery" in m2.matched_exclusions


def test_match_in_list_case_insensitive():
    assert match_in_list("teeth whitening", ["Teeth Whitening"]) == "Teeth Whitening"
    assert match_in_list("Root Canal Treatment", ["Teeth Whitening"]) is None


def test_match_in_list_one_direction_no_generic_over_reject():
    """Excluded matching is one-direction: the policy phrase must be in the line
    text. A generic 'Treatment' line must NOT match the longer excluded
    'Orthodontic Treatment (Braces)'; a line containing the full phrase does."""
    assert match_in_list("Treatment", ["Orthodontic Treatment (Braces)"]) is None
    assert match_in_list("Teeth Whitening Procedure", ["Teeth Whitening"]) == \
        "Teeth Whitening"


def test_dental_non_covered_line_routes_to_review(policy):
    """A dental line on neither the covered nor the excluded list is not
    auto-paid — it routes to review for a human to confirm coverage."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP002", "policy_id": "PLUM_GHI_2024",
        "claim_category": "DENTAL", "treatment_date": "2024-10-15",
        "claimed_amount": 5000,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Priya Singh",
              "line_items": [{"description": "Wisdom Tooth Surgery", "amount": 5000}],
              "total": 5000}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW


def test_dental_partial_label_matches_covered_via_token_overlap(policy):
    """A partial covered-procedure label ('Root Canal') still matches the
    covered entry ('Root Canal Treatment') via token overlap and is approved,
    not bounced to review."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP002", "policy_id": "PLUM_GHI_2024",
        "claim_category": "DENTAL", "treatment_date": "2024-10-15",
        "claimed_amount": 6000,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Priya Singh",
              "line_items": [{"description": "Root Canal", "amount": 6000}],
              "total": 6000}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 6000   # dental, 0% co-pay


def test_unknown_member_rejected(policy):
    orch = ClaimsOrchestrator(policy=policy)
    claim = _base_claim(member_id="EMP999")
    result = orch.process(claim)
    assert result.decision == Decision.REJECTED
    assert any("EMP999" in r for r in result.reasons)


def test_below_minimum_amount_rejected(policy):
    orch = ClaimsOrchestrator(policy=policy)
    claim = _base_claim(claimed_amount=300)  # minimum is 500
    result = orch.process(claim)
    assert result.decision == Decision.REJECTED
    assert any("minimum" in r.lower() for r in result.reasons)


def test_is_network_hospital_token_subset(policy):
    """Network match is token-subset: an address/branch suffix still matches,
    a short partial token does not over-match."""
    assert policy.is_network_hospital("Apollo Hospitals")
    assert policy.is_network_hospital("Apollo Hospitals, Bannerghatta Road, Bengaluru")
    assert policy.is_network_hospital("Max Healthcare Saket")
    assert not policy.is_network_hospital("Apollo Pharmacy")
    assert not policy.is_network_hospital("Max")     # partial token, not the hospital
    assert not policy.is_network_hospital("City Clinic, Bengaluru")


def test_network_discount_applies_with_address_suffix(policy):
    """A network hospital name with a branch/address suffix still earns the
    network discount: Apollo at 4,500 -> 20% -> 3,600 -> 10% co-pay -> 3,240."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP010", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-03",
        "claimed_amount": 4500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Deepak Shah", "diagnosis": "Acute Bronchitis"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Deepak Shah",
              "hospital_name": "Apollo Hospitals, Bannerghatta Road, Bengaluru",
              "line_items": [{"description": "Consultation Fee", "amount": 1500},
                             {"description": "Medicines", "amount": 3000}],
              "total": 4500}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 3240
    assert result.amount_breakdown.network_discount_amount == 900


def test_policy_loader_reads_runtime_values(policy):
    # Spot-check that nothing is hardcoded: values come from the JSON.
    assert policy.coverage.per_claim_limit == 5000
    assert policy.waiting_periods.specific_conditions["diabetes"] == 90
    assert "Apollo Hospitals" in policy.network_hospitals
    assert policy.fraud_thresholds.same_day_claims_limit == 2


def test_trace_reconstructs_decision(policy):
    """Observability contract: the trace alone explains the outcome."""
    orch = ClaimsOrchestrator(policy=policy)
    claim = _base_claim()
    result = orch.process(claim)
    stages = {s.stage for s in result.trace}
    assert {"orchestrator", "intake", "document_verification",
            "extraction", "adjudication"} <= stages
    assert all(s.detail for s in result.trace), "Every step carries a detail"


class PerDocExtractor:
    """Mock DocumentExtractor returning canned fields keyed by file_id."""

    def __init__(self, by_id):
        self.by_id = by_id
        self.calls = 0

    def extract(self, document):
        self.calls += 1
        return dict(self.by_id[document.file_id])


def test_parallel_extraction_reads_all_documents_in_order(policy):
    """Multiple distinct documents are extracted (in parallel) and each file_id
    gets its own extracted record in submission order — the parallel rewrite
    preserves per-document assembly."""
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "diagnosis": "Viral Fever", "date": "2024-11-01",
               "readability": "GOOD", "_extraction_confidence": 0.95},
        "lab": {"document_type": "LAB_REPORT", "patient_name": "Rajesh Kumar",
                "test_name": "CBC", "readability": "GOOD",
                "_extraction_confidence": 0.95},
        "bill": {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
                 "hospital_name": "City Clinic", "date": "2024-11-01",
                 "line_items": [{"description": "Consultation Fee", "amount": 1000}],
                 "total": 1000, "readability": "GOOD", "_extraction_confidence": 0.95},
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "rx", "file_data": "cng=", "media_type": "image/jpeg"},
            {"file_id": "lab", "file_data": "bGFi", "media_type": "image/jpeg"},
            {"file_id": "bill", "file_data": "YmlsbA==", "media_type": "image/jpeg"},
        ]}))
    ids = [d["file_id"] for d in result.extracted_documents]
    assert ids == ["rx", "lab", "bill"]   # all read, submission order preserved
    assert result.decision == Decision.APPROVED


def test_primary_diagnosis_field_scopes_waiting_period(policy):
    """PART 2B: the waiting check reads primary_diagnosis, not the full
    diagnosis text. EMP005 is inside the diabetes window; this visit's
    primary_diagnosis is a viral fever, with diabetes present only as recorded
    history (in the full diagnosis line and the comorbidities field). Reading
    primary_diagnosis approves it; reading the full diagnosis line — which names
    'diabetic' — would wrongly trip the diabetes waiting period."""
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Vikram Joshi",
               "primary_diagnosis": "Viral Fever",
               "diagnosis": "Viral fever; k/c/o diabetic, on metformin",
               "comorbidities": "Type 2 Diabetes Mellitus",
               "date": "2024-10-15", "readability": "GOOD",
               "_extraction_confidence": 0.95},
        "bill": {"document_type": "HOSPITAL_BILL", "patient_name": "Vikram Joshi",
                 "hospital_name": "City Clinic", "date": "2024-10-15",
                 "line_items": [{"description": "Consultation Fee", "amount": 1000}],
                 "total": 1000, "readability": "GOOD",
                 "_extraction_confidence": 0.95},
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP005", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "rx", "file_data": "cng=", "media_type": "image/jpeg"},
            {"file_id": "bill", "file_data": "YmlsbA==", "media_type": "image/jpeg"},
        ]}))
    assert result.decision == Decision.APPROVED
    assert not any(s.check.startswith("waiting_period:") for s in result.trace)


def test_low_amount_confidence_holds_even_when_overall_confidence_high(policy):
    """PART 2B: the gate keys on amount-field confidence specifically. A bill
    read with HIGH overall confidence but LOW confidence in the amounts is held
    for review — the amount drives the payout, so an uncertain amount is held
    even when the rest of the document reads cleanly."""
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "primary_diagnosis": "Viral Fever", "date": "2024-11-01",
               "readability": "GOOD", "_extraction_confidence": 0.96},
        "bill": {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
                 "hospital_name": "City Clinic", "date": "2024-11-01",
                 "line_items": [{"description": "Consultation Fee", "amount": 1500}],
                 "total": 1500, "readability": "GOOD",
                 "_extraction_confidence": 0.96,   # overall: a clean read
                 "amount_confidence": 0.40},        # but the amounts are uncertain
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "rx", "file_data": "cng=", "media_type": "image/jpeg"},
            {"file_id": "bill", "file_data": "YmlsbA==", "media_type": "image/jpeg"},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert any(s.stage == "confidence_gate" and "amount-field confidence" in s.detail
               for s in result.trace)


def test_extracted_type_overrides_declared_and_gate_catches(policy):
    """Real-upload path: both files are actually prescriptions per the
    extractor, even though one is *declared* a hospital bill. The gate must
    use the EXTRACTED document_type and stop the claim for the missing
    HOSPITAL_BILL — before any adjudication."""
    extractor = PerDocExtractor({
        "U1": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "readability": "GOOD", "_extraction_confidence": 0.95},
        "U2": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "readability": "GOOD", "_extraction_confidence": 0.95},
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    claim = ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "U1", "actual_type": "PRESCRIPTION",
             "file_data": "ZmFrZS1yeDE=", "media_type": "image/jpeg"},
            # Declared a hospital bill, but the extractor detects a prescription.
            {"file_id": "U2", "actual_type": "HOSPITAL_BILL",
             "file_data": "ZmFrZS1yeDI=", "media_type": "image/jpeg"},
        ],
    })
    result = orch.process(claim)
    assert result.decision is None
    assert result.status == "NEEDS_RESUBMISSION"
    assert any(i.issue_code in ("WRONG_TYPE", "MISSING_REQUIRED")
               for i in result.document_issues)
    assert "HOSPITAL_BILL" in " ".join(i.message for i in result.document_issues)
    assert not any(s.stage == "adjudication" for s in result.trace)


def test_claimed_amount_derived_from_extracted_bill(policy):
    """Real-upload path: no amount/date/hospital is typed; all are read from
    the documents. The bill total drives claimed_amount and the deterministic
    co-pay math is applied to it unchanged."""
    extractor = PerDocExtractor({
        "U1": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "diagnosis": "Viral Fever", "date": "2024-11-01",
               "readability": "GOOD", "_extraction_confidence": 0.95},
        "U2": {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
               "hospital_name": "City Clinic, Bengaluru", "date": "2024-11-01",
               "line_items": [{"description": "Consultation Fee", "amount": 1000},
                              {"description": "CBC Test", "amount": 500}],
               "total": 1500, "readability": "GOOD", "_extraction_confidence": 0.95},
    })
    orch = ClaimsOrchestrator(policy=policy, extractor=extractor)
    claim = ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "U1", "actual_type": "PRESCRIPTION",
             "file_data": "ZmFrZS1yeA==", "media_type": "image/jpeg"},
            {"file_id": "U2", "actual_type": "HOSPITAL_BILL",
             "file_data": "ZmFrZS1iaWxs", "media_type": "image/jpeg"},
        ],
    })
    assert claim.claimed_amount is None  # nothing typed by the member
    result = orch.process(claim)
    assert result.decision == Decision.APPROVED
    # Amount derived from the bill total (1500); 10% consultation co-pay -> 1350.
    assert result.amount_breakdown.claimed_amount == 1500
    assert result.approved_amount == 1350
    # The submission was populated with the derived effective values.
    assert claim.claimed_amount == 1500
    assert str(claim.treatment_date) == "2024-11-01"


def test_claimed_amount_sums_across_multiple_bills(policy):
    """Multi-bill derived claim: claimed_amount is the SUM of all bill
    documents, matching the line-item aggregation, so claimed == eligible
    (not just the first bill's total)."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "hospital_name": "City Clinic",
              "line_items": [{"description": "Consultation Fee", "amount": 1000}],
              "total": 1000}},
            {"file_id": "F3", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "hospital_name": "City Clinic",
              "line_items": [{"description": "Follow-up Consultation", "amount": 1000}],
              "total": 1000}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.amount_breakdown.claimed_amount == 2000   # sum of both bills
    assert result.amount_breakdown.eligible_amount == 2000  # claimed == eligible
    assert result.approved_amount == 1800                   # 10% consultation co-pay


def test_parse_flexible_date_formats_and_ambiguity():
    """The derived-path date parser: ISO and named-month parse directly; a
    day>12 numeric date is unambiguous DD-MM; a day<=12 numeric date that reads
    validly both ways is flagged ambiguous; junk yields no date."""
    from datetime import date as _date
    from app.agents.derivation import parse_flexible_date
    assert parse_flexible_date("2024-11-01") == (_date(2024, 11, 1), False)
    assert parse_flexible_date("2024-11-1") == (_date(2024, 11, 1), False)   # non-padded
    assert parse_flexible_date("15-11-2024") == (_date(2024, 11, 15), False)  # DD-MM, day>12
    assert parse_flexible_date("01-Nov-2024") == (_date(2024, 11, 1), False)  # named month
    assert parse_flexible_date("01 Nov 2024") == (_date(2024, 11, 1), False)
    assert parse_flexible_date("05-11-2024") == (None, True)   # 5 Nov vs 11 May -> ambiguous
    assert parse_flexible_date("not a date") == (None, False)


def test_dd_mm_treatment_date_parsed_unambiguously(policy):
    """A DD-MM date with day>12 (no ISO) is parsed and the claim adjudicates —
    the ISO-only path previously held it as 'no treatment date'."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "hospital_name": "City Clinic",
              "date": "15-11-2024",
              "line_items": [{"description": "Consultation Fee", "amount": 1000}],
              "total": 1000}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 900   # 1000 less 10% consultation co-pay


def test_ambiguous_treatment_date_routes_to_review(policy):
    """A numeric date that reads validly as both DD-MM and MM-DD is not guessed
    — it routes to manual review for a human to confirm the date."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "hospital_name": "City Clinic",
              "date": "05-11-2024",
              "line_items": [{"description": "Consultation Fee", "amount": 1000}],
              "total": 1000}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert any("ambiguous" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Batch 2: confidence-gate PARTIAL relaxation, future-date check, format msg.
# (The "PARTIAL holds" assertion previously lived only in an inline mock
# check, never in the committed suite; these formalize the new behavior.)
# ---------------------------------------------------------------------------

def _two_doc_live_claim(bill_readability="GOOD", bill_conf=0.95):
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "diagnosis": "Viral Fever", "date": "2024-11-01",
               "readability": "GOOD", "_extraction_confidence": 0.95},
        "bill": {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
                 "hospital_name": "City Clinic", "date": "2024-11-01",
                 "line_items": [{"description": "Consultation Fee", "amount": 1000},
                                {"description": "CBC Test", "amount": 500}],
                 "total": 1500, "readability": bill_readability,
                 "_extraction_confidence": bill_conf},
    })
    submission = ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "rx", "file_data": "ZmFrZS1yeA==", "media_type": "image/jpeg"},
            {"file_id": "bill", "file_data": "ZmFrZS1iaWxs", "media_type": "image/jpeg"},
        ]})
    return extractor, submission


def test_partial_readability_high_confidence_passes(policy):
    """A high-confidence PARTIAL bill (a normal stamped/folded clinic bill that
    was still read) no longer holds — PARTIAL alone is not a hold."""
    extractor, submission = _two_doc_live_claim(bill_readability="PARTIAL",
                                                bill_conf=0.95)
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(submission)
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 1350


def test_low_confidence_document_still_holds(policy):
    """A low-confidence read still holds (the confidence trigger, not the
    PARTIAL flag, does the holding)."""
    extractor, submission = _two_doc_live_claim(bill_readability="PARTIAL",
                                                bill_conf=0.6)
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(submission)
    assert result.decision == Decision.MANUAL_REVIEW


def test_unreadable_document_still_holds(policy):
    """An UNREADABLE decision-critical document still holds (caught at the
    document gate as a re-upload request — never paid)."""
    extractor, submission = _two_doc_live_claim(bill_readability="UNREADABLE",
                                                bill_conf=0.95)
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(submission)
    assert result.decision is None
    assert result.status == "NEEDS_RESUBMISSION"


def test_future_treatment_date_routes_to_review(policy):
    """Treatment date after the submission date is impossible — held for
    review (deterministic: both dates are provided inputs)."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-12-01",
        "submission_date": "2024-11-15", "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Consultation Fee", "amount": 1500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert any(s.check == "future_treatment_date" for s in result.trace)


def test_past_treatment_date_is_not_flagged(policy):
    """A normal past treatment date is not flagged by the future-date check."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "submission_date": "2024-11-15", "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Consultation Fee", "amount": 1500}],
              "total": 1500}},
        ]}))
    assert result.decision == Decision.APPROVED
    assert not any(s.check == "future_treatment_date" for s in result.trace)


def test_unsupported_media_type_has_clear_message():
    """An unsupported photo format (e.g. iPhone HEIC) is rejected with a
    clear, member-facing message, not a raw validation string."""
    import pytest as _pytest
    from pydantic import ValidationError
    with _pytest.raises(ValidationError) as exc:
        ClaimSubmission.model_validate({
            "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
            "claim_category": "CONSULTATION",
            "documents": [{"file_id": "F1", "file_data": "eA==",
                           "media_type": "image/heic"}],
        })
    msg = str(exc.value)
    assert "not supported" in msg and "JPG, PNG, or PDF" in msg


# ---------------------------------------------------------------------------
# Batch 3 — C4 pre-authorization
# ---------------------------------------------------------------------------

def _diagnostic_mri_live(with_pre_auth=False):
    """3-doc DIAGNOSTIC claim with an MRI over the pre-auth threshold, live
    upload path (file_data + mock extractor)."""
    bill = {"document_type": "HOSPITAL_BILL", "patient_name": "Suresh Patil",
            "date": "2024-11-02",
            "line_items": [{"description": "MRI Lumbar Spine", "amount": 15000}],
            "total": 15000, "readability": "GOOD", "_extraction_confidence": 0.95}
    if with_pre_auth:
        bill["pre_auth_number"] = "AUTH-2024-0001"
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Suresh Patil",
               "diagnosis": "Suspected Lumbar Disc Herniation",
               "tests_ordered": ["MRI Lumbar Spine"], "date": "2024-11-02",
               "readability": "GOOD", "_extraction_confidence": 0.95},
        "lab": {"document_type": "LAB_REPORT", "patient_name": "Suresh Patil",
                "test_name": "MRI Lumbar Spine", "date": "2024-11-02",
                "readability": "GOOD", "_extraction_confidence": 0.95},
        "bill": bill,
    })
    submission = ClaimSubmission.model_validate({
        "member_id": "EMP007", "policy_id": "PLUM_GHI_2024",
        "claim_category": "DIAGNOSTIC",
        "documents": [
            {"file_id": "rx", "file_data": "ZmFrZS1yeA==", "media_type": "image/jpeg"},
            {"file_id": "lab", "file_data": "ZmFrZS1sYWI=", "media_type": "image/jpeg"},
            {"file_id": "bill", "file_data": "ZmFrZS1iaWxs", "media_type": "image/jpeg"},
        ]})
    return extractor, submission


def test_annual_limit_accrues_eligible_not_gross_claimed(policy):
    """The annual OPD limit accrues the post-exclusion eligible amount, not the
    gross claimed. A dental claim of 12,000 with 4,000 excluded (whitening) is
    an 8,000 eligible claim; with YTD 41,000 the eligible total (49,000) is
    within the 50,000 annual limit, where the gross (53,000) would wrongly
    reject it."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP002", "policy_id": "PLUM_GHI_2024",
        "claim_category": "DENTAL", "treatment_date": "2024-10-15",
        "claimed_amount": 12000, "ytd_claims_amount": 41000,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Priya Singh",
              "line_items": [{"description": "Root Canal Treatment", "amount": 8000},
                             {"description": "Teeth Whitening", "amount": 4000}],
              "total": 12000}},
        ]}))
    assert result.decision == Decision.PARTIAL
    assert result.approved_amount == 8000


def test_high_value_claim_routes_to_review(policy):
    """A claim above the high-value auto-review threshold (25,000) is held for a
    reviewer rather than auto-rejected by the per-claim cap. No official case
    exceeds 25,000, so this is byte-identical for them."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "PHARMACY", "treatment_date": "2024-11-01",
        "claimed_amount": 30000,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Chronic condition"}},
            {"file_id": "F2", "actual_type": "PHARMACY_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Specialty Medicines", "amount": 30000}],
              "total": 30000}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert any("high-value" in r.lower() for r in result.reasons)


def test_total_only_dental_bill_routes_to_review(policy):
    """A dental bill with no itemized lines (total only) cannot be checked for
    covered/excluded procedures, so it is held for review, not auto-paid.
    Consultation total-only bills (TC005/TC009) are unaffected."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP002", "policy_id": "PLUM_GHI_2024",
        "claim_category": "DENTAL", "treatment_date": "2024-10-15",
        "claimed_amount": 5000,
        "documents": [
            {"file_id": "F1", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Priya Singh", "total": 5000}},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW


def test_dental_report_typed_doc_satisfies_dental_bill_requirement(policy):
    """The dental document gate accepts a DENTAL_REPORT-typed document as the
    required hospital bill, so a dental-clinic bill the extractor mis-types is
    not wrongly bounced. Scoped to DENTAL (TC001 still flags a missing
    consultation bill — covered by the official suite)."""
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP002", "policy_id": "PLUM_GHI_2024",
        "claim_category": "DENTAL", "treatment_date": "2024-10-15",
        "claimed_amount": 8000,
        "documents": [
            {"file_id": "F1", "actual_type": "DENTAL_REPORT", "content":
             {"patient_name": "Priya Singh", "treatment": "Root Canal Treatment"}},
        ]}))
    assert result.status == "DECIDED"   # the gate did not bounce it
    assert any(s.stage == "adjudication" for s in result.trace)


def test_global_pre_auth_catches_high_value_test_under_any_category(policy):
    """An MRI over the pre-auth threshold must require pre-auth even when the
    claim's resolved category (here CONSULTATION) has no pre-auth config of its
    own — pre-auth is enforced on the line, not the category. Without global
    pre-auth the decisive reason would be the per-claim cap, masking the missing
    pre-auth; with it, the correct PRE_AUTH_MISSING is reported. (The pharmacy
    variant is separately caught by the consistency category check.)"""
    from app.models.decision import RejectionReason
    orch = ClaimsOrchestrator(policy=policy)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-02",
        "claimed_amount": 13000,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Headache"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Consultation Fee", "amount": 1000},
                             {"description": "MRI Brain", "amount": 12000}],
              "total": 13000}},
        ]}))
    assert result.decision == Decision.REJECTED
    assert RejectionReason.PRE_AUTH_MISSING in result.rejection_reasons


def test_c4_live_no_preauth_holds(policy):
    """Live upload, MRI over threshold, no pre-auth found: held for review
    (the pre-auth document may not have extracted), not auto-rejected."""
    extractor, submission = _diagnostic_mri_live(with_pre_auth=False)
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(submission)
    assert result.decision == Decision.MANUAL_REVIEW
    assert not result.rejection_reasons


def test_c4_present_preauth_string_holds(policy):
    """A present pre-auth reference no longer auto-satisfies — it cannot be
    verified without a registry, so it is held for a human."""
    extractor, submission = _diagnostic_mri_live(with_pre_auth=True)
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(submission)
    assert result.decision == Decision.MANUAL_REVIEW


def test_c4_structured_no_preauth_rejects(policy):
    """Structured/known path, MRI over threshold, no pre-auth: absence is
    definitive, so it remains a REJECTED PRE_AUTH_MISSING (official contract)."""
    from app.models.decision import RejectionReason
    result = ClaimsOrchestrator(policy=policy).process(ClaimSubmission.model_validate({
        "member_id": "EMP007", "policy_id": "PLUM_GHI_2024",
        "claim_category": "DIAGNOSTIC", "treatment_date": "2024-11-02",
        "claimed_amount": 15000,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"diagnosis": "Suspected Lumbar Disc Herniation",
              "tests_ordered": ["MRI Lumbar Spine"]}},
            {"file_id": "F2", "actual_type": "LAB_REPORT", "content":
             {"test_name": "MRI Lumbar Spine"}},
            {"file_id": "F3", "actual_type": "HOSPITAL_BILL", "content":
             {"line_items": [{"description": "MRI Lumbar Spine", "amount": 15000}],
              "total": 15000}},
        ]}))
    assert result.decision == Decision.REJECTED
    assert RejectionReason.PRE_AUTH_MISSING in result.rejection_reasons


# ---------------------------------------------------------------------------
# Batch 3 — H4 amount sanity & reconciliation
# ---------------------------------------------------------------------------

def _consult_with_bill(policy, line_items, total, claimed=1500):
    return ClaimsOrchestrator(policy=policy).process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": claimed,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "line_items": line_items,
              "total": total}},
        ]}))


def test_h4_exact_match_passes(policy):
    r = _consult_with_bill(
        policy, [{"description": "Consultation Fee", "amount": 1000},
                 {"description": "CBC Test", "amount": 500}], total=1500)
    assert r.decision == Decision.APPROVED
    assert r.approved_amount == 1350


def test_h4_within_20pct_passes(policy):
    # total 1700 vs itemized 1500: ~12% gap (a normal discount/tax/fee) -> pass.
    r = _consult_with_bill(
        policy, [{"description": "Consultation Fee", "amount": 1000},
                 {"description": "CBC Test", "amount": 500}], total=1700,
        claimed=1700)
    assert r.decision == Decision.APPROVED


def test_h4_over_20pct_holds(policy):
    # total 3000 vs itemized 1500: 50% gap -> hold for review.
    r = _consult_with_bill(
        policy, [{"description": "Consultation Fee", "amount": 1000},
                 {"description": "CBC Test", "amount": 500}], total=3000,
        claimed=3000)
    assert r.decision == Decision.MANUAL_REVIEW


def test_h4_over_ceiling_holds(policy):
    # A single line item above the sum insured (500,000) is absurd -> hold.
    r = _consult_with_bill(
        policy, [{"description": "Consultation Fee", "amount": 600000}],
        total=600000, claimed=600000)
    assert r.decision == Decision.MANUAL_REVIEW


def test_h4_negative_net_holds_and_does_not_underpay(policy):
    # Net-negative itemized bill is nonsensical -> hold; payout never negative.
    r = _consult_with_bill(
        policy, [{"description": "Consultation Fee", "amount": 1000},
                 {"description": "Adjustment", "amount": -1500}], total=-500,
        claimed=1000)
    assert r.decision == Decision.MANUAL_REVIEW
    assert (r.approved_amount or 0) >= 0


# ---------------------------------------------------------------------------
# 4a — lifecycle predicates (explicit over status, decision, review_status)
# ---------------------------------------------------------------------------

def test_lifecycle_predicates():
    from app.models.decision import (ClaimResult, Decision, is_final,
                                     is_reviewer_resolvable)

    def mk(**kw):
        base = dict(claim_reference="X", member_id="EMP001",
                    claim_category="CONSULTATION")
        base.update(kw)
        return ClaimResult(**base)

    auto = mk(status="DECIDED", decision=Decision.APPROVED)
    held = mk(status="DECIDED", decision=Decision.MANUAL_REVIEW,
              review_status="PENDING_REVIEW")
    resolved = mk(status="DECIDED", decision=Decision.APPROVED,
                  review_status="RESOLVED")
    resub = mk(status="NEEDS_RESUBMISSION", decision=None)

    assert is_final(auto) and not is_reviewer_resolvable(auto)
    assert is_reviewer_resolvable(held) and not is_final(held)
    assert is_final(resolved) and not is_reviewer_resolvable(resolved)
    # NEEDS_RESUBMISSION is neither final nor reviewer-resolvable.
    assert not is_final(resub) and not is_reviewer_resolvable(resub)


# ---------------------------------------------------------------------------
# 4c — per-claim AI cost cap
# ---------------------------------------------------------------------------

def test_extraction_cost_cap_routes_to_review(policy):
    from app.config import EngineConfig

    calls = {"n": 0}

    class CountingExtractor:
        def extract(self, document):
            calls["n"] += 1
            return {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
                    "date": "2024-11-01",
                    "line_items": [{"description": "Consultation Fee", "amount": 1000}],
                    "total": 1000, "readability": "GOOD", "_extraction_confidence": 0.95}

    cfg = EngineConfig(max_extraction_calls_per_claim=1)
    orch = ClaimsOrchestrator(policy=policy, extractor=CountingExtractor(), config=cfg)
    result = orch.process(ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [  # distinct bytes -> no dedupe, 3 paid calls without a cap
            {"file_id": "D1", "file_data": "ZA==", "media_type": "image/jpeg"},
            {"file_id": "D2", "file_data": "ZGQ=", "media_type": "image/jpeg"},
            {"file_id": "D3", "file_data": "ZGRk", "media_type": "image/jpeg"},
        ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert calls["n"] == 1                      # stopped after the cap
    assert any(cf.component == "extraction.cost_cap"
               for cf in result.component_failures)   # not a silent drop
    assert any(s.check == "cost_cap" for s in result.trace)


# ---------------------------------------------------------------------------
# PART 3 — transparent weighted fraud score
# ---------------------------------------------------------------------------

def _corroborated_fraud_claim():
    """A claim that stays at/under EVERY binary fraud gate (same-day = limit 2,
    monthly = limit 6, amount < high-value, no byte-duplicate) but whose
    graduated signals corroborate: a same-day near-duplicate re-file, six
    claims in the month, and an amount well above the member's own average."""
    return {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "claimed_amount": 2000, "treatment_date": "2024-11-01",
        "claims_history": [
            {"claim_id": "CLM_A", "date": "2024-11-01", "amount": 2000,
             "provider": "Apollo Clinic"},   # same-day AND near-duplicate
            {"claim_id": "CLM_B", "date": "2024-11-05", "amount": 100,
             "provider": "Apollo Clinic"},
            {"claim_id": "CLM_C", "date": "2024-11-10", "amount": 100,
             "provider": "Apollo Clinic"},
            {"claim_id": "CLM_D", "date": "2024-11-15", "amount": 100,
             "provider": "Apollo Clinic"},
            {"claim_id": "CLM_E", "date": "2024-11-20", "amount": 100,
             "provider": "Apollo Clinic"},
        ],
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Consultation Fee", "amount": 2000}],
              "total": 2000}},
        ]}


def test_fraud_score_routes_to_review_without_tripping_a_binary_gate(policy):
    """PART 3: the weighted score catches a corroborated pattern that no single
    binary gate trips. This claim sits at/under every binary threshold, yet the
    graduated signals corroborate above 0.80, so it routes to review on the
    score alone — not on any same-day/monthly/high-value gate."""
    threshold = policy.fraud_thresholds.fraud_score_manual_review_threshold
    result = ClaimsOrchestrator(policy=policy).process(
        ClaimSubmission.model_validate(_corroborated_fraud_claim()))
    assert result.fraud_score >= threshold
    assert result.decision == Decision.MANUAL_REVIEW
    # Routed by the score itself...
    assert any("fraud score" in fs.lower() for fs in result.fraud_signals)
    # ...not by any binary gate (none of which tripped).
    assert not any(("same-day limit" in fs) or ("monthly limit" in fs)
                   or ("high-value" in fs) for fs in result.fraud_signals)


def test_fraud_score_breakdown_is_transparent_and_sums(policy):
    """PART 3: every signal's contribution is visible in the trace, and the
    published score is exactly the sum of those contributions (weight ×
    sub-score) — the number is auditable, not a black box."""
    result = ClaimsOrchestrator(policy=policy).process(
        ClaimSubmission.model_validate(_corroborated_fraud_claim()))
    step = next(s for s in result.trace
                if s.stage == "adjudication" and s.check == "fraud_score")
    rows = step.data["breakdown"]
    assert {r["signal"] for r in rows} == {
        "same_day_frequency", "monthly_frequency",
        "amount_vs_history", "near_duplicate"}
    for r in rows:
        # contribution is weight × sub-score, rounded to 4 dp for display.
        assert r["contribution"] == round(r["weight"] * r["sub_score"], 4)
        assert r["signal"] in step.detail        # each signal named in the line
    assert abs(sum(r["contribution"] for r in rows) - result.fraud_score) < 1e-3


# ---------------------------------------------------------------------------
# Tier 1 / Fix 6 — submission deadline (submission_rules.deadline_days_from_treatment)
# ---------------------------------------------------------------------------

def _consult_claim_received_on(received_date):
    """A clean EMP001 consultation (TC004 shape) with an explicit received_date,
    so the deadline window can be exercised. The 12 official cases set no
    received_date and therefore skip the deadline check entirely (the API's
    auto-stamped submission_date does not count toward the deadline)."""
    return ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "received_date": received_date, "claimed_amount": 1500,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar",
              "line_items": [{"description": "Consultation Fee", "amount": 1500}],
              "total": 1500}},
        ]})


def test_submission_deadline_rejects_late_claim(policy):
    """Filed 45 days after treatment (> the 30-day policy deadline) -> REJECTED
    with SUBMISSION_DEADLINE_PASSED; the limit is read from policy_terms.json."""
    limit = policy.submission_rules.deadline_days_from_treatment
    orch = ClaimsOrchestrator(policy=policy)
    late = orch.process(_consult_claim_received_on("2024-12-16"))  # +45d
    assert late.decision == Decision.REJECTED
    assert RejectionReason.SUBMISSION_DEADLINE_PASSED in late.rejection_reasons
    assert str(limit) in " ".join(late.reasons)          # explainable: states the limit
    assert any(s.check == "submission_deadline" and s.status.value == "FAILED"
               for s in late.trace)


def test_submission_within_deadline_is_unaffected(policy):
    """Filed 10 days after treatment (within 30) -> the deadline does not fire;
    the claim approves exactly as the no-submission_date path would."""
    orch = ClaimsOrchestrator(policy=policy)
    ontime = orch.process(_consult_claim_received_on("2024-11-11"))  # +10d
    assert ontime.decision == Decision.APPROVED
    assert ontime.approved_amount == 1350
    assert RejectionReason.SUBMISSION_DEADLINE_PASSED not in ontime.rejection_reasons


# ---------------------------------------------------------------------------
# Tier 1 / Fix 2 — alt-med registered practitioner (requires_registered_practitioner)
# ---------------------------------------------------------------------------

def _altmed_claim(registration):
    """A clean alternative-medicine claim (TC011 shape, no simulated failure).
    `registration` set on the prescription, or None to omit it entirely."""
    rx = {"doctor_name": "Vaidya T. Krishnan", "diagnosis": "Chronic Joint Pain",
          "treatment": "Panchakarma Therapy"}
    if registration is not None:
        rx["doctor_registration"] = registration
    return ClaimSubmission.model_validate({
        "member_id": "EMP006", "policy_id": "PLUM_GHI_2024",
        "claim_category": "ALTERNATIVE_MEDICINE", "treatment_date": "2024-10-28",
        "claimed_amount": 4000,
        "documents": [
            {"file_id": "RX", "actual_type": "PRESCRIPTION", "content": rx},
            {"file_id": "BILL", "actual_type": "HOSPITAL_BILL", "content":
             {"hospital_name": "Ayur Wellness Centre", "total": 4000,
              "line_items": [{"description": "Panchakarma Therapy", "amount": 3000},
                             {"description": "Consultation", "amount": 1000}]}},
        ]})


def test_altmed_missing_registration_routes_to_review(policy):
    """Alt-med requires a registered practitioner; absent a registration number,
    hold for review (never auto-reject)."""
    result = ClaimsOrchestrator(policy=policy).process(_altmed_claim(None))
    assert result.decision == Decision.MANUAL_REVIEW
    assert any("registered practitioner" in r.lower() for r in result.reasons)
    assert any(s.check == "practitioner_registration" and s.status.value == "FAILED"
               for s in result.trace)


def test_altmed_with_registration_is_not_held_for_that(policy):
    """Registration present (present-only, no format check) -> not held for a
    missing practitioner registration; the alt-med claim approves at ₹4,000."""
    result = ClaimsOrchestrator(policy=policy).process(
        _altmed_claim("AYUR/KL/2345/2019"))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 4000
    assert not any("registered practitioner" in r.lower() for r in result.reasons)


# ---------------------------------------------------------------------------
# Tier 1 / Fix 1-short — diagnosis shorthand + excluded-list word-boundary
# ---------------------------------------------------------------------------

def test_t2dm_shorthand_maps_to_diabetes_and_triggers_waiting(policy):
    """'T2DM' shorthand maps to the diabetes condition and trips the 90-day
    diabetes waiting period for a within-window member (EMP005, joined
    2024-09-01, treated 2024-10-15)."""
    assert "diabetes" in map_text_to_conditions("T2DM").matched_waiting_conditions
    result = ClaimsOrchestrator(policy=policy).process(ClaimSubmission.model_validate({
        "member_id": "EMP005", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-10-15",
        "claimed_amount": 3000,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Vikram Joshi", "diagnosis": "T2DM"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Vikram Joshi", "total": 3000}},
        ]}))
    assert result.decision == Decision.REJECTED
    assert RejectionReason.WAITING_PERIOD in result.rejection_reasons


def test_match_in_list_word_boundary_stops_substring_overmatch():
    """A short excluded term embedded in a larger word no longer over-matches,
    but a real whole-word occurrence still does — and punctuated policy phrases
    (which a trailing \\b would break) still match via the lookarounds."""
    assert match_in_list("Bleachingkit", ["Bleaching"]) is None   # was a false hit
    assert match_in_list("Tooth Bleaching", ["Bleaching"]) == "Bleaching"
    assert match_in_list("Orthodontic Treatment (Braces)",
                         ["Orthodontic Treatment (Braces)"]) == \
        "Orthodontic Treatment (Braces)"


# ---------------------------------------------------------------------------
# Tier 2 / Fix 1-long — canonical diagnosis normalization
# ---------------------------------------------------------------------------

def _emp005_diabetes_window_claim(rx_content):
    """EMP005 (joined 2024-09-01) treated 2024-10-15 — inside the 90-day diabetes
    window. `rx_content` sets the prescription's diagnosis fields."""
    return ClaimSubmission.model_validate({
        "member_id": "EMP005", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-10-15",
        "claimed_amount": 3000,
        "documents": [
            {"file_id": "RX", "actual_type": "PRESCRIPTION",
             "content": {"patient_name": "Vikram Joshi", **rx_content}},
            {"file_id": "BILL", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Vikram Joshi", "total": 3000}},
        ]})


def test_canonical_condition_drives_waiting_over_keyword(policy):
    """A confident canonical_condition='diabetes' trips the diabetes waiting
    period even though the raw diagnosis text has NO diabetes keyword — proving
    the canonical field, not the keyword list, drove the match."""
    result = ClaimsOrchestrator(policy=policy).process(
        _emp005_diabetes_window_claim({
            "diagnosis": "Raised blood sugar on routine screening",  # no keyword
            "canonical_condition": "diabetes",
            "canonical_condition_confidence": 0.95,
        }))
    assert result.decision == Decision.REJECTED
    assert RejectionReason.WAITING_PERIOD in result.rejection_reasons


def test_absent_canonical_falls_back_to_keyword(policy):
    """No canonical_condition -> the keyword path decides, exactly as before (the
    12 official path); the diabetes keyword in the raw text trips the waiting."""
    result = ClaimsOrchestrator(policy=policy).process(
        _emp005_diabetes_window_claim({"diagnosis": "Type 2 Diabetes Mellitus"}))
    assert result.decision == Decision.REJECTED
    assert RejectionReason.WAITING_PERIOD in result.rejection_reasons


def test_low_confidence_canonical_routes_to_review(policy):
    """A canonical_condition below the confidence threshold on a decision-critical
    diagnosis -> hold for review rather than deciding on an uncertain mapping."""
    result = ClaimsOrchestrator(policy=policy).process(
        _emp005_diabetes_window_claim({
            "diagnosis": "Raised blood sugar on routine screening",
            "canonical_condition": "diabetes",
            "canonical_condition_confidence": 0.3,
        }))
    assert result.decision == Decision.MANUAL_REVIEW
    assert any(s.check == "diagnosis_certainty" for s in result.trace)


# ---------------------------------------------------------------------------
# Tier 2 / Fix 5 — pharmacy branded/generic per-line co-pay
# ---------------------------------------------------------------------------

def _pharmacy_claim(line):
    """A pharmacy claim (PRESCRIPTION + PHARMACY_BILL) with one medicine line.
    `line` is the bill line-item dict (description/amount/drug_type...)."""
    return ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "PHARMACY", "treatment_date": "2024-11-01",
        "claimed_amount": line["amount"],
        "documents": [
            {"file_id": "RX", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "BILL", "actual_type": "PHARMACY_BILL", "content":
             {"patient_name": "Rajesh Kumar", "total": line["amount"],
              "line_items": [line]}},
        ]})


def test_pharmacy_branded_line_gets_branded_copay(policy):
    """A branded medicine line pays branded_drug_copay_percent (from policy)."""
    rate = policy.category_terms("pharmacy").branded_drug_copay_percent  # 30
    result = ClaimsOrchestrator(policy=policy).process(_pharmacy_claim(
        {"description": "Brand Med 500", "amount": 1000, "drug_type": "BRANDED",
         "drug_type_confidence": 0.95}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == round(1000 * (1 - rate / 100), 2)  # 700


def test_pharmacy_generic_line_gets_zero_copay(policy):
    """A generic medicine line pays 0 co-pay."""
    result = ClaimsOrchestrator(policy=policy).process(_pharmacy_claim(
        {"description": "Generic Salt 500", "amount": 1000, "drug_type": "GENERIC",
         "drug_type_confidence": 0.95}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 1000


def test_pharmacy_unknown_drug_type_routes_to_review(policy):
    """A medicine line with no branded/generic status is held for review."""
    result = ClaimsOrchestrator(policy=policy).process(_pharmacy_claim(
        {"description": "Some Medicine", "amount": 1000}))  # no drug_type
    assert result.decision == Decision.MANUAL_REVIEW
    assert any(s.check == "pharmacy_drug_type" for s in result.trace)


# ---------------------------------------------------------------------------
# Tier 2 / Fix 3 — alt-med session cap (this-claim only)
# ---------------------------------------------------------------------------

def _altmed_session_claim(line_desc):
    """A clean alt-med claim (registration present) whose first bill line carries
    `line_desc`, so the session count on this claim can be varied."""
    return ClaimSubmission.model_validate({
        "member_id": "EMP006", "policy_id": "PLUM_GHI_2024",
        "claim_category": "ALTERNATIVE_MEDICINE", "treatment_date": "2024-10-28",
        "claimed_amount": 4000,
        "documents": [
            {"file_id": "RX", "actual_type": "PRESCRIPTION", "content":
             {"doctor_registration": "AYUR/KL/2345/2019",
              "diagnosis": "Chronic Joint Pain", "treatment": "Panchakarma Therapy"}},
            {"file_id": "BILL", "actual_type": "HOSPITAL_BILL", "content":
             {"hospital_name": "Ayur Wellness Centre", "total": 4000,
              "line_items": [{"description": line_desc, "amount": 3000},
                             {"description": "Consultation", "amount": 1000}]}},
        ]})


def test_altmed_over_session_cap_routes_to_review(policy):
    """More sessions on this claim than max_sessions_per_year -> hold for review."""
    cap = policy.category_terms("alternative_medicine").max_sessions_per_year  # 20
    result = ClaimsOrchestrator(policy=policy).process(
        _altmed_session_claim(f"Panchakarma Therapy ({cap + 5} sessions)"))
    assert result.decision == Decision.MANUAL_REVIEW
    assert any(s.check == "session_cap" for s in result.trace)


def test_altmed_within_session_cap_passes(policy):
    """Sessions under the cap -> approved (the TC011 shape: 5 sessions)."""
    result = ClaimsOrchestrator(policy=policy).process(
        _altmed_session_claim("Panchakarma Therapy (5 sessions)"))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 4000


def test_altmed_no_session_count_passes_through(policy):
    """No parseable session count -> pass-through (do NOT route to review)."""
    result = ClaimsOrchestrator(policy=policy).process(
        _altmed_session_claim("Panchakarma Therapy"))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 4000
    assert not any(s.check == "session_cap" for s in result.trace)


# ---------------------------------------------------------------------------
# Tier 3 / Fix 7 — confidence hold tied to the deciding field
# ---------------------------------------------------------------------------

def test_reject_on_confident_field_not_held_by_unrelated_low_amount(policy):
    """A deadline rejection reads the date (confident); the amount is a
    low-confidence read but did NOT drive the decision, so the claim FINALIZES as
    REJECTED rather than being held."""
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "diagnosis": "Viral Fever", "date": "2024-11-01",
               "readability": "GOOD", "_extraction_confidence": 0.95},
        "bill": {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
                 "hospital_name": "City Clinic", "date": "2024-11-01",
                 "line_items": [{"description": "Consultation Fee", "amount": 1500}],
                 "total": 1500, "readability": "GOOD",
                 "_extraction_confidence": 0.95, "amount_confidence": 0.40},
    })
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(
        ClaimSubmission.model_validate({
            "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
            "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
            "received_date": "2024-12-20",   # 49 days later -> past 30-day deadline
            "documents": [
                {"file_id": "rx", "file_data": "cng=", "media_type": "image/jpeg"},
                {"file_id": "bill", "file_data": "YmlsbA==", "media_type": "image/jpeg"},
            ]}))
    assert result.decision == Decision.REJECTED
    assert RejectionReason.SUBMISSION_DEADLINE_PASSED in result.rejection_reasons


def test_low_confidence_deciding_field_is_held(policy):
    """The diagnosis behind a waiting-period rejection is a low-confidence read,
    so the claim is held for review instead of finalizing the rejection."""
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Vikram Joshi",
               "diagnosis": "Type 2 Diabetes Mellitus", "date": "2024-10-15",
               "readability": "GOOD", "_extraction_confidence": 0.40},  # low read
        "bill": {"document_type": "HOSPITAL_BILL", "patient_name": "Vikram Joshi",
                 "date": "2024-10-15", "total": 3000,
                 "line_items": [{"description": "Consultation Fee", "amount": 3000}],
                 "readability": "GOOD", "_extraction_confidence": 0.95},
    })
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(
        ClaimSubmission.model_validate({
            "member_id": "EMP005", "policy_id": "PLUM_GHI_2024",
            "claim_category": "CONSULTATION", "treatment_date": "2024-10-15",
            "documents": [
                {"file_id": "rx", "file_data": "cng=", "media_type": "image/jpeg"},
                {"file_id": "bill", "file_data": "YmlsbA==", "media_type": "image/jpeg"},
            ]}))
    assert result.decision == Decision.MANUAL_REVIEW
    assert any(s.stage == "confidence_gate"
               and "diagnosis-field confidence" in s.detail for s in result.trace)


def test_all_deciding_fields_confident_finalizes(policy):
    """Every deciding field read confidently -> the gate does not hold; the
    claim finalizes (APPROVED ₹1,350)."""
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "diagnosis": "Viral Fever", "date": "2024-11-01",
               "readability": "GOOD", "_extraction_confidence": 0.95},
        "bill": {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
                 "hospital_name": "City Clinic", "date": "2024-11-01",
                 "line_items": [{"description": "Consultation Fee", "amount": 1500}],
                 "total": 1500, "readability": "GOOD", "_extraction_confidence": 0.95},
    })
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(
        ClaimSubmission.model_validate({
            "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
            "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
            "documents": [
                {"file_id": "rx", "file_data": "cng=", "media_type": "image/jpeg"},
                {"file_id": "bill", "file_data": "YmlsbA==", "media_type": "image/jpeg"},
            ]}))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 1350


def test_diagnosis_not_held_by_low_canonical_mapping_confidence(policy):
    """Regression: a cleanly-READ diagnosis must not be held just because the
    model could not MAP it to a policy condition.

    Reproduces the reported live-path bug: the extractor reads a clean printed
    prescription at high confidence (extraction_confidence 0.97) but reports a LOW
    canonical_condition_confidence (0.10) — correct, since 'Viral Fever' maps to
    none of the policy's waiting/exclusion conditions. The confidence gate keys
    the diagnosis field on the document-level READ confidence, NOT the
    canonical-mapping confidence, so the claim approves cleanly (₹1,350) instead
    of being stuck in MANUAL_REVIEW. Before the fix, the gate read
    canonical_condition_confidence and held every live claim carrying a diagnosis.
    """
    extractor = PerDocExtractor({
        "rx": {"document_type": "PRESCRIPTION", "patient_name": "Rajesh Kumar",
               "diagnosis": "Viral Fever", "primary_diagnosis": "Viral Fever",
               "canonical_condition": None,             # maps to no policy condition
               "canonical_condition_confidence": 0.10,  # low MAPPING conf, clean READ
               "date": "2024-11-01", "readability": "GOOD",
               "_extraction_confidence": 0.97},
        "bill": {"document_type": "HOSPITAL_BILL", "patient_name": "Rajesh Kumar",
                 "hospital_name": "City Clinic", "date": "2024-11-01",
                 "line_items": [{"description": "Consultation Fee", "amount": 1500}],
                 "total": 1500, "readability": "GOOD", "_extraction_confidence": 0.97},
    })
    result = ClaimsOrchestrator(policy=policy, extractor=extractor).process(
        ClaimSubmission.model_validate({
            "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
            "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
            "documents": [
                {"file_id": "rx", "file_data": "cng=", "media_type": "image/jpeg"},
                {"file_id": "bill", "file_data": "YmlsbA==", "media_type": "image/jpeg"},
            ]}))
    # Approves cleanly — a low canonical-mapping confidence is not a low read.
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 1350
    # The gate evaluated and PASSED; it did not hold on the diagnosis field.
    assert any(s.stage == "confidence_gate" and s.check == "passed"
               for s in result.trace)
    assert not any(s.stage == "confidence_gate" and s.check == "hold"
                   for s in result.trace)
    assert not any("diagnosis-field confidence" in s.detail for s in result.trace)


# ---------------------------------------------------------------------------
# Tier 3 / Fix 4 — family floater combined limit
# ---------------------------------------------------------------------------

def _family_claim(amount, family_ytd, ytd=0):
    """EMP001 consultation with an injected family year-to-date. The bill line
    avoids the 'consultation' keyword so it is not capped at the consultation
    sub-limit, keeping the eligible amount equal to `amount`."""
    return ClaimSubmission.model_validate({
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "claimed_amount": amount, "ytd_claims_amount": ytd,
        "family_ytd_amount": family_ytd,
        "documents": [
            {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
             {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
            {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
             {"patient_name": "Rajesh Kumar", "total": amount,
              "line_items": [{"description": "Day Care Procedure",
                              "amount": amount}]}},
        ]})


def test_family_over_combined_limit_rejected(policy):
    """Family YTD + this claim over the family floater limit -> rejected, the
    same shape as an over-annual-limit claim."""
    limit = policy.coverage.family_floater.combined_limit  # 150000
    result = ClaimsOrchestrator(policy=policy).process(
        _family_claim(4000, family_ytd=limit - 3000))  # 147000 + 4000 = 151000
    assert result.decision == Decision.REJECTED
    assert RejectionReason.FAMILY_LIMIT_EXCEEDED in result.rejection_reasons


def test_family_under_combined_limit_passes(policy):
    """Family YTD + this claim under the family floater limit -> approved."""
    result = ClaimsOrchestrator(policy=policy).process(
        _family_claim(4000, family_ytd=100000))
    assert result.decision == Decision.APPROVED
    assert result.approved_amount == 3600          # 4000 less 10% consultation co-pay
    assert RejectionReason.FAMILY_LIMIT_EXCEEDED not in result.rejection_reasons


def test_per_member_annual_still_fires_independently(policy):
    """The per-member annual limit fires on its own even when the family is well
    under the combined limit (annual is checked before family)."""
    result = ClaimsOrchestrator(policy=policy).process(
        _family_claim(4000, family_ytd=0, ytd=48000))  # 48000 + 4000 > 50000 annual
    assert result.decision == Decision.REJECTED
    assert RejectionReason.ANNUAL_LIMIT_EXCEEDED in result.rejection_reasons
    assert RejectionReason.FAMILY_LIMIT_EXCEEDED not in result.rejection_reasons
