"""Document-alteration detection.

A document the model flags as visibly altered (`alteration_suspected`) routes the
claim to manual review with the cause in the trace; a clean document does not;
and the validation layer accepts/normalizes the new field. The 12 official cases
carry no such field, so the rule stays silent (the golden-trace test guards
that).
"""

import types

from app.llm.extractor import (
    SYSTEM_PROMPT,
    ClaudeVisionExtractor,
    _system_prompt_with_conditions,
)
from app.models.claim import ClaimSubmission, SubmittedDocument
from app.models.decision import Decision
from app.models.extraction import normalize_extraction
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


# --- validation layer -------------------------------------------------------

def test_validation_normalizes_alteration_fields():
    out = normalize_extraction({
        "document_type": "HOSPITAL_BILL", "readability": "GOOD",
        "extraction_confidence": 0.9,
        "alteration_suspected": "true",          # string -> True
        "alteration_reason": "  total overwritten  "})
    assert out["alteration_suspected"] is True
    assert out["alteration_reason"] == "total overwritten"


def test_validation_alteration_defaults_clean():
    out = normalize_extraction({
        "document_type": "HOSPITAL_BILL", "readability": "GOOD",
        "extraction_confidence": 0.9})
    assert out["alteration_suspected"] is False    # absent -> clean
    assert out["alteration_reason"] is None


def test_extractor_carries_alteration_from_response():
    tool_input = {"document_type": "HOSPITAL_BILL", "readability": "GOOD",
                  "extraction_confidence": 0.9, "alteration_suspected": True,
                  "alteration_reason": "DUPLICATE stamp present"}
    block = types.SimpleNamespace(type="tool_use", input=tool_input)
    resp = types.SimpleNamespace(
        content=[block],
        usage=types.SimpleNamespace(input_tokens=1, output_tokens=1),
        stop_reason="tool_use")

    class _Client:
        class messages:
            @staticmethod
            def create(**kw):
                return resp

    ext = ClaudeVisionExtractor(client=_Client())
    out = ext.extract(SubmittedDocument(file_id="D", file_data="ZmFrZQ==",
                                        media_type="image/png"))
    assert out["alteration_suspected"] is True
    assert out["alteration_reason"] == "DUPLICATE stamp present"


# --- routing rule -----------------------------------------------------------

def _consultation(bill_extra=None):
    bill_content = {"hospital_name": "City Clinic", "patient_name": "Rajesh Kumar",
                    "date": "2024-11-01", "total": 1000,
                    "line_items": [{"description": "Consultation Fee",
                                    "amount": 1000}]}
    if bill_extra:
        bill_content.update(bill_extra)
    return ClaimSubmission(
        member_id="EMP001", policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION", treatment_date="2024-11-01",
        claimed_amount=1000,
        documents=[
            SubmittedDocument(file_id="F1", actual_type="PRESCRIPTION",
                              content={"patient_name": "Rajesh Kumar",
                                       "diagnosis": "Viral Fever",
                                       "doctor_registration": "KA/45678/2015"}),
            SubmittedDocument(file_id="F2", actual_type="HOSPITAL_BILL",
                              content=bill_content),
        ])


def test_altered_document_routes_to_review_with_trace():
    orch = ClaimsOrchestrator(policy=load_policy())
    result = orch.process(_consultation(
        {"alteration_suspected": True, "alteration_reason": "total amount overwritten"}))

    assert result.decision == Decision.MANUAL_REVIEW
    steps = [s for s in result.trace if s.check == "document_alteration"]
    assert len(steps) == 1
    assert steps[0].status.value == "FAILED"
    detail = steps[0].detail.lower()
    assert "alteration" in detail and "overwritten" in detail
    assert steps[0].data.get("altered_documents") == ["F2"]
    assert any("alteration" in r.lower() for r in result.reasons)


def test_clean_document_not_routed_for_alteration():
    orch = ClaimsOrchestrator(policy=load_policy())
    result = orch.process(_consultation())   # no alteration field
    assert not any(s.check == "document_alteration" for s in result.trace)
    assert result.decision == Decision.APPROVED


# --- prompt coverage (live extraction path) ---------------------------------

def test_alteration_prompt_explicitly_covers_struck_through_name():
    """The alteration instruction must explicitly cover a PATIENT NAME that was
    struck through and rewritten (not only amounts), so the broadened coverage
    cannot be silently dropped. Checked in both the base prompt and the built
    prompt actually sent on the live path."""
    for text in (SYSTEM_PROMPT.lower(), _system_prompt_with_conditions().lower()):
        assert "alteration_suspected" in text
        assert "struck through" in text
        assert "patient name" in text
        # the explicit name-rewrite example locks the broadened coverage
        assert "patient name struck through and rewritten" in text
        # original framing preserved: visible signs, not digital-forgery guessing
        assert "digital forgery" in text
