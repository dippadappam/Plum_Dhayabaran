"""Truncation guard: when the extraction response is cut off (stop_reason
max_tokens), the tail line items may be missing, so the claim must NOT be
auto-decided on the incomplete extraction — it is routed to manual review with
the cause visible in the trace.

The live extractor is exercised with an injected fake client (no network). The
12 official cases provide structured content (no model call, no stop_reason) and
never hit this path; the golden-trace test guards that.
"""

import types

from app.llm.extractor import ClaudeVisionExtractor
from app.models.claim import ClaimSubmission, SubmittedDocument
from app.models.decision import Decision
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator


def _response(tool_input, *, stop_reason):
    block = types.SimpleNamespace(type="tool_use", input=tool_input)
    usage = types.SimpleNamespace(input_tokens=100, output_tokens=2000)
    return types.SimpleNamespace(content=[block], usage=usage,
                                 stop_reason=stop_reason)


class _FakeClient:
    """Returns one scripted response for every messages.create() call."""

    def __init__(self, response):
        outer = self
        outer._response = response

        class _Messages:
            def create(self, **kwargs):
                return outer._response

        self.messages = _Messages()


_BILL = {"document_type": "HOSPITAL_BILL", "readability": "GOOD",
         "extraction_confidence": 0.9, "total": 1500.0,
         "line_items": [{"description": "Consultation Fee", "amount": 1500.0}]}


def _doc():
    return SubmittedDocument(file_id="D1", file_data="ZmFrZQ==",
                             media_type="image/png")


def test_extractor_flags_truncation_on_max_tokens():
    ext = ClaudeVisionExtractor(
        client=_FakeClient(_response(_BILL, stop_reason="max_tokens")))
    out = ext.extract(_doc())
    assert out.get("_truncated") is True


def test_extractor_does_not_flag_complete_response():
    ext = ClaudeVisionExtractor(
        client=_FakeClient(_response(_BILL, stop_reason="tool_use")))
    out = ext.extract(_doc())
    assert "_truncated" not in out


def test_truncated_extraction_routes_to_review_with_trace():
    ext = ClaudeVisionExtractor(
        client=_FakeClient(_response(_BILL, stop_reason="max_tokens")))
    orch = ClaimsOrchestrator(policy=load_policy(), extractor=ext)
    sub = ClaimSubmission(
        member_id="EMP001", policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION", treatment_date="2024-11-01",
        documents=[SubmittedDocument(file_id="A", actual_type="HOSPITAL_BILL",
                                     file_data="ZmFrZQ==", media_type="image/png")])
    result = orch.process(sub)

    # Routed to review rather than auto-decided on the incomplete extraction.
    assert result.decision == Decision.MANUAL_REVIEW
    assert result.manual_review_recommended is True

    # The truncation cause is visible in the trace, the reasons, and the
    # component failures.
    trunc = [s for s in result.trace if s.check == "truncated"]
    assert len(trunc) == 1
    assert trunc[0].status.value == "FAILED"
    detail = trunc[0].detail.lower()
    assert "truncat" in detail and "max_tokens" in detail
    assert trunc[0].data.get("truncated_files") == ["A"]
    assert any("truncat" in r.lower() for r in result.reasons)
    assert any(cf.component == "extraction.truncated"
               for cf in result.component_failures)


def test_complete_extraction_is_not_routed_for_truncation():
    # A normal (tool_use) response does not trigger the truncation route.
    ext = ClaudeVisionExtractor(
        client=_FakeClient(_response(_BILL, stop_reason="tool_use")))
    orch = ClaimsOrchestrator(policy=load_policy(), extractor=ext)
    sub = ClaimSubmission(
        member_id="EMP001", policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION", treatment_date="2024-11-01",
        documents=[SubmittedDocument(file_id="A", actual_type="HOSPITAL_BILL",
                                     file_data="ZmFrZQ==", media_type="image/png")])
    result = orch.process(sub)
    assert not any(s.check == "truncated" for s in result.trace)
