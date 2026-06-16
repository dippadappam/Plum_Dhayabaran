"""Tests for the live extractor's API-call handling.

Two concerns: (1) retry with exponential backoff as ONE coherent layer — the
SDK's own retries disabled, our loop honoring Retry-After and not hammering on a
rate-limit; (2) per-claim token usage captured from response.usage, aggregated
across a claim's calls, and logged on the per-claim line.

The live extractor is never hit over the network: a fake client is injected and
the backoff sleep is stubbed. The 12 official cases bypass the live extractor
(the golden-trace test guards that).
"""

import importlib
import json
import logging
import sys
import types
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.agents.extraction import ExtractionAgent
from app.llm.extractor import (
    ClaudeVisionExtractor,
    ExtractionError,
    _parse_retry_after,
    _usage_dict,
)
from app.models.claim import ClaimSubmission, SubmittedDocument
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator

CASES_PATH = Path(__file__).parent / "test_cases.json"
with open(CASES_PATH, "r", encoding="utf-8") as f:
    _CASES = {c["case_id"]: c for c in json.load(f)["test_cases"]}

_GOOD_INPUT = {"document_type": "HOSPITAL_BILL", "readability": "GOOD",
               "extraction_confidence": 0.9}


# --- fakes ------------------------------------------------------------------

def _tool_response(tool_input, *, input_tokens=0, output_tokens=0):
    usage = types.SimpleNamespace(input_tokens=input_tokens,
                                  output_tokens=output_tokens)
    block = types.SimpleNamespace(type="tool_use", input=tool_input)
    return types.SimpleNamespace(content=[block], usage=usage,
                                 stop_reason="tool_use")


class _ApiError(Exception):
    """Stand-in for an anthropic API error: a status_code and an optional
    Retry-After header on a .response."""

    def __init__(self, status_code=None, retry_after=None):
        super().__init__(f"api error {status_code}")
        self.status_code = status_code
        if retry_after is not None:
            self.response = types.SimpleNamespace(
                headers={"retry-after": str(retry_after)})


class _FakeClient:
    """Minimal stand-in for anthropic.Anthropic with a scripted create(): each
    entry is either an Exception to raise or a response to return; the last entry
    repeats once the script is exhausted."""

    def __init__(self, behaviors):
        self._behaviors = list(behaviors)
        self.calls = 0
        outer = self

        class _Messages:
            def create(self, **kwargs):
                b = outer._behaviors[min(outer.calls, len(outer._behaviors) - 1)]
                outer.calls += 1
                if isinstance(b, Exception):
                    raise b
                return b

        self.messages = _Messages()


def _extractor(behaviors, monkeypatch):
    ext = ClaudeVisionExtractor(client=_FakeClient(behaviors))
    delays: list[float] = []
    monkeypatch.setattr(ext, "_sleep", lambda s: delays.append(s))
    return ext, delays


def _doc():
    return SubmittedDocument(file_id="D1", file_data="ZmFrZQ==",
                             media_type="image/png")


# --- retry / backoff --------------------------------------------------------

def test_sdk_max_retries_disabled(monkeypatch):
    """Constructing the real client sets max_retries=0, so the SDK and our own
    loop don't both retry (no compounding layers)."""
    captured: dict = {}

    class _FakeAnthropic:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake = types.ModuleType("anthropic")
    fake.Anthropic = _FakeAnthropic
    monkeypatch.setitem(sys.modules, "anthropic", fake)
    ClaudeVisionExtractor(api_key="test-key")
    assert captured.get("max_retries") == 0


def test_transient_failure_retried_with_backoff(monkeypatch):
    ext, delays = _extractor(
        [_ApiError(status_code=500), _ApiError(status_code=503),
         _tool_response(_GOOD_INPUT, input_tokens=100, output_tokens=20)],
        monkeypatch)
    out = ext.extract(_doc())
    assert ext._client.calls == 3       # 2 failures + 1 success
    assert len(delays) == 2             # backed off between attempts (no hammering)
    assert all(d >= 0 for d in delays)
    assert out["_usage"] == {"input_tokens": 100, "output_tokens": 20}


def test_rate_limit_honors_retry_after(monkeypatch):
    ext, delays = _extractor(
        [_ApiError(status_code=429, retry_after=7),
         _tool_response(_GOOD_INPUT)],
        monkeypatch)
    ext.extract(_doc())
    assert delays == [7.0]              # honored Retry-After, not jittered backoff


def test_permanent_error_not_retried(monkeypatch):
    ext, delays = _extractor([_ApiError(status_code=400)], monkeypatch)
    with pytest.raises(ExtractionError):
        ext.extract(_doc())
    assert ext._client.calls == 1      # 400 is permanent -> single attempt
    assert delays == []                # no backoff


def test_retry_exhaustion_raises_after_max(monkeypatch):
    ext, delays = _extractor([_ApiError(status_code=500)], monkeypatch)
    with pytest.raises(ExtractionError):
        ext.extract(_doc())
    assert ext._client.calls == 3      # MAX_RETRIES + 1 attempts
    assert len(delays) == 2            # slept before each retry, not after the last


def test_retry_after_and_usage_helpers():
    assert _parse_retry_after(_ApiError(status_code=429, retry_after=12)) == 12.0
    assert _parse_retry_after(_ApiError(status_code=500)) is None
    assert _usage_dict(None) == {"input_tokens": 0, "output_tokens": 0}
    assert _usage_dict(types.SimpleNamespace(input_tokens=3, output_tokens=4)) == \
        {"input_tokens": 3, "output_tokens": 4}


# --- token usage capture / aggregation --------------------------------------

def test_token_usage_captured_from_response(monkeypatch):
    ext, _ = _extractor(
        [_tool_response(_GOOD_INPUT, input_tokens=321, output_tokens=45)],
        monkeypatch)
    out = ext.extract(_doc())
    assert out["_usage"] == {"input_tokens": 321, "output_tokens": 45}


class _UsageExtractor:
    """Mock extractor returning minimal valid fields plus per-call token usage."""

    def __init__(self, input_tokens=100, output_tokens=20):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

    def extract(self, document):
        dtype = document.actual_type.value if document.actual_type else "UNKNOWN"
        fields = {"document_type": dtype, "readability": "GOOD",
                  "_extraction_confidence": 0.9, "patient_name": "Rajesh Kumar",
                  "_usage": {"input_tokens": self.input_tokens,
                             "output_tokens": self.output_tokens}}
        if dtype == "HOSPITAL_BILL":
            fields["total"] = 1500.0
            fields["hospital_name"] = "City Clinic"
            fields["line_items"] = [{"description": "Consultation Fee",
                                     "amount": 1500.0}]
        else:
            fields["diagnosis"] = "Viral Fever"
            fields["date"] = "2024-11-01"
        return fields


def _two_doc_submission():
    return ClaimSubmission(
        member_id="EMP001", policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION", treatment_date="2024-11-01",
        documents=[
            SubmittedDocument(file_id="A", actual_type="PRESCRIPTION",
                              file_data="b25l", media_type="image/png"),
            SubmittedDocument(file_id="B", actual_type="HOSPITAL_BILL",
                              file_data="dHdv", media_type="image/png"),
        ])


def test_token_usage_aggregated_per_claim():
    orch = ClaimsOrchestrator(policy=load_policy(),
                              extractor=_UsageExtractor(100, 20))
    result = orch.process(_two_doc_submission())
    assert result.token_usage["input_tokens"] == 200    # two distinct paid calls
    assert result.token_usage["output_tokens"] == 40
    assert result.token_usage["calls"] == 2


def test_token_usage_dedupe_counts_once():
    orch = ClaimsOrchestrator(policy=load_policy(),
                              extractor=_UsageExtractor(100, 20))
    sub = ClaimSubmission(
        member_id="EMP001", policy_id="PLUM_GHI_2024",
        claim_category="CONSULTATION", treatment_date="2024-11-01",
        documents=[
            SubmittedDocument(file_id="A", actual_type="HOSPITAL_BILL",
                              file_data="c2FtZQ==", media_type="image/png"),
            SubmittedDocument(file_id="B", actual_type="HOSPITAL_BILL",
                              file_data="c2FtZQ==", media_type="image/png"),
        ])
    result = orch.process(sub)
    assert result.token_usage["calls"] == 1             # identical bytes: one call
    assert result.token_usage["input_tokens"] == 100


def test_structured_path_reports_zero_tokens():
    # The 12-style structured-content path makes no paid calls.
    orch = ClaimsOrchestrator(policy=load_policy())
    result = orch.process(
        ClaimSubmission.model_validate(_CASES["TC004"]["input"]))
    assert result.token_usage == {"input_tokens": 0, "output_tokens": 0, "calls": 0}


# --- per-claim logging + metrics (via the API) ------------------------------

@pytest.fixture()
def api_client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app.api as api_module
    importlib.reload(api_module)
    with TestClient(api_module.app) as c:
        yield c


class _LogCapture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


def _capture_claims_log():
    cap = _LogCapture()
    logging.getLogger("claims").addHandler(cap)
    return cap


def test_claim_decided_log_carries_token_fields(api_client):
    cap = _capture_claims_log()
    try:
        api_client.post("/api/claims", json=_CASES["TC004"]["input"])
    finally:
        logging.getLogger("claims").removeHandler(cap)
    decided = [m for m in cap.messages if m.startswith("claim_decided")]
    assert decided
    assert "tokens_in=" in decided[0]
    assert "tokens_out=" in decided[0]
    assert "extraction_calls=" in decided[0]


def _live_two_doc_payload():
    return {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
        "documents": [
            {"file_id": "A", "actual_type": "PRESCRIPTION",
             "file_data": "b25l", "media_type": "image/png"},
            {"file_id": "B", "actual_type": "HOSPITAL_BILL",
             "file_data": "dHdv", "media_type": "image/png"},
        ]}


def test_token_usage_logged_nonzero_via_api(api_client):
    for agent in api_client.app.state.orchestrator.agents:
        if isinstance(agent, ExtractionAgent):
            agent.extractor = _UsageExtractor(150, 30)
    cap = _capture_claims_log()
    try:
        api_client.post("/api/claims", json=_live_two_doc_payload())
    finally:
        logging.getLogger("claims").removeHandler(cap)
    decided = [m for m in cap.messages if m.startswith("claim_decided")]
    assert decided
    assert "tokens_in=300" in decided[0]    # 2 docs x 150
    assert "tokens_out=60" in decided[0]
    assert "extraction_calls=2" in decided[0]


def test_metrics_aggregates_tokens(api_client):
    import app.metrics as m
    m.metrics.reset()
    for agent in api_client.app.state.orchestrator.agents:
        if isinstance(agent, ExtractionAgent):
            agent.extractor = _UsageExtractor(150, 30)
    api_client.post("/api/claims", json=_live_two_doc_payload())
    snap = api_client.get("/metrics").json()
    assert snap["tokens"]["input"] == 300
    assert snap["tokens"]["output"] == 60
