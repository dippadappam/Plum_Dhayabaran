"""Observability tests: the /metrics snapshot moves as claims are processed,
and the health checks verify real dependencies — /api/ready reports unhealthy
when the database is down, while /api/health stays the plain liveness check.

No network, no LLM; mirrors test_api.py's fixture (temp DB via CLAIMS_DB_PATH,
api module reloaded to bind the patched env).
"""

import importlib
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

CASES_PATH = Path(__file__).parent / "test_cases.json"
with open(CASES_PATH, "r", encoding="utf-8") as f:
    _CASES = {c["case_id"]: c for c in json.load(f)["test_cases"]}


@pytest.fixture()
def api_module(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app.api as api_module
    importlib.reload(api_module)
    return api_module


@pytest.fixture()
def client(api_module):
    with TestClient(api_module.app) as c:
        yield c


# --- metrics ----------------------------------------------------------------

def test_metrics_counters_move_as_claims_processed(api_module, client):
    # The metrics singleton persists across api-module reloads; reset for a
    # deterministic slate, then process one claim of each decision class.
    api_module.metrics.reset()
    assert client.post("/api/claims",
                       json=_CASES["TC004"]["input"]).json()["decision"] == "APPROVED"
    assert client.post("/api/claims",
                       json=_CASES["TC006"]["input"]).json()["decision"] == "PARTIAL"
    assert client.post("/api/claims",
                       json=_CASES["TC005"]["input"]).json()["decision"] == "REJECTED"
    assert client.post("/api/claims",
                       json=_CASES["TC009"]["input"]).json()["decision"] == "MANUAL_REVIEW"

    snap = client.get("/metrics").json()
    assert snap["claims_processed"] == 4
    assert snap["decisions"]["APPROVED"] == 1
    assert snap["decisions"]["PARTIAL"] == 1
    assert snap["decisions"]["REJECTED"] == 1
    assert snap["decisions"]["MANUAL_REVIEW"] == 1
    assert snap["manual_review_rate"] == 0.25          # 1 of 4
    assert snap["latency_ms"]["count"] == 4            # latency recorded per claim
    assert sum(snap["confidence_buckets"].values()) == 4  # one bucket per claim


def test_metrics_endpoint_is_json_snapshot(client):
    snap = client.get("/metrics").json()
    for key in ("claims_processed", "decisions", "manual_review_rate",
                "degraded_claims", "confidence_buckets", "latency_ms"):
        assert key in snap
    assert set(snap["decisions"]) >= {"APPROVED", "PARTIAL", "REJECTED",
                                      "MANUAL_REVIEW"}


def test_degraded_claim_counted(api_module, client):
    """A component failure (TC011's simulated failure) increments the degraded
    counter without changing the decision."""
    api_module.metrics.reset()
    body = client.post("/api/claims", json=_CASES["TC011"]["input"]).json()
    assert body["component_failures"]            # the failure happened
    snap = client.get("/metrics").json()
    assert snap["degraded_claims"] == 1


# --- health / readiness -----------------------------------------------------

def test_health_is_simple_liveness(client):
    # /api/health stays the plain liveness contract (unchanged).
    assert client.get("/api/health").json() == {"status": "ok"}


def test_readiness_reports_healthy(client):
    r = client.get("/api/ready")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["database"]["status"] == "ok"
    assert body["checks"]["extractor"]["status"] == "not_configured"  # no key in tests


def test_readiness_reports_unhealthy_when_database_down(client):
    """A down database dependency makes readiness report unhealthy with a 503,
    instead of the process falsely claiming it is ready."""
    def boom():
        raise RuntimeError("database file is locked")
    client.app.state.store.ping = boom   # simulate the DB being unreachable

    r = client.get("/api/ready")
    assert r.status_code == 503
    body = r.json()
    assert body["status"] == "unhealthy"
    assert body["checks"]["database"]["status"] == "error"
    assert "locked" in body["checks"]["database"]["detail"]
