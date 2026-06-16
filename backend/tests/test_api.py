"""API-layer tests with FastAPI TestClient and a temp database. No network,
no LLM. Verifies the HTTP contracts, persistence, history-fed fraud checks,
and the no-500 degradation guarantee.
"""

import base64
import json
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

CASES_PATH = Path(__file__).parent / "test_cases.json"
with open(CASES_PATH, "r", encoding="utf-8") as f:
    _CASES = {c["case_id"]: c for c in json.load(f)["test_cases"]}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # Re-import to bind the patched env.
    import importlib
    import app.api as api_module
    importlib.reload(api_module)
    with TestClient(api_module.app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_policy_summary(client):
    data = client.get("/api/policy").json()
    assert data["policy_id"] == "PLUM_GHI_2024"
    assert "consultation" in data["categories"]
    assert data["per_claim_limit"] == 5000


def test_submit_clean_claim_and_retrieve(client):
    payload = _CASES["TC004"]["input"]
    r = client.post("/api/claims", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["decision"] == "APPROVED"
    assert body["approved_amount"] == 1350
    assert body["trace"], "Trace must be returned"

    ref = body["claim_reference"]
    fetched = client.get(f"/api/claims/{ref}").json()
    assert fetched["claim_reference"] == ref
    assert client.get("/api/claims").json()[0]["claim_reference"] == ref


def test_simulated_failure_returns_200_not_500(client):
    payload = _CASES["TC011"]["input"]
    r = client.post("/api/claims", json=payload)
    assert r.status_code == 200, "Component failure must never surface as 500"
    body = r.json()
    assert body["decision"] == "APPROVED"
    assert body["component_failures"]
    assert body["manual_review_recommended"] is True


def test_stored_history_feeds_fraud_check(client):
    """Submit 3 claims for one member on one day, then a 4th: the stored
    history alone must trigger the same-day fraud signal."""
    base = dict(_CASES["TC004"]["input"])  # EMP001, 2024-11-01, clean docs
    for _ in range(3):
        assert client.post("/api/claims", json=base).status_code == 200
    r = client.post("/api/claims", json=base)
    body = r.json()
    assert body["decision"] == "MANUAL_REVIEW"
    assert body["fraud_signals"]


def test_invalid_submission_is_422(client):
    r = client.post("/api/claims", json={"member_id": "EMP001"})
    assert r.status_code == 422


def test_h3_duplicate_of_decided_claim_holds(client):
    """A byte-identical resubmission of an already-decided claim is flagged as
    a duplicate and held for review (not paid again)."""
    payload = _CASES["TC004"]["input"]
    first = client.post("/api/claims", json=payload).json()
    assert first["decision"] == "APPROVED"
    second = client.post("/api/claims", json=payload).json()
    assert second["decision"] == "MANUAL_REVIEW"
    assert any("duplicate" in s.lower() for s in second["fraud_signals"])


def test_h3_resubmission_with_changed_document_flows(client):
    """A resubmission that changes a document (different document-set hash) is
    not a duplicate and flows through — the resubmission whitelist works."""
    import copy
    base = _CASES["TC004"]["input"]
    assert client.post("/api/claims", json=base).json()["decision"] == "APPROVED"
    changed = copy.deepcopy(base)
    # Change a non-amount field on one document -> different content hash.
    changed["documents"][0]["content"]["doctor_name"] = "Dr. A. Different"
    body = client.post("/api/claims", json=changed).json()
    assert body["decision"] == "APPROVED"
    assert not any("duplicate" in s.lower() for s in body["fraud_signals"])


def test_h3_held_claim_is_not_a_duplicate_target(client):
    """A claim still awaiting review (PENDING_REVIEW) is NOT in the duplicate
    set: a byte-identical resubmission of a held claim must not be flagged as a
    duplicate of an 'already-decided' claim (it has not been decided). Only
    finished claims — auto-final or reviewer-resolved — count."""
    payload = _CASES["TC009"]["input"]  # 4th same-day -> held (PENDING_REVIEW)
    first = client.post("/api/claims", json=payload).json()
    assert first["decision"] == "MANUAL_REVIEW"
    assert first["review_status"] == "PENDING_REVIEW"
    second = client.post("/api/claims", json=payload).json()
    # Still held on the same-day signal, but NOT flagged as a duplicate of the
    # pending first claim.
    assert not any("duplicate" in s.lower() for s in second["fraud_signals"])


# ---------------------------------------------------------------------------
# 4a — reviewer resolve
# ---------------------------------------------------------------------------

def _submit_held_claim(client):
    """TC009: a 4th same-day claim is held (MANUAL_REVIEW) and carries the
    engine-computed approved amount (4320)."""
    r = client.post("/api/claims", json=_CASES["TC009"]["input"]).json()
    assert r["decision"] == "MANUAL_REVIEW"
    assert r["review_status"] == "PENDING_REVIEW"
    assert r["approved_amount"] == 4320  # engine-computed, carried into the hold
    return r["claim_reference"]


def test_resolve_happy_path_keeps_approved_amount(client):
    ref = _submit_held_claim(client)
    resp = client.post(
        f"/api/claims/{ref}/resolve",
        json={"action": "approve", "reviewer_id": "alice",
              "reason": "Verified the same-day visits are legitimate."})
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "APPROVED"
    assert body["approved_amount"] == 4320      # correct non-zero amount kept
    assert body["review_status"] == "RESOLVED"
    assert body["resolved_by"] == "alice"
    assert body["resolution"] == "APPROVED"
    assert body["status"] == "DECIDED"          # status never left DECIDED
    # The resolution is in the trace.
    assert any(s["stage"] == "reviewer" and s["check"] == "resolution"
               for s in body["trace"])


def test_resolve_409_on_non_held_claim(client):
    """An auto-final claim is not reviewer-resolvable."""
    r = client.post("/api/claims", json=_CASES["TC004"]["input"]).json()
    assert r["decision"] == "APPROVED" and r.get("review_status") is None
    resp = client.post(
        f"/api/claims/{r['claim_reference']}/resolve",
        json={"action": "approve", "reviewer_id": "alice", "reason": "n/a"})
    assert resp.status_code == 409


def test_resolve_409_on_double_resolve_names_prior_resolver(client):
    ref = _submit_held_claim(client)
    first = client.post(
        f"/api/claims/{ref}/resolve",
        json={"action": "approve", "reviewer_id": "alice", "reason": "ok"})
    assert first.status_code == 200
    second = client.post(
        f"/api/claims/{ref}/resolve",
        json={"action": "reject", "reviewer_id": "bob", "reason": "changed mind"})
    assert second.status_code == 409
    assert "alice" in second.json()["detail"]   # 409 names the prior resolver


def test_reviewer_approved_claim_is_in_duplicate_set(client):
    """After a reviewer approves a held claim it stays DECIDED, so the
    duplicate-hash set picks up a byte-identical resubmission."""
    ref = _submit_held_claim(client)
    client.post(f"/api/claims/{ref}/resolve",
                json={"action": "approve", "reviewer_id": "alice", "reason": "ok"})
    again = client.post("/api/claims", json=_CASES["TC009"]["input"]).json()
    assert any("duplicate" in s.lower() for s in again["fraud_signals"])


# ---------------------------------------------------------------------------
# 4b — evidence persistence, document viewer, parent link, early-hold approve
# ---------------------------------------------------------------------------

def _live_none_amount_payload():
    """A live claim (file_data) the engine cannot derive an amount for in tests
    (no extractor configured -> extraction skipped) -> held with amount None."""
    return {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "U1", "actual_type": "PRESCRIPTION",
             "file_data": "ZmFrZS1yeA==", "media_type": "image/jpeg"},
            {"file_id": "U2", "actual_type": "HOSPITAL_BILL",
             "file_data": "ZmFrZS1iaWxs", "media_type": "image/jpeg"},
        ],
    }


def test_live_documents_persisted_and_fetchable(client):
    body = client.post("/api/claims", json=_live_none_amount_payload()).json()
    ref = body["claim_reference"]
    doc = client.get(f"/api/claims/{ref}/documents/U1")
    assert doc.status_code == 200
    assert doc.content == base64.b64decode("ZmFrZS1yeA==")  # decoded raw bytes
    assert client.get(f"/api/claims/{ref}/documents/NOPE").status_code == 404


def test_structured_submission_persists_no_documents(client):
    body = client.post("/api/claims", json=_CASES["TC004"]["input"]).json()
    ref = body["claim_reference"]
    # TC004 docs are structured content (no file_data): nothing stored.
    assert client.get(f"/api/claims/{ref}/documents/F007").status_code == 404


def test_extracted_record_present_in_result(client):
    body = client.post("/api/claims", json=_CASES["TC004"]["input"]).json()
    docs = body["extracted_documents"]
    assert isinstance(docs, list) and len(docs) == 2
    assert {d["file_id"] for d in docs} == {"F007", "F008"}


def test_resubmission_stores_parent_link(client, tmp_path):
    base = _CASES["TC004"]["input"]
    parent = client.post("/api/claims", json=base).json()["claim_reference"]
    resub = dict(base)
    resub["parent_claim_reference"] = parent
    child = client.post("/api/claims", json=resub).json()["claim_reference"]
    con = sqlite3.connect(str(tmp_path / "test.db"))
    row = con.execute("SELECT parent_reference FROM claims WHERE claim_reference=?",
                      (child,)).fetchone()
    con.close()
    assert row[0] == parent


def test_resolve_none_amount_hold_requires_explicit_amount(client):
    held = client.post("/api/claims", json=_live_none_amount_payload()).json()
    assert held["decision"] == "MANUAL_REVIEW"
    assert held["approved_amount"] is None
    ref = held["claim_reference"]
    # Approve with no amount -> 400.
    r400 = client.post(f"/api/claims/{ref}/resolve",
                       json={"action": "approve", "reviewer_id": "alice",
                             "reason": "ok"})
    assert r400.status_code == 400
    # Approve with an explicit amount -> APPROVED with that amount.
    r200 = client.post(f"/api/claims/{ref}/resolve",
                       json={"action": "approve", "reviewer_id": "alice",
                             "reason": "read the bill: 1234", "approved_amount": 1234.0})
    assert r200.status_code == 200
    body = r200.json()
    assert body["decision"] == "APPROVED"
    assert body["approved_amount"] == 1234.0


# ---------------------------------------------------------------------------
# 4c — reviewer-amount bounds validation
# ---------------------------------------------------------------------------

def test_resolve_approve_amount_bounds(client):
    held = client.post("/api/claims", json=_live_none_amount_payload()).json()
    ref = held["claim_reference"]
    # Negative amount -> 400 (the 400 is checked before resolve, so the claim
    # stays held and can still be resolved with a valid amount afterwards).
    neg = client.post(f"/api/claims/{ref}/resolve",
                      json={"action": "approve", "reviewer_id": "a",
                            "reason": "x", "approved_amount": -5})
    assert neg.status_code == 400
    # Above the ceiling (sum insured 500,000) -> 400.
    big = client.post(f"/api/claims/{ref}/resolve",
                      json={"action": "approve", "reviewer_id": "a",
                            "reason": "x", "approved_amount": 600000})
    assert big.status_code == 400
    # A valid amount approves.
    ok = client.post(f"/api/claims/{ref}/resolve",
                     json={"action": "approve", "reviewer_id": "a",
                           "reason": "read the bill", "approved_amount": 1500})
    assert ok.status_code == 200
    assert ok.json()["approved_amount"] == 1500
