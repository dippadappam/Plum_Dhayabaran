"""API error-path tests: the handled-500 branches that have no other coverage.

Two paths return a 500 rather than crashing the process:
- a stored document whose base64 payload cannot be decoded (get_claim_document);
- a storage I/O error during persistence, after the decision is computed.

Follows the test_api.py pattern: temp DB via CLAIMS_DB_PATH, no network, no LLM
(ANTHROPIC_API_KEY removed), the api module reloaded to bind the patched env.
"""

import importlib
import sqlite3

import pytest
from fastapi.testclient import TestClient


def _reload_api(tmp_path, monkeypatch):
    monkeypatch.setenv("CLAIMS_DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    import app.api as api_module
    importlib.reload(api_module)
    return api_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    api_module = _reload_api(tmp_path, monkeypatch)
    with TestClient(api_module.app) as c:
        yield c


def test_corrupt_stored_document_returns_handled_500(client):
    """A live document whose stored base64 cannot be decoded returns a handled
    500 ('Stored document is corrupt'), not a crash. 'AAA' is stored verbatim
    and is invalid base64 (length not a multiple of 4)."""
    payload = {
        "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
        "claim_category": "CONSULTATION",
        "documents": [
            {"file_id": "U1", "actual_type": "PRESCRIPTION",
             "file_data": "AAA", "media_type": "image/jpeg"},
        ],
    }
    post = client.post("/api/claims", json=payload)
    assert post.status_code == 200  # the submission itself does not 500
    ref = post.json()["claim_reference"]

    resp = client.get(f"/api/claims/{ref}/documents/U1")
    assert resp.status_code == 500
    assert "corrupt" in resp.json()["detail"].lower()


def test_storage_io_error_on_persist_surfaces_as_500(tmp_path, monkeypatch):
    """A storage I/O error during persistence (after the decision is computed)
    surfaces as a 500 from the API rather than crashing. raise_server_exceptions
    is False so the unhandled server error is returned as a 500 response instead
    of being re-raised into the test."""
    api_module = _reload_api(tmp_path, monkeypatch)
    with TestClient(api_module.app, raise_server_exceptions=False) as c:
        def boom(*args, **kwargs):
            raise sqlite3.OperationalError("disk I/O error")

        # The store exists after lifespan startup (entered by the with-block);
        # shadow save() so persistence fails after process() has decided.
        monkeypatch.setattr(c.app.state.store, "save", boom)
        resp = c.post("/api/claims", json={
            "member_id": "EMP001", "policy_id": "PLUM_GHI_2024",
            "claim_category": "CONSULTATION", "treatment_date": "2024-11-01",
            "claimed_amount": 1500,
            "documents": [
                {"file_id": "F1", "actual_type": "PRESCRIPTION", "content":
                 {"patient_name": "Rajesh Kumar", "diagnosis": "Viral Fever"}},
                {"file_id": "F2", "actual_type": "HOSPITAL_BILL", "content":
                 {"patient_name": "Rajesh Kumar",
                  "line_items": [{"description": "Consultation Fee", "amount": 1500}],
                  "total": 1500}},
            ]})
    assert resp.status_code == 500
