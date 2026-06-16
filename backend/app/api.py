"""FastAPI layer: a thin HTTP shell over the framework-free engine.

Endpoints:
  POST /api/claims            submit a claim, returns the full ClaimResult
  GET  /api/claims            recent claims (review queue)
  GET  /api/claims/{ref}      one claim with its complete trace
  GET  /api/policy            policy summary for the UI form
  GET  /api/health            liveness

Document content can be provided as structured `content` (deterministic
path) or as base64 `file_data` (live Claude vision extraction, if an
ANTHROPIC_API_KEY is configured). Stored member history is merged into the
submission so fraud checks see prior claims automatically.
"""

import base64
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Literal, Optional

from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.models.claim import ClaimSubmission
from app.models.decision import ClaimResult
from app.config import EngineConfig
from app.logging_config import get_logger, setup_logging
from app.metrics import metrics
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator
from app.storage import ClaimStore, claim_documents_hash

DB_PATH = os.environ.get("CLAIMS_DB_PATH", "claims.db")
STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

# Logging is configured once at import (idempotent across the api-module reloads
# the test suite performs). It is observability only: no log statement below
# affects any decision, amount, trace, or control flow.
setup_logging()
logger = get_logger("api")


def _build_extractor():
    """Live vision extractor only if a key is configured; the engine and
    all tests run without it."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        from app.llm.extractor import ClaudeVisionExtractor
        return ClaudeVisionExtractor()
    return None


def _family_member_ids(policy, member_id: str) -> list[str]:
    """The member_ids sharing one family floater: the employee (primary member)
    and all dependents linked to that primary. Works whether the claimant is the
    employee or a dependent."""
    member = policy.get_member(member_id)
    if member is None:
        return [member_id]
    primary_id = member.primary_member_id or member.member_id
    primary = policy.get_member(primary_id) or member
    ids = {primary.member_id, member_id, *(primary.dependents or [])}
    return list(ids)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.policy = load_policy()
    app.state.config = EngineConfig.from_env()
    app.state.store = ClaimStore(DB_PATH)
    app.state.orchestrator = ClaimsOrchestrator(
        policy=app.state.policy, extractor=_build_extractor(),
        config=app.state.config,
    )
    logger.info("api_startup db_path=%s live_extractor=%s",
                DB_PATH, bool(os.environ.get("ANTHROPIC_API_KEY")))
    yield


app = FastAPI(
    title="Health Insurance Claims Processing System",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """Liveness: the process is up and serving. Cheap and dependency-free, so a
    crash-looping container is detected. Dependency health lives in /api/ready."""
    return {"status": "ok"}


@app.get("/api/ready")
def readiness(response: Response):
    """Readiness: verify the real dependencies, not just process liveness.

    - database: a quick `SELECT 1` must succeed (the hard dependency).
    - extractor: whether the live vision extractor is configured. The engine
      also runs in structured-content mode without it, so `not_configured` is a
      valid state, not a failure.

    Returns 503 with status `unhealthy` when the database is unreachable, so a
    load balancer can pull the instance; 200 `ok` otherwise.
    """
    checks: dict = {}
    db_ok = True
    try:
        app.state.store.ping()
        checks["database"] = {"status": "ok"}
    except Exception as e:  # noqa: BLE001 — report, don't crash the probe
        db_ok = False
        checks["database"] = {"status": "error", "detail": str(e)}
        logger.exception("readiness_database_check_failed")
    checks["extractor"] = {
        "status": "configured" if os.environ.get("ANTHROPIC_API_KEY")
        else "not_configured"}
    if not db_ok:
        response.status_code = 503
    return {"status": "ok" if db_ok else "unhealthy", "checks": checks}


@app.get("/metrics")
def metrics_snapshot():
    """In-process runtime metrics as a JSON snapshot (no external metrics
    backend): claims processed, the decision mix, manual-review rate, degraded
    claims, confidence buckets, and request latency."""
    return metrics.snapshot()


@app.get("/api/policy")
def policy_summary():
    p = app.state.policy
    return {
        "policy_id": p.policy_id,
        "policy_name": p.policy_name,
        "insurer": p.insurer,
        "categories": list(p.opd_categories.keys()),
        "per_claim_limit": p.coverage.per_claim_limit,
        "annual_opd_limit": p.coverage.annual_opd_limit,
        "network_hospitals": p.network_hospitals,
        "document_requirements": p.document_requirements,
        "members": [
            {"member_id": m.member_id, "name": m.name,
             "relationship": m.relationship}
            for m in p.members
        ],
    }


@app.post("/api/claims", response_model=ClaimResult)
def submit_claim(submission: ClaimSubmission):
    """Process a claim through the full pipeline and persist the result.

    The orchestrator guarantees graceful degradation: component failures
    inside the pipeline never surface as a 500; they appear in the result
    as component_failures with reduced confidence.
    """
    # Wall-clock start for the request-latency metric (side effect only — never
    # read by the engine and never affects the decision, amount, or trace).
    t0 = time.perf_counter()

    # Short per-request id so a request that fails before a claim reference is
    # minted (e.g. a persistence error) is still correlatable in the logs.
    request_id = uuid.uuid4().hex[:8]
    logger.info(
        "claim_received request_id=%s member_id=%s category=%s "
        "claimed_amount=%s documents=%d",
        request_id, submission.member_id,
        submission.claim_category.value if submission.claim_category else None,
        submission.claimed_amount, len(submission.documents))

    # Stamp the submission time if the caller did not provide one, so the
    # future-date sanity check has a reference point (engine stays
    # deterministic given inputs; this records the real submission time).
    if submission.submission_date is None:
        submission.submission_date = date.today()

    # H3 duplicate detection: a byte-identical document set already decided is
    # flagged so the engine's fraud stage routes a payable duplicate to review.
    documents_hash = claim_documents_hash(submission.documents)
    if submission.duplicate_of is None:
        submission.duplicate_of = app.state.store.find_decided_by_hash(documents_hash)

    # Merge stored history into the submission for fraud checks.
    stored = app.state.store.member_history(submission.member_id)
    known_ids = {c.claim_id for c in submission.claims_history}
    submission.claims_history.extend(
        c for c in stored if c.claim_id not in known_ids
    )

    # Family-floater year-to-date: sum the family's approved spend this policy
    # year from storage and inject it. The engine reads this injected total and
    # never touches storage itself (it stays a pure function of its inputs).
    policy = app.state.policy
    family_member_ids = None
    if policy.coverage.family_floater.enabled:
        family_member_ids = _family_member_ids(policy, submission.member_id)
        submission.family_ytd_amount = app.state.store.family_ytd_approved(
            family_member_ids,
            policy.policy_holder.policy_start_date,
            policy.policy_holder.policy_end_date,
        )

    result = app.state.orchestrator.process(submission)

    # Write the computed decision to the record BEFORE persistence. If the save
    # below fails, the decision and the reasoning behind it still exist in the
    # log (keyed by the claim reference) and are not lost. This is a log line
    # only — it changes no decision, amount, trace, or control flow.
    tu = result.token_usage or {}
    logger.info(
        "claim_decided request_id=%s claim_ref=%s decision=%s "
        "approved_amount=%s confidence=%.2f tokens_in=%s tokens_out=%s "
        "extraction_calls=%s",
        request_id, result.claim_reference,
        result.decision.value if result.decision else None,
        result.approved_amount, result.confidence_score,
        tu.get("input_tokens", 0), tu.get("output_tokens", 0),
        tu.get("calls", 0))

    try:
        app.state.store.save(
            result,
            treatment_date=submission.treatment_date,
            claimed_amount=submission.claimed_amount,
            documents_hash=documents_hash,
            same_day_limit=app.state.policy.fraud_thresholds.same_day_claims_limit,
            parent_reference=submission.parent_claim_reference,
            # Save-time family-floater backstop: re-check the shared cap under the
            # write lock so two concurrent same-family claims cannot both slip past
            # the stale total. Limit and policy year come from policy_terms.json.
            family_combined_limit=(policy.coverage.family_floater.combined_limit
                                   if policy.coverage.family_floater.enabled else None),
            family_member_ids=family_member_ids,
            policy_start=policy.policy_holder.policy_start_date,
            policy_end=policy.policy_holder.policy_end_date,
        )
        # 4b: persist raw uploaded documents (live submissions) for the reviewer
        # and the audit; structured submissions write nothing.
        app.state.store.save_documents(result.claim_reference, submission.documents)
    except Exception:
        # Persistence failed after the decision was computed. Capture the stack
        # trace keyed by the claim reference, then re-raise unchanged so the
        # response is exactly the 500 it is today (response behavior unchanged).
        logger.exception("claim_save_failed request_id=%s claim_ref=%s",
                         request_id, result.claim_reference)
        raise

    logger.info("claim_saved request_id=%s claim_ref=%s",
                request_id, result.claim_reference)

    # Runtime metrics: record the fully-processed, persisted claim. Pure side
    # effect — counts the FINAL (post-save-backstop) decision; never mutates it.
    metrics.record_claim(
        decision=result.decision.value if result.decision else None,
        status=result.status,
        confidence=result.confidence_score,
        degraded=bool(result.component_failures),
        latency_ms=(time.perf_counter() - t0) * 1000.0,
        input_tokens=tu.get("input_tokens", 0),
        output_tokens=tu.get("output_tokens", 0),
    )
    return result


@app.get("/api/claims")
def list_claims(limit: int = 50):
    return app.state.store.list_recent(limit=limit)


@app.get("/api/claims/{claim_reference}")
def get_claim(claim_reference: str):
    found = app.state.store.get(claim_reference)
    if not found:
        logger.info("claim_not_found claim_ref=%s", claim_reference)
        raise HTTPException(status_code=404, detail="Claim not found")
    return found


@app.get("/api/review-queue")
def review_queue():
    """Claims held for a human reviewer (review_status = PENDING_REVIEW)."""
    return app.state.store.list_held()


@app.get("/api/claims/{claim_reference}/documents/{file_id}")
def get_claim_document(claim_reference: str, file_id: str):
    """Fetch a stored raw document (live submissions only) for the reviewer."""
    found = app.state.store.get_document(claim_reference, file_id)
    if not found:
        logger.info("document_not_found claim_ref=%s file_id=%s",
                    claim_reference, file_id)
        raise HTTPException(status_code=404, detail="Document not found")
    media_type, b64 = found
    try:
        data = base64.b64decode(b64)
    except (ValueError, TypeError):
        logger.exception("stored_document_corrupt claim_ref=%s file_id=%s",
                         claim_reference, file_id)
        raise HTTPException(status_code=500, detail="Stored document is corrupt")
    return Response(content=data, media_type=media_type or "application/octet-stream")


class ResolveRequest(BaseModel):
    action: Literal["approve", "reject", "close"]
    reviewer_id: str
    reason: str  # required: 422 if missing
    approved_amount: Optional[float] = None  # required for a None-amount hold


@app.post("/api/claims/{claim_reference}/resolve")
def resolve_claim(claim_reference: str, body: ResolveRequest):
    """Reviewer resolves a held claim. reviewer_id is a passed-in stand-in for
    real auth (deferred). resolved_at is a server audit timestamp."""
    # Validate an explicit reviewer amount against the config bounds so a
    # fat-fingered or absurd amount is rejected rather than approved.
    if body.action == "approve" and body.approved_amount is not None:
        ceiling = (app.state.config.amount_ceiling
                   if app.state.config.amount_ceiling is not None
                   else app.state.policy.coverage.sum_insured_per_employee)
        if body.approved_amount < 0 or body.approved_amount > ceiling:
            logger.warning(
                "resolve_amount_out_of_range claim_ref=%s reviewer=%s amount=%s",
                claim_reference, body.reviewer_id, body.approved_amount)
            raise HTTPException(
                status_code=400,
                detail=(f"Approved amount must be between ₹0 and "
                        f"₹{ceiling:,.0f}."))
    resolved_at = datetime.now(timezone.utc).isoformat()
    outcome, result = app.state.store.resolve(
        claim_reference, body.action, body.reviewer_id, body.reason, resolved_at,
        approved_amount=body.approved_amount)
    if outcome == "NOT_FOUND":
        logger.info("resolve_claim_not_found claim_ref=%s", claim_reference)
        raise HTTPException(status_code=404, detail="Claim not found")
    if outcome == "NEEDS_AMOUNT":
        logger.warning("resolve_needs_amount claim_ref=%s reviewer=%s",
                       claim_reference, body.reviewer_id)
        raise HTTPException(
            status_code=400,
            detail="This claim was held before an amount was computed; approving "
                   "it requires an explicit approved_amount (read the bill in the "
                   "document viewer and enter the amount).")
    if outcome == "CONFLICT":
        if result and result.get("review_status") == "RESOLVED":
            detail = (f"Claim already resolved by {result.get('resolved_by')} "
                      f"at {result.get('resolved_at')}.")
        else:
            state = (result.get("review_status") or result.get("status")
                     if result else "unknown")
            detail = (f"Claim is not awaiting review (state: {state}); only a "
                      "claim held for review can be resolved.")
        logger.warning("resolve_conflict claim_ref=%s reviewer=%s",
                       claim_reference, body.reviewer_id)
        raise HTTPException(status_code=409, detail=detail)
    logger.info("claim_resolved claim_ref=%s action=%s reviewer=%s",
                claim_reference, body.action, body.reviewer_id)
    return result


# Serve the built frontend (single-service deploy) when present.
if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
