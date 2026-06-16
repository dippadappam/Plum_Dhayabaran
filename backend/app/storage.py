"""SQLite persistence.

Stores every processed claim with its full result (decision, trace, all
audit data) and serves the member claim history that fraud checks read.
SQLite is deliberate for this scale: persistent across restarts, zero
separate server, a single file to deploy; the storage interface is thin so
production would swap in Postgres behind the same methods.
"""

import hashlib
import json
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

from app.models.claim import PriorClaim
from app.models.decision import (
    ClaimResult,
    Decision,
    RejectionReason,
    StepStatus,
    TraceStep,
)


def claim_documents_hash(documents) -> str:
    """A stable, order-independent hash of the full document SET.

    Catches a byte-identical resubmission of the same files (H3). It does NOT
    catch a re-photographed copy of the same bill — that near-duplicate case
    belongs with account-level behavioral fraud detection, not here.
    """
    per_doc: list[str] = []
    for d in documents:
        if d.file_data:
            h = hashlib.sha256(d.file_data.encode("utf-8")).hexdigest()
        elif d.content is not None:
            h = hashlib.sha256(
                json.dumps(d.content, sort_keys=True, default=str)
                .encode("utf-8")).hexdigest()
        else:
            h = hashlib.sha256(
                f"{d.file_id}:{d.actual_type}".encode("utf-8")).hexdigest()
        per_doc.append(h)
    return hashlib.sha256("|".join(sorted(per_doc)).encode("utf-8")).hexdigest()


# Columns added after the original schema, applied idempotently (CREATE TABLE
# IF NOT EXISTS does not alter an existing table, so a PRAGMA-based migration
# is required). Batch 4 will extend this dict with its lifecycle columns; the
# reserved names — review_status, resolved_by, resolved_at, resolution,
# resolution_reason, parent_reference, extraction_json — must be added as
# their own columns and must NOT overload `status` (H3 filters status='DECIDED').
_MIGRATION_COLUMNS: dict[str, str] = {
    "documents_hash": "TEXT",
    # Batch 4 (4a lifecycle + 4b parent link). Additive; the review lifecycle
    # lives in review_status so `status` is never overloaded. The reserved
    # `extraction_json` column is intentionally NOT added — the extracted
    # record rides in result_json (4b).
    "review_status": "TEXT",
    "resolved_by": "TEXT",
    "resolved_at": "TEXT",
    "resolution": "TEXT",
    "resolution_reason": "TEXT",
    "parent_reference": "TEXT",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS claims (
    claim_reference TEXT PRIMARY KEY,
    member_id       TEXT NOT NULL,
    claim_category  TEXT NOT NULL,
    treatment_date  TEXT,
    claimed_amount  REAL,
    status          TEXT NOT NULL,
    decision        TEXT,
    approved_amount REAL,
    confidence      REAL,
    result_json     TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_claims_member_date
    ON claims (member_id, treatment_date);

-- Raw uploaded documents (4b evidence home), kept out of the claims row and
-- result_json because they are large and read only when a reviewer opens one.
CREATE TABLE IF NOT EXISTS claim_documents (
    claim_reference TEXT NOT NULL,
    file_id         TEXT NOT NULL,
    media_type      TEXT,
    content_hash    TEXT,
    data            BLOB,
    PRIMARY KEY (claim_reference, file_id)
);
"""


class ClaimStore:
    def __init__(self, db_path: str | Path = "claims.db"):
        self.db_path = str(db_path)
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
            self._migrate(conn)

    def _conn(self) -> sqlite3.Connection:
        # Explicit busy timeout so a second concurrent BEGIN IMMEDIATE WAITS
        # for the write lock and serializes, instead of getting SQLITE_BUSY
        # and erroring immediately — that is what actually closes the same-day
        # race. (Python's sqlite3 default is 5s; we set it explicitly and
        # generously rather than depend on a library default.)
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 30000")
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotently add columns missing from an existing table."""
        existing = {r["name"] for r in conn.execute("PRAGMA table_info(claims)")}
        for col, decl in _MIGRATION_COLUMNS.items():
            if col not in existing:
                conn.execute(f"ALTER TABLE claims ADD COLUMN {col} {decl}")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_claims_hash "
                     "ON claims (documents_hash)")

    def ping(self) -> bool:
        """Readiness probe for the storage backend: a trivial query that raises
        if the database is unreachable. Additive — used only by the /api/ready
        health check, never on the decision path."""
        with self._conn() as conn:
            conn.execute("SELECT 1").fetchone()
        return True

    def save(self, result: ClaimResult, treatment_date: Optional[date],
             claimed_amount: Optional[float],
             documents_hash: Optional[str] = None,
             same_day_limit: Optional[int] = None,
             parent_reference: Optional[str] = None,
             family_combined_limit: Optional[float] = None,
             family_member_ids: Optional[list[str]] = None,
             policy_start: Optional[date] = None,
             policy_end: Optional[date] = None) -> None:
        # Explicit transaction control so the same-day backstop count and the
        # insert are atomic (BEGIN IMMEDIATE serializes concurrent writers).
        conn = self._conn()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            # Same-day backstop (closes the read-merge-write race): if a
            # concurrent insert already pushed this member over the same-day
            # limit since the engine read its snapshot, override an otherwise
            # payable decision to MANUAL_REVIEW. Skipped when the engine already
            # routed to review (decision is then not APPROVED/PARTIAL).
            if (same_day_limit is not None and treatment_date is not None
                    and result.status == "DECIDED"
                    and result.decision in (Decision.APPROVED, Decision.PARTIAL)):
                prior = conn.execute(
                    "SELECT COUNT(*) AS n FROM claims WHERE member_id = ? "
                    "AND treatment_date = ? AND status = 'DECIDED'",
                    (result.member_id, treatment_date.isoformat()),
                ).fetchone()["n"]
                if prior + 1 > same_day_limit:
                    original = result.decision
                    result.decision = Decision.MANUAL_REVIEW
                    result.manual_review_recommended = True
                    # Match the engine's normal MANUAL_REVIEW path so the
                    # race-loser surfaces in the review queue (list_held), not
                    # limbo. (REJECTED overrides below are terminal and need no
                    # review_status.)
                    result.review_status = "PENDING_REVIEW"
                    result.fraud_signals.append(
                        f"This is same-day claim number {prior + 1} for member "
                        f"{result.member_id} on {treatment_date.isoformat()}, "
                        f"above the same-day limit of {same_day_limit} (detected "
                        "at persistence; concurrent submissions).")
                    result.reasons.append(
                        "Routed to manual review: the member's same-day claim "
                        "count exceeded the limit (detected during persistence).")
                    # Trace the override so the persisted decision is fully
                    # reconstructable from the trace alone, not only from
                    # reasons[]. (The 12 official cases never reach save(), so
                    # this step never fires for them.)
                    result.trace.append(TraceStep(
                        stage="persistence", check="same_day_backstop",
                        status=StepStatus.FAILED,
                        detail=(f"Same-day backstop: claim {prior + 1} for member "
                                f"{result.member_id} on "
                                f"{treatment_date.isoformat()} is above the "
                                f"same-day limit of {same_day_limit} (detected at "
                                "persistence, after concurrent submissions). "
                                f"Decision overridden from {original.value} to "
                                "MANUAL_REVIEW."),
                        data={"prior_same_day": prior,
                              "same_day_limit": same_day_limit,
                              "overridden_from": original.value,
                              "overridden_to": Decision.MANUAL_REVIEW.value}))
            # Family-floater backstop (closes the same read-merge-write race for
            # the shared family cap): re-compute the family's approved spend this
            # policy year INSIDE this transaction — now accurate, including any
            # same-family claim saved since submission — and re-apply the family
            # limit. If this claim's approved amount plus the recomputed total
            # crosses the cap, override to the same FAMILY_LIMIT_EXCEEDED outcome
            # the engine produces, so the race-loser is handled exactly as it
            # would have been on an accurate total. Mirrors the same-day backstop
            # above; the combined limit is read from policy (passed in), never
            # hardcoded. Skipped when the engine already routed away from a
            # payable decision.
            if (family_combined_limit is not None and family_member_ids
                    and policy_start is not None and policy_end is not None
                    and result.status == "DECIDED"
                    and result.decision in (Decision.APPROVED, Decision.PARTIAL)):
                placeholders = ",".join("?" for _ in family_member_ids)
                fam_total = conn.execute(
                    f"SELECT COALESCE(SUM(approved_amount), 0) AS total "
                    f"FROM claims WHERE member_id IN ({placeholders}) "
                    f"AND status = 'DECIDED' AND approved_amount IS NOT NULL "
                    f"AND treatment_date IS NOT NULL "
                    f"AND treatment_date >= ? AND treatment_date <= ?",
                    (*family_member_ids, policy_start.isoformat(),
                     policy_end.isoformat()),
                ).fetchone()["total"]
                family_total = float(fam_total or 0.0)
                eligible = result.approved_amount or 0
                if family_total + eligible > family_combined_limit:
                    original = result.decision
                    result.decision = Decision.REJECTED
                    if RejectionReason.FAMILY_LIMIT_EXCEEDED \
                            not in result.rejection_reasons:
                        result.rejection_reasons.append(
                            RejectionReason.FAMILY_LIMIT_EXCEEDED)
                    result.approved_amount = 0
                    result.reasons.append(
                        f"This claim of ₹{eligible:,.0f} on top of "
                        f"₹{family_total:,.0f} already approved for the family "
                        f"this year exceeds the family floater limit of "
                        f"₹{family_combined_limit:,.0f} (detected at persistence; "
                        "concurrent submissions).")
                    # Trace the override so the persisted decision is fully
                    # reconstructable from the trace alone, not only from
                    # reasons[]. (The 12 official cases never reach save(), so
                    # this step never fires for them.)
                    result.trace.append(TraceStep(
                        stage="persistence", check="family_floater_backstop",
                        status=StepStatus.FAILED,
                        detail=(f"Family-floater backstop: family year-to-date "
                                f"₹{family_total:,.0f} + this claim "
                                f"₹{eligible:,.0f} exceeds the family floater "
                                f"limit ₹{family_combined_limit:,.0f} (detected at "
                                "persistence, after concurrent submissions). "
                                f"Decision overridden from {original.value} to "
                                "REJECTED."),
                        data={"family_ytd": family_total, "eligible": eligible,
                              "family_combined_limit": family_combined_limit,
                              "overridden_from": original.value,
                              "overridden_to": Decision.REJECTED.value}))
            conn.execute(
                """INSERT INTO claims (claim_reference, member_id, claim_category,
                   treatment_date, claimed_amount, status, decision,
                   approved_amount, confidence, result_json, documents_hash,
                   review_status, parent_reference)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    result.claim_reference,
                    result.member_id,
                    result.claim_category,
                    treatment_date.isoformat() if treatment_date else None,
                    claimed_amount,
                    result.status,
                    result.decision.value if result.decision else None,
                    result.approved_amount,
                    result.confidence_score,
                    result.model_dump_json(),
                    documents_hash,
                    result.review_status,
                    parent_reference,
                ),
            )
            conn.execute("COMMIT")
        except Exception:
            # Best-effort rollback; never let a failed rollback mask the
            # original error (e.g. if BEGIN IMMEDIATE itself could not acquire
            # the lock, there is no transaction to roll back).
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            conn.close()

    def save_documents(self, claim_reference: str, documents) -> None:
        """Persist raw uploaded documents (live submissions only) into
        claim_documents. Structured submissions (no file_data) write nothing.
        The base64 payload is stored as-is; it is decoded on fetch."""
        rows = [
            (claim_reference, d.file_id, d.media_type,
             hashlib.sha256(d.file_data.encode("utf-8")).hexdigest(), d.file_data)
            for d in documents if d.file_data
        ]
        if not rows:
            return
        with self._conn() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO claim_documents "
                "(claim_reference, file_id, media_type, content_hash, data) "
                "VALUES (?, ?, ?, ?, ?)", rows)

    def get_document(self, claim_reference: str,
                     file_id: str) -> Optional[tuple[Optional[str], str]]:
        """(media_type, base64_data) for a stored raw document, or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT media_type, data FROM claim_documents "
                "WHERE claim_reference = ? AND file_id = ?",
                (claim_reference, file_id)).fetchone()
        return (row["media_type"], row["data"]) if row else None

    def list_held(self) -> list[dict]:
        """Full result dicts of claims awaiting a reviewer (the review queue)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT result_json FROM claims "
                "WHERE review_status = 'PENDING_REVIEW' "
                "ORDER BY created_at DESC",
            ).fetchall()
        return [json.loads(r["result_json"]) for r in rows]

    def resolve(self, reference: str, action: str, reviewer_id: str,
                reason: str, resolved_at: str,
                approved_amount: Optional[float] = None) -> tuple[str, Optional[dict]]:
        """Atomic reviewer resolution of a held claim.

        Resolvable iff review_status == 'PENDING_REVIEW' (finality is
        review_status in {None auto-final, 'RESOLVED'}; NEEDS_RESUBMISSION and
        auto-final DECIDED claims are not resolvable). The SELECT-check and the
        UPDATE run in one BEGIN IMMEDIATE transaction, so two concurrent
        resolves cannot both win — the second sees 'RESOLVED' and conflicts.

        Approve amount (4b): if the held claim already carries a computed amount
        (fraud/identity/confidence holds), approve keeps it unless an explicit
        amount is given. If it carries no amount (pre-auth/H4/future-date/
        derivation holds), approve REQUIRES an explicit amount — the reviewer
        reads the bill in the document viewer and enters it; the amount is never
        recomputed from the stored record.

        Returns (outcome, result_dict): outcome in
        {"OK", "NOT_FOUND", "CONFLICT", "BAD_ACTION", "NEEDS_AMOUNT"}.
        """
        if action not in ("approve", "reject", "close"):
            return ("BAD_ACTION", None)
        conn = self._conn()
        conn.isolation_level = None
        try:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                "SELECT result_json, review_status FROM claims "
                "WHERE claim_reference = ?", (reference,)).fetchone()
            if row is None:
                conn.execute("ROLLBACK")
                return ("NOT_FOUND", None)
            if row["review_status"] != "PENDING_REVIEW":
                existing = json.loads(row["result_json"])
                conn.execute("ROLLBACK")
                return ("CONFLICT", existing)

            result = ClaimResult.model_validate_json(row["result_json"])
            if action == "approve":
                if approved_amount is not None:
                    result.approved_amount = approved_amount
                elif result.approved_amount is None:
                    # Held before the money math; reviewer must supply an amount.
                    existing = json.loads(row["result_json"])
                    conn.execute("ROLLBACK")
                    return ("NEEDS_AMOUNT", existing)
                result.decision = Decision.APPROVED        # keep/override amount
                resolution = "APPROVED"
            elif action == "reject":
                result.decision = Decision.REJECTED
                result.approved_amount = 0
                resolution = "REJECTED"
            else:  # close: leave the decision, record it was closed out
                resolution = "CLOSED"

            result.review_status = "RESOLVED"
            result.resolved_by = reviewer_id
            result.resolved_at = resolved_at
            result.resolution = resolution
            result.resolution_reason = reason
            result.manual_review_recommended = False
            note = (f"Reviewer {reviewer_id} resolved this claim at "
                    f"{resolved_at}: {action} ({resolution}): {reason}")
            result.reasons.append(note)
            result.trace.append(TraceStep(
                stage="reviewer", check="resolution",
                status=StepStatus.INFO, detail=note))

            conn.execute(
                "UPDATE claims SET status = ?, decision = ?, approved_amount = ?, "
                "result_json = ?, review_status = ?, resolved_by = ?, "
                "resolved_at = ?, resolution = ?, resolution_reason = ? "
                "WHERE claim_reference = ?",
                (
                    result.status,
                    result.decision.value if result.decision else None,
                    result.approved_amount,
                    result.model_dump_json(),
                    result.review_status,
                    result.resolved_by,
                    result.resolved_at,
                    result.resolution,
                    result.resolution_reason,
                    reference,
                ),
            )
            conn.execute("COMMIT")
            return ("OK", result.model_dump(mode="json"))
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.OperationalError:
                pass
            raise
        finally:
            conn.close()

    def find_decided_by_hash(self, documents_hash: Optional[str]) -> Optional[str]:
        """The reference of a FINISHED claim with the same full document-set
        hash, or None. Finished = status DECIDED AND review_status in {NULL
        (auto-final), 'RESOLVED'}. A claim still PENDING_REVIEW is explicitly
        NOT a duplicate target: a member resubmitting a claim that is still in
        the review queue must not be flagged as duplicating an "already-decided"
        claim (it has not been decided yet). A corrected resubmission also
        changes ≥1 document, so its set-hash differs regardless."""
        if not documents_hash:
            return None
        with self._conn() as conn:
            row = conn.execute(
                "SELECT claim_reference FROM claims WHERE documents_hash = ? "
                "AND status = 'DECIDED' "
                "AND (review_status IS NULL OR review_status = 'RESOLVED') "
                "ORDER BY created_at LIMIT 1",
                (documents_hash,),
            ).fetchone()
        return row["claim_reference"] if row else None

    def get(self, claim_reference: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT result_json FROM claims WHERE claim_reference = ?",
                (claim_reference,),
            ).fetchone()
        return json.loads(row["result_json"]) if row else None

    def list_recent(self, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT claim_reference, member_id, claim_category,
                          treatment_date, claimed_amount, status, decision,
                          approved_amount, confidence, created_at
                   FROM claims ORDER BY created_at DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def member_history(self, member_id: str) -> list[PriorClaim]:
        """Decided claims for a member, as fraud-check input."""
        with self._conn() as conn:
            rows = conn.execute(
                """SELECT claim_reference, treatment_date, claimed_amount
                   FROM claims
                   WHERE member_id = ? AND status = 'DECIDED'
                     AND treatment_date IS NOT NULL
                     AND claimed_amount IS NOT NULL
                   ORDER BY treatment_date""",
                (member_id,),
            ).fetchall()
        return [
            PriorClaim(
                claim_id=r["claim_reference"],
                date=date.fromisoformat(r["treatment_date"]),
                amount=r["claimed_amount"],
            )
            for r in rows
        ]

    def family_ytd_approved(self, member_ids: list[str], start: date,
                            end: date) -> float:
        """Total approved_amount across a family's members for DECIDED claims
        whose treatment date is within [start, end] (the policy year). The API
        injects this as the family-floater year-to-date; the engine never reads
        storage itself. Mirrors the same-day backstop's storage-query pattern
        (a real SQL rollup), not the caller-supplied annual input field."""
        if not member_ids:
            return 0.0
        placeholders = ",".join("?" for _ in member_ids)
        with self._conn() as conn:
            row = conn.execute(
                f"""SELECT COALESCE(SUM(approved_amount), 0) AS total
                    FROM claims
                    WHERE member_id IN ({placeholders})
                      AND status = 'DECIDED'
                      AND approved_amount IS NOT NULL
                      AND treatment_date IS NOT NULL
                      AND treatment_date >= ? AND treatment_date <= ?""",
                (*member_ids, start.isoformat(), end.isoformat()),
            ).fetchone()
        return float(row["total"] or 0.0)
