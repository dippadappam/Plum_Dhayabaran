"""Storage-level tests, including the same-day backstop under concurrency.

The backstop relies on BEGIN IMMEDIATE serializing concurrent writers; that
only holds if a blocked writer WAITS for the lock (busy timeout) rather than
erroring with SQLITE_BUSY. This test spawns several threads that submit a
same-member, same-date, otherwise-payable claim at the same instant and
asserts the limit is enforced with no unhandled lock error.
"""

import sqlite3
import threading
import uuid
from datetime import date

from app.models.decision import ClaimResult, Decision, RejectionReason
from app.storage import ClaimStore

SAME_DAY_LIMIT = 2


def _approved_result() -> ClaimResult:
    return ClaimResult(
        claim_reference=f"CLM-{uuid.uuid4().hex[:10].upper()}",
        member_id="EMP001",
        claim_category="CONSULTATION",
        status="DECIDED",
        decision=Decision.APPROVED,
        approved_amount=1350.0,
        confidence_score=0.95,
    )


def test_same_day_backstop_is_concurrency_safe(tmp_path):
    db_path = str(tmp_path / "concur.db")
    store = ClaimStore(db_path)
    n_threads = 8
    treatment = date(2024, 11, 1)
    barrier = threading.Barrier(n_threads)
    errors: list[Exception] = []

    def worker():
        result = _approved_result()
        try:
            barrier.wait()  # release all threads together to maximize contention
            store.save(result, treatment_date=treatment, claimed_amount=1500.0,
                       documents_hash=uuid.uuid4().hex,
                       same_day_limit=SAME_DAY_LIMIT)
        except Exception as e:  # capture SQLITE_BUSY / any unhandled error
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 1. No unhandled lock errors: the blocked writers waited and serialized.
    assert not errors, f"unhandled errors under contention: {errors!r}"

    con = sqlite3.connect(db_path)
    try:
        total = con.execute(
            "SELECT COUNT(*) FROM claims WHERE member_id='EMP001' "
            "AND treatment_date=?", (treatment.isoformat(),)).fetchone()[0]
        approved = con.execute(
            "SELECT COUNT(*) FROM claims WHERE member_id='EMP001' "
            "AND treatment_date=? AND decision='APPROVED'",
            (treatment.isoformat(),)).fetchone()[0]
        manual = con.execute(
            "SELECT COUNT(*) FROM claims WHERE member_id='EMP001' "
            "AND treatment_date=? AND decision='MANUAL_REVIEW'",
            (treatment.isoformat(),)).fetchone()[0]
    finally:
        con.close()

    # 2. Every claim persisted exactly once.
    assert total == n_threads
    # 3. The same-day limit was enforced atomically: no more than the limit
    #    ended APPROVED, and the rest were held — even though all raced.
    assert approved <= SAME_DAY_LIMIT, f"expected <= {SAME_DAY_LIMIT}, got {approved}"
    assert approved + manual == n_threads


def test_family_ytd_approved_sums_across_family(tmp_path):
    """Fix 4: the family-floater year-to-date is a real storage rollup — the sum
    of approved_amount across the family's members for the policy year — which
    the API injects into the submission."""
    store = ClaimStore(str(tmp_path / "fam.db"))

    def approved(ref, member, amt):
        return ClaimResult(
            claim_reference=ref, member_id=member, claim_category="CONSULTATION",
            status="DECIDED", decision=Decision.APPROVED, approved_amount=amt,
            confidence_score=0.95)

    store.save(approved("CLM-A", "EMP001", 1350.0),
               treatment_date=date(2024, 11, 1), claimed_amount=1500.0)
    store.save(approved("CLM-B", "DEP001", 800.0),
               treatment_date=date(2024, 12, 1), claimed_amount=900.0)
    # A claim outside the policy year must NOT count.
    store.save(approved("CLM-C", "DEP002", 5000.0),
               treatment_date=date(2023, 1, 1), claimed_amount=5000.0)

    total = store.family_ytd_approved(
        ["EMP001", "DEP001", "DEP002"], date(2024, 4, 1), date(2025, 3, 31))
    assert total == 2150.0  # 1350 + 800; the 2023 claim is out of the policy year


def test_family_limit_save_time_backstop(tmp_path):
    """Fix 4 race-safety: a same-family claim that crosses the combined limit on
    the now-accurate total is caught by the save-time re-check and rejected as
    FAMILY_LIMIT_EXCEEDED — not saved as approved on a stale total. A claim that
    stays under still saves as approved."""
    db_path = str(tmp_path / "famrace.db")
    store = ClaimStore(db_path)
    fam = ["EMP001", "DEP001", "DEP002"]
    start, end = date(2024, 4, 1), date(2025, 3, 31)
    limit = 150000.0

    def mk(ref, member, amt):
        return ClaimResult(
            claim_reference=ref, member_id=member, claim_category="CONSULTATION",
            status="DECIDED", decision=Decision.APPROVED, approved_amount=amt,
            confidence_score=0.95)

    # Seed the family near the cap (stands in for prior approved spend).
    store.save(mk("CLM-SEED", "EMP001", 148000.0),
               treatment_date=date(2024, 5, 1), claimed_amount=148000.0)

    # Race loser: a same-family claim computed APPROVED on the stale total; the
    # save-time re-check sees 148000 + 3000 > 150000 and rejects it.
    crossing = mk("CLM-CROSS", "DEP001", 3000.0)
    store.save(crossing, treatment_date=date(2024, 11, 1), claimed_amount=3000.0,
               family_combined_limit=limit, family_member_ids=fam,
               policy_start=start, policy_end=end)
    assert crossing.decision == Decision.REJECTED
    assert RejectionReason.FAMILY_LIMIT_EXCEEDED in crossing.rejection_reasons
    assert crossing.approved_amount == 0
    # Part B: the save-time override is explained by a trace step whose recorded
    # outcome matches the persisted decision (the trace alone reconstructs it).
    bk = [s for s in crossing.trace if s.check == "family_floater_backstop"]
    assert len(bk) == 1
    assert bk[0].data["overridden_to"] == crossing.decision.value  # "REJECTED"

    # A same-family claim that stays under the cap still saves as APPROVED.
    under = mk("CLM-UNDER", "DEP002", 1000.0)
    store.save(under, treatment_date=date(2024, 11, 2), claimed_amount=1000.0,
               family_combined_limit=limit, family_member_ids=fam,
               policy_start=start, policy_end=end)
    assert under.decision == Decision.APPROVED
    assert under.approved_amount == 1000.0
    assert not any(s.check == "family_floater_backstop" for s in under.trace)

    # Persisted outcomes match (the crossing stored REJECTED, not approved).
    con = sqlite3.connect(db_path)
    try:
        decisions = dict(con.execute(
            "SELECT claim_reference, decision FROM claims").fetchall())
    finally:
        con.close()
    assert decisions["CLM-CROSS"] == "REJECTED"
    assert decisions["CLM-UNDER"] == "APPROVED"


def test_same_day_backstop_emits_trace_step(tmp_path):
    """Part B: the same-day save-time override is recorded as a trace step, so a
    race-loser's persisted MANUAL_REVIEW is reconstructable from the trace, not
    only from reasons[]."""
    store = ClaimStore(str(tmp_path / "sdtrace.db"))
    treatment = date(2024, 11, 1)
    # Two prior same-day DECIDED claims (saved without a limit -> APPROVED).
    store.save(_approved_result(), treatment_date=treatment,
               claimed_amount=1500.0, documents_hash="sd1")
    store.save(_approved_result(), treatment_date=treatment,
               claimed_amount=1500.0, documents_hash="sd2")
    # The third would-be-APPROVED claim crosses the same-day limit at save time.
    third = _approved_result()
    store.save(third, treatment_date=treatment, claimed_amount=1500.0,
               documents_hash="sd3", same_day_limit=SAME_DAY_LIMIT)
    assert third.decision == Decision.MANUAL_REVIEW
    bk = [s for s in third.trace if s.check == "same_day_backstop"]
    assert len(bk) == 1
    assert bk[0].data["overridden_to"] == third.decision.value  # "MANUAL_REVIEW"
    # Part 1: the race-loser is marked PENDING_REVIEW and surfaces in the review
    # queue (it was previously stuck in limbo — MANUAL_REVIEW but not listed).
    assert third.review_status == "PENDING_REVIEW"
    held_refs = {h["claim_reference"] for h in store.list_held()}
    assert third.claim_reference in held_refs
