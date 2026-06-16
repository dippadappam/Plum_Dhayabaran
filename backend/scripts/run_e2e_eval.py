"""End-to-end eval runner. PAID and MANUAL — live Claude vision calls.
Never part of pytest.

Runs each claim bundle (real images, no category/amount/date provided)
through the full in-process pipeline and scores decision-level accuracy:
decision, approved amount, derived category, rejection reason.

Bundles flagged `expected_fail_until` (Batch 6 targets — per-line
categorization for multi-service bills, multi-bill aggregation) are bucketed
separately: a red there is expected and does NOT count against accuracy; a
green there is an XPASS, signalling the feature may now be implemented and the
flag can drop.

Usage (from backend/, ANTHROPIC_API_KEY required):
    python -m scripts.run_e2e_eval
"""

import base64
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.llm.extractor import ClaudeVisionExtractor  # noqa: E402
from app.models.claim import ClaimSubmission  # noqa: E402
from app.models.policy import load_policy  # noqa: E402
from app.orchestrator import ClaimsOrchestrator  # noqa: E402
from eval.e2e_bundles import BUNDLES_DIR, load_manifests, partition  # noqa: E402

RESULTS_DIR = BACKEND / "eval" / "results"
REPORT_PATH = BACKEND.parent / "docs" / "eval_baseline.md"


def score_bundle(orch: ClaimsOrchestrator, manifest: dict) -> dict:
    """Run one bundle through the full pipeline and score it against its
    expected outcome. The bundle directory is named by bundle_id."""
    bdir = BUNDLES_DIR / manifest["bundle_id"]
    docs = []
    for d in manifest["documents"]:
        data = base64.b64encode((bdir / d["file"]).read_bytes()).decode()
        docs.append({"file_id": d["file_id"], "file_name": d["file"],
                     "file_data": data, "media_type": "image/png"})
    submission = ClaimSubmission.model_validate({
        "member_id": manifest["member_id"],
        "policy_id": manifest["policy_id"],
        "documents": docs,
    })
    result = orch.process(submission)
    exp = manifest["expected"]
    outcome = result.decision.value if result.decision else result.status

    checks = []
    if "decision_any" in exp:
        checks.append(("decision", outcome in exp["decision_any"],
                       f"{outcome} in {exp['decision_any']}"))
    elif "decision" in exp:
        checks.append(("decision", outcome == exp["decision"],
                       f"{outcome} == {exp['decision']}"))
    if "approved_amount" in exp:
        ok = (result.approved_amount is not None
              and abs(result.approved_amount - exp["approved_amount"]) <= 1)
        checks.append(("amount", ok,
                       f"{result.approved_amount} ~= {exp['approved_amount']}"))
    if "category" in exp:
        checks.append(("category", result.claim_category == exp["category"],
                       f"{result.claim_category} == {exp['category']}"))
    if "rejection_reason" in exp:
        got = [r.value for r in result.rejection_reasons]
        checks.append(("reason", exp["rejection_reason"] in got,
                       f"{exp['rejection_reason']} in {got}"))

    ok_all = all(ok for _, ok, _ in checks)
    return {"bundle": manifest["bundle_id"], "outcome": outcome,
            "approved_amount": result.approved_amount,
            "category": result.claim_category,
            "confidence": result.confidence_score,
            "passed": ok_all,
            "expected_fail_until": manifest.get("expected_fail_until"),
            "checks": [{"check": c, "ok": ok, "detail": d}
                       for c, ok, d in checks]}


def main() -> int:
    manifests = load_manifests()
    if not manifests:
        print("No bundles found. Run: python -m scripts.build_eval_dataset")
        return 1
    active, xfail = partition(manifests)
    orch = ClaimsOrchestrator(policy=load_policy(),
                              extractor=ClaudeVisionExtractor())

    active_rows = [score_bundle(orch, m) for m in active]
    xfail_rows = [score_bundle(orch, m) for m in xfail]
    passed = sum(r["passed"] for r in active_rows)

    for r in active_rows:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"[{status}] {r['bundle']}: {r['outcome']}, "
              f"amount={r['approved_amount']}, category={r['category']}, "
              f"conf={r['confidence']}")
        for c in r["checks"]:
            mark = "ok " if c["ok"] else "BAD"
            print(f"        {mark} {c['check']}: {c['detail']}")
    print(f"\nDecision-level accuracy (active bundles): "
          f"{passed}/{len(active_rows)}")

    if xfail_rows:
        print("\nExpected-fail bundles (not yet implemented; bucketed separately "
              "so they do not read as regressions):")
        for r in xfail_rows:
            tag = ("XPASS — now green, the expected_fail_until flag can drop"
                   if r["passed"] else "red, as expected")
            print(f"  [{tag}] {r['bundle']}: {r['outcome']}, "
                  f"amount={r['approved_amount']} "
                  f"(target until {r['expected_fail_until']})")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    payload = {"kind": "e2e", "generated": datetime.now(timezone.utc).isoformat(),
               "passed": passed, "total": len(active_rows),
               "active_bundles": active_rows,
               "expected_fail_bundles": xfail_rows}
    (RESULTS_DIR / f"e2e_{stamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # Append the e2e section to the baseline report.
    md = ["\n## End-to-end decision accuracy\n",
          f"Generated: {payload['generated']}  ",
          f"Result: **{passed} of {len(active_rows)} active bundles decided "
          "as expected.**\n",
          "| Bundle | Outcome | Amount | Category | Conf. | Result |",
          "|---|---|---|---|---|---|"]
    for r in active_rows:
        md.append(f"| {r['bundle']} | {r['outcome']} | {r['approved_amount']} "
                  f"| {r['category']} | {r['confidence']} "
                  f"| {'PASS' if r['passed'] else 'FAIL'} |")
    if xfail_rows:
        md.append("\n### Expected-fail bundles (Batch 6 targets, bucketed "
                  "separately)\n")
        md.append("| Bundle | Outcome | Amount | Flagged until | Status |")
        md.append("|---|---|---|---|---|")
        for r in xfail_rows:
            tag = "**XPASS**" if r["passed"] else "red (expected)"
            md.append(f"| {r['bundle']} | {r['outcome']} | {r['approved_amount']} "
                      f"| {r['expected_fail_until']} | {tag} |")
    with open(REPORT_PATH, "a", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Appended e2e section to {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
