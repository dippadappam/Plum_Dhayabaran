"""Eval report generator.

Runs all 12 official test cases through the real pipeline and writes a
Markdown report showing, for each case: the input summary, the full decision
output (decision, amounts, reasons, line items, fraud signals, component
failures, confidence), whether it matched the expected contract, and the
complete trace. Per the assignment note: "show the full decision output for
each case, not just pass/fail."

Usage:
    python -m scripts.generate_eval_report          (from backend/)
Writes docs/eval_report.md at the repo root.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.models.claim import ClaimSubmission  # noqa: E402
from app.models.decision import Decision, RejectionReason  # noqa: E402
from app.models.policy import load_policy  # noqa: E402
from app.orchestrator import ClaimsOrchestrator  # noqa: E402

CASES_PATH = BACKEND / "tests" / "test_cases.json"
OUTPUT_PATH = BACKEND.parent / "docs" / "eval_report.md"


def check_expectations(case: dict, result) -> tuple[bool, list[str]]:
    """Programmatic checks mirroring the acceptance suite, reported per case."""
    checks: list[str] = []
    ok = True

    def record(passed: bool, label: str):
        nonlocal ok
        checks.append(f"{'PASS' if passed else 'FAIL'} — {label}")
        if not passed:
            ok = False

    expected = case["expected"]
    exp_decision = expected.get("decision")

    if exp_decision is None:
        record(result.decision is None,
               "No claim decision made (decision is null)")
        record(result.status == "NEEDS_RESUBMISSION",
               "Status is NEEDS_RESUBMISSION with actionable document issues")
        record(bool(result.document_issues),
               "Specific document issue(s) surfaced to the member")
    else:
        record(result.decision is not None and result.decision.value == exp_decision,
               f"Decision is {exp_decision}")

    if "approved_amount" in expected:
        record(result.approved_amount == expected["approved_amount"],
               f"Approved amount is ₹{expected['approved_amount']:,}")

    if "rejection_reasons" in expected:
        got = [r.value for r in result.rejection_reasons]
        record(got == expected["rejection_reasons"],
               f"Rejection reasons are exactly {expected['rejection_reasons']}")

    conf = expected.get("confidence_score")
    if conf and conf.startswith("above"):
        threshold = float(conf.split()[-1])
        record(result.confidence_score > threshold,
               f"Confidence {result.confidence_score:.2f} is above {threshold}")

    # Case-specific behavioral contracts.
    cid = case["case_id"]
    text_all = " ".join(result.reasons) + " " + " ".join(s.detail for s in result.trace)
    if cid == "TC001":
        msgs = " ".join(i.message for i in result.document_issues)
        record("PRESCRIPTION" in msgs and "HOSPITAL_BILL" in msgs,
               "Message names the uploaded type and the required type")
    if cid == "TC002":
        unread = [i for i in result.document_issues if i.issue_code == "UNREADABLE"]
        record(bool(unread) and unread[0].file_id == "F004",
               "The specific unreadable document (F004) is identified")
        record(bool(unread) and "re-upload" in unread[0].message.lower(),
               "Member asked to re-upload that document, claim not rejected")
    if cid == "TC003":
        mism = [i for i in result.document_issues if i.issue_code == "PATIENT_MISMATCH"]
        record(bool(mism) and "Rajesh Kumar" in mism[0].message
               and "Arjun Mehta" in mism[0].message,
               "Both patient names found on the documents are surfaced")
    if cid == "TC005":
        record("2024-11-30" in text_all,
               "States the date the member becomes eligible (2024-11-30)")
    if cid == "TC006":
        record(len(result.line_items) == 2 and
               any(li.status == "REJECTED" and li.reason for li in result.line_items),
               "Line items itemized with a reason on each rejection")
    if cid == "TC007":
        record("resubmit" in " ".join(result.reasons).lower(),
               "Tells the member how to resubmit with pre-auth")
    if cid == "TC008":
        record("5,000" in " ".join(result.reasons) and "7,500" in " ".join(result.reasons),
               "States the per-claim limit and the claimed amount")
    if cid == "TC009":
        record(bool(result.fraud_signals),
               "The specific triggering signals are included in the output")
    if cid == "TC010":
        b = result.amount_breakdown
        record(b is not None and b.network_discount_amount == 900 and b.copay_amount == 360,
               "Breakdown shows discount (₹900) before co-pay (₹360)")
    if cid == "TC011":
        record(bool(result.component_failures),
               "Output indicates a component failed and was skipped")
        record(result.manual_review_recommended,
               "Manual review recommended due to incomplete processing")
    return ok, checks


def fmt_result(result) -> str:
    lines = []
    lines.append(f"- Status: `{result.status}`")
    lines.append(f"- Decision: `{result.decision.value if result.decision else 'null'}`")
    if result.approved_amount is not None:
        lines.append(f"- Approved amount: ₹{result.approved_amount:,.0f}")
    if result.rejection_reasons:
        lines.append("- Rejection reasons: "
                     + ", ".join(f"`{r.value}`" for r in result.rejection_reasons))
    lines.append(f"- Confidence: {result.confidence_score:.2f} "
                 f"({'; '.join(result.confidence_factors)})")
    if result.reasons:
        lines.append("- Explanations:")
        lines.extend(f"  - {r}" for r in result.reasons)
    if result.document_issues:
        lines.append("- Document issues:")
        lines.extend(
            f"  - [{i.issue_code}] {i.message} Action: {i.action_required}"
            for i in result.document_issues
        )
    if result.line_items:
        lines.append("- Line items:")
        lines.extend(
            f"  - {li.status}: {li.description} — claimed ₹{li.claimed_amount:,.0f}, "
            f"approved ₹{li.approved_amount:,.0f}. {li.reason}"
            for li in result.line_items
        )
    if result.amount_breakdown:
        b = result.amount_breakdown
        lines.append(
            f"- Amount math: claimed ₹{b.claimed_amount:,.0f} → eligible "
            f"₹{b.eligible_amount:,.0f} → network discount "
            f"{b.network_discount_percent:.0f}% (−₹{b.network_discount_amount:,.0f}) → "
            f"₹{b.amount_after_discount:,.0f} → co-pay {b.copay_percent:.0f}% "
            f"(−₹{b.copay_amount:,.0f}) → approved ₹{b.approved_amount:,.0f}"
        )
    if result.fraud_signals:
        lines.append("- Fraud signals:")
        lines.extend(f"  - {s}" for s in result.fraud_signals)
    if result.component_failures:
        lines.append("- Component failures:")
        lines.extend(
            f"  - `{c.component}`: {c.error} Impact: {c.impact}"
            for c in result.component_failures
        )
    if result.manual_review_recommended:
        lines.append("- Manual review recommended: yes")
    return "\n".join(lines)


def fmt_trace(result) -> str:
    rows = ["| # | Stage | Check | Status | Detail |", "|---|---|---|---|---|"]
    for i, s in enumerate(result.trace, 1):
        detail = s.detail.replace("|", "\\|")
        rows.append(f"| {i} | {s.stage} | {s.check} | {s.status.value} | {detail} |")
    return "\n".join(rows)


def main() -> int:
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        cases = json.load(f)["test_cases"]

    orchestrator = ClaimsOrchestrator(policy=load_policy())
    sections = []
    passed_count = 0

    for case in cases:
        submission = ClaimSubmission.model_validate(case["input"])
        result = orchestrator.process(submission)
        ok, checks = check_expectations(case, result)
        passed_count += ok

        inp = case["input"]
        sections.append(f"""## {case['case_id']}: {case['case_name']} — {'MATCHED' if ok else 'MISMATCH'}

**Scenario:** {case['description']}

**Input:** member `{inp['member_id']}`, category `{inp['claim_category']}`, treatment date {inp['treatment_date']}, claimed ₹{inp['claimed_amount']:,}, {len(inp['documents'])} document(s).

**Expected contract checks:**
{chr(10).join('- ' + c for c in checks)}

**Full decision output:**
{fmt_result(result)}

**Complete trace:**
{fmt_trace(result)}
""")

    report = f"""# Eval Report — Health Insurance Claims Processing System

Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}
Result: **{passed_count} of {len(cases)} cases matched the expected contract.**

Method: every case below was run through the real pipeline (orchestrator and
all agents) with no LLM involvement; all twelve cases carry structured
document content, so decisions are fully deterministic and reproducible.
This report is generated by `scripts/generate_eval_report.py`; re-running it
reproduces identical decisions. The same contracts are enforced in CI by
`tests/test_official_cases.py`.

---

{chr(10).join(sections)}
"""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH} — {passed_count}/{len(cases)} matched")
    return 0 if passed_count == len(cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
