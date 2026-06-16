"""Extraction eval runner. PAID and MANUAL — live Claude vision calls.
Never part of pytest.

Runs every dataset item through ClaudeVisionExtractor, scores field-level
precision/recall/F1 with normalization-aware comparison and asymmetric
hallucination accounting, breaks results down per document type and per
degradation profile, reports confidence reliability, and diffs against the
previous run.

Usage (from backend/, ANTHROPIC_API_KEY required):
    python -m scripts.run_extraction_eval [--workers 3] [--limit N] [--runs N]

With --runs > 1 the dataset is evaluated N times and a run-to-run variance
table (mean / min / max / stdev per metric) is reported, so the N=3 variance
run needs no code change.
"""

import argparse
import base64
import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.llm.extractor import ClaudeVisionExtractor, ExtractionError  # noqa: E402
from app.models.claim import SubmittedDocument  # noqa: E402
from eval.compare import compare_document, summarize  # noqa: E402

DATASET_DIR = BACKEND / "eval" / "dataset"
RESULTS_DIR = BACKEND / "eval" / "results"
REPORT_PATH = BACKEND.parent / "docs" / "eval_baseline.md"

CONF_BUCKETS = [(0.0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]


def load_items(limit=None):
    items = []
    for d in sorted(DATASET_DIR.iterdir()):
        if not (d / "truth.json").exists():
            continue
        items.append({
            "id": d.name,
            "image": d / "image.png",
            "truth": json.loads((d / "truth.json").read_text(encoding="utf-8")),
            "profile": json.loads((d / "degradations.json")
                                  .read_text(encoding="utf-8"))["profile"],
            "doc_type": d.name.split("__")[0],
        })
    return items[:limit] if limit else items


def run_item(extractor, item):
    data = base64.b64encode(item["image"].read_bytes()).decode()
    doc = SubmittedDocument(file_id=item["id"], file_data=data,
                            media_type="image/png")
    try:
        pred = extractor.extract(doc)
        error = None
    except ExtractionError as e:
        pred, error = {}, str(e)
    records = compare_document(item["truth"], pred)
    return {**item, "image": str(item["image"]), "pred": pred,
            "records": records, "error": error,
            "confidence": pred.get("_extraction_confidence")}


def group_summaries(results, key):
    groups = {}
    for r in results:
        groups.setdefault(r[key], []).extend(r["records"])
    return {k: summarize(v) for k, v in sorted(groups.items())}


def reliability(results):
    rows = []
    for lo, hi in CONF_BUCKETS:
        bucket = [r for r in results
                  if r["confidence"] is not None and lo <= r["confidence"] < hi]
        records = [rec for r in bucket for rec in r["records"]]
        s = summarize(records)
        accuracy = (s["counts"]["correct"] / s["scored_fields"]
                    if s["scored_fields"] else None)
        rows.append({"bucket": f"{lo:.1f}-{min(hi, 1.0):.1f}",
                     "documents": len(bucket),
                     "mean_stated": (sum(r["confidence"] for r in bucket)
                                     / len(bucket)) if bucket else None,
                     "field_accuracy": accuracy})
    return rows


def fmt_pct(v):
    return f"{v * 100:5.1f}%" if v is not None else "    --"


def previous_results():
    if not RESULTS_DIR.exists():
        return None
    files = sorted(RESULTS_DIR.glob("extraction_*.json"))
    return json.loads(files[-1].read_text(encoding="utf-8")) if files else None


def variance_across_runs(overalls: list[dict]) -> dict:
    """Per-metric mean / min / max / stdev across N independent eval runs.
    Sample stdev (n-1); a single run reports stdev 0.0. Live extraction is
    non-deterministic, so this quantifies how much a single run can be trusted.
    """
    metrics = ["precision", "recall", "f1", "hallucination_rate", "quality"]
    out: dict = {}
    for m in metrics:
        vals = [o[m] for o in overalls if o.get(m) is not None]
        if not vals:
            out[m] = None
            continue
        mean = sum(vals) / len(vals)
        stdev = (math.sqrt(sum((v - mean) ** 2 for v in vals) / (len(vals) - 1))
                 if len(vals) > 1 else 0.0)
        out[m] = {"mean": mean, "min": min(vals), "max": max(vals),
                  "stdev": stdev, "n": len(vals)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--runs", type=int, default=1,
                    help="Evaluate the dataset N times and report run-to-run "
                         "variance (mean/min/max/stdev per metric).")
    args = ap.parse_args()

    items = load_items(args.limit)
    if not items:
        print("No dataset found. Run: python -m scripts.build_eval_dataset")
        return 1
    extractor = ClaudeVisionExtractor()
    n_runs = max(1, args.runs)
    print(f"Evaluating {len(items)} documents with {args.workers} workers, "
          f"{n_runs} run(s)...")

    # N-run loop: each run re-extracts the whole dataset (live extraction is
    # non-deterministic). Detailed breakdowns below report the last run; the
    # variance table aggregates every run.
    per_run_overall: list[dict] = []
    results: list = []
    for run_idx in range(n_runs):
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(pool.map(lambda it: run_item(extractor, it), items))
        per_run_overall.append(
            summarize([rec for r in results for rec in r["records"]]))
        if n_runs > 1:
            ov = per_run_overall[-1]
            print(f"  run {run_idx + 1}/{n_runs}: "
                  f"P={fmt_pct(ov['precision']).strip()} "
                  f"R={fmt_pct(ov['recall']).strip()} "
                  f"F1={fmt_pct(ov['f1']).strip()}")

    overall = per_run_overall[-1]
    variance = variance_across_runs(per_run_overall)
    by_type = group_summaries(results, "doc_type")
    by_profile = group_summaries(results, "profile")
    by_field = {}
    for r in results:
        for rec in r["records"]:
            by_field.setdefault(rec["field"], []).append(rec)
    by_field = {k: summarize(v) for k, v in sorted(by_field.items())}
    rel = reliability(results)
    failures = [{"id": r["id"], "error": r["error"]}
                for r in results if r["error"]]

    prev = previous_results()
    payload = {
        "kind": "extraction", "generated": datetime.now(timezone.utc).isoformat(),
        "items": len(results), "runs": n_runs,
        "per_run_overall": per_run_overall, "variance": variance,
        "overall": overall, "by_doc_type": by_type,
        "by_profile": by_profile, "by_field": by_field,
        "reliability": rel, "extraction_failures": failures,
        "details": [{"id": r["id"], "profile": r["profile"],
                     "confidence": r["confidence"],
                     "summary": summarize(r["records"]),
                     "mismatches": [rec for rec in r["records"]
                                    if rec["outcome"] != "correct"]}
                    for r in results],
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (RESULTS_DIR / f"extraction_{stamp}.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    # ----- console + markdown report -----
    def table(title, summaries):
        lines = [f"### {title}", "",
                 "| Group | P | R | F1 | Halluc. | Quality | Fields |",
                 "|---|---|---|---|---|---|---|"]
        for name, s in summaries.items():
            lines.append(
                f"| {name} | {fmt_pct(s['precision'])} | {fmt_pct(s['recall'])} "
                f"| {fmt_pct(s['f1'])} | {fmt_pct(s['hallucination_rate'])} "
                f"| {fmt_pct(s['quality'])} | {s['scored_fields']} |")
        return "\n".join(lines) + "\n"

    md = [f"# Extraction Eval Baseline\n",
          f"Generated: {payload['generated']}  ",
          f"Documents: {len(results)} | Extraction failures: {len(failures)}\n",
          "## Overall\n",
          f"- Precision: **{fmt_pct(overall['precision']).strip()}**",
          f"- Recall: **{fmt_pct(overall['recall']).strip()}**",
          f"- F1: **{fmt_pct(overall['f1']).strip()}**",
          f"- **Hallucination rate: {fmt_pct(overall['hallucination_rate']).strip()}**"
          " (wrong values + spurious fields / all predictions; a wrong value"
          " is graded worse than an abstained null)",
          f"- Quality score: {fmt_pct(overall['quality']).strip()}"
          " (correct=1, abstained=0.5, wrong=0)",
          f"- Outcome counts: {overall['counts']}\n",
          table("Per document type", by_type),
          table("Per degradation profile", by_profile),
          table("Per field", by_field),
          "### Confidence reliability (stated vs measured)\n",
          "| Stated confidence | Documents | Mean stated | Field accuracy |",
          "|---|---|---|---|"]
    for row in rel:
        md.append(f"| {row['bucket']} | {row['documents']} "
                  f"| {fmt_pct(row['mean_stated'])} "
                  f"| {fmt_pct(row['field_accuracy'])} |")
    if n_runs > 1:
        md.append(f"\n### Run-to-run variance ({n_runs} runs)\n")
        md.append("| Metric | Mean | Min | Max | Stdev |")
        md.append("|---|---|---|---|---|")
        for m in ["precision", "recall", "f1", "hallucination_rate", "quality"]:
            v = variance[m]
            if v is None:
                continue
            md.append(f"| {m} | {fmt_pct(v['mean'])} | {fmt_pct(v['min'])} "
                      f"| {fmt_pct(v['max'])} | {v['stdev'] * 100:.2f} pts |")
    if prev:
        md.append("\n### Regression vs previous run\n")
        po = prev["overall"]
        for metric in ["precision", "recall", "f1", "hallucination_rate"]:
            cur, old = overall[metric], po.get(metric)
            if cur is not None and old is not None:
                md.append(f"- {metric}: {fmt_pct(old).strip()} -> "
                          f"{fmt_pct(cur).strip()} ({(cur - old) * 100:+.1f} pts)")
    else:
        md.append("\n_No previous run: this is the baseline._")

    report = "\n".join(md) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote {REPORT_PATH} and eval/results/extraction_{stamp}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
