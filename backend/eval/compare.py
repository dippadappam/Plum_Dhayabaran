"""Normalization-aware field comparison and scoring.

Outcome taxonomy per field (the asymmetric-scoring contract):
- correct   : truth present, prediction matches (normalized)
- wrong     : truth present, prediction present but does not match  [hallucination]
- missed    : truth present, prediction null                        [abstention]
- spurious  : truth null, prediction present                        [hallucination]
- (truth null + prediction null is not scored)

precision = correct / (correct + wrong + spurious)
recall    = correct / (correct + wrong + missed)
hallucination rate = (wrong + spurious) / all predictions made
quality   = mean(correct=1.0, missed=0.5, wrong=0, spurious=0)
            -- a wrong value is penalized harder than an abstained null.

Text matching is DIRECTIONAL (label -> prediction): a prediction counts as
correct only when it equals or *contains* the whole label (or is a close fuzzy
variant). A prediction that is merely a *fragment* of the label does not count.
The earlier rule matched in both directions, so a truncated read scored as
correct and flattered both precision and recall.
"""

import re
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any, Optional

SCALAR_TEXT_FIELDS = ["patient_name", "doctor_name", "doctor_registration",
                      "hospital_name", "diagnosis", "treatment"]
DATE_FIELDS = ["date"]
NUMBER_FIELDS = ["total"]
EXACT_FIELDS = ["document_type"]

_DATE_FORMATS = ["%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d",
                 "%d %b %Y", "%d %B %Y", "%b %d, %Y", "%B %d, %Y"]


def _norm_text(s: str) -> str:
    s = re.sub(r"[.,;:()\-–—|]", " ", str(s).casefold())
    return " ".join(s.split())


def _text_match(label: str, pred: str) -> bool:
    """Directional: the PREDICTION must reproduce the LABEL, not the reverse.

    Matches when the normalized prediction equals the label or *contains* the
    whole label (the model may wrap it in surrounding text), or is a close
    fuzzy variant (OCR/casing/punctuation noise). A prediction that is only a
    *fragment* of the label (label contains pred) no longer counts: the old
    bidirectional rule credited a truncated read as correct and flattered both
    precision and recall. Direction is label -> prediction only.
    """
    nl, npred = _norm_text(label), _norm_text(pred)
    if not nl or not npred:
        return nl == npred
    if nl == npred or nl in npred:
        return True
    return SequenceMatcher(None, nl, npred).ratio() >= 0.85


def _parse_date(s: str) -> Optional[str]:
    s = str(s).strip()
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _parse_number(v: Any) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = re.sub(r"[^\d.\-]", "", str(v).replace(",", ""))
    try:
        return float(s)
    except ValueError:
        return None


def _date_match(a: str, b: str) -> bool:
    pa, pb = _parse_date(a), _parse_date(b)
    return pa is not None and pa == pb


def _number_match(a: Any, b: Any, tolerance: float = 1.0) -> bool:
    pa, pb = _parse_number(a), _parse_number(b)
    return pa is not None and pb is not None and abs(pa - pb) <= tolerance


def _present(v: Any) -> bool:
    if v is None:
        return False
    if isinstance(v, str):
        return v.strip() != ""
    if isinstance(v, (list, dict)):
        return len(v) > 0
    return True


def _outcome(truth_present: bool, pred_present: bool, matches: bool) -> Optional[str]:
    if truth_present and pred_present:
        return "correct" if matches else "wrong"
    if truth_present:
        return "missed"
    if pred_present:
        return "spurious"
    return None  # both absent: not scored


def compare_document(truth: dict[str, Any],
                     pred: dict[str, Any]) -> list[dict[str, Any]]:
    """Returns one record per scored field/sub-item:
    {field, outcome, truth, predicted}."""
    records: list[dict[str, Any]] = []

    def record(field: str, outcome: Optional[str], t: Any, p: Any):
        if outcome is not None:
            records.append({"field": field, "outcome": outcome,
                            "truth": t, "predicted": p})

    for f in SCALAR_TEXT_FIELDS:
        t, p = truth.get(f), pred.get(f)
        record(f, _outcome(_present(t), _present(p),
                           _present(t) and _present(p) and _text_match(t, p)), t, p)

    for f in DATE_FIELDS:
        t, p = truth.get(f), pred.get(f)
        record(f, _outcome(_present(t), _present(p),
                           _present(t) and _present(p) and _date_match(t, p)), t, p)

    for f in NUMBER_FIELDS:
        t, p = truth.get(f), pred.get(f)
        record(f, _outcome(_present(t), _present(p),
                           _present(t) and _present(p) and _number_match(t, p)), t, p)

    for f in EXACT_FIELDS:
        t, p = truth.get(f), pred.get(f)
        record(f, _outcome(_present(t), _present(p),
                           str(t) == str(p)), t, p)

    # Medicines: per-item greedy matching.
    t_meds = list(truth.get("medicines") or [])
    p_meds = list(pred.get("medicines") or [])
    unmatched_pred = list(p_meds)
    for tm in t_meds:
        hit = next((pm for pm in unmatched_pred if _text_match(tm, pm)), None)
        if hit is not None:
            unmatched_pred.remove(hit)
            record("medicines[]", "correct", tm, hit)
        else:
            record("medicines[]", "missed", tm, None)
    for pm in unmatched_pred:
        record("medicines[]", "spurious", None, pm)

    # Line items: match by description, then check amount (both must hold).
    t_items = list(truth.get("line_items") or [])
    p_items = list(pred.get("line_items") or [])
    unmatched_p = list(p_items)
    for ti in t_items:
        hit = next((pi for pi in unmatched_p
                    if _text_match(ti.get("description", ""),
                                   pi.get("description", ""))), None)
        if hit is not None:
            unmatched_p.remove(hit)
            ok = _number_match(ti.get("amount"), hit.get("amount"))
            record("line_items[]", "correct" if ok else "wrong", ti, hit)
        else:
            record("line_items[]", "missed", ti, None)
    for pi in unmatched_p:
        record("line_items[]", "spurious", None, pi)

    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts = {"correct": 0, "wrong": 0, "missed": 0, "spurious": 0}
    for r in records:
        counts[r["outcome"]] += 1
    c, w, m, s = (counts["correct"], counts["wrong"],
                  counts["missed"], counts["spurious"])
    predictions_made = c + w + s
    truth_total = c + w + m
    precision = c / predictions_made if predictions_made else None
    recall = c / truth_total if truth_total else None
    f1 = (2 * precision * recall / (precision + recall)
          if precision and recall and (precision + recall) > 0 else 0.0
          ) if predictions_made and truth_total else None
    hallucination = (w + s) / predictions_made if predictions_made else None
    scored = c + w + m + s
    quality = (c * 1.0 + m * 0.5) / scored if scored else None
    return {"counts": counts, "precision": precision, "recall": recall,
            "f1": f1, "hallucination_rate": hallucination, "quality": quality,
            "scored_fields": scored}
