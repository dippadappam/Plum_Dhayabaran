"""Unit tests for the directional extraction field matching (eval.compare).

No LLM, no network: synthetic predicted-vs-label pairs only. The eval scripts
themselves are paid/manual; this exercises the scoring logic that grades them.
The contract under test: a prediction counts as correct only when it matches
the label (equals or contains it, or a close fuzzy variant) — never the reverse
direction, where a fragment of the label used to score as correct.
"""

from eval.compare import _text_match, compare_document, summarize


# --- the directional unit, on _text_match(label, pred) directly -------------

def test_exact_and_normalized_match_counts():
    assert _text_match("Rajesh Kumar", "rajesh  kumar") is True
    assert _text_match("Dr. S. Iyer", "Dr S Iyer") is True


def test_prediction_containing_the_label_counts():
    # Forward direction kept: the prediction may wrap the label in extra text.
    assert _text_match("Rajesh Kumar", "Patient: Rajesh Kumar (M/39)") is True


def test_fragment_prediction_no_longer_counts():
    # The reverse direction: a prediction that is only a fragment of the label
    # used to count under the old bidirectional rule. It must not now.
    assert _text_match("Rajesh Kumar", "Rajesh") is False
    assert _text_match("Root Canal Treatment", "Root Canal") is False


def test_fuzzy_typo_still_tolerated():
    assert _text_match("Rajesh Kumar", "Rajesh Kumat") is True  # 1-char OCR slip


# --- the same contract through the public compare_document/summarize ---------

def _doc(**kw):
    base = {"document_type": None, "patient_name": None, "doctor_name": None,
            "doctor_registration": None, "hospital_name": None, "date": None,
            "diagnosis": None, "treatment": None, "medicines": [],
            "line_items": [], "total": None}
    base.update(kw)
    return base


def test_matching_prediction_is_correct():
    recs = compare_document(_doc(patient_name="Rajesh Kumar"),
                            _doc(patient_name="Rajesh Kumar"))
    pn = [r for r in recs if r["field"] == "patient_name"][0]
    assert pn["outcome"] == "correct"


def test_predicted_field_absent_from_label_is_spurious():
    # Label has no diagnosis; the prediction invents one -> spurious (counts
    # against precision, never as correct).
    recs = compare_document(_doc(), _doc(diagnosis="Viral Fever"))
    diag = [r for r in recs if r["field"] == "diagnosis"][0]
    assert diag["outcome"] == "spurious"


def test_fragment_prediction_is_wrong_not_correct():
    # Reverse-direction fragment: label present, prediction present but only a
    # fragment -> "wrong", not "correct".
    recs = compare_document(_doc(patient_name="Rajesh Kumar"),
                            _doc(patient_name="Rajesh"))
    pn = [r for r in recs if r["field"] == "patient_name"][0]
    assert pn["outcome"] == "wrong"


def test_reverse_direction_no_longer_flatters_precision_and_recall():
    """A fragment prediction scored as correct under the old bidirectional rule.
    Now it is wrong, so precision and recall both drop from a flattering 1.0."""
    truth = _doc(patient_name="Rajesh Kumar", doctor_name="Dr. Arun Sharma")
    pred = _doc(patient_name="Rajesh",            # fragment: now wrong
                doctor_name="Dr. Arun Sharma")    # exact: correct
    s = summarize(compare_document(truth, pred))
    # 1 correct, 1 wrong: precision = recall = 0.5, not the old flattering 1.0.
    assert s["counts"] == {"correct": 1, "wrong": 1, "missed": 0, "spurious": 0}
    assert s["precision"] == 0.5
    assert s["recall"] == 0.5


def test_line_item_fragment_description_does_not_match():
    """Line items match by description first; a fragment description must not
    pair with the labeled item (the reverse direction is gone there too)."""
    truth = _doc(line_items=[{"description": "Root Canal Treatment",
                              "amount": 8000}])
    pred = _doc(line_items=[{"description": "Root Canal", "amount": 8000}])
    recs = [r for r in compare_document(truth, pred) if r["field"] == "line_items[]"]
    outcomes = sorted(r["outcome"] for r in recs)
    # No description match -> the truth item is missed and the pred is spurious,
    # never a (flattering) correct.
    assert outcomes == ["missed", "spurious"]
    assert "correct" not in outcomes
