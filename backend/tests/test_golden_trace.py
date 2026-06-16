"""Golden-trace test — pins the full ordered trace of the 12 official cases.

Decision/amount assertions (test_official_cases.py) do not catch a trace
regression: a reordered step, reworded detail, flipped status, or an
added/dropped step passes those tests as long as the decision is unchanged.
This test captures each official case's complete trace and asserts it against a
committed golden fixture, so any such change fails the suite. It replaces the
throwaway byte-identical scripts that were written by hand for each refactor.

The 12 cases are run through the orchestrator DIRECTLY — the same path the
byte-identical verification used — not through the API or the save() path. All
12 provide a category.

Run-varying fields (see the step-1 discovery): the claim reference CLM-<uuid> is
embedded in the orchestrator/start detail, and every step carries a wall-clock
timestamp. The reference is masked to a fixed placeholder; the timestamp is not
part of the asserted shape. Nothing else varies between runs.

Regenerate the golden ONLY after an intentional, reviewed trace change, from the
backend/ directory:

    python -m tests.test_golden_trace
"""

import json
import re
from pathlib import Path

import pytest

from app.models.claim import ClaimSubmission
from app.models.policy import load_policy
from app.orchestrator import ClaimsOrchestrator

CASES_PATH = Path(__file__).parent / "test_cases.json"
GOLDEN_PATH = Path(__file__).parent / "fixtures" / "golden_traces.json"

# The only run-varying value embedded in trace text (step-1 discovery): the
# generated claim reference. The per-step timestamp is the other run-varying
# field and is simply not captured below.
_CLM_RE = re.compile(r"CLM-[0-9A-F]{10}")
_CLM_PLACEHOLDER = "CLM-XXXXXXXXXX"


def _mask(value):
    """Recursively mask the run-varying claim reference in any string, including
    strings nested inside a step's `data` dict/list."""
    if isinstance(value, str):
        return _CLM_RE.sub(_CLM_PLACEHOLDER, value)
    if isinstance(value, dict):
        return {k: _mask(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_mask(v) for v in value]
    return value


def normalize_trace(result):
    """The deterministic trace shape the golden pins: the ordered list of steps,
    each {stage, check, status, detail, data}, with the claim reference masked
    and the timestamp dropped. Round-tripped through JSON so the runtime objects
    compare exactly equal to the JSON-loaded fixture (no type drift)."""
    steps = [
        {
            "stage": s.stage,
            "check": s.check,
            "status": s.status.value,
            "detail": _mask(s.detail),
            "data": _mask(s.data),
        }
        for s in result.trace
    ]
    return json.loads(json.dumps(steps, ensure_ascii=False, default=str))


def _load_cases():
    with open(CASES_PATH, "r", encoding="utf-8") as f:
        return {c["case_id"]: c for c in json.load(f)["test_cases"]}


_CASES = _load_cases()
CASE_IDS = sorted(_CASES)  # TC001 .. TC012


def _run(orchestrator, case_id):
    return orchestrator.process(
        ClaimSubmission.model_validate(_CASES[case_id]["input"]))


@pytest.fixture(scope="module")
def orchestrator():
    return ClaimsOrchestrator(policy=load_policy())


@pytest.fixture(scope="module")
def golden():
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def test_golden_covers_all_twelve(golden):
    """The golden fixture has exactly the 12 official cases — no missing or
    stray entries."""
    assert sorted(golden) == CASE_IDS


@pytest.mark.parametrize("case_id", CASE_IDS)
def test_golden_trace(case_id, orchestrator, golden):
    assert case_id in golden, (
        f"{case_id} is missing from {GOLDEN_PATH.name}; regenerate the golden "
        "with `python -m tests.test_golden_trace` after an intentional change.")
    actual = normalize_trace(_run(orchestrator, case_id))
    expected = golden[case_id]

    # Step count first — a clear message naming the (stage, check) of each step.
    assert len(actual) == len(expected), (
        f"{case_id}: trace step count changed (golden {len(expected)}, got "
        f"{len(actual)}).\n"
        f"  golden: {[(s['stage'], s['check']) for s in expected]}\n"
        f"  actual: {[(s['stage'], s['check']) for s in actual]}")

    # First differing step — point straight at it.
    for i, (a, e) in enumerate(zip(actual, expected)):
        assert a == e, (
            f"{case_id}: trace step {i} differs from the golden "
            f"(stage/check golden={e['stage']}/{e['check']}, "
            f"actual={a['stage']}/{a['check']}).\n"
            f"  golden: {e}\n"
            f"  actual: {a}")

    # Whole-trace equality — guarantees nothing else slipped past the loop.
    assert actual == expected, f"{case_id}: normalized trace != golden."


def _regenerate():
    """Write the golden fixture from the CURRENT output (the known-correct
    byte-identical state). Run from backend/: python -m tests.test_golden_trace"""
    orch = ClaimsOrchestrator(policy=load_policy())
    data = {cid: normalize_trace(orch.process(
        ClaimSubmission.model_validate(_CASES[cid]["input"]))) for cid in CASE_IDS}
    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {GOLDEN_PATH} ({len(data)} cases, "
          f"{sum(len(v) for v in data.values())} steps)")


if __name__ == "__main__":
    _regenerate()
