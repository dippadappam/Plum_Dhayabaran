"""Tests for the e2e claim-bundle index.

Confirms the new bundles parse, are wired into the runner's loader, and carry
the right expected-fail flags. No paid execution: bundles are only loaded and
classified, never run through the vision extractor (that is the manual paid
step). The runner (scripts.run_e2e_eval) uses the very same load_manifests()/
partition() helpers, so "flagged + bucketed here" == "flagged + bucketed there".
"""

import pytest

from eval.e2e_bundles import (
    BUNDLES_DIR,
    is_expected_fail,
    load_manifests,
    partition,
    validate_manifest,
)

MANIFESTS = load_manifests()
BY_ID = {m["bundle_id"]: m for m in MANIFESTS}

NEGATIVE_CONTROLS = ["neg_out_of_period", "neg_excluded_cosmetic", "neg_over_cap"]
# Both Batch-6 capabilities have landed: two_bill_aggregate in 6a (multi-bill
# aggregation) and multi_service_visit in 6b (per-line money math). All three
# real-shape bundles are now regular green bundles; none is flagged.
GREEN_REAL_SHAPE = {"dependent_consultation": 1170, "two_bill_aggregate": 2160,
                    "multi_service_visit": 3350}
REAL_SHAPE = list(GREEN_REAL_SHAPE)


def test_all_bundles_parse_and_validate():
    assert MANIFESTS, "no bundles found — run python -m scripts.build_eval_dataset"
    for m in MANIFESTS:
        validate_manifest(m)  # raises ValueError on missing keys / empty docs


@pytest.mark.parametrize("bundle_id", REAL_SHAPE + NEGATIVE_CONTROLS)
def test_new_bundle_present_and_document_files_exist(bundle_id):
    """Parses and is wired into the runner: every referenced image is on disk
    where the runner (BUNDLES_DIR / bundle_id / file) will look for it."""
    assert bundle_id in BY_ID, f"{bundle_id} bundle missing"
    m = BY_ID[bundle_id]
    bdir = BUNDLES_DIR / bundle_id
    assert m["documents"], f"{bundle_id} has no documents"
    for d in m["documents"]:
        assert (bdir / d["file"]).exists(), f"{bundle_id}/{d['file']} missing"


def test_no_bundle_is_flagged_after_batch6():
    """Both Batch-6 capabilities have landed (6a aggregation, 6b per-line money
    math), so no bundle carries an expected_fail_until flag anymore."""
    flagged = [m["bundle_id"] for m in MANIFESTS if is_expected_fail(m)]
    assert flagged == [], f"unexpected expected-fail bundles: {flagged}"


@pytest.mark.parametrize("bundle_id,amount", list(GREEN_REAL_SHAPE.items()))
def test_green_real_shape_not_flagged(bundle_id, amount):
    """Green real-shape bundles are NOT flagged expected-fail: dependent already
    passed (directional-dependent logic); two_bill went green in 6a (multi-bill
    aggregation); multi_service in 6b (per-line money math). Each asserts its
    policy-correct approved amount."""
    m = BY_ID[bundle_id]
    assert is_expected_fail(m) is False
    assert m["expected"]["decision"] == "APPROVED"
    assert m["expected"]["approved_amount"] == amount
    assert m.get("expected_math")


def test_partition_buckets_batch6_targets_apart_from_active():
    active, xfail = partition(MANIFESTS)
    active_ids = {m["bundle_id"] for m in active}
    xfail_ids = {m["bundle_id"] for m in xfail}
    assert xfail_ids == set()                # nothing flagged after Batch 6
    for green in GREEN_REAL_SHAPE:
        assert green in active_ids           # all three real-shape bundles active
    for nc in NEGATIVE_CONTROLS:
        assert nc in active_ids


@pytest.mark.parametrize("bundle_id,reason", [
    ("neg_out_of_period", "NOT_COVERED"),
    ("neg_excluded_cosmetic", "EXCLUDED_CONDITION"),
    ("neg_over_cap", "PER_CLAIM_EXCEEDED"),
])
def test_negative_controls_expect_rejection(bundle_id, reason):
    """Negative controls should be REJECTED today (engine already handles them),
    so they are active bundles, not expected-fail."""
    m = BY_ID[bundle_id]
    assert is_expected_fail(m) is False
    assert m["expected"]["decision"] == "REJECTED"
    assert m["expected"]["rejection_reason"] == reason
