"""E2E claim-bundle index: load and classify bundles.

Deliberately free of any extractor / app import so it can be used by BOTH the
paid run_e2e_eval runner and the free pytest suite. The runner uses partition()
to bucket expected-fail bundles (e.g. Batch 6 targets) separately from the
active set, so a known-red bundle never reads as a regression; the test suite
uses load_manifests()/validate_manifest() to confirm bundles parse and are
flagged as intended — without running a single paid extraction.
"""

import json
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
BUNDLES_DIR = BACKEND / "eval" / "bundles"

REQUIRED_KEYS = ("bundle_id", "member_id", "policy_id", "documents", "expected")


def load_manifests(bundles_dir: Path = BUNDLES_DIR) -> list[dict]:
    """Every bundle.json under bundles_dir, parsed, sorted by directory name.
    The bundle directory is named by bundle_id, so the runner reconstructs the
    image path as bundles_dir / bundle_id."""
    manifests: list[dict] = []
    if not bundles_dir.exists():
        return manifests
    for d in sorted(bundles_dir.iterdir()):
        mf = d / "bundle.json"
        if mf.exists():
            manifests.append(json.loads(mf.read_text(encoding="utf-8")))
    return manifests


def validate_manifest(manifest: dict) -> None:
    """Raise ValueError if a manifest is missing required structure."""
    missing = [k for k in REQUIRED_KEYS if k not in manifest]
    if missing:
        raise ValueError(
            f"bundle {manifest.get('bundle_id', '?')} is missing keys: {missing}")
    if not manifest["documents"]:
        raise ValueError(f"bundle {manifest['bundle_id']} has no documents")


def is_expected_fail(manifest: dict) -> bool:
    """True for a bundle that encodes a policy-correct outcome the engine
    cannot produce yet (flagged via `expected_fail_until`, e.g. "batch6")."""
    return bool(manifest.get("expected_fail_until"))


def partition(manifests: list[dict]) -> tuple[list[dict], list[dict]]:
    """(active, expected_fail). Active bundles count toward decision accuracy;
    expected-fail bundles are reported on their own so a known-red Batch 6
    target does not read as a regression (and a green one reads as an XPASS)."""
    active = [m for m in manifests if not is_expected_fail(m)]
    xfail = [m for m in manifests if is_expected_fail(m)]
    return active, xfail
