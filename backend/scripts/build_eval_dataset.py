"""Build the labeled eval dataset and the e2e claim bundles.

Free to run (no API calls). Deterministic from --seed.

Dataset layout (per the design):
    eval/dataset/{id}/image.png
    eval/dataset/{id}/truth.json
    eval/dataset/{id}/degradations.json   {"profile": ..., "applied": [...]}

Bundles layout:
    eval/bundles/{id}/{*.png}
    eval/bundles/{id}/bundle.json         claim input + expected outcome

Usage (from backend/):
    python -m scripts.build_eval_dataset [--seed 20260613]
"""

import argparse
import json
import random
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from eval.degrade import apply_degradations  # noqa: E402
from eval.generator import (  # noqa: E402
    GENERATORS,
    make_hospital_bill,
    make_prescription,
)

DATASET_DIR = BACKEND / "eval" / "dataset"
BUNDLES_DIR = BACKEND / "eval" / "bundles"

SINGLE_DEGRADATIONS = ["rotate90", "skew", "blur", "shadow", "glare",
                       "noise", "lowres", "crumple", "stamp"]
PROFILES = (
    [("clean", [], 0.0)]
    + [(name, [name], 0.5) for name in SINGLE_DEGRADATIONS]
    + [("combined_moderate", ["skew", "shadow", "noise", "signature"], 0.3),
       ("combined_heavy", ["skew", "blur", "shadow", "glare", "noise",
                           "crumple"], 0.8)]
)


def _save_item(item_id: str, image, truth: dict, profile: str,
               applied: list) -> None:
    out = DATASET_DIR / item_id
    out.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(out / "image.png")
    (out / "truth.json").write_text(
        json.dumps(truth, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "degradations.json").write_text(
        json.dumps({"profile": profile, "applied": applied}, indent=2),
        encoding="utf-8")


def build_dataset(seed: int) -> int:
    count = 0
    for doc_type, maker in GENERATORS.items():
        for profile, names, severity in PROFILES:
            rng = random.Random(f"{seed}:{doc_type}:{profile}")
            gen = maker(rng)
            img, applied = apply_degradations(gen.image, names, rng, severity)
            item_id = f"{doc_type}__{profile}"
            _save_item(item_id, img, gen.truth, profile, applied)
            count += 1
    # Handwritten prescriptions (the sample guide's headline variation).
    for profile, names, severity in [("clean", [], 0.0), ("blur", ["blur"], 0.5),
                                     ("skew", ["skew"], 0.5)]:
        rng = random.Random(f"{seed}:prescription_hw:{profile}")
        gen = make_prescription(rng, handwritten=True)
        img, applied = apply_degradations(gen.image, names, rng, severity)
        applied.insert(0, {"name": "handwriting_font", "severity": 1.0,
                           "params": {}})
        item_id = f"prescription_hw__{profile}"
        _save_item(item_id, img, gen.truth,
                   f"handwritten_{profile}" if profile != "clean"
                   else "handwritten", applied)
        count += 1
    return count


# ---------------------------------------------------------------------------
# E2E claim bundles: full claims with known expected decisions. Members and
# names align with the policy roster so the real pipeline adjudicates them.
# ---------------------------------------------------------------------------

def build_bundles(seed: int) -> int:
    rng = random.Random(f"{seed}:bundles")
    date_nov = ("15-11-2024", "2024-11-15")
    date_oct = ("18-10-2024", "2024-10-18")
    date_pre = ("15-02-2024", "2024-02-15")  # before the policy start date

    def bundle(bundle_id, member_id, docs, expected, degrade_map=None,
               expected_fail_until=None, expected_math=None):
        out = BUNDLES_DIR / bundle_id
        out.mkdir(parents=True, exist_ok=True)
        manifest = {"bundle_id": bundle_id, "member_id": member_id,
                    "policy_id": "PLUM_GHI_2024", "documents": [],
                    "expected": expected}
        # A bundle that encodes a policy-correct outcome the engine cannot
        # produce yet is flagged so the runner buckets it separately (a red
        # there is expected, not a regression). The math is carried with the
        # bundle so the target is auditable before Batch 6 makes it green.
        if expected_fail_until:
            manifest["expected_fail_until"] = expected_fail_until
        if expected_math:
            manifest["expected_math"] = expected_math
        for name, gen in docs:
            img = gen.image
            if degrade_map and name in degrade_map:
                for deg_names, severity in degrade_map[name]:
                    img, _ = apply_degradations(img, deg_names, rng, severity)
            img.convert("RGB").save(out / f"{name}.png")
            manifest["documents"].append({"file": f"{name}.png",
                                          "file_id": f"{bundle_id}:{name}"})
        (out / "bundle.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return 1

    n = 0
    arun = ("Dr. Arun Sharma", "KA/45678/2015", "MBBS, MD (General Medicine)")

    # 1. Clean consultation -> APPROVED 1350 (10% co-pay on 1500).
    n += bundle(
        "clean_consult", "EMP001",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "diagnosis": "Viral Fever",
            "meds": ["Paracetamol 650mg", "Vitamin C 500mg"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "items": [{"description": "Consultation Fee", "amount": 1000.0},
                      {"description": "CBC Test", "amount": 300.0},
                      {"description": "Dengue NS1 Test", "amount": 200.0}]}))],
        {"decision": "APPROVED", "approved_amount": 1350,
         "category": "CONSULTATION"},
        degrade_map={"bill": [(["skew"], 0.2)]})

    # 2. Network hospital -> discount before co-pay: 3000 -> 2400 -> 2160.
    iyer = ("Dr. S. Iyer", "TN/56789/2013", "MBBS, MD (Internal Medicine)")
    n += bundle(
        "network_discount", "EMP010",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Deepak Shah", 44, "M"), "doctor": iyer,
            "hospital": "Apollo Hospitals", "date": date_nov,
            "diagnosis": "Acute Bronchitis",
            "meds": ["Amoxicillin 500mg", "Salbutamol Inhaler"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Deepak Shah", 44, "M"), "doctor": iyer,
            "hospital": "Apollo Hospitals", "date": date_nov,
            "items": [{"description": "Consultation Fee", "amount": 1500.0},
                      {"description": "Medicines", "amount": 1500.0}]}))],
        {"decision": "APPROVED", "approved_amount": 2160,
         "category": "CONSULTATION"})

    # 3. Dental partial -> root canal approved, whitening excluded: 8000.
    n += bundle(
        "dental_partial", "EMP002",
        [("bill", make_hospital_bill(rng, dental=True, overrides={
            "patient": ("Priya Singh", 34, "F"),
            "hospital": "Smile Dental Clinic", "date": date_oct,
            "items": [{"description": "Root Canal Treatment", "amount": 8000.0},
                      {"description": "Teeth Whitening", "amount": 4000.0}]}))],
        {"decision": "PARTIAL", "approved_amount": 8000, "category": "DENTAL"})

    # 4. Excluded obesity treatment -> REJECTED.
    banerjee = ("Dr. P. Banerjee", "WB/34567/2015", "MBBS, MD")
    n += bundle(
        "excluded_obesity", "EMP009",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Anita Desai", 31, "F"), "doctor": banerjee,
            "hospital": "Wellness Care Clinic", "date": date_oct,
            "diagnosis": "Morbid Obesity - BMI 37",
            "meds": ["Customised Diet Plan"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Anita Desai", 31, "F"), "doctor": banerjee,
            "hospital": "Wellness Care Clinic", "date": date_oct,
            "items": [{"description": "Bariatric Consultation",
                       "amount": 3000.0},
                      {"description": "Personalised Diet and Nutrition Program",
                       "amount": 5000.0}]}))],
        {"decision": "REJECTED", "category": "CONSULTATION",
         "rejection_reason": "EXCLUDED_CONDITION"})

    # 5. Diabetes inside the 90-day waiting period -> REJECTED.
    gupta = ("Dr. R. Gupta", "DL/34567/2016", "MBBS, DNB")
    n += bundle(
        "waiting_diabetes", "EMP005",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Vikram Joshi", 45, "M"), "doctor": gupta,
            "hospital": "Wellness Care Clinic", "date": date_oct,
            "diagnosis": "Type 2 Diabetes Mellitus",
            "meds": ["Metformin 500mg"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Vikram Joshi", 45, "M"), "doctor": gupta,
            "hospital": "Wellness Care Clinic", "date": date_oct,
            "items": [{"description": "Consultation Fee", "amount": 1200.0},
                      {"description": "Medicines", "amount": 800.0}]}))],
        {"decision": "REJECTED", "category": "CONSULTATION",
         "rejection_reason": "WAITING_PERIOD"})

    # 6. Unreadable bill -> must NOT sail through as a confident approval.
    n += bundle(
        "unreadable_bill", "EMP001",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "diagnosis": "Viral Fever", "meds": ["Paracetamol 650mg"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "items": [{"description": "Consultation Fee", "amount": 1000.0},
                      {"description": "CBC Test", "amount": 500.0}]}))],
        {"decision_any": ["NEEDS_RESUBMISSION", "MANUAL_REVIEW"],
         "note": "Heavily blurred bill: acceptable outcomes are a re-upload "
                 "request or manual review; a clean auto-approval is a fail."},
        degrade_map={"bill": [(["blur"], 1.0), (["blur"], 1.0),
                              (["lowres"], 1.0), (["noise"], 0.8)]})

    # -----------------------------------------------------------------------
    # Negative controls: each MUST be rejected. They measure that good-claim
    # accuracy is not bought with rubber-stamping (the eval can over-approve and
    # still look "accurate" if it is only ever shown payable claims). Outcomes
    # confirmed via the derived path before encoding.
    # -----------------------------------------------------------------------

    # NC1. Treatment date before the policy start -> NOT_COVERED.
    n += bundle(
        "neg_out_of_period", "EMP001",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_pre,
            "diagnosis": "Viral Fever", "meds": ["Paracetamol 650mg"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_pre,
            "items": [{"description": "Consultation Fee", "amount": 1000.0},
                      {"description": "CBC Test", "amount": 300.0}]}))],
        {"decision": "REJECTED", "category": "CONSULTATION",
         "rejection_reason": "NOT_COVERED"})

    # NC2. Excluded cosmetic dental (whitening + bleaching) -> EXCLUDED_CONDITION.
    n += bundle(
        "neg_excluded_cosmetic", "EMP002",
        [("bill", make_hospital_bill(rng, dental=True, overrides={
            "patient": ("Priya Singh", 34, "F"),
            "hospital": "Smile Dental Clinic", "date": date_oct,
            "items": [{"description": "Teeth Whitening", "amount": 4000.0},
                      {"description": "Bleaching", "amount": 3000.0}]}))],
        {"decision": "REJECTED", "category": "DENTAL",
         "rejection_reason": "EXCLUDED_CONDITION"})

    # NC3. Consultation above the per-claim cap (7,500 > 5,000) -> PER_CLAIM_EXCEEDED.
    n += bundle(
        "neg_over_cap", "EMP001",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "diagnosis": "Hypertension follow-up", "meds": ["Amlodipine 5mg"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "items": [{"description": "Consultation Fee", "amount": 2000.0},
                      {"description": "Specialist Consultation", "amount": 3000.0},
                      {"description": "Follow-up Consultation", "amount": 2500.0}]}))],
        {"decision": "REJECTED", "category": "CONSULTATION",
         "rejection_reason": "PER_CLAIM_EXCEEDED"})

    # -----------------------------------------------------------------------
    # Three real-shape bundles, each encoding the POLICY-CORRECT outcome. The
    # dependent one already passes (directional-dependent logic) and is a green
    # positive control; the other two need Batch 6 (per-line categorization,
    # multi-bill aggregation) and are flagged expected-fail-until-batch6 so the
    # runner buckets them apart from regressions. Current vs policy-correct
    # confirmed via the derived path.
    # -----------------------------------------------------------------------

    # B1. Dependent claim: EMP001 files for Sunita Kumar (DEP001, SPOUSE), a
    # rostered dependent of the filer. GREEN today -> positive control.
    n += bundle(
        "dependent_consultation", "EMP001",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Sunita Kumar", 37, "F"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "diagnosis": "Viral Fever", "meds": ["Paracetamol 650mg"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Sunita Kumar", 37, "F"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "items": [{"description": "Consultation Fee", "amount": 1000.0},
                      {"description": "CBC Test", "amount": 300.0}]}))],
        {"decision": "APPROVED", "approved_amount": 1170, "category": "CONSULTATION"},
        expected_math=(
            "Patient Sunita Kumar is DEP001 (SPOUSE), a rostered dependent of the "
            "filer EMP001, so the directional-dependent identity check passes. "
            "Consultation, City Medical Centre (off-network). Eligible = 1000 + "
            "300 = 1300. No network discount. Co-pay 10%: 1300 x 0.90 = 1170. "
            "Within the consultation per-claim cap max(5000, 2000) = 5000. "
            "Expected APPROVED 1170. Already green today -> positive control for "
            "dependent handling."))

    # B2. Multi-service: ONE bill spanning two categories (consultation +
    # dental). Batch 6 target: per-line categorization. Expected-fail today.
    n += bundle(
        "multi_service_visit", "EMP001",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "diagnosis": "Tooth Pain", "meds": ["Ibuprofen 400mg"]})),
         ("bill", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "City Medical Centre, Bengaluru", "date": date_nov,
            "items": [{"description": "Consultation Fee", "amount": 1500.0},
                      {"description": "Root Canal Treatment", "amount": 2000.0}]}))],
        {"decision": "APPROVED", "approved_amount": 3350},
        expected_math=(
            "One bill spanning two categories. Batch 6b per-line categorization "
            "adjudicates each line under its own category: Consultation 1500 -> "
            "10% co-pay -> 1350; Dental (Root Canal Treatment, a covered "
            "procedure) 2000 -> 0% co-pay -> 2000. Off-network (no discount). "
            "Sub-limits OK (consult 1500 < 2000 per-line; dental 2000 < 10000); "
            "aggregate 3500 < max(5000, 10000) per-claim cap. APPROVED 3350 "
            "(regular green bundle now that per-line money math landed in 6b)."))

    # B3. Two-bill claim: two hospital bills in one submission that must
    # aggregate. Batch 6 target: multi-bill aggregation. Expected-fail today.
    n += bundle(
        "two_bill_aggregate", "EMP001",
        [("prescription", make_prescription(rng, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "Apollo Hospitals", "date": date_nov,
            "diagnosis": "Viral Fever", "meds": ["Paracetamol 650mg"]})),
         ("bill1", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "Apollo Hospitals", "date": date_nov,
            "items": [{"description": "Consultation Fee", "amount": 1500.0}]})),
         ("bill2", make_hospital_bill(rng, dental=False, overrides={
            "patient": ("Rajesh Kumar", 39, "M"), "doctor": arun,
            "hospital": "Apollo Hospitals", "date": date_nov,
            "items": [{"description": "Follow-up Consultation", "amount": 1500.0}]}))],
        {"decision": "APPROVED", "approved_amount": 2160, "category": "CONSULTATION"},
        expected_math=(
            "Two single-line consultation bills in one submission, 1500 each, "
            "both at Apollo Hospitals (network). Batch 6a multi-bill aggregation "
            "sums line items across both bills: eligible = 1500 + 1500 = 3000 "
            "(each line < 2000 consultation per-line cap; total < 5000 per-claim "
            "cap). Network discount 20% -> 2400. Co-pay 10% -> 2160. APPROVED "
            "2160 (regular green bundle now that aggregation has landed in 6a)."))

    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=20260613)
    args = ap.parse_args()
    n_items = build_dataset(args.seed)
    n_bundles = build_bundles(args.seed)
    print(f"dataset: {n_items} labeled items -> {DATASET_DIR}")
    print(f"bundles: {n_bundles} e2e claim bundles -> {BUNDLES_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
