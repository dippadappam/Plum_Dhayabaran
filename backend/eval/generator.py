"""Synthetic labeled-document generator.

Renders the four document types from the sample guide (hospital bill,
prescription, pharmacy bill, diagnostic/lab report) with PIL. The generator
knows the truth it wrote, so every image ships with exact ground truth in
the extraction schema — labeled data for free.

Determinism: every document is generated from a seeded random.Random, so
the dataset is reproducible from (seed, doc index).
"""

import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from PIL import Image, ImageDraw, ImageFont

W, H = 840, 1090
INK = (28, 31, 36)
MUTED = (107, 113, 120)
ACCENT = (12, 76, 99)

_FONT_DIR = Path("C:/Windows/Fonts")
_FONT_FILES = {
    "regular": ["arial.ttf", "calibri.ttf"],
    "bold": ["arialbd.ttf", "calibrib.ttf"],
    "hand": ["segoesc.ttf", "Inkfree.ttf", "segoepr.ttf"],
}


def _font(kind: str, size: int) -> ImageFont.FreeTypeFont:
    for name in _FONT_FILES.get(kind, []):
        p = _FONT_DIR / name
        if p.exists():
            return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


# ---------------------------------------------------------------------------
# Data pools (Indian-medical flavored, per the sample guide)
# ---------------------------------------------------------------------------

PATIENTS = [
    ("Rajesh Kumar", 39, "M"), ("Priya Singh", 34, "F"), ("Amit Verma", 36, "M"),
    ("Sneha Reddy", 32, "F"), ("Vikram Joshi", 45, "M"), ("Kavita Nair", 41, "F"),
    ("Suresh Patil", 49, "M"), ("Deepak Shah", 44, "M"), ("Anita Desai", 31, "F"),
    ("Meena Iyer", 38, "F"), ("Arjun Mehta", 29, "M"), ("Farhan Ali", 35, "M"),
]
DOCTORS = [
    ("Dr. Arun Sharma", "KA/45678/2015", "MBBS, MD (General Medicine)"),
    ("Dr. S. Iyer", "TN/56789/2013", "MBBS, MD (Internal Medicine)"),
    ("Dr. R. Gupta", "DL/34567/2016", "MBBS, DNB"),
    ("Dr. Meena Pillai", "KL/78901/2012", "MD (Pathology)"),
    ("Dr. Venkat Rao", "AP/67890/2017", "MBBS, MS (Ortho)"),
    ("Dr. P. Banerjee", "WB/34567/2015", "MBBS, MD"),
]
HOSPITALS = [
    "City Medical Centre, Bengaluru", "Apollo Hospitals", "Sunrise Clinic, Pune",
    "Fortis Healthcare", "Green Valley Hospital, Chennai", "Wellness Care Clinic",
]
PHARMACIES = [
    "Health First Pharmacy", "MedPlus Chemists", "Apollo Pharmacy",
    "City Care Medicals",
]
LABS = [
    "Precision Diagnostics Pvt Ltd", "SRL Diagnostics", "Metropolis Labs",
]
DIAGNOSES = [
    "Viral Fever", "Acute Bronchitis", "Gastroenteritis", "Migraine",
    "Urinary Tract Infection", "Type 2 Diabetes Mellitus", "Hypertension",
    "Acute Pharyngitis", "Lumbar Spondylosis",
]
MEDICINES = [
    ("Paracetamol 650mg", 2.5), ("Amoxicillin 500mg", 9.0),
    ("Azithromycin 500mg", 18.0), ("Cetirizine 10mg", 2.0),
    ("Pantoprazole 40mg", 6.5), ("Metformin 500mg", 3.0),
    ("Vitamin C 500mg", 4.0), ("Ibuprofen 400mg", 3.5),
    ("Salbutamol Inhaler", 165.0), ("ORS Sachet", 21.0),
]
BILL_SERVICES = [
    ("Consultation Fee", (600, 2000)), ("CBC Test", (200, 450)),
    ("Dengue NS1 Test", (250, 600)), ("X-Ray Chest PA", (300, 700)),
    ("Dressing Charges", (150, 400)), ("Injection Charges", (100, 300)),
    ("ECG", (250, 500)), ("Urine Routine", (120, 280)),
]
LAB_TESTS = [
    ("Hemoglobin", "g/dL", (11.0, 16.5), "13.0 - 17.0"),
    ("WBC Count", "/uL", (4800, 10800), "4,500 - 11,000"),
    ("Platelet Count", "x10^3/uL", (160, 420), "150 - 450"),
    ("Fasting Glucose", "mg/dL", (78, 150), "70 - 100"),
    ("Serum Creatinine", "mg/dL", (0.6, 1.3), "0.6 - 1.2"),
]
DENTAL_SERVICES = [
    ("Root Canal Treatment", (5000, 9000)), ("Dental X-Ray", (250, 500)),
    ("Scaling and Polishing", (800, 1800)), ("Tooth Extraction", (900, 2200)),
]


@dataclass
class GeneratedDocument:
    image: Image.Image
    truth: dict[str, Any]
    notes: dict[str, Any] = field(default_factory=dict)


def _money(n: float) -> str:
    return f"Rs. {n:,.2f}"


def _pick_date(rng: random.Random) -> tuple[str, str]:
    """(printed DD-MM-YYYY with unambiguous day>12, ISO truth)."""
    day = rng.randint(13, 28)
    month = rng.randint(4, 12)
    return f"{day:02d}-{month:02d}-2024", f"2024-{month:02d}-{day:02d}"


def _truth(**kw) -> dict[str, Any]:
    base: dict[str, Any] = {
        "document_type": None, "patient_name": None, "doctor_name": None,
        "doctor_registration": None, "hospital_name": None, "date": None,
        "diagnosis": None, "treatment": None, "medicines": [],
        "line_items": [], "total": None,
    }
    base.update(kw)
    return base


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def make_hospital_bill(rng: random.Random,
                       dental: Optional[bool] = None,
                       overrides: Optional[dict] = None) -> GeneratedDocument:
    o = overrides or {}
    dental = rng.random() < 0.25 if dental is None else dental
    patient, age, sex = o.get("patient") or rng.choice(PATIENTS)
    doctor = o.get("doctor") or rng.choice(DOCTORS)
    hospital = o.get("hospital") or ("Smile Dental Clinic" if dental else rng.choice(HOSPITALS))
    printed_date, iso_date = o.get("date") or _pick_date(rng)
    if o.get("items"):
        items = [dict(i) for i in o["items"]]
    else:
        pool = DENTAL_SERVICES if dental else BILL_SERVICES
        n_items = rng.randint(2, 4)
        items = []
        for svc, (lo, hi) in rng.sample(pool, n_items):
            items.append({"description": svc, "amount": float(rng.randrange(lo, hi, 50))})
    total = sum(i["amount"] for i in items)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 96], fill=ACCENT)
    d.text((48, 22), hospital.upper(), font=_font("bold", 30), fill="white")
    d.text((48, 62), "Bengaluru | Ph: 080-2345 6789", font=_font("regular", 15),
           fill=(214, 228, 234))
    d.text((48, 130), "TAX INVOICE / FINAL BILL", font=_font("bold", 24), fill=ACCENT)
    bill_no = f"BL/2024/{rng.randint(1000, 9999)}"
    rows = [
        (f"Bill No: {bill_no}", f"Date: {printed_date}"),
        (f"Patient Name: {patient}", f"Age / Sex: {age} / {sex}"),
        (f"Referring Doctor: {doctor[0]}", "Payment: Cash"),
    ]
    y = 180
    for left, right in rows:
        d.text((48, y), left, font=_font("regular", 17), fill=INK)
        d.text((470, y), right, font=_font("regular", 17), fill=INK)
        y += 30
    y += 18
    d.line([48, y, W - 48, y], fill=MUTED, width=1)
    d.text((48, y + 10), "Description", font=_font("bold", 17), fill=INK)
    d.text((620, y + 10), "Amount", font=_font("bold", 17), fill=INK)
    y += 44
    d.line([48, y, W - 48, y], fill=(220, 220, 220), width=1)
    for item in items:
        y += 14
        d.text((48, y), item["description"], font=_font("regular", 17), fill=INK)
        d.text((620, y), _money(item["amount"]), font=_font("regular", 17), fill=INK)
        y += 26
    y += 12
    d.line([48, y, W - 48, y], fill=MUTED, width=1)
    d.text((48, y + 14), "TOTAL AMOUNT", font=_font("bold", 20), fill=INK)
    d.text((620, y + 14), _money(total), font=_font("bold", 20), fill=ACCENT)
    d.text((48, H - 70), f"{hospital} | GSTIN: 29AABCA{rng.randint(1000, 9999)}D1Z5",
           font=_font("regular", 13), fill=MUTED)

    truth = _truth(document_type="HOSPITAL_BILL", patient_name=patient,
                   doctor_name=doctor[0], hospital_name=hospital,
                   date=iso_date, line_items=items, total=total)
    return GeneratedDocument(img, truth, {"dental": dental, "bill_no": bill_no})


def make_prescription(rng: random.Random,
                      handwritten: bool = False,
                      overrides: Optional[dict] = None) -> GeneratedDocument:
    o = overrides or {}
    patient, age, sex = o.get("patient") or rng.choice(PATIENTS)
    doctor_name, reg, quals = o.get("doctor") or rng.choice(DOCTORS)
    hospital = o.get("hospital") or rng.choice(HOSPITALS)
    printed_date, iso_date = o.get("date") or _pick_date(rng)
    diagnosis = o.get("diagnosis") or rng.choice(DIAGNOSES)
    meds = o.get("meds") or [m for m, _ in rng.sample(MEDICINES, rng.randint(2, 4))]
    body = _font("hand", 26) if handwritten else _font("regular", 19)
    body_b = _font("hand", 28) if handwritten else _font("bold", 19)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((48, 30), doctor_name, font=_font("bold", 30), fill=INK)
    d.text((48, 72), quals, font=_font("regular", 16), fill=MUTED)
    d.text((48, 96), f"Reg. No: {reg}", font=_font("regular", 16), fill=MUTED)
    d.text((560, 36), hospital, font=_font("bold", 17), fill=ACCENT)
    d.text((560, 64), "Bengaluru - 560076", font=_font("regular", 14), fill=MUTED)
    d.line([48, 140, W - 48, 142], fill=ACCENT, width=2)
    d.text((48, 165), f"Date: {printed_date}", font=body, fill=INK)
    d.text((48, 205), f"Patient: {patient}", font=body_b, fill=INK)
    d.text((560, 205), f"Age/Sex: {age} / {sex}", font=body, fill=INK)
    d.text((48, 245), f"Diagnosis: {diagnosis}", font=body, fill=INK)
    d.text((48, 305), "Rx", font=_font("bold", 40), fill=ACCENT)
    y = 370
    for i, med in enumerate(meds, 1):
        dose = rng.choice(["1-0-1 x 5 days", "0-0-1 x 7 days", "1-1-1 x 3 days",
                           "SOS for fever", "2 puffs when needed"])
        d.text((80, y), f"{i}.  {med}", font=body, fill=INK)
        d.text((430, y + 2), dose, font=body, fill=MUTED)
        y += 52
    d.text((48, y + 30), "Advice: Rest, plenty of fluids. Review after 5 days.",
           font=body, fill=MUTED)
    d.line([560, H - 130, 740, H - 130], fill=INK, width=1)
    d.text((575, H - 118), doctor_name, font=_font("bold", 16), fill=INK)
    d.text((575, H - 94), "Signature & Seal", font=_font("regular", 13), fill=MUTED)

    truth = _truth(document_type="PRESCRIPTION", patient_name=patient,
                   doctor_name=doctor_name, doctor_registration=reg,
                   hospital_name=hospital, date=iso_date, diagnosis=diagnosis,
                   medicines=meds)
    return GeneratedDocument(img, truth, {"handwritten": handwritten})


def make_pharmacy_bill(rng: random.Random) -> GeneratedDocument:
    patient, _, _ = rng.choice(PATIENTS)
    doctor_name, _, _ = rng.choice(DOCTORS)
    pharmacy = rng.choice(PHARMACIES)
    printed_date, iso_date = _pick_date(rng)
    n = rng.randint(2, 4)
    items = []
    for med, mrp in rng.sample(MEDICINES, n):
        qty = rng.randint(5, 15)
        items.append({"description": med, "amount": round(mrp * qty, 2)})
    total = round(sum(i["amount"] for i in items), 2)

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.text((48, 34), pharmacy.upper(), font=_font("bold", 28), fill=INK)
    d.text((48, 76), f"Drug Lic. No: KA-BLR-{rng.randint(10000, 99999)}",
           font=_font("regular", 15), fill=MUTED)
    d.text((48, 98), "22 Brigade Road, Bengaluru", font=_font("regular", 15), fill=MUTED)
    d.line([48, 132, W - 48, 132], fill=INK, width=2)
    d.text((48, 150), f"Bill No: HFP-24-{rng.randint(1000, 9999)}",
           font=_font("regular", 17), fill=INK)
    d.text((470, 150), f"Date: {printed_date}", font=_font("regular", 17), fill=INK)
    d.text((48, 182), f"Patient: {patient}", font=_font("regular", 17), fill=INK)
    d.text((470, 182), f"Dr: {doctor_name}", font=_font("regular", 17), fill=INK)
    y = 240
    d.text((48, y), "MEDICINE", font=_font("bold", 16), fill=INK)
    d.text((620, y), "AMOUNT", font=_font("bold", 16), fill=INK)
    y += 30
    d.line([48, y, W - 48, y], fill=(220, 220, 220), width=1)
    for item in items:
        y += 14
        d.text((48, y), item["description"], font=_font("regular", 17), fill=INK)
        d.text((620, y), _money(item["amount"]), font=_font("regular", 17), fill=INK)
        y += 26
    y += 16
    d.line([48, y, W - 48, y], fill=INK, width=1)
    d.text((48, y + 14), "Net Amount:", font=_font("bold", 19), fill=INK)
    d.text((620, y + 14), _money(total), font=_font("bold", 19), fill=INK)
    d.text((48, H - 70), "Pharmacist: R. Sharma", font=_font("regular", 14), fill=MUTED)

    truth = _truth(document_type="PHARMACY_BILL", patient_name=patient,
                   doctor_name=doctor_name, hospital_name=pharmacy,
                   date=iso_date, line_items=items, total=total,
                   medicines=[i["description"] for i in items])
    return GeneratedDocument(img, truth, {})


def make_lab_report(rng: random.Random) -> GeneratedDocument:
    patient, age, sex = rng.choice(PATIENTS)
    doctor_name, _, _ = rng.choice(DOCTORS)
    lab = rng.choice(LABS)
    printed_date, iso_date = _pick_date(rng)
    tests = rng.sample(LAB_TESTS, rng.randint(3, 5))

    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, W, 86], fill=(243, 246, 247))
    d.text((48, 22), lab.upper(), font=_font("bold", 26), fill=ACCENT)
    d.text((48, 58), "NABL Accredited Lab | 45 Jayanagar, Bengaluru",
           font=_font("regular", 14), fill=MUTED)
    rows = [
        (f"Patient: {patient}", f"Age/Sex: {age} / {sex}"),
        (f"Ref Doctor: {doctor_name}", f"Report Date: {printed_date}"),
        (f"Sample ID: PD-2024-{rng.randint(10000, 99999)}", ""),
    ]
    y = 110
    for left, right in rows:
        d.text((48, y), left, font=_font("regular", 17), fill=INK)
        if right:
            d.text((470, y), right, font=_font("regular", 17), fill=INK)
        y += 30
    y += 16
    headers = [("TEST NAME", 48), ("RESULT", 380), ("UNIT", 530), ("NORMAL RANGE", 640)]
    for text, x in headers:
        d.text((x, y), text, font=_font("bold", 15), fill=INK)
    y += 28
    d.line([48, y, W - 48, y], fill=MUTED, width=1)
    for name, unit, (lo, hi), normal in tests:
        y += 16
        val = round(rng.uniform(lo, hi), 1)
        sval = f"{val:,.1f}" if val < 1000 else f"{int(val):,}"
        d.text((48, y), name, font=_font("regular", 17), fill=INK)
        d.text((380, y), sval, font=_font("regular", 17), fill=INK)
        d.text((530, y), unit, font=_font("regular", 15), fill=MUTED)
        d.text((640, y), normal, font=_font("regular", 15), fill=MUTED)
        y += 26
    y += 30
    d.text((48, y), "Remarks: Clinical correlation advised.",
           font=_font("regular", 15), fill=MUTED)
    d.text((560, H - 110), "Dr. Meena Pillai, MD (Pathology)",
           font=_font("bold", 15), fill=INK)
    d.text((560, H - 88), "Reg. No: KL/78901/2012", font=_font("regular", 13),
           fill=MUTED)

    truth = _truth(document_type="LAB_REPORT", patient_name=patient,
                   doctor_name=doctor_name, hospital_name=lab, date=iso_date)
    return GeneratedDocument(img, truth, {"tests": [t[0] for t in tests]})


GENERATORS = {
    "hospital_bill": make_hospital_bill,
    "prescription": make_prescription,
    "pharmacy_bill": make_pharmacy_bill,
    "lab_report": make_lab_report,
}
