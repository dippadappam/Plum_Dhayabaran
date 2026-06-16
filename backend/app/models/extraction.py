"""Validation and normalization of the model's extraction output.

The vision extractor's tool result arrives as a raw dict. Before the engine
trusts it, it passes through this Pydantic layer, which:
  - clamps every confidence to [0, 1];
  - coerces numeric fields (line-item amounts, total) to float, turning an
    unreadable value into None instead of crashing deep in a rule;
  - validates the enum fields (document_type, readability, per-line drug_type)
    against their allowed values, falling back safely;
  - drops unknown fields (extra="ignore").

It runs at exactly one place — `normalize_extraction`, called on the tool output
inside `ClaudeVisionExtractor.extract`. The structured-content test path (the 12
official cases provide `content` directly) never reaches the live extractor, so
it is unaffected.
"""

import math
import re
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

# Allowed enum values, mirroring the extraction tool schema.
_DOCUMENT_TYPES = {
    "PRESCRIPTION", "HOSPITAL_BILL", "PHARMACY_BILL", "LAB_REPORT",
    "DIAGNOSTIC_REPORT", "DISCHARGE_SUMMARY", "DENTAL_REPORT", "UNKNOWN",
}
_READABILITY = {"GOOD", "PARTIAL", "UNREADABLE"}
_DRUG_TYPES = {"BRANDED", "GENERIC"}


def _coerce_number(v: Any) -> Optional[float]:
    """Best-effort numeric coercion. Accepts int/float and numeric strings
    (stripping commas and currency symbols); returns None for anything that is
    not a finite number, so a bad amount becomes a clean null rather than a
    crash deep in a rule."""
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v) if math.isfinite(v) else None
    s = re.sub(r"[^\d.\-]", "", str(v).replace(",", ""))
    try:
        f = float(s)
    except ValueError:
        return None
    return f if math.isfinite(f) else None


def _clamp01(v: Any) -> Optional[float]:
    """Coerce to a number and clamp to [0, 1]; None when not a number."""
    n = _coerce_number(v)
    if n is None:
        return None
    return 0.0 if n < 0.0 else 1.0 if n > 1.0 else n


def _opt_str(v: Any) -> Optional[str]:
    """A trimmed string, or None for empty/None (graceful for non-string input)."""
    if v is None:
        return None
    s = str(v).strip()
    return s or None


def _coerce_bool(v: Any) -> bool:
    """Lenient truthiness: None/absent -> False; accepts bools, numbers, and
    'true'/'yes'/'1' strings (anything else -> False)."""
    if v is None:
        return False
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    return str(v).strip().lower() in ("true", "yes", "1")


class ExtractedLineItem(BaseModel):
    """One validated bill line. Amount coerces (bad → None); drug_type is checked
    against the allowed enum (anything else → None); unknown keys are dropped."""

    model_config = ConfigDict(extra="ignore")

    description: str = ""
    amount: Optional[float] = None
    drug_type: Optional[str] = None
    drug_type_confidence: Optional[float] = None

    @field_validator("description", mode="before")
    @classmethod
    def _v_desc(cls, v: Any) -> str:
        return "" if v is None else str(v)

    @field_validator("amount", mode="before")
    @classmethod
    def _v_amount(cls, v: Any) -> Optional[float]:
        return _coerce_number(v)

    @field_validator("drug_type", mode="before")
    @classmethod
    def _v_drug_type(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip().upper()
        return s if s in _DRUG_TYPES else None

    @field_validator("drug_type_confidence", mode="before")
    @classmethod
    def _v_dtc(cls, v: Any) -> Optional[float]:
        return _clamp01(v)


class ExtractedDocument(BaseModel):
    """Validated, normalized mirror of the extraction tool schema. Constructed
    from the model's raw tool output; everything outside this shape is dropped."""

    model_config = ConfigDict(extra="ignore")

    document_type: str = "UNKNOWN"
    readability: str = "PARTIAL"
    patient_name: Optional[str] = None
    doctor_name: Optional[str] = None
    doctor_registration: Optional[str] = None
    hospital_name: Optional[str] = None
    date: Optional[str] = None
    diagnosis: Optional[str] = None
    primary_diagnosis: Optional[str] = None
    comorbidities: Optional[str] = None
    canonical_condition: Optional[str] = None
    canonical_condition_confidence: Optional[float] = None
    treatment: Optional[str] = None
    medicines: list[str] = Field(default_factory=list)
    line_items: list[ExtractedLineItem] = Field(default_factory=list)
    total: Optional[float] = None
    amount_confidence: Optional[float] = None
    treatment_date_confidence: Optional[float] = None
    patient_name_confidence: Optional[float] = None
    hospital_confidence: Optional[float] = None
    category_confidence: Optional[float] = None
    pre_auth_number: Optional[str] = None
    # Visible-tampering signal (real extracted field; persisted). Safe default
    # False so an absent value is treated as a clean document.
    alteration_suspected: bool = False
    alteration_reason: Optional[str] = None
    extraction_confidence: float = 0.5

    @field_validator("document_type", mode="before")
    @classmethod
    def _v_doctype(cls, v: Any) -> str:
        s = str(v).strip().upper() if v is not None else ""
        return s if s in _DOCUMENT_TYPES else "UNKNOWN"

    @field_validator("readability", mode="before")
    @classmethod
    def _v_readability(cls, v: Any) -> str:
        s = str(v).strip().upper() if v is not None else ""
        return s if s in _READABILITY else "PARTIAL"

    @field_validator("patient_name", "doctor_name", "doctor_registration",
                     "hospital_name", "date", "diagnosis", "primary_diagnosis",
                     "comorbidities", "canonical_condition", "treatment",
                     "pre_auth_number", "alteration_reason", mode="before")
    @classmethod
    def _v_opt_str(cls, v: Any) -> Optional[str]:
        return _opt_str(v)

    @field_validator("alteration_suspected", mode="before")
    @classmethod
    def _v_alteration(cls, v: Any) -> bool:
        return _coerce_bool(v)

    @field_validator("medicines", mode="before")
    @classmethod
    def _v_medicines(cls, v: Any) -> list:
        if isinstance(v, list):
            return [str(x).strip() for x in v if x is not None and str(x).strip()]
        return []

    @field_validator("line_items", mode="before")
    @classmethod
    def _v_line_items(cls, v: Any) -> list:
        if isinstance(v, list):
            return [x for x in v if isinstance(x, dict)]
        return []

    @field_validator("total", mode="before")
    @classmethod
    def _v_total(cls, v: Any) -> Optional[float]:
        return _coerce_number(v)

    @field_validator("canonical_condition_confidence", "amount_confidence",
                     "treatment_date_confidence", "patient_name_confidence",
                     "hospital_confidence", "category_confidence", mode="before")
    @classmethod
    def _v_conf(cls, v: Any) -> Optional[float]:
        return _clamp01(v)

    @field_validator("extraction_confidence", mode="before")
    @classmethod
    def _v_extraction_conf(cls, v: Any) -> float:
        c = _clamp01(v)
        return c if c is not None else 0.5


def normalize_extraction(raw: Any) -> dict[str, Any]:
    """Validate and normalize a raw extraction tool result into the clean dict
    the engine consumes. The per-document confidence is exposed as
    `_extraction_confidence` (the engine's existing contract); the public
    `extraction_confidence` key is removed."""
    doc = ExtractedDocument.model_validate(dict(raw))
    out = doc.model_dump()
    out["_extraction_confidence"] = out.pop("extraction_confidence")
    return out
