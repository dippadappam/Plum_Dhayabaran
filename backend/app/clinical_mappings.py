"""Clinical mappings loader.

Loads the clinical knowledge tables (condition/diagnosis synonyms and category
keyword evidence) from `data/clinical_mappings.json` — the same way the policy
rulebook is loaded from `data/policy_terms.json` (see app/models/policy.py).

These tables are clinical knowledge, NOT policy spec, so they live in their own
file and never in policy_terms.json (Plum's given spec, untouched). The matcher
(`app/condition_mapping.py`, `app/category_evidence.py`) reads the validated
tables via the module-level `CLINICAL_MAPPINGS` below.

Fail-loud: a missing or malformed file raises at import (startup); the engine
does not start with empty tables.
"""

import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ValidationError


class ClinicalMappings(BaseModel):
    # Each table: a mapping of a target (waiting-period condition key, exclusion
    # condition string, or claim category) to its ordered list of keyword
    # synonyms. Order is significant — the matcher reports the first match.
    waiting_condition_keywords: dict[str, list[str]]
    exclusion_keywords: dict[str, list[str]]
    category_evidence: dict[str, list[str]]


_DEFAULT_PATH = Path(__file__).resolve().parents[1] / "data" / "clinical_mappings.json"


def load_clinical_mappings(path: Optional[Path] = None) -> ClinicalMappings:
    """Load and validate the clinical mappings from JSON. Raises a clear
    RuntimeError on a missing file, malformed JSON, or an invalid shape —
    never returns partial or empty tables."""
    mappings_path = path or _DEFAULT_PATH
    try:
        with open(mappings_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Clinical mappings file not found at {mappings_path}. The engine "
            "cannot start without the clinical keyword tables; restore "
            "data/clinical_mappings.json."
        ) from e
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Clinical mappings file at {mappings_path} is not valid JSON: {e}."
        ) from e
    try:
        return ClinicalMappings.model_validate(raw)
    except ValidationError as e:
        raise RuntimeError(
            f"Clinical mappings file at {mappings_path} is malformed (expected "
            "waiting_condition_keywords, exclusion_keywords, and category_evidence, "
            f"each a mapping of strings to lists of strings):\n{e}"
        ) from e


# Loaded ONCE at import (startup). A missing/malformed file raises here, so the
# matcher modules that import this fail loud at startup rather than running with
# empty tables.
CLINICAL_MAPPINGS = load_clinical_mappings()
