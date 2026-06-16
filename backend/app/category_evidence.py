"""Category evidence: deterministic keyword evidence over extracted document
fields, shared by two consumers with different stakes:

- CHECK grade (ConsistencyAgent): a lenient cross-check that only ever vetoes
  (routes to manual review). Wide corpus on purpose: diagnosis, treatment,
  test names, hospital name, line items.
- DECIDE grade (CategoryResolutionAgent): authoritative derivation when no
  category was provided — this chooses the rules the claim is adjudicated
  under, so only procedural evidence counts: line items and test names.
  Diagnosis-only, treatment, and hospital-name hits are excluded, so a
  diagnosis of "eye pain" never derives VISION, an invented `treatment`
  never selects a category, and a clinic named "Smile Dental" never derives
  DENTAL.

Tie-breaks for DECIDE grade (derive_category):
- exactly one specialty with evidence -> that specialty;
- consultation evidence only -> CONSULTATION (the generic category);
- two or more specialties, or no evidence at all -> ambiguous (None).

PHARMACY is special in DECIDE grade: hospital bills routinely carry a
"Medicines (Pharmacy)" sub-line, so the word "pharmacy" in a line item is
not evidence of a pharmacy claim. PHARMACY derives only from document-type
evidence: an actual PHARMACY_BILL among the documents. (Check grade keeps
the keyword: it only ever supports a filed PHARMACY claim.)
"""

from typing import Any, Optional

from app.clinical_mappings import CLINICAL_MAPPINGS
from app.condition_mapping import _contains_phrase, _norm

# Evidence keywords per category, matched whole-word against normalized text.
# Loaded once at startup from data/clinical_mappings.json
# (app/clinical_mappings.py), not hardcoded here. CONSULTATION is the generic
# category: in the consistency check its evidence only ever supports a filed
# CONSULTATION claim, and in derivation it loses every tie-break against a
# specialty.
CATEGORY_EVIDENCE: dict[str, list[str]] = CLINICAL_MAPPINGS.category_evidence

GENERIC_CATEGORY = "CONSULTATION"
SPECIALTY_CATEGORIES = [c for c in CATEGORY_EVIDENCE if c != GENERIC_CATEGORY]

# Scalar fields scanned per grade. Line items and tests_ordered are always
# scanned (procedural by nature). The DECIDE grade excludes `treatment`: the
# vision model invents that field when the document states none, so it must
# never steer the authoritative category. Procedural truth (line items, test
# names) decides; `treatment` remains a CHECK-grade signal only, where it can
# only ever veto, not select.
_CHECK_FIELDS = ("diagnosis", "treatment", "test_name", "hospital_name")
_DECIDE_FIELDS = ("test_name",)


def evidence_texts(extracted_documents: list[dict[str, Any]],
                   grade: str = "check") -> list[str]:
    """Collect the evidence corpus from extracted documents at the given
    grade ("check" = wide, "decide" = procedural only)."""
    scalar_fields = _CHECK_FIELDS if grade == "check" else _DECIDE_FIELDS
    texts: list[str] = []
    for doc in extracted_documents:
        fields = doc.get("fields", {})
        for key in scalar_fields:
            if fields.get(key):
                texts.append(str(fields[key]))
        for item in fields.get("line_items") or []:
            if item.get("description"):
                texts.append(str(item["description"]))
        for t in fields.get("tests_ordered") or []:
            texts.append(str(t))
    return texts


def category_hits(corpus_norm: str, category: str) -> list[str]:
    """Keywords of `category` found whole-word in the normalized corpus."""
    return [kw for kw in CATEGORY_EVIDENCE.get(category, [])
            if _contains_phrase(corpus_norm, _norm(kw))]


def normalize_corpus(texts: list[str]) -> str:
    return _norm(" | ".join(texts))


def _pharmacy_bill_present(extracted_documents: list[dict[str, Any]]) -> bool:
    for doc in extracted_documents:
        dtype = str(doc.get("fields", {}).get("document_type")
                    or doc.get("type") or "").upper()
        if dtype == "PHARMACY_BILL":
            return True
    return False


def derive_category(
    extracted_documents: list[dict[str, Any]],
) -> tuple[Optional[str], Optional[str]]:
    """DECIDE-grade authoritative derivation.

    Returns (category, matched_keyword), or (None, None) when ambiguous:
    no procedural evidence at all, or two or more specialties in evidence.
    """
    corpus = normalize_corpus(evidence_texts(extracted_documents, grade="decide"))
    has_pharmacy_bill = _pharmacy_bill_present(extracted_documents)
    if not corpus and not has_pharmacy_bill:
        return None, None

    specialty_hits = {
        c: h for c in SPECIALTY_CATEGORIES
        if c != "PHARMACY" and (h := category_hits(corpus, c))
    }
    # PHARMACY: document-type evidence only (see module docstring).
    if has_pharmacy_bill:
        specialty_hits["PHARMACY"] = ["PHARMACY_BILL document"]
    if len(specialty_hits) == 1:
        category, keywords = next(iter(specialty_hits.items()))
        return category, keywords[0]
    if len(specialty_hits) >= 2:
        return None, None  # multiple specialties: genuinely ambiguous

    generic = category_hits(corpus, GENERIC_CATEGORY)
    if generic:
        return GENERIC_CATEGORY, generic[0]
    return None, None
