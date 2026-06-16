"""Shared adjudication helpers (Option A: free-function support module).

These were private methods on the monolithic AdjudicationAgent; they are the
cross-cutting helpers that more than one rule depends on. Extracted verbatim as
free functions that take the shared ``ClaimContext`` (and a threshold where the
original read it from config), so each rule object can call them without a
reference to the agent. Behavior is identical to the original methods.
"""

from typing import Optional

from app.agents.base import ClaimContext
from app.category_evidence import (
    GENERIC_CATEGORY,
    SPECIALTY_CATEGORIES,
    category_hits,
    normalize_corpus,
)
from app.condition_mapping import (
    ConditionMapping,
    covered_in_list,
    map_canonical_condition,
    map_text_to_conditions,
)
from app.models.decision import LineItemDecision

# Documents whose extracted line items are billed charges. Prescriptions and
# lab reports never carry billable lines, so only these contribute to the
# adjudicated amount (Batch 6a multi-bill aggregation).
BILL_DOCUMENT_TYPES = {"HOSPITAL_BILL", "PHARMACY_BILL"}


def all_text_fields(ctx: ClaimContext) -> list[str]:
    """Collect diagnosis, treatment, and line-item text from extracted docs."""
    texts: list[str] = []
    for doc in ctx.extracted_documents:
        fields = doc.get("fields", {})
        for key in ("diagnosis", "treatment", "test_name"):
            if fields.get(key):
                texts.append(str(fields[key]))
        for item in fields.get("line_items", []) or []:
            if item.get("description"):
                texts.append(str(item["description"]))
        for t in fields.get("tests_ordered", []) or []:
            texts.append(str(t))
    return texts


def diagnosis_texts(ctx: ClaimContext) -> list[str]:
    """The PRIMARY diagnosis text only — the condition actually being claimed.
    Prefers the extractor's `primary_diagnosis` (the condition treated this
    visit) and falls back to the full `diagnosis` field when the extractor did
    not separate them. The `comorbidities`/history field is deliberately never
    read, so a condition noted only as history cannot trip a waiting period or a
    diagnosis-level exclusion for an unrelated claim. Per-line exclusion matching,
    which reads each line's own description, is separate and unchanged."""
    texts: list[str] = []
    for doc in ctx.extracted_documents:
        fields = doc.get("fields", {})
        primary = fields.get("primary_diagnosis") or fields.get("diagnosis")
        if primary:
            texts.append(str(primary))
    return texts


def diagnosis_mapping(ctx: ClaimContext,
                      threshold: float) -> tuple[ConditionMapping, bool]:
    """Diagnosis-level condition mapping shared by the waiting and exclusion
    checks. PREFERS the extractor's `canonical_condition` (the model's single
    best mapping, constrained to the policy vocabulary) when it is present and
    confident; otherwise FALLS BACK to the keyword dictionary over the raw
    diagnosis text. Returns (mapping, uncertain): `uncertain` is True when a
    diagnosis carried a canonical mapping BELOW the confidence threshold, so the
    caller can hold the claim for review.

    The 12 official cases carry no canonical_condition (structured content), so
    this is pure keyword fallback and byte-identical."""
    policy = ctx.policy
    waiting_keys = list(policy.waiting_periods.specific_conditions.keys())
    exclusion_conditions = list(policy.exclusions.conditions)
    confident: list[str] = []
    uncertain = False
    for doc in ctx.extracted_documents:
        fields = doc.get("fields", {})
        canon = fields.get("canonical_condition")
        if not canon or not str(canon).strip():
            continue
        conf = fields.get("canonical_condition_confidence")
        if conf is not None and conf < threshold:
            uncertain = True   # low-confidence canonical -> fall back + review
            continue
        confident.append(str(canon).strip())
    if confident:
        merged = ConditionMapping()
        for c in confident:
            m = map_canonical_condition(c, waiting_keys, exclusion_conditions)
            for w in m.matched_waiting_conditions:
                if w not in merged.matched_waiting_conditions:
                    merged.matched_waiting_conditions.append(w)
            for e in m.matched_exclusions:
                if e not in merged.matched_exclusions:
                    merged.matched_exclusions.append(e)
            merged.matched_terms.update(m.matched_terms)
        # A confident but out-of-vocabulary canonical yields an empty mapping;
        # only trust the canonical path when it actually recognized something.
        if merged.matched_waiting_conditions or merged.matched_exclusions:
            return merged, uncertain
    return map_text_to_conditions(*diagnosis_texts(ctx)), uncertain


def line_items(ctx: ClaimContext) -> list[dict]:
    """Line items aggregated across ALL bill documents in the submission
    (Batch 6a multi-bill aggregation), not just the first. A submission can
    carry more than one bill — a consultation visit plus a follow-up, say — and
    every billed line must be adjudicated and aggregated.

    Only bill documents (HOSPITAL_BILL / PHARMACY_BILL) contribute: a
    prescription or lab report carries no billable lines, so a stray line_items
    field on a non-bill document is ignored. The type is read from the extracted
    document_type, falling back to the declared type."""
    items: list[dict] = []
    for doc in ctx.extracted_documents:
        dtype = str(doc.get("fields", {}).get("document_type")
                    or doc.get("type") or "").upper()
        if dtype not in BILL_DOCUMENT_TYPES:
            continue
        for item in doc.get("fields", {}).get("line_items") or []:
            items.append(item)
    return items


def documented_bill_total(ctx: ClaimContext) -> Optional[float]:
    """The documented bill total, for cross-checking against the member's stated
    amount: the summed `total` of the bill documents (a doc's line-item sum when
    it carries no total), else any single doc carrying a total, else None.
    Mirrors the derivation agent's bill-total logic."""

    def _doc_total(fields: dict) -> Optional[float]:
        total = fields.get("total")
        if total is not None:
            try:
                return float(total)
            except (TypeError, ValueError):
                return None
        items = fields.get("line_items") or []
        if not items:
            return None
        s = 0.0
        for i in items:
            try:
                s += float(i.get("amount", 0) or 0)
            except (TypeError, ValueError):
                return None
        return s

    bill_totals: list[float] = []
    for doc in ctx.extracted_documents:
        fields = doc.get("fields", {})
        dtype = str(fields.get("document_type") or doc.get("type") or "").upper()
        if "BILL" in dtype:
            t = _doc_total(fields)
            if t is not None:
                bill_totals.append(t)
    if bill_totals:
        return sum(bill_totals)
    for doc in ctx.extracted_documents:
        t = _doc_total(doc.get("fields", {}))
        if t is not None:
            return t
    return None


def bill_hospital_name(ctx: ClaimContext) -> Optional[str]:
    """The extracted bill is the source of truth for the hospital; the form value
    is only a fallback. (The network discount must come from what the documents
    show, not from what the submitter typed.)"""
    for doc in ctx.extracted_documents:
        name = doc.get("fields", {}).get("hospital_name")
        if name:
            return str(name)
    if ctx.submission.hospital_name:
        return ctx.submission.hospital_name
    return None


def line_category(description: str, primary: str) -> str:
    """The category a single line is adjudicated under: a specialty keyword wins;
    a generic consultation signal -> CONSULTATION; no signal -> the claim's
    primary category. PHARMACY is never derived from a line — a 'Medicines
    (Pharmacy)' line on a hospital bill stays in the base category (pharmacy
    derives only from a PHARMACY_BILL document). This rule is load-bearing for
    TC010 and the network_discount bundle."""
    corpus = normalize_corpus([description])
    for cat in SPECIALTY_CATEGORIES:
        if cat == "PHARMACY":
            continue
        if category_hits(corpus, cat):
            return cat
    if category_hits(corpus, GENERIC_CATEGORY):
        return GENERIC_CATEGORY
    return primary


def coverage_uncertain(desc: str, primary: str, policy) -> bool:
    """True when a non-excluded line belongs to a whitelist category (one
    shipping a covered list — dental/vision) and is NOT on that category's
    covered list. The line's OWN category is used, so a consultation line on a
    dental claim is judged against consultation (which has no covered list), not
    dental — only genuinely non-covered specialty procedures are held."""
    terms = policy.category_terms(line_category(desc, primary))
    if terms is None:
        return False
    covered = list(terms.covered_procedures) + list(terms.covered_items)
    return bool(covered) and covered_in_list(desc, covered) is None


def category_groups(ctx: ClaimContext):
    """APPROVED line items grouped by their own category — but ONLY on the
    derived path and ONLY when two or more categories are present. Returns None
    otherwise (provided category, no line items, or a single category), so the
    caller uses the unchanged single-category path."""
    if not ctx.category_was_derived or not ctx.result.line_items:
        return None
    primary = ctx.submission.claim_category.value
    groups: dict[str, list[LineItemDecision]] = {}
    for li in ctx.result.line_items:
        if li.status != "APPROVED":
            continue
        groups.setdefault(line_category(li.description, primary), []).append(li)
    return groups if len(groups) > 1 else None
