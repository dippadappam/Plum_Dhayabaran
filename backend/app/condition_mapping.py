"""Deterministic mapping from free-text diagnosis/treatment to policy conditions.

Decision 5 of the architecture: dictionary-first so the entire decision path
stays deterministic and testable. Anything unmatched returns no mapping and
never silently produces a wrong decision. A constrained LLM fallback can be
added behind the same interface for long-tail text.

Matching is keyword containment over normalized text, against a small fixed
set of policy targets (waiting-period conditions and exclusion conditions).
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from app.clinical_mappings import CLINICAL_MAPPINGS


def _norm(text: str) -> str:
    return " ".join(text.lower().replace("-", " ").replace("—", " ").split())


def _contains_phrase(corpus: str, phrase: str) -> bool:
    """Whole-word phrase match: 'hernia' must not match 'herniation'."""
    return re.search(rf"\b{re.escape(phrase)}\b", corpus) is not None


# Clinical keyword tables — loaded once at startup from
# data/clinical_mappings.json (app/clinical_mappings.py), not hardcoded here.
# WAITING_CONDITION_KEYWORDS maps the policy's waiting-period condition keys
# (policy_terms.json: waiting_periods.specific_conditions) and EXCLUSION_KEYWORDS
# maps the policy's exclusion conditions (exclusions.conditions), each to its
# ordered keyword synonyms. Read below by map_text_to_conditions.
WAITING_CONDITION_KEYWORDS: dict[str, list[str]] = \
    CLINICAL_MAPPINGS.waiting_condition_keywords
EXCLUSION_KEYWORDS: dict[str, list[str]] = CLINICAL_MAPPINGS.exclusion_keywords


@dataclass
class ConditionMapping:
    """Result of mapping diagnosis/treatment text to policy conditions."""

    matched_waiting_conditions: list[str] = field(default_factory=list)
    matched_exclusions: list[str] = field(default_factory=list)
    matched_terms: dict[str, str] = field(default_factory=dict)  # condition -> keyword that matched


def map_text_to_conditions(*texts: Optional[str]) -> ConditionMapping:
    """Map one or more free-text fields (diagnosis, treatment, line-item
    descriptions) to waiting-period conditions and exclusions.

    Deterministic: same input always produces the same mapping.
    """
    result = ConditionMapping()
    corpus = _norm(" | ".join(t for t in texts if t))
    if not corpus:
        return result

    for condition, keywords in WAITING_CONDITION_KEYWORDS.items():
        for kw in keywords:
            if _contains_phrase(corpus, _norm(kw)):
                if condition not in result.matched_waiting_conditions:
                    result.matched_waiting_conditions.append(condition)
                    result.matched_terms[condition] = kw
                break

    for exclusion, keywords in EXCLUSION_KEYWORDS.items():
        for kw in keywords:
            if _contains_phrase(corpus, _norm(kw)):
                if exclusion not in result.matched_exclusions:
                    result.matched_exclusions.append(exclusion)
                    result.matched_terms[exclusion] = kw
                break

    return result


def map_canonical_condition(
    canonical: Optional[str],
    waiting_keys: list[str],
    exclusion_conditions: list[str],
) -> ConditionMapping:
    """Map a single extractor-provided `canonical_condition` (already constrained
    by the prompt to the policy's vocabulary) to a ConditionMapping, matched
    case-/separator-insensitively against the policy's waiting-period condition
    keys (e.g. 'diabetes', 'obesity_treatment') and exclusion condition strings
    (e.g. 'Bariatric surgery'). Returns an EMPTY mapping when the value is not
    recognized, so the caller can fall back to the keyword path rather than
    trusting an out-of-vocabulary canonical."""
    result = ConditionMapping()
    if not canonical:
        return result
    v = _norm(canonical)
    v_key = v.replace(" ", "_")
    for wk in waiting_keys:
        if v == _norm(wk) or v_key == _norm(wk).replace(" ", "_"):
            result.matched_waiting_conditions.append(wk)
            result.matched_terms[wk] = canonical
            return result
    for ex in exclusion_conditions:
        if v == _norm(ex):
            result.matched_exclusions.append(ex)
            result.matched_terms[ex] = canonical
            return result
    return result


def match_in_list(text: str, items: list[str]) -> Optional[str]:
    """One-direction WORD-BOUNDARY containment for EXCLUDED-list matching: the
    policy phrase must appear in the line text as whole words (not embedded in a
    larger word). This stops a short entry ('Bleaching') from over-matching
    inside an unrelated word, while still letting a line that contains the full
    phrase match, and still stopping a generic line label ('Treatment') from
    matching a longer excluded entry ('Orthodontic Treatment (Braces)').

    Lookarounds (`(?<!\\w)…(?!\\w)`) are used rather than `\\b` because policy
    phrases can end in punctuation ('(Braces)'), where a trailing `\\b` would
    fail to match. Returns the matched policy entry, or None."""
    n = _norm(text)
    for item in items:
        ni = _norm(item)
        if ni and re.search(rf"(?<!\w){re.escape(ni)}(?!\w)", n):
            return item
    return None


_NAME_HONORIFICS = {"mr", "mrs", "ms", "miss", "dr", "smt", "shri", "sri",
                    "kumari", "master"}
_NAME_REL_MARKERS = {"s/o", "d/o", "w/o", "c/o"}
# Minimum token-set overlap (shared tokens / the larger name's token count) for
# two normalized names to be treated as the same person in names_match.
_NAME_OVERLAP_THRESHOLD = 0.6


def normalize_person_name(name: str) -> str:
    """Normalize a person name for matching: lowercase, drop punctuation, drop
    honorific prefixes (Mr/Dr/…) and everything from a relationship marker
    (S/o, D/o, W/o, C/o) onward, collapse whitespace."""
    s = re.sub(r"[^\w\s/]", " ", str(name).lower())
    out: list[str] = []
    for tok in s.split():
        if tok in _NAME_REL_MARKERS:
            break
        if tok in _NAME_HONORIFICS:
            continue
        cleaned = tok.replace("/", "")
        if cleaned:
            out.append(cleaned)
    return " ".join(out)


def names_match(a: str, b: str) -> bool:
    """Whether two person names refer to the same person — tolerant of
    honorifics, punctuation, and S/o-style suffixes, and of middle initials or
    an extra surname (high-band token-set overlap). Conservative: unrelated
    names, or names differing in surname, do not match, so a genuine patient
    mismatch is still flagged."""
    na, nb = normalize_person_name(a), normalize_person_name(b)
    if not na or not nb:
        return na == nb
    if na == nb:
        return True
    ta, tb = set(na.split()), set(nb.split())
    if not ta or not tb:
        return False
    return len(ta & tb) / max(len(ta), len(tb)) >= _NAME_OVERLAP_THRESHOLD


def covered_in_list(text: str, items: list[str]) -> Optional[str]:
    """Lenient token-overlap match for COVERED-list (whitelist) checking, so a
    partial label ('Root Canal') still matches the covered entry ('Root Canal
    Treatment'). Matches when the line's tokens are a subset of a covered
    entry's tokens or vice versa. Returns the matched entry, or None."""
    text_tokens = set(_norm(text).split())
    if not text_tokens:
        return None
    for item in items:
        item_tokens = set(_norm(item).split())
        if item_tokens and (text_tokens <= item_tokens
                             or item_tokens <= text_tokens):
            return item
    return None
