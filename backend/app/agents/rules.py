"""Adjudication rules — the decomposed rules engine.

Each rule is a small object with a uniform ``apply(ctx) -> bool`` (True ⇒ the
rule resolved the claim and the gate runner stops; False ⇒ continue). The check
ORDER is data — ``build_gates`` (the resolving gates) and ``build_terminal`` (the
always-run money/fraud/finalize sequence, then the identity and confidence holds
relocated from the orchestrator) — not control flow buried in one 1,600-line
method. Every rule's logic, its trace steps (stage/check/status/
detail/data), and its Fix-7 field attribution (``read_fields`` →
``ctx.deciding_fields``) are identical to the ``_check_*`` methods they replaced.

Cross-cutting helpers shared by several rules live in
``app/agents/adjudication_support.py`` (free functions over the shared context).
"""

import math
import re
from decimal import ROUND_HALF_UP, Decimal
from typing import Optional

from app.agents.adjudication_support import (
    all_text_fields,
    bill_hospital_name,
    category_groups,
    coverage_uncertain,
    diagnosis_mapping,
    diagnosis_texts,
    documented_bill_total,
    line_items,
)
from app.agents.base import AgentError, ClaimContext
from app.condition_mapping import map_text_to_conditions, match_in_list
from app.models.decision import (
    AmountBreakdown,
    CategoryBreakdown,
    ComponentFailure,
    Decision,
    LineItemDecision,
    RejectionReason,
    StepStatus,
)

STAGE = "adjudication"

# Weighted fraud-score signal weights. They sum to 1.0, so the score is in
# [0, 1]. Each signal is graduated (unlike the binary same-day/monthly/
# high-value gates) and its contribution (weight x sub-score) is shown in the
# trace, so the final number is fully explainable. Corroboration is the point:
# no single signal can cross the review threshold alone — it takes several.
FRAUD_WEIGHTS = {
    "same_day_frequency": 0.30,
    "monthly_frequency": 0.20,
    "amount_vs_history": 0.25,
    "near_duplicate": 0.25,
}

# Near-duplicate thresholds for the `near_duplicate` fraud sub-score: a prior
# claim whose amount is within _NEAR_DUPLICATE_AMOUNT_BAND of this claim's amount
# is a near-duplicate on the same date, or within _NEAR_DUPLICATE_DAY_WINDOW days
# of it.
_NEAR_DUPLICATE_AMOUNT_BAND = 0.02   # fraction of the claimed amount (2%)
_NEAR_DUPLICATE_DAY_WINDOW = 3       # days on either side of the treatment date


def round_money(x: float) -> float:
    """Round a rupee amount to 2 decimals, HALF-UP (e.g. ₹0.005 -> ₹0.01).

    Currency amounts use round-half-up by convention (the conventional per-claim
    rounding), NOT Python's built-in round(), which is round-half-to-even
    (banker's) and would send a half-paisa to the nearest even digit. Applied at
    exactly the points the money math previously called round(value, 2), at the
    same 2-decimal precision. Only currency uses this — confidence scores and the
    fraud sub-scores keep round() (they are not currency)."""
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


# --- shared rejection / routing helpers (free functions; mutate ctx.result) --
def reject(ctx: ClaimContext, reason: RejectionReason, message: str,
           read_fields: Optional[set[str]]) -> None:
    ctx.result.decision = Decision.REJECTED
    if reason not in ctx.result.rejection_reasons:
        ctx.result.rejection_reasons.append(reason)
    ctx.result.reasons.append(message)
    ctx.result.approved_amount = 0
    # Fix 7: record the extracted fields THIS rejection rested on, so the
    # confidence gate holds only when one of them was a low-confidence read.
    # read_fields is REQUIRED (no default): a rule cannot reject without
    # declaring which fields it read. Pass an explicit None only to declare
    # "no extracted field drove this rejection"; omitting the argument is now a
    # hard error at the call site, so the confidence gate can never silently act
    # on stale deciding_fields.
    if read_fields is not None:
        ctx.deciding_fields = set(read_fields)


def route_to_review(ctx: ClaimContext, message: str) -> None:
    """Hold a claim for a human without auto-rejecting or auto-paying.

    No read_fields here (unlike reject()): this sets MANUAL_REVIEW, and the
    confidence gate self-suppresses on a MANUAL_REVIEW decision (it acts only on
    APPROVED/PARTIAL/REJECTED), so a routed-to-review claim never consults
    ctx.deciding_fields. Recording fields here would be dead input, so the
    required-field lock that reject() carries is intentionally not mirrored."""
    ctx.result.decision = Decision.MANUAL_REVIEW
    ctx.result.manual_review_recommended = True
    ctx.result.reasons.append(message)


# --- rule base + runner ------------------------------------------------------
class Rule:
    """A single adjudication rule. ``apply`` returns True when it resolves the
    claim (the runner then stops); False to continue to the next rule."""

    name: str = "rule"

    def apply(self, ctx: ClaimContext) -> bool:
        raise NotImplementedError


def run_rules(rules: list[Rule], ctx: ClaimContext) -> bool:
    """Run the gate rules in order; return True if one resolved the claim."""
    for rule in rules:
        if rule.apply(ctx):
            return True
    return False


# --- gate rules --------------------------------------------------------------
class AmountSanityRule(Rule):
    """Amount sanity & reconciliation (runs before any decision math).

    NOTE on framing: an amount-sanity and reconciliation layer, NOT a fraud stop.
    The payout follows the line items and is bounded by the per-claim cap, so a
    *matched* inflation is bounded by the cap, not by this check. The protective
    piece is the absurd-amount bound; the reconciliation surfaces a likely misread
    (or an altered total) for a human. Tolerances/ceiling come from EngineConfig."""

    name = "amount_sanity"

    def __init__(self, config):
        self.config = config

    def apply(self, ctx: ClaimContext) -> bool:
        tolerance = self.config.reconcile_tolerance
        band = self.config.reconcile_band
        ceiling = (self.config.amount_ceiling
                   if self.config.amount_ceiling is not None
                   else ctx.policy.coverage.sum_insured_per_employee)
        # Per document and per line, so this composes with Batch 6's per-line
        # categorization and multi-bill aggregation (no single-bill/category
        # assumption).
        for doc in ctx.extracted_documents:
            fields = doc.get("fields", {})
            fid = doc.get("file_id", "?")
            items = fields.get("line_items") or []

            def _num(v):
                try:
                    return float(v)
                except (TypeError, ValueError):
                    return float("nan")

            # Absurd-amount bound (per line). Non-finite or beyond the entire
            # sum insured is impossible for an OPD line — hold for a human.
            for item in items:
                amt = _num(item.get("amount", 0))
                if not math.isfinite(amt) or abs(amt) > ceiling:
                    route_to_review(
                        ctx,
                        f"A line item on document {fid} has an out-of-range "
                        f"amount ({item.get('amount')!r}). A reviewer will check "
                        "the bill before the claim is processed.")
                    ctx.trace(STAGE, "amount_sanity", StepStatus.FAILED,
                              f"Out-of-range line-item amount on {fid}: "
                              f"{item.get('amount')!r} (ceiling ₹{ceiling:,.0f}).")
                    return True

            total = fields.get("total")
            if total is not None:
                t = _num(total)
                if not math.isfinite(t) or abs(t) > ceiling:
                    route_to_review(
                        ctx,
                        f"The bill total on document {fid} is out of range "
                        f"({total!r}). A reviewer will check the bill.")
                    ctx.trace(STAGE, "amount_sanity", StepStatus.FAILED,
                              f"Out-of-range total on {fid}: {total!r}.")
                    return True

            if items:
                net = sum(_num(i.get("amount", 0)) for i in items)
                if net < 0:
                    route_to_review(
                        ctx,
                        f"The itemized charges on document {fid} sum to a "
                        f"negative amount. A reviewer will check the bill.")
                    ctx.trace(STAGE, "amount_sanity", StepStatus.FAILED,
                              f"Net-negative line items on {fid} (₹{net:,.0f}).")
                    return True

            # Reconciliation: bill total vs the itemized sum (per document).
            if total is not None and items:
                t = _num(total)
                s = sum(_num(i.get("amount", 0)) for i in items)
                diff = abs(t - s)
                larger = max(abs(t), abs(s), 1.0)
                if diff <= tolerance:
                    ctx.trace(STAGE, "amount_reconciliation", StepStatus.PASSED,
                              f"Document {fid}: total ₹{t:,.0f} matches the "
                              f"itemized sum ₹{s:,.0f}.")
                elif diff <= band * larger:
                    ctx.trace(STAGE, "amount_reconciliation", StepStatus.INFO,
                              f"Document {fid}: total ₹{t:,.0f} differs from the "
                              f"itemized sum ₹{s:,.0f} by ₹{diff:,.0f}, within "
                              f"{band:.0%} (normal discounts, taxes, or fees).")
                else:
                    route_to_review(
                        ctx,
                        f"On document {fid}, the bill total (₹{t:,.0f}) and the "
                        f"itemized charges (₹{s:,.0f}) differ by more than "
                        f"{band:.0%}. This may be a misread "
                        "of a legitimate bill or an altered total; a reviewer "
                        "will confirm the amount.")
                    ctx.trace(STAGE, "amount_reconciliation", StepStatus.FAILED,
                              f"Document {fid}: total ₹{t:,.0f} vs itemized "
                              f"₹{s:,.0f} diverge beyond "
                              f"{band:.0%}; routed to "
                              "MANUAL_REVIEW.")
                    return True
        return False


class DocumentAlterationRule(Rule):
    """Route to review when an extracted document shows a visible sign of
    alteration (the extractor's ``alteration_suspected`` flag) — a crossed-out or
    overwritten amount, ink/font inconsistent with the rest of the document, or
    an ORIGINAL/DUPLICATE stamp. Silent when no document carries the flag, so a
    clean document and the 12 structured cases (which have no such field) are
    untouched."""

    name = "document_alteration"

    def apply(self, ctx: ClaimContext) -> bool:
        flagged = [
            (doc.get("file_id", "?"),
             str(doc.get("fields", {}).get("alteration_reason")
                 or "no detail given"))
            for doc in ctx.extracted_documents
            if doc.get("fields", {}).get("alteration_suspected")
        ]
        if not flagged:
            return False
        detail = "; ".join(f"{fid}: {reason}" for fid, reason in flagged)
        route_to_review(
            ctx,
            "One or more uploaded documents show visible signs of alteration "
            f"({detail}); a reviewer must verify the document(s) before the "
            "claim can be processed.")
        ctx.trace(STAGE, "document_alteration", StepStatus.FAILED,
                  f"Document alteration suspected — {detail}. Routed to "
                  "MANUAL_REVIEW for manual verification.",
                  {"altered_documents": [fid for fid, _ in flagged]})
        return True


class ClaimedAmountMismatchRule(Rule):
    """Cross-check the member's stated claim amount against the documented bill
    total. When they diverge by more than ``config.claimed_amount_band``, route
    to review — a mismatch is a data-quality signal, not proof of fraud, so it is
    never an auto-reject. Silent when the member stated no amount (the real-upload
    derive path), when there is no documented total, or when the two agree within
    the band — so the 12 official cases (claimed == bill) are untouched."""

    name = "claimed_amount_mismatch"

    def __init__(self, config):
        self.config = config

    def apply(self, ctx: ClaimContext) -> bool:
        stated = ctx.original_claimed_amount
        if stated is None or stated <= 0:
            return False  # no member-stated amount to compare
        bill_total = documented_bill_total(ctx)
        if bill_total is None or bill_total <= 0:
            return False  # no documented total to compare against
        band = self.config.claimed_amount_band
        if abs(stated - bill_total) <= band * max(stated, bill_total):
            return False  # within tolerance (rounding / small discounts)
        route_to_review(
            ctx,
            f"The amount claimed (₹{stated:,.0f}) differs materially from the "
            f"documented bill total (₹{bill_total:,.0f}); a reviewer must verify "
            "the amount before the claim is processed.")
        ctx.trace(STAGE, "claimed_amount_mismatch", StepStatus.FAILED,
                  f"Claimed amount ₹{stated:,.0f} differs from the documented "
                  f"bill total ₹{bill_total:,.0f} by more than {band:.0%}; routed "
                  "to MANUAL_REVIEW for amount verification.",
                  {"claimed_amount": stated, "bill_total": bill_total,
                   "band": band})
        return True


class EligibilityRule(Rule):
    """Eligibility: future-date sanity (treatment after submission → review) and
    the policy-period out-of-coverage check."""

    name = "eligibility"

    def apply(self, ctx: ClaimContext) -> bool:
        sub, policy = ctx.submission, ctx.policy
        ph = policy.policy_holder

        # Future-date sanity check (only when the submission date is known): a
        # treatment date after the submission date is impossible — hold for a
        # human to correct rather than adjudicating on a bad date.
        if sub.submission_date is not None and sub.treatment_date is not None \
                and sub.treatment_date > sub.submission_date:
            ctx.result.decision = Decision.MANUAL_REVIEW
            ctx.result.manual_review_recommended = True
            ctx.result.reasons.append(
                f"The treatment date {sub.treatment_date} is after the claim "
                f"submission date {sub.submission_date}, which is not possible. "
                "Held for manual review to correct the date before the claim "
                "can be processed."
            )
            ctx.trace(STAGE, "future_treatment_date", StepStatus.FAILED,
                      f"Treatment date {sub.treatment_date} is in the future "
                      f"relative to submission {sub.submission_date}; routed to "
                      "MANUAL_REVIEW.")
            return True

        if not (ph.policy_start_date <= sub.treatment_date <= ph.policy_end_date):
            reject(
                ctx, RejectionReason.NOT_COVERED,
                f"Treatment date {sub.treatment_date} falls outside the policy period "
                f"({ph.policy_start_date} to {ph.policy_end_date}).",
                read_fields={"treatment_date"},
            )
            ctx.trace(STAGE, "policy_period", StepStatus.FAILED,
                      f"Treatment date {sub.treatment_date} outside policy period.")
            return True
        ctx.trace(STAGE, "policy_period", StepStatus.PASSED,
                  f"Treatment date {sub.treatment_date} is within the policy period "
                  f"({ph.policy_start_date} to {ph.policy_end_date}).")
        return False


class SubmissionDeadlineRule(Rule):
    """Submission-deadline window: a claim received more than
    `deadline_days_from_treatment` days after the treatment date is out of time
    and rejected. Measured against `received_date` (a real intake date), never
    the auto-stamped `submission_date`; absent a received_date, skipped and
    silent (the 12 official cases set none)."""

    name = "submission_deadline"

    def apply(self, ctx: ClaimContext) -> bool:
        sub, policy = ctx.submission, ctx.policy
        if sub.received_date is None or sub.treatment_date is None:
            return False
        limit = policy.submission_rules.deadline_days_from_treatment
        gap = (sub.received_date - sub.treatment_date).days
        if gap > limit:
            reject(
                ctx, RejectionReason.SUBMISSION_DEADLINE_PASSED,
                f"This claim was received on {sub.received_date}, "
                f"{gap} days after the treatment date {sub.treatment_date}, "
                f"beyond the {limit}-day submission deadline. Claims must be "
                f"filed within {limit} days of treatment.",
                read_fields={"treatment_date"},
            )
            ctx.trace(STAGE, "submission_deadline", StepStatus.FAILED,
                      f"Received {gap} days after treatment (limit {limit} "
                      f"days); beyond the submission deadline.",
                      {"gap_days": gap, "deadline_days": limit})
            return True
        ctx.trace(STAGE, "submission_deadline", StepStatus.PASSED,
                  f"Received {gap} day(s) after treatment, within the "
                  f"{limit}-day submission deadline.")
        return False


class DiagnosisCertaintyRule(Rule):
    """If a decision-critical diagnosis carries a `canonical_condition` mapping
    BELOW the confidence threshold, hold for a human rather than deciding the
    waiting-period/exclusion checks on an uncertain read. The 12 carry no
    canonical_condition, so this never fires for them and is silent."""

    name = "diagnosis_certainty"

    def __init__(self, config):
        self.config = config

    def apply(self, ctx: ClaimContext) -> bool:
        _, uncertain = diagnosis_mapping(ctx, self.config.confidence_threshold)
        if not uncertain:
            return False
        route_to_review(
            ctx,
            "The diagnosis could not be confidently mapped to a policy condition "
            "(the extractor's canonical mapping was below the confidence "
            "threshold). A reviewer will confirm the diagnosis before the "
            "waiting-period and exclusion rules are applied.")
        ctx.trace(STAGE, "diagnosis_certainty", StepStatus.FAILED,
                  "Low-confidence canonical diagnosis mapping; routed to "
                  "MANUAL_REVIEW rather than deciding on an uncertain diagnosis.")
        return True


class ExclusionsRule(Rule):
    """Exclusions at the line-item level (before waiting periods, because an
    excluded condition is permanently not covered). All excluded ⇒ REJECTED;
    mixed ⇒ PARTIAL; a whitelist-category line on neither list ⇒ review."""

    name = "exclusions"

    def __init__(self, config):
        self.config = config

    def apply(self, ctx: ClaimContext) -> bool:
        sub, policy = ctx.submission, ctx.policy
        terms = policy.category_terms(sub.claim_category.value)
        items_list = line_items(ctx)

        # Total-only bill (no itemized lines) in a category that has procedure/
        # item exclusions (dental, vision): an excluded procedure could be
        # hidden in the lump total and cannot be checked line by line, so hold
        # for a human rather than auto-pay. Categories with no such lists
        # (consultation, diagnostic, pharmacy, alt-med) are unaffected — a
        # total-only consultation bill (TC005/TC009) still flows unchanged.
        has_proc_lists = bool(terms and (
            terms.excluded_procedures or terms.excluded_items
            or terms.covered_procedures or terms.covered_items))
        if not items_list and has_proc_lists:
            route_to_review(
                ctx,
                f"This {sub.claim_category.value} claim has no itemized bill "
                f"lines, so individual covered/excluded "
                f"{sub.claim_category.value.lower()} procedures cannot be "
                "verified. A reviewer will itemize and confirm coverage before "
                "payment.")
            ctx.trace(STAGE, "total_only_bill", StepStatus.FAILED,
                      f"Total-only bill in {sub.claim_category.value} (which has "
                      "procedure/item exclusions); routed to MANUAL_REVIEW.")
            return True

        diag_texts = diagnosis_texts(ctx)
        treatment_texts = [
            t for t in (
                doc.get("fields", {}).get("treatment")
                for doc in ctx.extracted_documents
            ) if t
        ]

        decisions: list[LineItemDecision] = []
        any_excluded = False
        any_covered = False
        uncertain: list[str] = []

        items = items_list or [
            {"description": " / ".join(treatment_texts + diag_texts) or
             sub.claim_category.value, "amount": sub.claimed_amount}
        ]

        # Diagnosis-level exclusion applies to the whole claim: if the PRIMARY
        # DIAGNOSIS is an excluded condition, every line item is excluded.
        # Scoped to the diagnosis field only (not the treatment field, which the
        # vision model sometimes invents) so a comorbidity cannot taint an
        # unrelated claim. Per-line exclusion matching (below) is unchanged.
        # Prefers the extractor's canonical_condition when present/confident,
        # else falls back to the keyword dictionary (byte-identical for the 12).
        diag_mapping, _ = diagnosis_mapping(ctx, self.config.confidence_threshold)

        for item in items:
            desc = str(item.get("description", ""))
            amount = float(item.get("amount", 0))
            reason = None

            # Category-specific excluded procedures/items (dental, vision).
            if terms:
                hit = match_in_list(desc, terms.excluded_procedures) or \
                      match_in_list(desc, terms.excluded_items)
                if hit:
                    reason = (f"'{desc}' matches excluded "
                              f"{sub.claim_category.value.lower()} item "
                              f"'{hit}' under the policy.")

            # General policy exclusions on the line item text.
            if reason is None:
                item_mapping = map_text_to_conditions(desc)
                if item_mapping.matched_exclusions:
                    excl = item_mapping.matched_exclusions[0]
                    reason = (f"'{desc}' falls under policy exclusion "
                              f"'{excl}' (matched on "
                              f"'{item_mapping.matched_terms[excl]}').")

            # Diagnosis-level exclusion taints all line items.
            if reason is None and diag_mapping.matched_exclusions:
                excl = diag_mapping.matched_exclusions[0]
                reason = (f"The diagnosis/treatment "
                          f"('{'; '.join(diag_texts + treatment_texts)}') falls "
                          f"under policy exclusion '{excl}', so this line item is "
                          f"not payable.")

            if reason:
                any_excluded = True
                decisions.append(LineItemDecision(
                    description=desc, claimed_amount=amount, approved_amount=0,
                    status="REJECTED", reason=reason,
                ))
                ctx.trace(STAGE, "exclusions", StepStatus.FAILED,
                          f"Line item '{desc}' (₹{amount:,.0f}) rejected: {reason}")
            elif coverage_uncertain(desc, sub.claim_category.value, policy):
                # Whitelist category (dental/vision): the line is not on its own
                # category's covered list -> coverage uncertain, hold for a human
                # (judged by the line's own category, so a consultation line on a
                # dental claim is not measured against the dental list).
                uncertain.append(desc)
                decisions.append(LineItemDecision(
                    description=desc, claimed_amount=amount, approved_amount=0,
                    status="REVIEW",
                    reason=(f"'{desc}' is on neither the covered nor the excluded "
                            f"{sub.claim_category.value.lower()} list; a reviewer "
                            "must confirm coverage before payment."),
                ))
                ctx.trace(STAGE, "exclusions", StepStatus.FAILED,
                          f"Line item '{desc}' (₹{amount:,.0f}) is on neither the "
                          f"covered nor excluded {sub.claim_category.value} list; "
                          "held for review.")
            else:
                any_covered = True
                decisions.append(LineItemDecision(
                    description=desc, claimed_amount=amount, approved_amount=amount,
                    status="APPROVED", reason="Covered under the policy.",
                    drug_type=item.get("drug_type"),
                    drug_type_confidence=item.get("drug_type_confidence"),
                ))
                ctx.trace(STAGE, "exclusions", StepStatus.PASSED,
                          f"Line item '{desc}' (₹{amount:,.0f}) is covered.")

        ctx.result.line_items = decisions

        if uncertain:
            route_to_review(
                ctx,
                f"This {sub.claim_category.value} claim includes item(s) on "
                f"neither the policy's covered nor excluded list "
                f"({', '.join(uncertain)}); a reviewer must confirm coverage "
                "before any payment.")
            ctx.trace(STAGE, "covered_list", StepStatus.FAILED,
                      f"{len(uncertain)} item(s) not on the covered/excluded list "
                      "for this category; routed to MANUAL_REVIEW.")
            return True

        if any_excluded and not any_covered:
            reject(
                ctx, RejectionReason.EXCLUDED_CONDITION,
                "All items in this claim fall under policy exclusions: "
                + "; ".join(d.reason for d in decisions if d.status == "REJECTED"),
                read_fields={"diagnosis", "amount"},
            )
            return True

        if not any_excluded:
            ctx.trace(STAGE, "exclusions_summary", StepStatus.PASSED,
                      "No policy exclusions apply to this claim.")
        return False


class WaitingPeriodsRule(Rule):
    """Initial waiting period, then condition-specific periods from the member's
    join date (dependents inherit the primary's). Matched on the primary /
    canonical diagnosis only."""

    name = "waiting_periods"

    def __init__(self, config):
        self.config = config

    def apply(self, ctx: ClaimContext) -> bool:
        from datetime import timedelta
        sub, policy = ctx.submission, ctx.policy
        member = policy.get_member(sub.member_id)
        join = policy.member_join_date(member) if member else None
        if join is None:
            ctx.trace(STAGE, "waiting_period", StepStatus.SKIPPED,
                      "Member join date unavailable; waiting-period check skipped.")
            return False

        wp = policy.waiting_periods

        # Initial waiting period applies to all claims.
        initial_end = join + timedelta(days=wp.initial_waiting_period_days)
        if sub.treatment_date < initial_end:
            reject(
                ctx, RejectionReason.WAITING_PERIOD,
                f"Treatment date {sub.treatment_date} falls within the initial "
                f"{wp.initial_waiting_period_days}-day waiting period. The member "
                f"joined on {join} and is eligible for claims from {initial_end}.",
                read_fields={"treatment_date"},
            )
            ctx.trace(STAGE, "initial_waiting_period", StepStatus.FAILED,
                      f"Within initial waiting period; eligible from {initial_end}.")
            return True
        ctx.trace(STAGE, "initial_waiting_period", StepStatus.PASSED,
                  f"Initial {wp.initial_waiting_period_days}-day waiting period "
                  f"completed on {initial_end}.")

        # Condition-specific waiting periods from the PRIMARY DIAGNOSIS only.
        # A comorbidity mentioned in a line item, test name, or treatment field
        # must not trip a waiting period for an unrelated claim (a within-window
        # member seeing a doctor for a viral fever, whose prescription notes
        # "k/c/o diabetes", is not making a diabetes claim).
        mapping, _ = diagnosis_mapping(ctx, self.config.confidence_threshold)
        for condition in mapping.matched_waiting_conditions:
            days = wp.specific_conditions.get(condition)
            if days is None:
                continue
            eligible_from = join + timedelta(days=days)
            if sub.treatment_date < eligible_from:
                reject(
                    ctx, RejectionReason.WAITING_PERIOD,
                    f"The diagnosis maps to '{condition}', which has a {days}-day "
                    f"waiting period. The member joined on {join}, so "
                    f"{condition.replace('_', ' ')}-related claims are eligible "
                    f"from {eligible_from}. The treatment date {sub.treatment_date} "
                    f"is before that.",
                    read_fields={"diagnosis", "treatment_date"},
                )
                ctx.trace(
                    STAGE, f"waiting_period:{condition}", StepStatus.FAILED,
                    f"'{condition}' waiting period of {days} days not completed; "
                    f"eligible from {eligible_from} (matched on "
                    f"'{mapping.matched_terms.get(condition)}').",
                    {"eligible_from": str(eligible_from), "condition": condition},
                )
                return True
            ctx.trace(STAGE, f"waiting_period:{condition}", StepStatus.PASSED,
                      f"'{condition}' waiting period of {days} days completed on "
                      f"{eligible_from}.")

        if not mapping.matched_waiting_conditions:
            ctx.trace(STAGE, "condition_waiting_periods", StepStatus.PASSED,
                      "No condition-specific waiting periods apply to this diagnosis.")
        return False


class PreAuthorizationRule(Rule):
    """Global pre-auth: any high-value test (MRI/CT/PET) above its threshold in
    ANY category. Structured-path absence ⇒ REJECTED; live path or an
    unverifiable reference ⇒ review."""

    name = "pre_authorization"

    def apply(self, ctx: ClaimContext) -> bool:
        sub, policy = ctx.submission, ctx.policy

        # Global pre-auth rules: every (high-value test, threshold) the policy
        # defines in ANY category. Enforced regardless of the claim's resolved
        # category, so an MRI/CT/PET line cannot evade pre-auth by being filed
        # or derived under a category with no pre-auth config of its own.
        rules: list[tuple[str, float]] = []
        for cat_terms in policy.opd_categories.values():
            if cat_terms.high_value_tests_requiring_pre_auth and \
                    cat_terms.pre_auth_threshold is not None:
                for test in cat_terms.high_value_tests_requiring_pre_auth:
                    rules.append((test, cat_terms.pre_auth_threshold))

        # Find any line item (then free-text) naming a high-value test above its
        # threshold.
        triggering: list[tuple[str, float, float]] = []
        for item in line_items(ctx):
            desc = str(item.get("description", ""))
            amount = float(item.get("amount", 0))
            for test, threshold in rules:
                if test.lower() in desc.lower() and amount > threshold:
                    triggering.append((desc, amount, threshold))
        for text in all_text_fields(ctx):
            for test, threshold in rules:
                if test.lower() in text.lower() and sub.claimed_amount \
                        and sub.claimed_amount > threshold and not triggering:
                    triggering.append((text, sub.claimed_amount, threshold))

        if not triggering:
            # Preserve the pre-existing trace text, keyed on the claim's own
            # category, so single-category claims stay byte-identical.
            terms = policy.category_terms(sub.claim_category.value)
            if terms and terms.high_value_tests_requiring_pre_auth \
                    and terms.pre_auth_threshold is not None:
                ctx.trace(STAGE, "pre_authorization", StepStatus.PASSED,
                          f"No high-value test above "
                          f"₹{terms.pre_auth_threshold:,.0f} requiring "
                          "pre-authorization found in this claim.")
            else:
                ctx.trace(STAGE, "pre_authorization", StepStatus.PASSED,
                          "No pre-authorization requirement applies to this category.")
            return False

        # A pre-auth reference in any document. Without an internal pre-auth
        # registry we cannot VERIFY it, so a present reference no longer
        # auto-satisfies — it is held for a human to confirm. (Registry-backed
        # verification is the scheduled real fix.)
        has_pre_auth = any(
            doc.get("fields", {}).get("pre_auth_number")
            or doc.get("fields", {}).get("pre_authorization")
            for doc in ctx.extracted_documents
        )
        is_live = any(d.file_data for d in ctx.submission.documents)
        desc, amount, threshold = triggering[0]

        if has_pre_auth:
            route_to_review(
                ctx,
                f"'{desc}' (₹{amount:,.0f}) is above the ₹{threshold:,.0f} "
                "pre-authorization threshold and a pre-authorization reference "
                "was provided, but it could not be automatically verified. A "
                "reviewer will confirm the pre-authorization before payment.")
            ctx.trace(STAGE, "pre_authorization", StepStatus.FAILED,
                      f"Pre-authorization claimed for '{desc}' (₹{amount:,.0f}); "
                      "not auto-verifiable, routed to MANUAL_REVIEW.")
            return True

        if is_live:
            # Real upload with no pre-auth found: the pre-auth document may
            # simply not have extracted, so hold for a human rather than
            # auto-rejecting. Honest message — no impossible "obtain it after
            # the fact" instruction.
            route_to_review(
                ctx,
                f"'{desc}' (₹{amount:,.0f}) is above the ₹{threshold:,.0f} "
                "threshold that requires pre-authorization, and no "
                "pre-authorization was found in the uploaded documents. A "
                "reviewer will check whether a valid pre-authorization exists "
                "before the claim is decided.")
            ctx.trace(STAGE, "pre_authorization", StepStatus.FAILED,
                      f"No pre-authorization found for '{desc}' (₹{amount:,.0f}) "
                      "on the live-upload path; routed to MANUAL_REVIEW.")
            return True

        # Structured/known path with no pre-auth: definitive (nothing failed to
        # extract), so reject per policy — preserves the official contract.
        reject(
            ctx, RejectionReason.PRE_AUTH_MISSING,
            f"'{desc}' costs ₹{amount:,.0f}, which is above the ₹{threshold:,.0f} "
            f"threshold requiring pre-authorization, and no pre-authorization was "
            f"obtained. To resubmit: obtain pre-authorization from the insurer "
            f"(valid for {policy.pre_authorization.validity_days} days) and attach "
            f"the pre-authorization number to the claim.",
            read_fields={"amount"},
        )
        ctx.trace(STAGE, "pre_authorization", StepStatus.FAILED,
                  f"Pre-authorization required for '{desc}' (₹{amount:,.0f} > "
                  f"₹{threshold:,.0f}) but not obtained.")
        return True


class PractitionerRegistrationRule(Rule):
    """Categories with `requires_registered_practitioner` (alt-med) must carry a
    non-empty practitioner registration number, else review (present-only — no
    format regex). Silent on success, so TC011 (which carries one) is unchanged."""

    name = "practitioner_registration"

    def apply(self, ctx: ClaimContext) -> bool:
        sub, policy = ctx.submission, ctx.policy
        terms = policy.category_terms(sub.claim_category.value)
        if terms is None or not terms.requires_registered_practitioner:
            return False
        registration = None
        for doc in ctx.extracted_documents:
            reg = doc.get("fields", {}).get("doctor_registration")
            if reg and str(reg).strip():
                registration = str(reg).strip()
                break
        if registration:
            return False
        route_to_review(
            ctx,
            f"{sub.claim_category.value} claims require a registered "
            "practitioner, but no practitioner registration number was found in "
            "the submitted documents. A reviewer will confirm the practitioner's "
            "registration before payment.")
        ctx.trace(STAGE, "practitioner_registration", StepStatus.FAILED,
                  f"No practitioner registration on a {sub.claim_category.value} "
                  "claim that requires a registered practitioner; routed to "
                  "MANUAL_REVIEW.")
        return True


class SessionCapRule(Rule):
    """Categories with `max_sessions_per_year` (alt-med, 20): held when THIS
    claim alone names more sessions than the cap. Pass-through (silent) when no
    count is parseable; cross-year accrual is deferred."""

    name = "session_cap"

    def _parse_session_count(self, ctx: ClaimContext) -> Optional[int]:
        """Total sessions named on THIS claim, parsed from line-item descriptions
        and the treatment field (e.g. 'Panchakarma Therapy (5 sessions)').
        Returns None when no count is stated, so the caller can pass through
        rather than guess."""
        total = 0
        found = False
        for doc in ctx.extracted_documents:
            fields = doc.get("fields", {})
            texts = [str(fields.get("treatment"))] if fields.get("treatment") else []
            for item in fields.get("line_items") or []:
                if item.get("description"):
                    texts.append(str(item["description"]))
            for t in texts:
                for m in re.finditer(r"(\d+)\s*sessions?\b", t.lower()):
                    total += int(m.group(1))
                    found = True
        return total if found else None

    def apply(self, ctx: ClaimContext) -> bool:
        sub, policy = ctx.submission, ctx.policy
        terms = policy.category_terms(sub.claim_category.value)
        if terms is None or terms.max_sessions_per_year is None:
            return False
        cap = terms.max_sessions_per_year
        sessions = self._parse_session_count(ctx)
        if sessions is not None and sessions > cap:
            route_to_review(
                ctx,
                f"This {sub.claim_category.value} claim names {sessions} sessions, "
                f"above the policy cap of {cap} sessions per year. A reviewer will "
                "confirm the session count and prior-year usage before payment.")
            ctx.trace(STAGE, "session_cap", StepStatus.FAILED,
                      f"{sessions} sessions on this claim exceed the "
                      f"{cap}-session cap; routed to MANUAL_REVIEW.",
                      {"sessions": sessions, "cap": cap})
            return True
        return False


class HighValueRule(Rule):
    """An otherwise-payable claim above `fraud_thresholds.auto_manual_review_above`
    is held BEFORE the limit/money rules, so a genuinely large claim is reviewed
    rather than auto-rejected by the per-claim cap. Silent within the threshold."""

    name = "high_value"

    def apply(self, ctx: ClaimContext) -> bool:
        ft = ctx.policy.fraud_thresholds
        amount = ctx.submission.claimed_amount or 0
        if amount > ft.auto_manual_review_above:
            route_to_review(
                ctx,
                f"The claimed amount ₹{amount:,.0f} is above the "
                f"₹{ft.auto_manual_review_above:,.0f} high-value threshold; held "
                "for a reviewer to verify before payment.")
            ctx.result.fraud_signals.append(
                f"High-value claim: ₹{amount:,.0f} is above the "
                f"₹{ft.auto_manual_review_above:,.0f} auto-review threshold.")
            ctx.trace(STAGE, "high_value", StepStatus.FAILED,
                      f"Claimed ₹{amount:,.0f} is above the auto-review threshold "
                      f"₹{ft.auto_manual_review_above:,.0f}; routed to MANUAL_REVIEW.")
            return True
        return False


class LimitsRule(Rule):
    """Per-claim cap (= max(per_claim_limit, category sub_limit)), annual OPD
    limit, and the family-floater combined limit — applied to the payable
    (post-exclusion) eligible amount. On the derived multi-category path, applied
    per category group; a provided category folds all lines into one group."""

    name = "limits"

    def apply(self, ctx: ClaimContext) -> bool:
        groups = category_groups(ctx)
        return (self._limits_multi(ctx, groups) if groups
                else self._limits_single(ctx))

    def _family_limit(self, ctx: ClaimContext, eligible: float) -> bool:
        """Family-floater combined limit: a family (employee + dependents) shares
        coverage.family_floater.combined_limit on TOP of each member's own annual
        OPD limit. Fails when the family's year-to-date approved spend (injected
        by the API as submission.family_ytd_amount — the engine never reads
        storage) plus this claim's eligible amount exceeds the combined limit.
        SILENT on pass, so the 12 (family_ytd 0) stay byte-identical."""
        ff = ctx.policy.coverage.family_floater
        if not ff.enabled:
            return False
        family_ytd = ctx.submission.family_ytd_amount
        if family_ytd + eligible > ff.combined_limit:
            reject(
                ctx, RejectionReason.FAMILY_LIMIT_EXCEEDED,
                f"This claim of ₹{eligible:,.0f} on top of ₹{family_ytd:,.0f} "
                f"already approved for the family this year exceeds the family "
                f"floater limit of ₹{ff.combined_limit:,.0f}.",
                read_fields={"amount"})
            ctx.trace(STAGE, "family_floater_limit", StepStatus.FAILED,
                      f"Family YTD ₹{family_ytd:,.0f} + eligible ₹{eligible:,.0f} "
                      f"> family floater limit ₹{ff.combined_limit:,.0f}.")
            return True
        return False

    # --- single-category (provided, or one derived category): UNCHANGED ------
    def _limits_single(self, ctx: ClaimContext) -> bool:
        sub, policy = ctx.submission, ctx.policy
        terms = policy.category_terms(sub.claim_category.value)
        per_claim = policy.coverage.per_claim_limit
        sub_limit = terms.sub_limit if terms else per_claim

        # Documented assumption: limits apply to the payable (eligible) amount
        # after line-item exclusions, not the raw claimed amount.
        eligible = sum(
            li.approved_amount for li in ctx.result.line_items
            if li.status == "APPROVED"
        ) if ctx.result.line_items else sub.claimed_amount

        # Documented assumption: claim-level cap = max(per_claim_limit,
        # category sub_limit). See module docstring.
        claim_cap = max(per_claim, sub_limit)

        if eligible > claim_cap:
            reject(
                ctx, RejectionReason.PER_CLAIM_EXCEEDED,
                f"The claimed amount ₹{eligible:,.0f} exceeds the "
                f"per-claim limit of ₹{claim_cap:,.0f} for "
                f"{sub.claim_category.value} claims.",
                read_fields={"amount"},
            )
            ctx.trace(STAGE, "per_claim_limit", StepStatus.FAILED,
                      f"Payable amount ₹{eligible:,.0f} > per-claim limit "
                      f"₹{claim_cap:,.0f}.",
                      {"per_claim_limit": claim_cap,
                       "claimed_amount": sub.claimed_amount,
                       "payable_amount": eligible})
            return True
        ctx.trace(STAGE, "per_claim_limit", StepStatus.PASSED,
                  f"Payable amount ₹{eligible:,.0f} is within the per-claim "
                  f"limit of ₹{claim_cap:,.0f}.")

        # Consultation sub_limit caps the consultation-fee line item.
        if sub.claim_category.value == "CONSULTATION" and terms:
            for li in ctx.result.line_items:
                if "consultation" in li.description.lower() and \
                        li.status == "APPROVED" and \
                        li.approved_amount > terms.sub_limit:
                    capped = terms.sub_limit
                    ctx.trace(STAGE, "consultation_fee_sub_limit", StepStatus.INFO,
                              f"Consultation fee ₹{li.approved_amount:,.0f} capped "
                              f"at the ₹{capped:,.0f} consultation sub-limit.")
                    li.approved_amount = capped
                    li.reason += (f" Capped at the ₹{capped:,.0f} consultation "
                                  "sub-limit.")

        # Annual OPD limit on year-to-date claims. Accrues the post-exclusion
        # ELIGIBLE amount (not the gross claimed).
        annual = policy.coverage.annual_opd_limit
        if sub.ytd_claims_amount + eligible > annual:
            reject(
                ctx, RejectionReason.ANNUAL_LIMIT_EXCEEDED,
                f"This claim of ₹{eligible:,.0f} on top of "
                f"₹{sub.ytd_claims_amount:,.0f} already claimed this year exceeds "
                f"the annual OPD limit of ₹{annual:,.0f}.",
                read_fields={"amount"},
            )
            ctx.trace(STAGE, "annual_opd_limit", StepStatus.FAILED,
                      f"YTD ₹{sub.ytd_claims_amount:,.0f} + eligible "
                      f"₹{eligible:,.0f} > annual limit ₹{annual:,.0f}.")
            return True
        ctx.trace(STAGE, "annual_opd_limit", StepStatus.PASSED,
                  f"YTD ₹{sub.ytd_claims_amount:,.0f} + eligible "
                  f"₹{eligible:,.0f} is within the annual OPD limit of "
                  f"₹{annual:,.0f}.")
        if self._family_limit(ctx, eligible):
            return True
        return False

    # --- multi-category (derived path, two or more categories) ---------------
    def _limits_multi(self, ctx: ClaimContext, groups) -> bool:
        sub, policy = ctx.submission, ctx.policy
        eligible = sum(li.approved_amount
                       for lines in groups.values() for li in lines)
        per_claim = policy.coverage.per_claim_limit
        present = [policy.category_terms(c) for c in groups]
        # Ratified multi-category cap: max(per_claim, max present sub_limit).
        claim_cap = max([per_claim] + [t.sub_limit for t in present if t])

        if eligible > claim_cap:
            reject(
                ctx, RejectionReason.PER_CLAIM_EXCEEDED,
                f"The claimed amount ₹{eligible:,.0f} exceeds the per-claim "
                f"limit of ₹{claim_cap:,.0f} for this claim.",
                read_fields={"amount"})
            ctx.trace(STAGE, "per_claim_limit", StepStatus.FAILED,
                      f"Aggregate payable ₹{eligible:,.0f} > per-claim limit "
                      f"₹{claim_cap:,.0f}.",
                      {"per_claim_limit": claim_cap, "payable_amount": eligible,
                       "categories": sorted(groups)})
            return True
        ctx.trace(STAGE, "per_claim_limit", StepStatus.PASSED,
                  f"Aggregate payable ₹{eligible:,.0f} is within the per-claim "
                  f"limit of ₹{claim_cap:,.0f} (max of ₹{per_claim:,.0f} and the "
                  f"sub-limits of the categories present: "
                  f"{', '.join(sorted(groups))}).")

        # Consultation per-line sub-limit cap, within the consultation group
        # (generalizes the single-path rule to a multi-category claim).
        consult_terms = policy.category_terms("CONSULTATION")
        if consult_terms:
            for li in groups.get("CONSULTATION", []):
                if "consultation" in li.description.lower() and \
                        li.approved_amount > consult_terms.sub_limit:
                    capped = consult_terms.sub_limit
                    ctx.trace(STAGE, "consultation_fee_sub_limit", StepStatus.INFO,
                              f"Consultation fee ₹{li.approved_amount:,.0f} capped "
                              f"at the ₹{capped:,.0f} consultation sub-limit.")
                    li.approved_amount = capped
                    li.reason += (f" Capped at the ₹{capped:,.0f} consultation "
                                  "sub-limit.")

        annual = policy.coverage.annual_opd_limit
        if sub.ytd_claims_amount + eligible > annual:
            reject(
                ctx, RejectionReason.ANNUAL_LIMIT_EXCEEDED,
                f"This claim of ₹{eligible:,.0f} on top of "
                f"₹{sub.ytd_claims_amount:,.0f} already claimed this year exceeds "
                f"the annual OPD limit of ₹{annual:,.0f}.",
                read_fields={"amount"})
            ctx.trace(STAGE, "annual_opd_limit", StepStatus.FAILED,
                      f"YTD ₹{sub.ytd_claims_amount:,.0f} + eligible "
                      f"₹{eligible:,.0f} > annual limit ₹{annual:,.0f}.")
            return True
        ctx.trace(STAGE, "annual_opd_limit", StepStatus.PASSED,
                  f"YTD ₹{sub.ytd_claims_amount:,.0f} + eligible "
                  f"₹{eligible:,.0f} is within the annual OPD limit of "
                  f"₹{annual:,.0f}.")
        if self._family_limit(ctx, eligible):
            return True
        return False


class PharmacyDrugTypeRule(Rule):
    """PHARMACY only: a payable medicine line whose branded/generic status is
    missing or low-confidence is held (the per-line co-pay cannot be computed).
    The 12 carry no pharmacy claim that reaches here (TC002 halts at the gate)."""

    name = "pharmacy_drug_type"

    def __init__(self, config):
        self.config = config

    def apply(self, ctx: ClaimContext) -> bool:
        sub, policy = ctx.submission, ctx.policy
        if sub.claim_category.value != "PHARMACY":
            return False
        terms = policy.category_terms("PHARMACY")
        if not terms or not terms.branded_drug_copay_percent:
            return False
        threshold = self.config.confidence_threshold
        unknown: list[str] = []
        for li in ctx.result.line_items:
            if li.status != "APPROVED":
                continue
            dt = (li.drug_type or "").upper()
            if dt not in ("BRANDED", "GENERIC"):
                unknown.append(li.description)
            elif li.drug_type_confidence is not None \
                    and li.drug_type_confidence < threshold:
                unknown.append(li.description)
        if not unknown:
            return False
        route_to_review(
            ctx,
            "This pharmacy claim has medicine line(s) whose branded/generic "
            f"status could not be determined ({', '.join(unknown)}), so the "
            "co-pay cannot be computed automatically. A reviewer will confirm "
            "before payment.")
        ctx.trace(STAGE, "pharmacy_drug_type", StepStatus.FAILED,
                  f"{len(unknown)} pharmacy line(s) with unknown branded/generic "
                  "status; routed to MANUAL_REVIEW.")
        return True


# --- terminal steps (always run once the gates pass; return False) -----------
class ComputeAmountStep(Rule):
    """Money math: network discount FIRST, then co-pay, rounding at each step.
    Single-category folds the whole bill into one group; the derived
    multi-category path runs per group and aggregates. Preserved exactly — the
    asserted breakdown fields are byte-identical."""

    name = "compute_amount"

    def apply(self, ctx: ClaimContext) -> bool:
        groups = category_groups(ctx)
        if groups:
            self._compute_multi(ctx, groups)
        else:
            self._compute_single(ctx)
        return False

    def _compute_single(self, ctx: ClaimContext) -> None:
        sub, policy = ctx.submission, ctx.policy
        terms = policy.category_terms(sub.claim_category.value)

        eligible = sum(li.approved_amount for li in ctx.result.line_items) \
            if ctx.result.line_items else sub.claimed_amount

        breakdown = AmountBreakdown(
            claimed_amount=sub.claimed_amount,
            eligible_amount=eligible,
        )

        hospital = bill_hospital_name(ctx)
        in_network = policy.is_network_hospital(hospital)
        amount = eligible

        if in_network and terms and terms.network_discount_percent > 0:
            breakdown.network_discount_percent = terms.network_discount_percent
            breakdown.network_discount_amount = round_money(
                amount * terms.network_discount_percent / 100)
            amount = round_money(amount - breakdown.network_discount_amount)
            ctx.trace(
                STAGE, "network_discount", StepStatus.PASSED,
                f"'{hospital}' is a network hospital. Network discount "
                f"({terms.network_discount_percent:.0f}%) applied first on "
                f"₹{eligible:,.0f} = ₹{breakdown.network_discount_amount:,.0f} "
                f"discount, leaving ₹{amount:,.0f}.",
                {"order": "discount_before_copay"},
            )
        elif in_network:
            ctx.trace(STAGE, "network_discount", StepStatus.INFO,
                      f"'{hospital}' is a network hospital, but "
                      f"{sub.claim_category.value} has no network discount; "
                      "none applies.")
        elif hospital:
            ctx.trace(STAGE, "network_discount", StepStatus.INFO,
                      f"Hospital '{hospital}' is not a network hospital; no "
                      "network discount applies.")
        else:
            ctx.trace(STAGE, "network_discount", StepStatus.INFO,
                      "No hospital specified; no network discount applies.")
        breakdown.amount_after_discount = amount

        copay_pct = terms.copay_percent if terms else 0
        branded_pct = (terms.branded_drug_copay_percent or 0) if terms else 0
        is_pharmacy = sub.claim_category.value == "PHARMACY"
        if is_pharmacy and branded_pct and ctx.result.line_items:
            # Pharmacy per-line co-pay: branded lines pay branded_drug_copay_percent,
            # generic lines pay 0 (generic-mandatory is enforced via the branded
            # rate). Pharmacy has no network discount, so this is on the eligible
            # line amounts directly. Unknown drug types were already held for
            # review by PharmacyDrugTypeRule, so every line here is known.
            total_copay = 0.0
            branded_total = 0.0
            for li in ctx.result.line_items:
                if li.status == "APPROVED" and (li.drug_type or "").upper() == "BRANDED":
                    total_copay = round_money(
                        total_copay + li.approved_amount * branded_pct / 100)
                    branded_total = round_money(branded_total + li.approved_amount)
            breakdown.copay_percent = branded_pct
            breakdown.copay_amount = total_copay
            amount = round_money(amount - total_copay)
            ctx.trace(
                STAGE, "copay", StepStatus.PASSED,
                f"Pharmacy per-line co-pay: {branded_pct:.0f}% on branded lines "
                f"(₹{branded_total:,.0f}) = ₹{total_copay:,.0f} deducted; 0% on "
                f"generic lines; leaving ₹{amount:,.0f}.")
        elif copay_pct > 0:
            breakdown.copay_percent = copay_pct
            breakdown.copay_amount = round_money(amount * copay_pct / 100)
            amount = round_money(amount - breakdown.copay_amount)
            ctx.trace(
                STAGE, "copay", StepStatus.PASSED,
                f"Co-pay ({copay_pct:.0f}%) applied on "
                f"₹{breakdown.amount_after_discount:,.0f} = "
                f"₹{breakdown.copay_amount:,.0f} deducted, leaving ₹{amount:,.0f}.",
            )
        else:
            ctx.trace(STAGE, "copay", StepStatus.INFO,
                      f"No co-pay applies to {sub.claim_category.value}.")

        # Payout is floored at zero (defensive; the amount-sanity rule already
        # holds net-negative bills before this runs).
        amount = max(0.0, amount)
        breakdown.approved_amount = amount
        # The breakdown carries the one category group (additive; the top-level
        # fields above are computed and unchanged, so single-category stays
        # byte-identical on every asserted field).
        breakdown.category_breakdowns = [CategoryBreakdown(
            category=sub.claim_category.value,
            eligible=breakdown.eligible_amount,
            network_discount_percent=breakdown.network_discount_percent,
            network_discount_amount=breakdown.network_discount_amount,
            amount_after_discount=breakdown.amount_after_discount,
            copay_percent=breakdown.copay_percent,
            copay_amount=breakdown.copay_amount,
            approved=amount,
        )]
        ctx.result.amount_breakdown = breakdown
        ctx.result.approved_amount = amount

    def _compute_multi(self, ctx: ClaimContext, groups) -> None:
        sub, policy = ctx.submission, ctx.policy
        hospital = bill_hospital_name(ctx)
        in_network = policy.is_network_hospital(hospital)
        total_eligible = sum(li.approved_amount
                             for lines in groups.values() for li in lines)
        breakdown = AmountBreakdown(claimed_amount=sub.claimed_amount,
                                    eligible_amount=total_eligible)
        cat_breakdowns: list[CategoryBreakdown] = []
        # One rounded discount + co-pay per CATEGORY GROUP (never per line),
        # mirroring _compute_single applied to each group; the aggregate is the
        # sum of the per-group approved amounts (sum of 2-dp is exact).
        for cat in sorted(groups):
            terms = policy.category_terms(cat)
            g_eligible = sum(li.approved_amount for li in groups[cat])
            amount = g_eligible
            disc_pct = (terms.network_discount_percent
                        if terms and in_network else 0)
            disc_amt = 0.0
            if disc_pct > 0:
                disc_amt = round_money(amount * disc_pct / 100)
                amount = round_money(amount - disc_amt)
            after = amount
            copay_pct = terms.copay_percent if terms else 0
            copay_amt = 0.0
            if copay_pct > 0:
                copay_amt = round_money(amount * copay_pct / 100)
                amount = round_money(amount - copay_amt)
            amount = max(0.0, amount)
            cat_breakdowns.append(CategoryBreakdown(
                category=cat, eligible=g_eligible,
                network_discount_percent=disc_pct,
                network_discount_amount=disc_amt, amount_after_discount=after,
                copay_percent=copay_pct, copay_amount=copay_amt, approved=amount))
            ctx.trace(
                STAGE, f"category_money:{cat}", StepStatus.PASSED,
                f"{cat}: eligible ₹{g_eligible:,.0f}"
                + (f"; network discount {disc_pct:.0f}% (−₹{disc_amt:,.0f})"
                   if disc_pct else "; no network discount")
                + (f"; co-pay {copay_pct:.0f}% (−₹{copay_amt:,.0f})"
                   if copay_pct else "; no co-pay")
                + f" → ₹{amount:,.0f}.",
                {"category": cat, "approved": amount})

        # Top-level fields hold the aggregates across the category groups.
        breakdown.category_breakdowns = cat_breakdowns
        breakdown.network_discount_amount = round_money(
            sum(c.network_discount_amount for c in cat_breakdowns))
        breakdown.amount_after_discount = round_money(
            sum(c.amount_after_discount for c in cat_breakdowns))
        breakdown.copay_amount = round_money(
            sum(c.copay_amount for c in cat_breakdowns))
        total_approved = round_money(sum(c.approved for c in cat_breakdowns))
        breakdown.approved_amount = total_approved
        ctx.result.amount_breakdown = breakdown
        ctx.result.approved_amount = total_approved
        ctx.trace(
            STAGE, "amount_aggregate", StepStatus.PASSED,
            "Per-category total: " + " + ".join(
                f"{c.category} ₹{c.approved:,.0f}" for c in cat_breakdowns)
            + f" = ₹{total_approved:,.0f}.",
            {"approved_amount": total_approved})


class FraudSignalsStep(Rule):
    """Binary fraud gates + a transparent weighted fraud score. Individually
    contained: a failure (incl. the `simulate_component_failure` hook) records a
    ComponentFailure + DEGRADED trace and recommends manual review while the
    decision still finalizes — preserved exactly from the old run() wrapper."""

    name = "fraud_signals"

    def apply(self, ctx: ClaimContext) -> bool:
        try:
            self._check(ctx)
        except Exception as e:
            # Component-level graceful degradation: record the failure,
            # continue to a decision, recommend manual review.
            ctx.result.component_failures.append(ComponentFailure(
                component="adjudication.fraud_signals",
                error=str(e),
                impact="Fraud-signal analysis was skipped. The claim history could "
                       "not be checked for same-day, monthly, or high-value "
                       "patterns.",
            ))
            ctx.result.manual_review_recommended = True
            ctx.trace(STAGE, "fraud_signals", StepStatus.DEGRADED,
                      f"Fraud-signal component failed and was skipped: {e}. "
                      "Pipeline continued; manual review recommended because "
                      "processing was incomplete.")
        return False

    def _fraud_score_breakdown(self, ctx: ClaimContext) -> tuple[float, list[dict]]:
        """Transparent weighted fraud score in [0, 1] and its per-signal
        breakdown (same-day frequency, monthly frequency, amount-vs-history,
        near-duplicate). Each row records the raw value, the 0..1 sub-score, its
        weight, and its contribution, so the trace shows exactly what drove the
        number. A SECONDARY graduated view on top of the binary gates."""
        sub = ctx.submission
        ft = ctx.policy.fraud_thresholds
        history = sub.claims_history

        def clamp(x: float) -> float:
            return 0.0 if x < 0 else 1.0 if x > 1 else x

        # 1) Same-day frequency.
        same_day = [c for c in history if c.date == sub.treatment_date]
        total_same_day = len(same_day) + 1
        s_sd = clamp((total_same_day - 1) / max(1, ft.same_day_claims_limit))

        # 2) Monthly frequency.
        same_month = [
            c for c in history
            if c.date.year == sub.treatment_date.year
            and c.date.month == sub.treatment_date.month
        ]
        total_month = len(same_month) + 1
        s_mo = clamp((total_month - 1) / max(1, ft.monthly_claims_limit))

        # 3) Amount vs the member's OWN recent average (4x saturates; no history
        #    -> 0, the absolute high-value gate covers raw size separately).
        amounts = [c.amount for c in history if c.amount]
        avg = sum(amounts) / len(amounts) if amounts else 0.0
        if avg > 0 and sub.claimed_amount:
            s_amt = clamp((sub.claimed_amount / avg - 1.0) / 3.0)
        else:
            s_amt = 0.0

        # 4) Near-duplicate (same amount within 2% on the same/near date; a
        #    byte-identical resubmission via duplicate_of saturates).
        if sub.duplicate_of:
            s_dup = 1.0
        else:
            s_dup = 0.0
            for c in history:
                if not c.amount or not sub.claimed_amount:
                    continue
                close = (abs(c.amount - sub.claimed_amount)
                         <= _NEAR_DUPLICATE_AMOUNT_BAND * sub.claimed_amount)
                if close and c.date == sub.treatment_date:
                    s_dup = 1.0
                    break
                if close and abs((c.date - sub.treatment_date).days) \
                        <= _NEAR_DUPLICATE_DAY_WINDOW:
                    s_dup = max(s_dup, 0.6)

        rows = [
            {"signal": "same_day_frequency", "sub_score": s_sd,
             "detail": f"{total_same_day} claim(s) on {sub.treatment_date} "
                       f"(limit {ft.same_day_claims_limit})"},
            {"signal": "monthly_frequency", "sub_score": s_mo,
             "detail": f"{total_month} claim(s) in "
                       f"{sub.treatment_date.strftime('%B %Y')} "
                       f"(limit {ft.monthly_claims_limit})"},
            {"signal": "amount_vs_history", "sub_score": s_amt,
             "detail": (f"₹{sub.claimed_amount:,.0f} vs member avg "
                        f"₹{avg:,.0f}" if avg > 0
                        else "no prior claim history to compare")},
            {"signal": "near_duplicate", "sub_score": s_dup,
             "detail": ("byte-identical to a decided claim" if sub.duplicate_of
                        else "same-date, same-amount prior claim" if s_dup >= 1.0
                        else "near-date, same-amount prior claim" if s_dup > 0
                        else "no near-duplicate in history")},
        ]
        for r in rows:
            r["weight"] = FRAUD_WEIGHTS[r["signal"]]
            r["contribution"] = round(r["weight"] * r["sub_score"], 4)
        score = clamp(sum(r["contribution"] for r in rows))
        return score, rows

    def _check(self, ctx: ClaimContext) -> None:
        if ctx.submission.simulate_component_failure:
            raise AgentError(
                "SIMULATED_FAILURE",
                "Fraud-signal analysis component failed (simulated failure flag).",
            )

        sub, policy = ctx.submission, ctx.policy
        ft = policy.fraud_thresholds
        signals: list[str] = []

        # H3: byte-identical duplicate of an already-decided claim (set by the
        # API from a full-document-set hash).
        if sub.duplicate_of:
            signals.append(
                f"This claim's documents are byte-identical to already-decided "
                f"claim {sub.duplicate_of}; it may be a duplicate submission."
            )

        same_day = [c for c in sub.claims_history if c.date == sub.treatment_date]
        total_same_day = len(same_day) + 1  # including this claim
        if total_same_day > ft.same_day_claims_limit:
            signals.append(
                f"This is claim number {total_same_day} from member {sub.member_id} "
                f"on {sub.treatment_date}, above the same-day limit of "
                f"{ft.same_day_claims_limit}. Prior same-day claims: "
                + ", ".join(f"{c.claim_id} (₹{c.amount:,.0f}, {c.provider})"
                            for c in same_day) + "."
            )

        same_month = [
            c for c in sub.claims_history
            if c.date.year == sub.treatment_date.year
            and c.date.month == sub.treatment_date.month
        ]
        total_month = len(same_month) + 1
        if total_month > ft.monthly_claims_limit:
            signals.append(
                f"{total_month} claims in {sub.treatment_date.strftime('%B %Y')}, "
                f"above the monthly limit of {ft.monthly_claims_limit}."
            )

        if sub.claimed_amount > ft.high_value_claim_threshold:
            signals.append(
                f"Claimed amount ₹{sub.claimed_amount:,.0f} exceeds the high-value "
                f"threshold of ₹{ft.high_value_claim_threshold:,.0f}."
            )

        # Transparent weighted fraud score: a SECONDARY graduated view on top of
        # the binary gates above. Always computed and traced; adds a routing
        # signal only when it crosses the policy threshold.
        score, rows = self._fraud_score_breakdown(ctx)
        ctx.result.fraud_score = round(score, 4)
        score_threshold = ft.fraud_score_manual_review_threshold
        breakdown_txt = "; ".join(
            f"{r['signal']} {r['sub_score']:.2f}×{r['weight']:.2f}"
            f"={r['contribution']:.3f} [{r['detail']}]" for r in rows)
        ctx.trace(STAGE, "fraud_score",
                  StepStatus.FAILED if score >= score_threshold else StepStatus.INFO,
                  f"Weighted fraud score {score:.2f} "
                  f"(review threshold {score_threshold:.2f}) = {breakdown_txt}.",
                  {"fraud_score": round(score, 4), "threshold": score_threshold,
                   "breakdown": rows})
        if score >= score_threshold:
            signals.append(
                f"Weighted fraud score {score:.2f} is at or above the review "
                f"threshold of {score_threshold:.2f} (contributing signals — "
                f"{breakdown_txt})."
            )

        if signals:
            ctx.result.fraud_signals.extend(signals)
            ctx.trace(STAGE, "fraud_signals", StepStatus.FAILED,
                      "Fraud signals detected: " + " | ".join(signals),
                      {"signal_count": len(signals)})
        else:
            ctx.trace(STAGE, "fraud_signals", StepStatus.PASSED,
                      "No fraud signals: same-day, monthly, and high-value "
                      "thresholds all clear.")


class FinalizeStep(Rule):
    """Decide APPROVED/PARTIAL from the line items, record the deciding fields
    (Fix 7), and route a payable claim with fraud signals to MANUAL_REVIEW."""

    name = "finalize"

    def apply(self, ctx: ClaimContext) -> bool:
        if ctx.result.decision == Decision.REJECTED:
            return False

        # Fix 7: an APPROVED/PARTIAL decision rests on every field that gated it —
        # the amounts (payout/limits), the diagnosis (waiting/exclusion passed),
        # the hospital (network discount), the dates, the patient, the category.
        # A misread in any could flip the outcome, so all are deciding fields.
        ctx.deciding_fields = {"amount", "diagnosis", "hospital",
                               "treatment_date", "patient_name", "category"}

        rejected_items = [li for li in ctx.result.line_items if li.status == "REJECTED"]
        approved_items = [li for li in ctx.result.line_items if li.status == "APPROVED"]

        if rejected_items and approved_items:
            ctx.result.decision = Decision.PARTIAL
            ctx.result.reasons.append(
                f"{len(approved_items)} item(s) approved for "
                f"₹{ctx.result.approved_amount:,.0f}; {len(rejected_items)} item(s) "
                "rejected. See line items for per-item reasons."
            )
        else:
            ctx.result.decision = Decision.APPROVED
            ctx.result.reasons.append(
                f"Claim approved for ₹{ctx.result.approved_amount:,.0f}."
            )

        # Fraud signals on an otherwise-payable claim route to manual review.
        if ctx.result.fraud_signals:
            ctx.result.decision = Decision.MANUAL_REVIEW
            ctx.result.manual_review_recommended = True
            ctx.result.reasons.append(
                "Routed to manual review (not auto-rejected) because of the fraud "
                "signals listed; a human reviewer must verify before payment."
            )
            ctx.trace(STAGE, "decision_routing", StepStatus.INFO,
                      "Otherwise-payable claim routed to MANUAL_REVIEW due to "
                      "fraud signals.")
        return False


# --- domain holds relocated from the orchestrator (terminal steps) -----------
# Item 2: these two checks were business decisions the orchestrator applied
# AFTER the agent pipeline. They are now terminal steps so the orchestrator is
# pure coordination. Both ALWAYS run and only SOMETIMES override (they return
# False like the other terminal steps; the terminal runner ignores the return).
# Order and precedence are preserved exactly: the identity name hold runs before
# the confidence gate, and both run after FinalizeStep (which sets the
# ctx.deciding_fields the gate reads). The ctx.trace stage labels ("orchestrator"
# and "confidence_gate") are kept verbatim so the emitted trace is byte-identical
# to when this logic lived in the orchestrator.
class IdentityNameHoldStep(Rule):
    """No-patient-name identity hold (moved verbatim from the orchestrator).

    A payable claim whose documents carried no patient name could not be
    identity-checked by name. On the live-upload path, at or above
    EngineConfig.no_name_high_value the otherwise-payable claim is held for a
    human; below it, it passes with an advisory trace note. Structured content
    (the known/official cases) provides fields directly rather than reading them,
    so a missing name there is a fixture choice, not an unread identity, and is
    never held on this basis. Genuine inconsistencies (a different covered
    person, an off-roster patient) are caught earlier by the consistency agent,
    which halts before adjudication ever runs."""

    name = "identity_advisory"

    def __init__(self, config):
        self.config = config

    def apply(self, ctx: ClaimContext) -> bool:
        if not ctx.halted and ctx.no_patient_names and \
                ctx.result.decision in (Decision.APPROVED, Decision.PARTIAL):
            # Live-upload path only: a real uploaded claim whose name could not
            # be read is unverified identity. Below the high-value threshold it
            # passes with an advisory; at or above it, it is held. Structured
            # content (the known/official cases) provides fields directly rather
            # than reading them, so a missing name there is a fixture choice,
            # not an unread identity, and is never held on this basis.
            is_live_upload = any(d.file_data for d in ctx.submission.documents)
            amount = ctx.submission.claimed_amount or ctx.result.approved_amount or 0
            threshold = self.config.no_name_high_value
            if is_live_upload and amount >= threshold:
                original = ctx.result.decision
                ctx.result.manual_review_recommended = True
                ctx.result.reasons.append(
                    f"Computed decision was {original.value} "
                    f"(₹{(ctx.result.approved_amount or 0):,.0f}); patient name "
                    f"not verified on a high-value claim (₹{amount:,.0f} ≥ "
                    f"₹{threshold:,.0f}). Held for manual "
                    "review to confirm the patient's identity before payment."
                )
                ctx.result.decision = Decision.MANUAL_REVIEW
                ctx.trace("orchestrator", "identity_advisory", StepStatus.FAILED,
                          f"No patient name on any document and claim amount "
                          f"₹{amount:,.0f} is at or above the "
                          f"₹{threshold:,.0f} high-value "
                          "threshold; otherwise-payable claim routed to "
                          "MANUAL_REVIEW (identity unverified on a high-value "
                          "claim).")
            else:
                ctx.trace("orchestrator", "identity_advisory", StepStatus.INFO,
                          "No patient name could be read from any document; "
                          "identity could not be verified by name. The filing "
                          "member and policy are otherwise consistent and the "
                          "amount is below the high-value threshold, so the claim "
                          "is not routed to review on this basis alone (advisory "
                          "only).")
        return False


# Fix 7: per-field confidence lookup for the confidence gate. Maps a logical
# deciding field (recorded by the deciding check in ctx.deciding_fields) to
# (does a document carry it?, the per-field confidence key, or None). When the
# per-field confidence is absent (key is None, or the document did not report
# it), the gate falls back to the document-level extraction confidence of the
# document(s) that carry the field — so a generally low-confidence read still
# holds even when no per-field confidence exists.
_FIELD_CONFIDENCE = {
    "amount": (lambda f: f.get("total") is not None or bool(f.get("line_items")),
               "amount_confidence"),
    # The diagnosis field has NO per-field READ-confidence key (None) on purpose.
    # The extraction schema reports no diagnosis-read score, and the only
    # diagnosis-adjacent score it does report, `canonical_condition_confidence`,
    # measures something different: the model's confidence MAPPING the diagnosis
    # onto one of the policy's waiting/exclusion conditions (diabetes,
    # hypertension, ...), NOT how confidently the diagnosis text was read. Most
    # real diagnoses (viral fever, bronchitis, gastroenteritis, ...) map to no
    # policy condition, so the model honestly returns a low mapping confidence
    # (~0.10) on a perfectly clean, high-confidence read. Keying the gate on it
    # therefore held essentially every live claim carrying a diagnosis for review
    # — the live path could never cleanly approve. A None key makes
    # _field_min_confidence fall back to the document-level extraction confidence,
    # which is the genuine read signal (and still counts an UNREADABLE carrier as
    # 0). Canonical-mapping uncertainty that ACTUALLY drives a decision is caught
    # separately and earlier by DiagnosisCertaintyRule, which routes to review
    # when a decision-critical canonical mapping is below threshold — so nothing
    # is lost by dropping the key here.
    "diagnosis": (lambda f: bool(f.get("primary_diagnosis") or f.get("diagnosis")),
                  None),
    "treatment_date": (lambda f: bool(f.get("date")), "treatment_date_confidence"),
    "patient_name": (lambda f: bool(f.get("patient_name")), "patient_name_confidence"),
    "hospital": (lambda f: bool(f.get("hospital_name")), "hospital_confidence"),
    "category": (lambda f: True, "category_confidence"),
}


class ConfidenceGateStep(Rule):
    """Fix 7 / C1 + M2: hold the computed claim for a human only when a field the
    DECIDING check actually read was a low-confidence (or unreadable) read — not
    because some unrelated field was fuzzy (moved verbatim from the orchestrator).
    The deciding check records the fields it read in ctx.deciding_fields; the gate
    checks the per-field confidence of exactly those.

    Holds APPROVED, PARTIAL, and REJECTED alike — a rejection built on a misread
    is as wrong as a misread payment. The computed decision and amount stay in
    the output for the reviewer; only the final status flips to MANUAL_REVIEW.
    Must run AFTER FinalizeStep, which sets ctx.deciding_fields. Acts on the live
    path only: structured/test content carries no per-field or document
    confidence, so nothing triggers and the 12 official cases stay byte-identical
    (outcome and trace)."""

    name = "confidence_gate"

    def __init__(self, config):
        self.config = config

    def _field_min_confidence(self, ctx: ClaimContext, presence, conf_key):
        """Minimum effective confidence for one logical field across the docs
        that carry it: the per-field confidence (conf_key) when the field has one
        and the document reported it, else the document-level extraction
        confidence; an UNREADABLE carrier counts as 0. A None conf_key means the
        field has no per-field READ-confidence score (e.g. diagnosis), so the
        document-level extraction confidence is used directly. Returns None when
        no carrying document has any confidence signal — so the 12 structured
        cases (no confidence at all) never trigger a hold."""
        vals: list[float] = []
        for entry in ctx.extracted_documents:
            fields = entry.get("fields", {})
            if not presence(fields):
                continue
            if str(fields.get("readability") or "").upper() == "UNREADABLE":
                vals.append(0.0)
                continue
            per = fields.get(conf_key) if conf_key is not None else None
            doc = fields.get("_extraction_confidence")
            eff = per if per is not None else doc
            if eff is not None:
                vals.append(float(eff))
        return min(vals) if vals else None

    def apply(self, ctx: ClaimContext) -> bool:
        if ctx.result.decision not in (
                Decision.APPROVED, Decision.PARTIAL, Decision.REJECTED):
            return False

        threshold = self.config.confidence_threshold
        reasons: list[str] = []
        for fieldname in sorted(ctx.deciding_fields):
            spec = _FIELD_CONFIDENCE.get(fieldname)
            if spec is None:
                continue
            presence, conf_key = spec
            conf = self._field_min_confidence(ctx, presence, conf_key)
            if conf is not None and conf < threshold:
                reasons.append(
                    f"{fieldname}-field confidence {conf:.2f} is below the "
                    f"{threshold:.2f} threshold")

        if not reasons:
            ctx.trace("confidence_gate", "passed", StepStatus.PASSED,
                      "All decision-critical documents were read with sufficient "
                      "confidence; no confidence hold applied.")
            return False

        original = ctx.result.decision
        ctx.result.manual_review_recommended = True
        ctx.result.reasons.append(
            f"Computed decision was {original.value} "
            f"(₹{(ctx.result.approved_amount or 0):,.0f}), but a field the "
            f"deciding check read was an uncertain read ({'; '.join(reasons)}). "
            "Held for manual review: a human must verify the read before the "
            "claim is finalized."
        )
        ctx.result.decision = Decision.MANUAL_REVIEW
        ctx.trace("confidence_gate", "hold", StepStatus.FAILED,
                  f"Otherwise-{original.value} claim held for MANUAL_REVIEW: a "
                  f"deciding field was an uncertain read ({'; '.join(reasons)}). "
                  "The computed decision and amount remain in the output for the "
                  "reviewer.",
                  {"computed_decision": original.value, "reasons": reasons,
                   "deciding_fields": sorted(ctx.deciding_fields)})
        return False


# --- factories: the check order as data --------------------------------------
def build_gates(config) -> list[Rule]:
    """The resolving gates, in order. The first to return True stops the run."""
    return [
        DocumentAlterationRule(),
        AmountSanityRule(config),
        ClaimedAmountMismatchRule(config),
        EligibilityRule(),
        SubmissionDeadlineRule(),
        DiagnosisCertaintyRule(config),
        ExclusionsRule(config),
        WaitingPeriodsRule(config),
        PreAuthorizationRule(),
        PractitionerRegistrationRule(),
        SessionCapRule(),
        HighValueRule(),
        LimitsRule(),
        PharmacyDrugTypeRule(config),
    ]


def build_terminal() -> list[Rule]:
    """The money sequence run once the gates pass WITHOUT resolving the claim:
    money math, the individually-contained fraud check, then finalize."""
    return [ComputeAmountStep(), FraudSignalsStep(), FinalizeStep()]


def build_terminal_holds(config) -> list[Rule]:
    """The two domain holds relocated from the orchestrator (Item 2): the
    identity name hold, then the per-field confidence gate, in that order.

    Unlike build_terminal (skipped when a gate resolves the claim), these ALWAYS
    run after the engine reaches a decision — whether a gate resolved it (e.g. a
    REJECTED waiting-period/pre-auth/limit) or the terminal sequence produced it
    (APPROVED/PARTIAL/MANUAL_REVIEW). That matches where they ran before: in the
    orchestrator, after the whole adjudication agent, regardless of how it
    decided — so a gate-rejected claim is still confidence-checked. The
    confidence gate must run after FinalizeStep (it reads ctx.deciding_fields,
    which FinalizeStep — or a gate's reject() — sets), and after the identity
    hold (precedence preserved)."""
    return [IdentityNameHoldStep(config), ConfidenceGateStep(config)]
