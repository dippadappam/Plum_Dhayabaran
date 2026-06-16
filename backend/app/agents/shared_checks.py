"""Shared intake-time validation rules (single source of truth per rule).

The minimum-claim-amount and category-coverage (NOT_COVERED) checks each run on
two paths: the provided-input path (IntakeAgent, when the member typed the value)
and the derived-input path (ClaimDerivationAgent / CategoryResolutionAgent, when
the value is read from the documents after extraction). The two paths share one
rule — the threshold/condition, the reason code, the REJECTED decision, and the
halt — and previously kept a second copy each in derivation.py /
category_resolution.py, which could drift from the intake.py copy.

Each rule now lives here once. The paths differ only in wording, the trace stage,
and (for the minimum check) whether `approved_amount` is zeroed; those per-path
specifics are selected by the `derived` flag and the `stage` argument, and the
wording/trace was copied verbatim from the original inline copies. Each helper
mutates `ctx.result` (and `ctx.halted`) and returns True when it rejected the
claim, so the caller can stop.
"""

from app.agents.base import ClaimContext
from app.models.decision import Decision, RejectionReason, StepStatus


def check_minimum_amount(ctx: ClaimContext, *, amount: float, stage: str,
                         derived: bool) -> bool:
    """Reject when `amount` is below the policy minimum claim amount.

    derived=False is the provided-input path (IntakeAgent); derived=True is the
    derived-input path (ClaimDerivationAgent), which additionally zeroes
    approved_amount. Returns True on rejection (REJECTED + BELOW_MINIMUM_AMOUNT +
    halt), False when the amount meets the minimum.
    """
    min_amount = ctx.policy.submission_rules.minimum_claim_amount
    if amount < min_amount:
        ctx.result.decision = Decision.REJECTED
        ctx.result.rejection_reasons.append(RejectionReason.BELOW_MINIMUM_AMOUNT)
        if derived:
            ctx.result.reasons.append(
                f"The amount ₹{amount:,.0f} derived from the bill is below "
                f"the minimum claim amount of ₹{min_amount:,.0f}."
            )
            ctx.result.approved_amount = 0
            ctx.trace(stage, "minimum_amount", StepStatus.FAILED,
                      f"Derived amount ₹{amount:,.0f} < minimum "
                      f"₹{min_amount:,.0f} (check was deferred at intake).")
        else:
            ctx.result.reasons.append(
                f"Claimed amount ₹{amount:,.0f} is below the minimum "
                f"claim amount of ₹{min_amount:,.0f}."
            )
            ctx.trace(stage, "minimum_amount", StepStatus.FAILED,
                      f"₹{amount:,.0f} < minimum ₹{min_amount:,.0f}.")
        ctx.halted = True
        return True
    if derived:
        ctx.trace(stage, "minimum_amount", StepStatus.PASSED,
                  f"Derived amount ₹{amount:,.0f} meets the "
                  f"₹{min_amount:,.0f} minimum (check deferred from intake).")
    else:
        ctx.trace(stage, "minimum_amount", StepStatus.PASSED,
                  f"₹{amount:,.0f} meets the ₹{min_amount:,.0f} minimum.")
    return False


def check_coverage(ctx: ClaimContext, *, category: str, stage: str,
                   derived: bool) -> bool:
    """Reject when `category` is not a covered policy category.

    derived=False is the provided-input path (IntakeAgent); derived=True is the
    derived-input path (CategoryResolutionAgent). Returns True on rejection
    (REJECTED + NOT_COVERED + halt), False when the category is covered.
    """
    terms = ctx.policy.category_terms(category)
    if terms is None or not terms.covered:
        ctx.result.decision = Decision.REJECTED
        ctx.result.rejection_reasons.append(RejectionReason.NOT_COVERED)
        if derived:
            ctx.result.reasons.append(
                f"The derived category '{category}' is not covered under this "
                f"policy."
            )
            ctx.trace(stage, "category_covered", StepStatus.FAILED,
                      f"Derived category '{category}' is not covered "
                      "(check was deferred at intake).")
        else:
            ctx.result.reasons.append(
                f"Category '{category}' is not covered under this policy."
            )
            ctx.trace(stage, "category_covered", StepStatus.FAILED,
                      f"Category '{category}' is not covered.")
        ctx.halted = True
        return True
    if derived:
        ctx.trace(stage, "category_covered", StepStatus.PASSED,
                  f"Derived category '{category}' is covered "
                  "(check deferred from intake).")
    else:
        ctx.trace(stage, "category_covered", StepStatus.PASSED,
                  f"Category '{category}' is covered.")
    return False
