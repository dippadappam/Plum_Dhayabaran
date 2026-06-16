"""Claim derivation agent.

Runs after extraction and the document gate, before adjudication. On the
real-upload path the member does not type the amount, hospital, or treatment
date; those come from the documents. This agent reads the extracted fields and
populates the effective submission values so the deterministic adjudication
engine (unchanged) can run exactly as it does on the structured/test path.

No-op when `claimed_amount` was already provided (structured/test path):
nothing is derived and no trace is added.

If the real-upload path yields no determinable bill total (or no treatment
date), the claim is routed to MANUAL_REVIEW rather than approved, and the
pipeline halts before adjudication. No decision rule, limit, or money math is
changed here.
"""

import re
from datetime import date, datetime

from app.agents.base import Agent, ClaimContext
from app.agents.shared_checks import check_minimum_amount
from app.models.decision import Decision, StepStatus

STAGE = "derivation"

# Unambiguous date formats (year-first ISO, or a named month). Numeric
# day/month/year strings are handled separately with ambiguity detection.
_UNAMBIGUOUS_DATE_FORMATS = [
    "%Y-%m-%d", "%Y/%m/%d",
    "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y",
    "%d %b, %Y", "%b %d, %Y", "%B %d, %Y",
]


def _safe_date(year: int, month: int, day: int):
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_flexible_date(raw) -> "tuple[date | None, bool]":
    """Parse an extracted date string into (date | None, ambiguous).

    ISO and named-month formats parse directly and are unambiguous. A purely
    numeric D-M-Y / M-D-Y string is read under BOTH the Indian (DD-MM) and US
    (MM-DD) conventions: if both are valid dates and they differ, the string is
    genuinely ambiguous (returns (None, True)) and the caller routes to a human
    rather than guessing; otherwise the single valid reading is used (DD-MM
    preferred, matching Indian convention and the eval's date parser)."""
    s = str(raw).strip()
    # ISO date, or the date prefix of an ISO datetime (zero-padded).
    try:
        return date.fromisoformat(s[:10]), False
    except ValueError:
        pass
    for fmt in _UNAMBIGUOUS_DATE_FORMATS:
        try:
            return datetime.strptime(s, fmt).date(), False
        except ValueError:
            continue
    m = re.match(r"^(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})$", s)
    if m:
        a, b, year = int(m.group(1)), int(m.group(2)), int(m.group(3))
        dd_mm = _safe_date(year, b, a)   # a = day,   b = month  (DD-MM-YYYY)
        mm_dd = _safe_date(year, a, b)   # a = month, b = day    (MM-DD-YYYY)
        if dd_mm and mm_dd and dd_mm != mm_dd:
            return None, True            # genuinely ambiguous — do not guess
        return (dd_mm or mm_dd), False
    return None, False


class ClaimDerivationAgent(Agent):
    name = STAGE

    def run(self, ctx: ClaimContext) -> None:
        sub = ctx.submission

        # Structured/test path: values already provided, nothing to derive.
        if sub.claimed_amount is not None:
            return

        bill_total = self._bill_total(ctx)
        derived_date, date_ambiguous = self._derived_date(ctx)

        if date_ambiguous:
            ctx.result.decision = Decision.MANUAL_REVIEW
            ctx.result.manual_review_recommended = True
            ctx.result.reasons.append(
                "The treatment date on the uploaded documents is ambiguous (it "
                "could be read as either DD-MM-YYYY or MM-DD-YYYY). A reviewer "
                "must confirm the date before the claim can be adjudicated."
            )
            ctx.trace(STAGE, "treatment_date", StepStatus.FAILED,
                      "Ambiguous numeric treatment date (DD-MM vs MM-DD); routed "
                      "to MANUAL_REVIEW rather than guessing an interpretation.")
            ctx.halted = True
            return

        if bill_total is None or derived_date is None:
            missing = []
            if bill_total is None:
                missing.append("claim amount")
            if derived_date is None:
                missing.append("treatment date")
            ctx.result.decision = Decision.MANUAL_REVIEW
            ctx.result.manual_review_recommended = True
            ctx.result.reasons.append(
                "Could not determine the "
                + " and ".join(missing)
                + " from the uploaded documents; routed to manual review for a "
                "human to read the documents and complete the claim."
            )
            ctx.trace(
                STAGE, "effective_values", StepStatus.FAILED,
                "No "
                + " and no ".join(missing)
                + " could be read from the extracted documents. Routed to "
                "manual review instead of auto-approving.",
            )
            ctx.halted = True
            return

        sub.claimed_amount = bill_total
        if not sub.hospital_name:
            sub.hospital_name = self._derived_hospital(ctx)
        if sub.treatment_date is None:
            sub.treatment_date = derived_date

        ctx.trace(
            STAGE, "effective_values", StepStatus.PASSED,
            f"Derived claim amount ₹{bill_total:,.0f} from the extracted bill"
            + (f", treatment date {sub.treatment_date}" if sub.treatment_date else "")
            + (f", hospital '{sub.hospital_name}'" if sub.hospital_name else "")
            + ". These values were read from the documents, not entered by the "
            "member.",
            {"derived_claimed_amount": bill_total,
             "derived_treatment_date": str(sub.treatment_date),
             "derived_hospital_name": sub.hospital_name},
        )

        # Intake deferred the minimum-amount check because the amount was not
        # yet known; it must run now against the derived amount. The rule lives
        # in shared_checks.check_minimum_amount (shared with intake.py).
        if check_minimum_amount(ctx, amount=bill_total, stage=STAGE,
                                derived=True):
            return

    # ------------------------------------------------------------------
    @staticmethod
    def _item_total(fields: dict) -> float | None:
        total = fields.get("total")
        if total is not None:
            try:
                return float(total)
            except (TypeError, ValueError):
                return None
        line_items = fields.get("line_items") or []
        if line_items:
            return sum(float(li.get("amount", 0) or 0) for li in line_items)
        return None

    def _bill_total(self, ctx: ClaimContext) -> float | None:
        """Sum the totals of ALL bill documents (multi-bill aggregation), so the
        derived claimed amount matches how adjudication aggregates line items
        across bills (Batch 6a) and what the accumulator records — not just the
        first bill. Falls back to any single doc carrying a total when no
        bill-typed document does."""
        bill_totals: list[float] = []
        for doc in ctx.extracted_documents:
            fields = doc.get("fields", {})
            dtype = str(fields.get("document_type") or doc.get("type") or "").upper()
            if "BILL" in dtype:
                total = self._item_total(fields)
                if total is not None:
                    bill_totals.append(total)
        if bill_totals:
            return sum(bill_totals)
        for doc in ctx.extracted_documents:
            total = self._item_total(doc.get("fields", {}))
            if total is not None:
                return total
        return None

    @staticmethod
    def _derived_hospital(ctx: ClaimContext) -> str | None:
        for doc in ctx.extracted_documents:
            name = doc.get("fields", {}).get("hospital_name")
            if name:
                return str(name)
        return None

    @staticmethod
    def _derived_date(ctx: ClaimContext) -> "tuple[date | None, bool]":
        """(treatment_date | None, ambiguous). A provided date wins and is
        never ambiguous; otherwise the first parseable extracted date is used,
        and a genuinely DD-MM/MM-DD-ambiguous string short-circuits to ambiguous
        so the caller routes to review rather than guessing."""
        if ctx.submission.treatment_date is not None:
            return ctx.submission.treatment_date, False
        for doc in ctx.extracted_documents:
            raw = doc.get("fields", {}).get("date")
            if raw:
                parsed, ambiguous = parse_flexible_date(raw)
                if ambiguous:
                    return None, True
                if parsed is not None:
                    return parsed, False
        return None, False
