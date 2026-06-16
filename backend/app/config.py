"""Engine configuration: system-level operational knobs (not per-policy
business rules). Defaults equal the values previously hardcoded in code, so a
default EngineConfig() changes no behavior.

Policy rules — the same-day limit and the pre-auth thresholds — deliberately do
NOT live here; they stay in policy_terms.json (fraud_thresholds.same_day_claims_limit,
opd_categories.*.pre_auth_threshold).
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class EngineConfig:
    # C1 fail-closed gate: a decision-critical document below this extraction
    # confidence holds a payable claim for a human.
    confidence_threshold: float = 0.8
    # Identity hold: a live no-name claim at or above this amount is held.
    no_name_high_value: float = 2500.0
    # H4 reconciliation: exact-match rupee slack and the acceptable
    # total-vs-itemized divergence band (normal discounts/taxes/fees).
    reconcile_tolerance: float = 1.0
    reconcile_band: float = 0.20
    # Claimed-vs-bill cross-check: the member's stated amount and the documented
    # bill total may diverge by up to this fraction before the claim routes to
    # review (loose enough for rounding/small discounts; the 12 official cases
    # have claimed == bill, so it never fires for them).
    claimed_amount_band: float = 0.20
    # Absurd-amount bound. None -> use the policy's sum_insured_per_employee.
    amount_ceiling: Optional[float] = None
    # Per-claim AI cost cap: max paid extraction calls before the claim is held.
    max_extraction_calls_per_claim: int = 10

    @classmethod
    def from_env(cls) -> "EngineConfig":
        """Build from env vars, falling back to the defaults above."""
        def _f(name: str, default: float) -> float:
            v = os.environ.get(name)
            return float(v) if v not in (None, "") else default

        def _opt_f(name: str) -> Optional[float]:
            v = os.environ.get(name)
            return float(v) if v not in (None, "") else None

        def _i(name: str, default: int) -> int:
            v = os.environ.get(name)
            return int(v) if v not in (None, "") else default

        return cls(
            confidence_threshold=_f("CLAIMS_CONFIDENCE_THRESHOLD", 0.8),
            no_name_high_value=_f("CLAIMS_NO_NAME_HIGH_VALUE", 2500.0),
            reconcile_tolerance=_f("CLAIMS_RECONCILE_TOLERANCE", 1.0),
            reconcile_band=_f("CLAIMS_RECONCILE_BAND", 0.20),
            claimed_amount_band=_f("CLAIMS_CLAIMED_AMOUNT_BAND", 0.20),
            amount_ceiling=_opt_f("CLAIMS_AMOUNT_CEILING"),
            max_extraction_calls_per_claim=_i("CLAIMS_MAX_EXTRACTION_CALLS", 10),
        )
