"""Deterministic confidence scoring.

The confidence score is computed by a documented formula, never generated
by an LLM, so identical claims always produce identical confidence.

Formula:
  confidence = BASE (0.95)
             x extraction_confidence        (1.0 for injected/clean content)
             - 0.25 if any component failed (degraded pipeline)
  clamped to [0.05, 0.95], rounded to 2 decimals.

Rationale: a full clean pipeline never claims certainty (cap 0.95); a
degraded pipeline must score visibly lower than any clean run (TC011
contract), and extraction uncertainty propagates multiplicatively because
every downstream check depends on the extracted fields.
"""

from app.agents.base import ClaimContext

BASE_CONFIDENCE = 0.95
COMPONENT_FAILURE_PENALTY = 0.25


def score_confidence(ctx: ClaimContext) -> None:
    factors: list[str] = [f"Base confidence {BASE_CONFIDENCE:.2f}"]
    confidence = BASE_CONFIDENCE

    if ctx.extraction_confidence < 1.0:
        confidence *= ctx.extraction_confidence
        factors.append(
            f"Multiplied by extraction confidence {ctx.extraction_confidence:.2f}"
        )
    else:
        factors.append("Extraction confidence 1.00 (structured content)")

    if ctx.result.component_failures:
        confidence -= COMPONENT_FAILURE_PENALTY
        factors.append(
            f"-{COMPONENT_FAILURE_PENALTY:.2f}: {len(ctx.result.component_failures)} "
            "component(s) failed during processing"
        )

    confidence = max(0.05, min(0.95, round(confidence, 2)))
    ctx.result.confidence_score = confidence
    ctx.result.confidence_factors = factors
