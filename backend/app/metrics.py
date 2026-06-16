"""In-process runtime metrics — standard library only.

A tiny, thread-safe counter set incremented on the claim path and exposed as a
JSON snapshot at GET /metrics. This is observability layered on top of the
engine: recording a metric is a pure side effect that never affects any
decision, amount, or trace.

No external metrics library (no Prometheus client), so the dependency set and
the lockfile are unchanged. FastAPI runs the sync claim handler in a worker
thread pool, so increments are guarded by a lock.
"""

import threading

# Decision classes counted in the mix (NEEDS_RESUBMISSION covers a document-gate
# stop, where the engine makes no claim decision).
_DECISIONS = ("APPROVED", "PARTIAL", "REJECTED", "MANUAL_REVIEW",
              "NEEDS_RESUBMISSION")
# Confidence buckets (the score is clamped to [0.05, 0.95]).
_CONF_BUCKETS = ("lt_0.5", "0.5_0.7", "0.7_0.85", "gte_0.85")
# Request-latency buckets, in milliseconds.
_LAT_BUCKETS = ("lt_50ms", "lt_200ms", "lt_1000ms", "gte_1000ms")


class Metrics:
    """Thread-safe in-process counters. One instance is shared process-wide."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        """Zero every counter. Used by tests for a deterministic slate (the
        module singleton persists across the suite's api-module reloads)."""
        with self._lock:
            self._claims = 0
            self._decisions = {d: 0 for d in _DECISIONS}
            self._degraded = 0
            self._conf = {b: 0 for b in _CONF_BUCKETS}
            self._lat_count = 0
            self._lat_sum_ms = 0.0
            self._lat = {b: 0 for b in _LAT_BUCKETS}
            self._input_tokens = 0
            self._output_tokens = 0

    @staticmethod
    def _conf_bucket(score: float) -> str:
        if score < 0.5:
            return "lt_0.5"
        if score < 0.7:
            return "0.5_0.7"
        if score < 0.85:
            return "0.7_0.85"
        return "gte_0.85"

    @staticmethod
    def _lat_bucket(ms: float) -> str:
        if ms < 50:
            return "lt_50ms"
        if ms < 200:
            return "lt_200ms"
        if ms < 1000:
            return "lt_1000ms"
        return "gte_1000ms"

    def record_claim(self, *, decision, status, confidence: float,
                     degraded: bool, latency_ms: float,
                     input_tokens: int = 0, output_tokens: int = 0) -> None:
        """Record one fully-processed claim. ``decision`` is the final decision
        value string (or None for a document-gate stop); ``status`` is the
        result status; ``degraded`` is True when the claim had any component
        failure; ``input_tokens``/``output_tokens`` are this claim's summed LLM
        usage. Pure side effect — reads the result, mutates only counters."""
        with self._lock:
            self._claims += 1
            key = decision if decision in self._decisions else (
                "NEEDS_RESUBMISSION" if status == "NEEDS_RESUBMISSION" else None)
            if key is not None:
                self._decisions[key] += 1
            if degraded:
                self._degraded += 1
            self._conf[self._conf_bucket(confidence)] += 1
            self._lat_count += 1
            self._lat_sum_ms += latency_ms
            self._lat[self._lat_bucket(latency_ms)] += 1
            self._input_tokens += int(input_tokens or 0)
            self._output_tokens += int(output_tokens or 0)

    def snapshot(self) -> dict:
        """A JSON-serializable point-in-time view of every counter."""
        with self._lock:
            claims = self._claims
            mr = self._decisions["MANUAL_REVIEW"]
            avg = (round(self._lat_sum_ms / self._lat_count, 2)
                   if self._lat_count else 0.0)
            return {
                "claims_processed": claims,
                "decisions": dict(self._decisions),
                "manual_review_rate": round(mr / claims, 4) if claims else 0.0,
                "degraded_claims": self._degraded,
                "confidence_buckets": dict(self._conf),
                "latency_ms": {
                    "count": self._lat_count,
                    "avg": avg,
                    "buckets": dict(self._lat),
                },
                "tokens": {
                    "input": self._input_tokens,
                    "output": self._output_tokens,
                },
            }


# Process-wide singleton. It persists across `importlib.reload(app.api)` (the
# app.metrics module stays cached in sys.modules), so counts survive the test
# suite's api reloads; tests call reset() when they need a clean slate.
metrics = Metrics()
