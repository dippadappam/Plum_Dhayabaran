"""Adjudication agent: the deterministic rules engine.

No LLM is involved in any decision or calculation. The engine is a list of small
rule objects (`app/agents/rules.py`) run in a fixed order over the shared
`ClaimContext`: the first gate to resolve the claim stops the gate run; the
money terminal sequence (money math → fraud → finalize) runs only when no gate
resolved the claim; then the domain holds (identity name hold → confidence gate)
ALWAYS run, whether a gate resolved the claim or the terminal sequence did. Each
rule appends its own trace steps; cross-cutting helpers live in
`app/agents/adjudication_support.py`.

Canonical check order (the order is itself a rule, defined as data in
`rules.build_gates` / `rules.build_terminal` / `rules.build_terminal_holds`):
  amount sanity → eligibility → submission deadline → diagnosis certainty →
  exclusions (line-item) → waiting periods → pre-authorization → registered
  practitioner → session cap → high-value auto-review → limits (per-claim,
  annual, family floater) → pharmacy drug-type → [money math → fraud signals →
  finalize, when no gate resolved] → identity name hold → confidence gate.
The identity name hold and the per-field confidence gate are domain decisions
relocated from the orchestrator (Item 2): they run after the engine reaches a
decision (the confidence gate reads `ctx.deciding_fields`, which finalize — or a
gate's reject() — sets) and are silent on the 12 structured cases.
The first failing rejection check resolves the claim and reports the decisive
reason (e.g. an MRI without pre-auth is PRE_AUTH_MISSING even though it would
also breach a limit; an obesity claim is EXCLUDED_CONDITION, not WAITING_PERIOD,
because exclusions run before waiting periods).

Claim-level cap assumption (reconciles the data): the cap for a category is
max(per_claim_limit, category sub_limit). Consultation: max(5000, 2000) = 5000,
so a 7,500 consultation claim is rejected. Dental: max(5000, 10000) = 10000, so
an 8,000 dental approval stands. The consultation sub_limit additionally caps the
consultation-fee line item at 2,000. (See docs/assumptions.md.)
"""

from typing import Optional

from app.agents.base import Agent, ClaimContext
from app.agents.rules import (
    STAGE,
    build_gates,
    build_terminal,
    build_terminal_holds,
    run_rules,
)
from app.config import EngineConfig


class AdjudicationAgent(Agent):
    name = STAGE

    def __init__(self, config: Optional[EngineConfig] = None):
        self.config = config or EngineConfig()
        # The check order is data: the resolving gates, the money terminal
        # sequence (run only when no gate resolved), and the always-run domain
        # holds. Rules hold only config (immutable), so one set is built once and
        # reused across claims.
        self._gates = build_gates(self.config)
        self._terminal = build_terminal()
        self._holds = build_terminal_holds(self.config)

    def run(self, ctx: ClaimContext) -> None:
        # First gate to resolve the claim stops the gate run; the money/fraud/
        # finalize terminal sequence runs only when NO gate resolved the claim.
        if not run_rules(self._gates, ctx):
            for step in self._terminal:
                step.apply(ctx)
        # The domain holds (identity name hold → confidence gate) ALWAYS run
        # after the engine reaches a decision — whether a gate resolved it (e.g.
        # a REJECTED) or the terminal sequence produced it — so a gate-rejected
        # claim is still confidence-checked, exactly as when these ran in the
        # orchestrator after the whole agent.
        for step in self._holds:
            step.apply(ctx)
