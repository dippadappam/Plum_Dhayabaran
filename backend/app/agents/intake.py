"""Intake agent: validates the submission against member roster and
submission rules before any document processing.
"""

from app.agents.base import Agent, ClaimContext
from app.agents.shared_checks import check_coverage, check_minimum_amount
from app.models.decision import Decision, RejectionReason, StepStatus


class IntakeAgent(Agent):
    name = "intake"

    def run(self, ctx: ClaimContext) -> None:
        sub = ctx.submission
        policy = ctx.policy

        # Policy ID must match the loaded policy.
        if sub.policy_id != policy.policy_id:
            ctx.result.decision = Decision.REJECTED
            ctx.result.rejection_reasons.append(RejectionReason.POLICY_MISMATCH)
            ctx.result.reasons.append(
                f"Policy ID '{sub.policy_id}' does not match the active policy "
                f"'{policy.policy_id}'."
            )
            ctx.trace(
                self.name, "policy_id", StepStatus.FAILED,
                f"Submitted policy_id '{sub.policy_id}' does not match active policy "
                f"'{policy.policy_id}'.",
            )
            ctx.halted = True
            return
        ctx.trace(self.name, "policy_id", StepStatus.PASSED,
                  f"Policy '{policy.policy_id}' is active.")

        # Member must exist on the roster.
        member = policy.get_member(sub.member_id)
        if member is None:
            ctx.result.decision = Decision.REJECTED
            ctx.result.rejection_reasons.append(RejectionReason.MEMBER_NOT_FOUND)
            ctx.result.reasons.append(
                f"Member '{sub.member_id}' is not on the policy member roster."
            )
            ctx.trace(self.name, "member_exists", StepStatus.FAILED,
                      f"Member '{sub.member_id}' not found on roster.")
            ctx.halted = True
            return
        ctx.trace(self.name, "member_exists", StepStatus.PASSED,
                  f"Member {member.member_id} ({member.name}) found on roster.",
                  {"member_name": member.name, "relationship": member.relationship})

        # Claim category must be a covered category in the policy. When no
        # category was provided (real-upload path) the check is deferred:
        # the CategoryResolutionAgent derives the category after extraction
        # and re-runs the coverage check there.
        if sub.claim_category is None:
            ctx.trace(self.name, "category_covered", StepStatus.INFO,
                      "No claim category provided; it will be derived from "
                      "the documents after extraction, and the coverage "
                      "check runs then.")
            self._check_minimum_amount(ctx)
            return
        if check_coverage(ctx, category=sub.claim_category.value,
                          stage=self.name, derived=False):
            return

        self._check_minimum_amount(ctx)

    def _check_minimum_amount(self, ctx: ClaimContext) -> None:
        """Minimum claim amount. On the real-upload path the amount is not
        yet known (it is derived from the bill after extraction), so the
        check is deferred rather than run against a missing value. The rule
        itself lives in shared_checks.check_minimum_amount, shared with the
        derived-input path (derivation.py)."""
        if ctx.submission.claimed_amount is None:
            ctx.trace(self.name, "minimum_amount", StepStatus.INFO,
                      "Claim amount will be derived from the uploaded documents; "
                      "minimum-amount check deferred to after extraction.")
            return
        check_minimum_amount(ctx, amount=ctx.submission.claimed_amount,
                             stage=self.name, derived=False)
