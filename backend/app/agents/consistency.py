"""Consistency agent: cross-checks the submission form against the documents.

Runs after extraction, the document gate, and derivation, and before
adjudication. Two checks, both deliberately lenient, and both route to
MANUAL_REVIEW rather than auto-rejecting — a mismatch here is usually an
honest mistake or potential misuse, and a human must decide which.

1. Patient identity: the patient named on the documents must be the filing
   member. A different covered person on the roster, or a person not on the
   roster at all, routes to manual review with the specific names. Documents
   carrying no patient names skip the check (never false-flag on absence).
   Roster entries are matched by normalized name; dependents listed in a
   member's `dependents` array without a roster row of their own simply
   produce no match and fall into the "not on roster" path.

2. Category consistency: flags ONLY when the documents contain no evidence
   for the filed category AND specific evidence for a different one. A
   supported filed category is never overridden, so generic items (the
   "Consultation" line on an ayurveda bill) can never flag a claim whose
   filed category has its own evidence.
"""

from app.agents.base import Agent, ClaimContext
from app.category_evidence import (
    CATEGORY_EVIDENCE,
    category_hits,
    evidence_texts,
    normalize_corpus,
)
from app.condition_mapping import names_match
from app.models.decision import Decision, StepStatus

STAGE = "consistency"

# The evidence keywords live in app/category_evidence.py, shared with the
# CategoryResolutionAgent. This agent uses the CHECK grade (wide corpus:
# diagnosis, treatment, line items, tests, hospital name) because it only
# ever vetoes. CONSULTATION evidence only ever *supports* a filed
# CONSULTATION claim; because a supported filed category is never flagged,
# consultation words on a specialty bill cannot override that specialty.


class ConsistencyAgent(Agent):
    name = STAGE

    def run(self, ctx: ClaimContext) -> None:
        flags: list[str] = []
        self._check_patient_identity(ctx, flags)
        self._check_category(ctx, flags)

        if flags:
            ctx.result.decision = Decision.MANUAL_REVIEW
            ctx.result.manual_review_recommended = True
            ctx.result.reasons.extend(flags)
            ctx.result.reasons.append(
                "Routed to manual review (not auto-rejected): the mismatch may "
                "be an honest mistake, so a human reviewer must verify before "
                "the claim can proceed."
            )
            ctx.halted = True
            ctx.trace(STAGE, "routing", StepStatus.INFO,
                      "Consistency mismatch(es) found; claim routed to "
                      "MANUAL_REVIEW before adjudication.",
                      {"flag_count": len(flags)})

    # ------------------------------------------------------------------
    # Check 1: the patient on the documents is the filing member
    # ------------------------------------------------------------------
    def _check_patient_identity(self, ctx: ClaimContext, flags: list[str]) -> None:
        member = ctx.policy.get_member(ctx.submission.member_id)
        extracted_by_id = {
            d.get("file_id"): d.get("fields", {}) for d in ctx.extracted_documents
        }

        names: list[str] = []
        for d in ctx.submission.documents:
            name = (
                extracted_by_id.get(d.file_id, {}).get("patient_name")
                or d.patient_name_on_doc
                or (d.content or {}).get("patient_name")
            )
            if name:
                names.append(str(name))

        if not names or member is None:
            if not names:
                # The identity check cannot run; flag it so a payable
                # decision carries a manual-review advisory (set by the
                # orchestrator), without halting or changing the decision.
                ctx.no_patient_names = True
            ctx.trace(STAGE, "patient_identity", StepStatus.SKIPPED,
                      "No patient names available on the documents to compare "
                      "against the filing member; check skipped. If the claim "
                      "is payable, manual review will be recommended because "
                      "identity was not verified.")
            return

        # Directional semantics: the patient must be the filing member, or a
        # dependent OF the filing member that has a roster row. Dependents
        # listed only as IDs (no name in the policy data) cannot be matched and
        # are deliberately not in this set, so those claims still route to
        # review. "Same household" is not enough — a dependent filing with the
        # primary's documents is not a dependent of the filer and still holds.
        covered = self._covered_names_for_filer(ctx.policy, member)
        mismatched = [n for n in names
                      if not any(names_match(n, c) for c in covered)]
        if not mismatched:
            ctx.trace(STAGE, "patient_identity", StepStatus.PASSED,
                      f"The documents are for {member.name} (the filing member) "
                      f"or a listed dependent of the filing member.")
            return

        other = mismatched[0]
        roster_hit = next(
            (m for m in ctx.policy.members if names_match(m.name, other)), None
        )
        if roster_hit:
            msg = (
                f"The documents are for '{other}' ({roster_hit.member_id}, a "
                f"covered person on this policy but not the filing member or a "
                f"dependent of the filing member), but the claim is filed for "
                f"'{member.name}' ({member.member_id})."
            )
            priority = "normal"
        else:
            msg = (
                f"The documents are for '{other}', who is not on the policy "
                f"member roster, but the claim is filed for '{member.name}' "
                f"({member.member_id})."
            )
            priority = "high"
        # Reviewer triage: never downgrade an already-high priority.
        if ctx.result.review_priority != "high":
            ctx.result.review_priority = priority
        flags.append(msg)
        ctx.trace(STAGE, "patient_identity", StepStatus.FAILED, msg,
                  {"document_patient": other,
                   "filing_member": member.name,
                   "roster_match": roster_hit.member_id if roster_hit else None,
                   "review_priority": priority})

    @staticmethod
    def _covered_names_for_filer(policy, member) -> list[str]:
        """Names the filer may legitimately claim for, under directional
        semantics: the filer, plus dependents OF the filer that have a roster
        row. A dependent listed only as an ID (DEP003-DEP006 have no row and no
        name in the policy data) cannot be matched and is deliberately absent
        here — its claims still route to review. Compared with names_match,
        which tolerates honorifics, suffixes, and middle initials."""
        covered = [member.name]
        for m in policy.members:
            if m.primary_member_id == member.member_id:
                covered.append(m.name)
        for dep_id in member.dependents:
            dep = policy.get_member(dep_id)
            if dep is not None:
                covered.append(dep.name)
        return covered

    # ------------------------------------------------------------------
    # Check 2: the documents support the filed category (lenient)
    # ------------------------------------------------------------------
    def _check_category(self, ctx: ClaimContext, flags: list[str]) -> None:
        filed = ctx.submission.claim_category.value
        corpus = normalize_corpus(
            evidence_texts(ctx.extracted_documents, grade="check"))
        if not corpus:
            ctx.trace(STAGE, "category_consistency", StepStatus.SKIPPED,
                      "No document text available to check against the filed "
                      "category; check skipped.")
            return

        def hits(category: str) -> list[str]:
            return category_hits(corpus, category)

        filed_hits = hits(filed)
        if filed_hits:
            ctx.result.derived_category = filed
            ctx.trace(STAGE, "category_consistency", StepStatus.PASSED,
                      f"The documents support the filed category {filed} "
                      f"(matched on '{filed_hits[0]}').",
                      {"derived_category": filed,
                       "matched_keywords": filed_hits})
            return

        other_hits = {
            c: h for c in CATEGORY_EVIDENCE
            if c != filed and (h := hits(c))
        }
        if not other_hits:
            ctx.trace(STAGE, "category_consistency", StepStatus.PASSED,
                      f"The documents carry no specific evidence for any other "
                      f"category; the filed category {filed} is accepted.",
                      {"derived_category": None})
            return

        derived = max(other_hits, key=lambda c: len(other_hits[c]))
        ctx.result.derived_category = derived
        msg = (
            f"The claim was filed as {filed}, but the documents contain no "
            f"{filed.replace('_', ' ').lower()}-related evidence and instead "
            f"describe {derived.replace('_', ' ')} (matched on "
            f"'{other_hits[derived][0]}')."
        )
        flags.append(msg)
        ctx.trace(STAGE, "category_consistency", StepStatus.FAILED, msg,
                  {"filed_category": filed, "derived_category": derived,
                   "matched_keywords": other_hits[derived]})
