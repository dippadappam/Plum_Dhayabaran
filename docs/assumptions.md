# Documented Assumptions

The policy data and test cases contain ambiguities and apparent
contradictions. Each assumption below states the conflict in the data, the
interpretation chosen, and why it is the reading that keeps every test case
consistent. All of these are implemented in the rules engine `app/agents/rules.py` (run by
`app/agents/adjudication.py`) and in `app/condition_mapping.py`, whose keyword
tables are loaded from `data/clinical_mappings.json`.

## A1. The consultation sub_limit caps the consultation-fee line item, not the claim total

**Conflict:** `opd_categories.consultation.sub_limit` is 2,000, yet TC010
approves a consultation claim at 3,240 and TC004's claim totals 1,500 with
only 1,000 of it being the consultation fee.

**Interpretation:** the consultation sub_limit is a cap on the
consultation-fee component of a bill (fee per visit), not on the claim. In
every consultation test case the "Consultation Fee" line item is ≤ 2,000
(TC004: 1,000; TC008: 2,000; TC010: 1,500), which is consistent with this
reading; a claim-total reading contradicts TC010 directly. Implementation:
consultation-fee line items above 2,000 are capped at 2,000 with a trace
entry.

## A2. Claim-level cap = max(per_claim_limit, category sub_limit)

**Conflict:** `coverage.per_claim_limit` is 5,000 and rejects TC008
(consultation, 7,500), yet TC006 approves 8,000 on a dental claim.

**Interpretation:** a category whose sub_limit exceeds the generic per-claim
limit is governed by its sub_limit at claim level (the category limit is the
more specific rule). Dental cap: max(5,000, 10,000) = 10,000, so 8,000
approves. Consultation cap: max(5,000, 2,000) = 5,000, so 7,500 rejects.
This single rule reproduces both expected outcomes.

## A3. Limits apply to the payable amount after exclusions

**Conflict:** TC006 claims 12,000 (above even the dental 10,000 cap) yet
expects PARTIAL at 8,000, not a limit rejection.

**Interpretation:** excluded line items are not payable, so they do not
count against limits. The limit check runs on the eligible amount after
line-item exclusions: 12,000 claimed − 4,000 excluded = 8,000 ≤ 10,000.
TC008 has no exclusions, so its eligible amount is the full 7,500 and the
rejection stands.

## A4. The decisive blocking reason is reported, and checks run in a fixed order

**Conflict:** TC007 (MRI, 15,000, diagnostic) breaches both the pre-auth
requirement and the 10,000 diagnostic cap, yet expects exactly
`["PRE_AUTH_MISSING"]`.

**Interpretation:** checks run in a canonical order (eligibility →
exclusions → waiting periods → pre-auth → limits → money → fraud) and the
first failing check resolves the claim; the member is told the decisive
blocker, not a list of everything that would also have failed. Pre-auth
precedes limits because it is actionable: the member can obtain pre-auth
and resubmit.

## A5. Exclusions precede waiting periods

**Conflict:** TC012's obesity diagnosis maps both to the
`obesity_treatment` 365-day waiting period and to the "Obesity and weight
loss programs"/"Bariatric surgery" exclusions; the expected reason is
EXCLUDED_CONDITION.

**Interpretation:** an excluded condition is permanently not covered, so a
waiting period for it is irrelevant. The exclusion check therefore runs
first. This also gives the member the truthful answer: waiting will not
make the claim payable.

## A6. Diagnosis-to-condition mapping is a deterministic whole-word dictionary

**Ambiguity:** the policy names conditions (`diabetes`, `hernia`, …) while
documents carry free text ("Type 2 Diabetes Mellitus", "Lumbar Disc
Herniation").

**Interpretation:** mapping is a keyword dictionary over the policy's fixed
condition set, with whole-word matching so "hernia" never matches
"herniation" (a disc herniation is not an abdominal hernia and TC007 must
not trip the hernia waiting period). Unmatched text maps to nothing; it can
never silently produce a wrong rejection. The constrained-LLM extension for
long-tail phrasing is now shipped as an *assist*: the extractor emits a
`canonical_condition`, constrained to the policy's condition set, with a
confidence; adjudication prefers it when confident, falls back to the keyword
dictionary otherwise, and holds for manual review when the mapping is
low-confidence — so the deterministic keyword path still backs every decision.

## A7. The waiting-period clock runs from the member's join date to the treatment date

**Ambiguity:** the policy does not state which dates the waiting period
compares.

**Interpretation:** eligibility = `join_date + waiting_days`; a claim is in
the waiting period when `treatment_date < eligibility`. TC005 confirms:
joined 2024-09-01, diabetes 90 days, eligible 2024-11-30, treatment
2024-10-15 → rejected, and the system states 2024-11-30 as required.
Dependents inherit the primary member's join date (dependents in the roster
carry no join_date of their own).

## A8. Document "hints" stand in for extraction output in the test harness

**Ambiguity:** test documents carry `actual_type`, `quality`, and
`patient_name_on_doc` instead of image bytes.

**Interpretation:** these fields represent what the extraction stage would
have produced from the raw file, allowing the document gate and decision
logic to be tested deterministically. On the live path the same information
comes from the vision extractor (document_type, readability, patient_name).
The verification checks consume both sources identically.

## A9. Fraud counting includes the claim being submitted

**Ambiguity:** whether "same-day claims limit 2" counts the incoming claim.

**Interpretation:** the incoming claim counts; TC009's three prior same-day
claims plus the submission make it the 4th, above the limit of 2, and the
flag fires. Signals route an otherwise-payable claim to MANUAL_REVIEW, never
to auto-rejection, per the TC009 contract.

## A10. NEEDS_RESUBMISSION is a status, not a decision

**Ambiguity:** TC001-TC003 expect `decision: null` while real members need a
clear state to act on.

**Interpretation:** the result separates `status` (DECIDED |
NEEDS_RESUBMISSION) from `decision`. Document problems set status
NEEDS_RESUBMISSION with decision null: no claim decision was made, the
member gets specific, actionable issues, and the claim is explicitly not
rejected.

## A11. Submission-deadline enforced against the recorded received date

The policy's 30-day submission deadline is enforced against `received_date` —
the genuine date a production intake records — which a production intake
supplies; this demo's auto-stamped `submission_date` ("today", set by the API
for the future-date sanity check) does not count toward the deadline.
Measuring against the auto-stamp would retroactively fail every 2024-dated test
and demo claim, so the check fires only when a real received date is present;
absent one (the 12 official cases, and any demo submission), it is skipped —
keeping the official cases byte-identical rather than enforcing with a
fabricated date.

## A12. Currency amounts round half-up

Every rupee amount in the money math (network discount, co-pay, and the
per-category and aggregate totals) is rounded to 2 decimals **half-up** — a
half-paisa (₹x.xx5) rounds up — via `round_money` in `app/agents/rules.py`, not
Python's built-in `round()`, which is round-half-to-even (banker's). Half-up is
the conventional per-claim rounding. Non-currency values (the confidence score,
the fraud sub-scores) keep `round()`. The 12 official cases never land on a
half-paisa, so this is byte-identical for them; the boundary is pinned by
`tests/test_money_rounding.py`.
