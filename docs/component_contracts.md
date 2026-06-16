# Component Contracts

Every contract below is enforced in code by the Pydantic models in
`backend/app/models/` (the source of truth) and exercised by the test suite.
This document states each component's inputs, outputs, and error behavior
precisely enough to rebuild the component without reading its implementation.
It is kept in sync with the code.

Shared convention: agents implement `run(ctx: ClaimContext) -> None`, reading
and writing a shared context. An agent never raises to the caller in normal
operation; expected failures are written into the result (document issues,
rejections, manual-review routing, component failures). Unexpected exceptions
are caught by the orchestrator and converted into degradation.

`ClaimContext` (shared blackboard, `app/agents/base.py`): `submission`,
`policy`, `result`, `extracted_documents`, `extraction_confidence`, `halted`,
`no_patient_names`, `category_was_derived`, `deciding_fields`.

---

## 1. Orchestrator (`app/orchestrator.py`)

**Pipeline (fixed order):**
Intake → Extraction → CategoryResolution → DocumentVerification → Derivation →
Consistency → Adjudication. Extraction runs **before** the document gate so the
gate and later stages use values detected from the documents; category
resolution runs before the gate because the required-document check is keyed by
category. The orchestrator is pure coordination: after the agent loop it runs
only coordination steps — no-decision fallback → uniform degradation policy →
review lifecycle → confidence score. The identity advisory and the per-field
confidence gate are now terminal steps inside AdjudicationAgent (§8), not the
orchestrator.

**Input:** `ClaimSubmission`
- `member_id: str`, `policy_id: str`
- `claim_category: enum | null` (CONSULTATION | DIAGNOSTIC | PHARMACY | DENTAL |
  VISION | ALTERNATIVE_MEDICINE) — optional; honored byte-for-byte when present,
  else derived by CategoryResolution or asked (CATEGORY_NEEDED)
- `treatment_date: date | null` — optional; derived on the real-upload path
- `submission_date: date | null` — when the claim was submitted; the API stamps
  `date.today()` when absent; used by the future-date sanity check. **Not** used
  by the submission-deadline check.
- `received_date: date | null` — the genuine date intake received the claim;
  **never auto-stamped**; the submission-deadline check measures against this
- `claimed_amount: float > 0 | null` (`allow_inf_nan=false`) — optional
- `hospital_name: str | null` (fallback only; the extracted bill wins)
- `ytd_claims_amount: float = 0` (`allow_inf_nan=false`) — caller-supplied
- `family_ytd_amount: float = 0` (`allow_inf_nan=false`) — the family's
  year-to-date approved spend; computed from storage and injected by the API
- `claims_history: PriorClaim[]` (claim_id, date, amount, provider)
- `duplicate_of: str | null` — set by the API when the document-set hash matches
  an already-DECIDED claim (H3)
- `parent_claim_reference: str | null` — explicit resubmission link (never
  inferred)
- `documents: SubmittedDocument[]` — **min 1, max 10**; per document `file_data`
  capped (~15M base64 ≈ 11 MB) and `media_type` ∈ {image/jpeg, image/png,
  image/webp, image/gif, application/pdf}; violations are free 422s
- `simulate_component_failure: bool = false` (test hook; see architecture.md §10)

**Output:** `ClaimResult`
- `claim_reference: str` (generated, unique)
- `status:` DECIDED | NEEDS_RESUBMISSION
- `decision:` APPROVED | PARTIAL | REJECTED | MANUAL_REVIEW | null (null iff NEEDS_RESUBMISSION)
- `approved_amount: float | null`
- `rejection_reasons: RejectionReason[]` — WAITING_PERIOD, EXCLUDED_CONDITION,
  PRE_AUTH_MISSING, PER_CLAIM_EXCEEDED, SUB_LIMIT_EXCEEDED, ANNUAL_LIMIT_EXCEEDED,
  FAMILY_LIMIT_EXCEEDED, MEMBER_NOT_FOUND, POLICY_MISMATCH, NOT_COVERED,
  BELOW_MINIMUM_AMOUNT, SUBMISSION_DEADLINE_PASSED
- `reasons: str[]` human-readable explanations
- `line_items: LineItemDecision[]`, `amount_breakdown: AmountBreakdown | null`
- `document_issues: DocumentIssue[]` (issue_code MISSING_REQUIRED | UNREADABLE |
  PATIENT_MISMATCH | EXTRACTION_FAILED | CATEGORY_NEEDED)
- `fraud_signals: str[]`, `fraud_score: float [0,1]`
- `component_failures: ComponentFailure[]`, `manual_review_recommended: bool`
- `review_priority: "high" | "normal" | null` (high = patient off-roster; normal
  = different covered person)
- `review_status: "PENDING_REVIEW" | "RESOLVED" | null`;
  `resolved_by`, `resolved_at`, `resolution` (APPROVED | REJECTED | CLOSED),
  `resolution_reason` — set by the review lifecycle (`POST .../resolve`)
- `derived_category: str | null` — the documents' indicated category (keyword
  evidence); transparency when a category was provided, the adjudicated category
  when none was
- `extracted_documents: dict[]` — the "what we read" record for the reviewer
- `confidence_score: float [0.05, 0.95]`, `confidence_factors: str[]`
- `trace: TraceStep[]` (stage, check, status PASSED|FAILED|SKIPPED|DEGRADED|INFO, detail, data, timestamp)

**Guarantees:**
- Never raises for any valid `ClaimSubmission`; component exceptions become
  `component_failures` + DEGRADED trace steps (a storage I/O error on
  persistence is the one path that can still surface as a 500 — the decision was
  already computed).
- Any component failure ⇒ `manual_review_recommended = true` + confidence penalty.
- A halt (intake rejection, document gate, category-needed, derivation failure,
  consistency mismatch) skips later stages with SKIPPED trace entries.
- **Identity advisory** *(now an AdjudicationAgent terminal step — §8)*: a
  live-upload payable claim with no readable patient name is held when amount ≥
  `EngineConfig.no_name_high_value` (₹2,500), else passes with an advisory note.
- **Confidence gate (per-field)** *(now an AdjudicationAgent terminal step —
  §8)*: an otherwise-payable-or-rejected claim is overridden to MANUAL_REVIEW
  when a field the deciding check read (`ctx.deciding_fields`) was below
  `EngineConfig.confidence_threshold`; the computed decision/amount remain in the
  output. Silent on structured content.
- The trace alone is sufficient to reconstruct the decision — **including the
  save-time backstops**, which now append `persistence`-stage trace steps.

---

## 2. IntakeAgent (`app/agents/intake.py`)

**Reads:** submission, policy. **Writes:** trace; on failure: REJECTED + reason + halt.
**Checks, in order:** policy_id matches (POLICY_MISMATCH); member on roster
(MEMBER_NOT_FOUND); category covered (NOT_COVERED); claimed amount ≥ policy
minimum (BELOW_MINIMUM_AMOUNT). Deferred (with an INFO trace) when the value
will be derived: coverage re-runs in CategoryResolution, minimum in Derivation.

---

## 3. ExtractionAgent (`app/agents/extraction.py`)

**Reads:** documents, `EngineConfig.max_extraction_calls_per_claim`.
**Writes:** `extracted_documents` (file_id, type, fields), `extraction_confidence`
(mean per-document), trace; failures append `component_failures`.

**Per-document behavior:**
- `content` present → used as-is, confidence 1.0 (deterministic test path).
- `file_data` + extractor configured → content-hash dedupe (identical sha256
  reuses the earlier extraction, INFO trace, no extra paid call); otherwise
  `DocumentExtractor.extract(document)`. Unique documents are extracted **in
  parallel** (`ThreadPoolExecutor`); the agent itself stays synchronous (the
  pool is joined before it returns).
- Extractor raises → confidence 0.0 for that document, ComponentFailure
  (`extraction:<file_id>`), DEGRADED trace, pipeline continues.
- Neither content nor extractable data → SKIPPED, confidence 0.5.
- **Per-claim paid-call cap:** more than `max_extraction_calls_per_claim` unique
  payloads → the first `cap` are read and the claim is routed to MANUAL_REVIEW
  (never a silent drop).

**DocumentExtractor protocol:** `extract(document: SubmittedDocument) -> dict`
returning: `document_type` (PRESCRIPTION | HOSPITAL_BILL | PHARMACY_BILL |
LAB_REPORT | DIAGNOSTIC_REPORT | DISCHARGE_SUMMARY | DENTAL_REPORT | UNKNOWN),
`patient_name`, `doctor_name`, `doctor_registration`, `hospital_name`, `date`,
`diagnosis`, `primary_diagnosis`, `comorbidities`, `canonical_condition`,
`canonical_condition_confidence`, `treatment`, `medicines[]`,
`line_items[{description, amount, drug_type, drug_type_confidence}]`, `total`,
`amount_confidence`, `treatment_date_confidence`, `patient_name_confidence`,
`hospital_confidence`, `category_confidence`, `pre_auth_number`, `readability`
(GOOD | PARTIAL | UNREADABLE), `extraction_confidence: float`. The
`canonical_condition` vocabulary is read from `policy_terms.json`. Raises
`ExtractionError` on missing data, timeout after bounded retries, malformed
output, or a permanent 4xx (≠429). Must not hang (hard timeout is part of the
contract).

---

## 4. CategoryResolutionAgent (`app/agents/category_resolution.py`)

**Reads:** submission, `extracted_documents`, `component_failures`, policy.
**Writes:** effective `claim_category` (back onto the submission),
`result.claim_category`, `category_was_derived`, trace; on ambiguity:
`status = NEEDS_RESUBMISSION`, `decision = null`, CATEGORY_NEEDED, halt.

- Provided category ⇒ **no-op** (honored byte-for-byte; no trace).
- No category ⇒ derive from DECIDE-grade evidence (`app/category_evidence.py`):
  **procedural text only** (line items, treatment, tests); diagnosis-only and
  hospital-name hits never decide. Exactly one specialty with evidence wins;
  consultation evidence only ⇒ CONSULTATION; two+ specialties or none ⇒
  ambiguous. PHARMACY derives only from an actual PHARMACY_BILL document type.
- Derived ⇒ written back; the intake-deferred coverage check re-runs here;
  `category_was_derived = True` (enables per-line categorization in adjudication).
- Ambiguous ⇒ **ask the member** (CATEGORY_NEEDED), halting before the gate; when
  extraction failures caused it, the message says the files could not be read.

The same evidence module serves the ConsistencyAgent at CHECK grade (wider
corpus) because that check only ever vetoes.

---

## 5. DocumentVerificationAgent (`app/agents/document_verification.py`)

**Reads:** submission documents, `extracted_documents`, `component_failures`,
policy `document_requirements`. **Writes:** `document_issues`, trace; on any
issue: `status = NEEDS_RESUBMISSION`, `decision = null`, halt.

Runs **after** extraction, using the extracted value when present
(`document_type`, `readability`, `patient_name`) and the declared hint
(`actual_type`, `quality`, `patient_name_on_doc`, structured `content`) as
fallback. Checks: (1) **required types** present — message names what was
uploaded and what is required; a missing required type whose candidate file
could not be read is **EXTRACTION_FAILED**, never a false MISSING_REQUIRED; a
dental claim accepts a DENTAL_REPORT/UNKNOWN substitute for the hospital bill.
(2) **readability** — any UNREADABLE document → UNREADABLE issue naming that
file, asks for re-upload, states the claim is not rejected. (3) **patient
consistency** across documents (tolerant name matching) → PATIENT_MISMATCH
listing each name; no names ⇒ SKIPPED. **Guarantee:** if this agent halts,
adjudication never runs (asserted in tests).

---

## 6. ClaimDerivationAgent (`app/agents/derivation.py`)

**Reads:** submission, `extracted_documents`, policy submission rules.
**Writes:** effective submission values, trace; on failure: MANUAL_REVIEW or
REJECTED + halt. No-op when `claimed_amount` was provided. Otherwise:
`claimed_amount` = extracted bill `total` (bill docs preferred, summed across
multiple bills), else the line-item sum; `hospital_name`/`treatment_date` from
extracted fields; an ambiguous date or no determinable total → MANUAL_REVIEW
(never auto-approval); the deferred minimum-amount check runs here.

---

## 7. ConsistencyAgent (`app/agents/consistency.py`)

**Reads:** submission, `extracted_documents`, policy roster. **Writes:**
`derived_category`, `review_priority`, `no_patient_names`, reasons, trace; on
mismatch: MANUAL_REVIEW + halt (never auto-reject).
- **Patient identity:** document patient (extracted first) must be the filing
  member or a roster dependent of the filer (tolerant name match). Different
  covered person ⇒ review (`normal`); off-roster ⇒ review (`high`); no names ⇒
  SKIPPED + `no_patient_names` (consumed by the engine's identity name hold,
  `IdentityNameHoldStep`, §8).
- **Category consistency (lenient):** flags ONLY when documents carry **no
  evidence for the filed category AND specific evidence for another**; a
  supported filed category is never overridden. Sets `derived_category` when
  evidence exists.

---

## 8. AdjudicationAgent (`app/agents/adjudication.py`, rules in `app/agents/rules.py`)

**Reads:** submission, policy, `extracted_documents`, line items. **Writes:**
decision, rejection_reasons, reasons, line_items, amount_breakdown,
fraud_signals, fraud_score, `deciding_fields`, trace.

**Check order (binding — the exact `run()` sequence):**
amount sanity → eligibility → submission deadline → diagnosis certainty →
exclusions (line-item) → waiting periods → pre-authorization → registered
practitioner → session cap → high-value auto-review → limits (per-claim, annual,
family floater) → pharmacy drug-type certainty → money math → fraud signals →
finalize → identity name hold → confidence gate. The first failing rejection
check resolves the claim and is the reason reported. The money/fraud/finalize
sequence runs only when no gate resolved the claim; the **identity name hold**
and **confidence gate** (relocated here from the orchestrator) ALWAYS run after
the engine reaches a decision, so a gate-rejected claim is still
confidence-checked. Each resolving check records the extracted fields it read
into `ctx.deciding_fields` (consumed by the confidence gate).

**Key sub-contracts:**
- *Submission deadline:* against `received_date` only; > policy deadline days ⇒
  REJECTED (SUBMISSION_DEADLINE_PASSED). Skipped when `received_date` is absent.
- *Exclusions:* every line item gets a LineItemDecision; all excluded ⇒ REJECTED
  (EXCLUDED_CONDITION), mixed ⇒ PARTIAL; non-covered/non-excluded lines in a
  whitelist category (dental/vision) ⇒ review.
- *Waiting periods:* rejection states the exact eligibility date (join + days);
  dependents inherit the primary's join date; matched on the primary/canonical
  diagnosis only.
- *Pre-auth:* global scan for MRI/CT/PET above threshold in any category;
  structured-path absence ⇒ REJECTED, live-path or unverifiable reference ⇒
  review.
- *Registered practitioner / session cap:* alt-med — missing registration ⇒
  review; this claim's session count over `max_sessions_per_year` ⇒ review
  (cross-year accrual deferred).
- *Limits:* per-claim cap = max(per_claim_limit, sub_limit); annual OPD; family
  floater (`family_ytd_amount` + eligible > combined_limit ⇒
  FAMILY_LIMIT_EXCEEDED). On the derived multi-category path, applied per group.
- *Money math:* `AmountBreakdown` shows claimed → eligible → network discount
  (first) → co-pay → approved. Pharmacy applies co-pay **per line** (0 generic,
  `branded_drug_copay_percent` branded); other categories group-level. Hospital
  for the network check comes from the extracted bill first.
- *Fraud:* binary gates + a transparent weighted `fraud_score` (each
  contribution in the trace); payable claim + signals ⇒ MANUAL_REVIEW. This
  component is individually contained; `simulate_component_failure` fails exactly
  it (ComponentFailure + DEGRADED + manual review; the decision still finalizes).
- *Terminal holds (always run after the engine decides; relocated from the
  orchestrator):* **identity name hold** — a live-upload payable claim with no
  readable patient name is held at/above `no_name_high_value`, else passes with
  an advisory note; then the per-field **confidence gate** (§10). Both are silent
  on the 12 structured cases (no patient name absent and no confidence present),
  preserving their decisions and traces byte-for-byte.

---

## 9. Condition mapper (`app/condition_mapping.py`)

**Input:** free-text fields (diagnosis, treatment, line-item descriptions).
**Output:** `ConditionMapping` (matched waiting-period keys, matched exclusion
conditions, the keyword each matched on).
**Guarantees:** deterministic; whole-word phrase matching ("hernia" never matches
"herniation"); unmatched ⇒ nothing, never a wrong condition.
- `match_in_list(text, items)` — one-direction **word-boundary** match (the
  policy phrase appears in the line as whole words; lookarounds, not `\b`, so
  punctuated entries like "Orthodontic Treatment (Braces)" still match);
  excluded-list matching.
- `covered_in_list(text, items)` — lenient token-overlap; covered-list matching.
- `map_canonical_condition(canonical, waiting_keys, exclusion_conditions)` — maps
  the extractor's `canonical_condition` to a `ConditionMapping`; empty when the
  value is out of the policy vocabulary (caller falls back to the keyword path).
- `names_match(a, b)` — tolerant person-name match (honorifics, S/o suffixes,
  token-set overlap) used by the identity checks.

---

## 10. Confidence scorer (`app/confidence.py`) and gate

**Scorer input:** context after all stages. **Output:** `confidence_score` and
`confidence_factors`. **Formula (binding):** `0.95 × extraction_confidence −
0.25 if any component failure`; clamped [0.05, 0.95]; rounded to 2 dp; every
applied factor listed. Identical inputs ⇒ identical confidence. Confidence means
certainty-of-output: a clear rejection scores as high as a clean approval.

**Confidence gate (`ConfidenceGateStep`, an AdjudicationAgent terminal step):**
after the engine reaches a decision, holds the claim (APPROVED/PARTIAL/REJECTED →
MANUAL_REVIEW) when a field the deciding check read (`ctx.deciding_fields`) has a
per-field confidence below `EngineConfig.confidence_threshold` (0.8), falling
back to the document-level extraction confidence when no per-field value exists.
Silent on structured content (no confidence present).

---

## 11. ClaimStore (`app/storage.py`)

SQLite; tables `claims` (+ idempotent column migrations) and `claim_documents`.
**Public methods (10):**
- `save(result, treatment_date, claimed_amount, documents_hash=None,
  same_day_limit=None, parent_reference=None, family_combined_limit=None,
  family_member_ids=None, policy_start=None, policy_end=None)` — persists the
  claim row + full result JSON inside `BEGIN IMMEDIATE`. Hosts two **save-time
  race backstops** that may override the decision and **append a trace step**:
  same-day (→ MANUAL_REVIEW) and family-floater (→ REJECTED). Raises sqlite3
  errors on I/O failure (the API surfaces a 500; the decision was already
  computed).
- `save_documents(claim_reference, documents)` — stores raw base64 payloads
  (live submissions only).
- `get(claim_reference) -> dict | null`
- `get_document(claim_reference, file_id) -> (media_type, base64) | null`
- `list_recent(limit=50) -> row[]`
- `list_held() -> dict[]` — the review queue (review_status = PENDING_REVIEW)
- `resolve(reference, action, reviewer_id, reason, resolved_at,
  approved_amount=None) -> (outcome, result)` — reviewer lifecycle; outcomes
  OK | NOT_FOUND | CONFLICT | BAD_ACTION | NEEDS_AMOUNT; appends a trace step; an
  explicit approve-amount on a None-amount hold is required.
- `find_decided_by_hash(documents_hash) -> reference | null` — H3 duplicate
  detection (DECIDED + (RESOLVED or no review) scope).
- `member_history(member_id) -> PriorClaim[]` — DECIDED claims (non-null
  date/amount), the fraud-check input.
- `family_ytd_approved(member_ids, start, end) -> float` — summed approved spend
  across the family for the policy year, injected by the API as
  `family_ytd_amount`.

The interface is small and SQLite-specific behavior is isolated, but a Postgres
swap is **not purely mechanical**: `save()` embeds the two domain backstops and
`BEGIN IMMEDIATE` semantics (architecture.md §9).

---

## 12. HTTP API (`app/api.py`)

- `POST /api/claims` → 200 `ClaimResult`. 422 on schema violations (document
  count, file size, media type, inf/nan — rejected before any extraction cost).
  Stamps `submission_date` if absent; sets `duplicate_of` from the document-set
  hash; merges stored `member_history` into `claims_history`; injects
  `family_ytd_amount`; passes the same-day and family backstop parameters to
  `save()`. Component failures are never a 500.
- `GET /api/claims?limit=` → 200 recent rows.
- `GET /api/claims/{reference}` → 200 full stored result | 404.
- `GET /api/review-queue` → 200 claims held for review.
- `GET /api/claims/{reference}/documents/{file_id}` → 200 raw document bytes |
  404 | 500 if the stored payload is corrupt.
- `POST /api/claims/{reference}/resolve` → 200 resolved result | 400
  (out-of-range amount / amount required) | 404 | 409 (not awaiting review /
  already resolved). Body: `action` (approve | reject | close), `reviewer_id`,
  `reason`, optional `approved_amount`.
- `GET /api/policy` → 200 policy summary.
- `GET /api/health` → 200 `{"status": "ok"}`.
