# Architecture — Health Insurance Claims Processing System

> This document is kept in sync with the code. Every claim below is verifiable
> against `backend/app/`. Where a value is policy data it lives in
> `data/policy_terms.json`; where it is an operational knob it lives in
> `app/config.py` (`EngineConfig`).

## 1. What this system does

A member submits a claim (member ID, optional category, optional treatment date,
optional amount, one or more documents). The system verifies the documents,
extracts structured data from them, applies the policy rules, and returns one of
APPROVED, PARTIAL, REJECTED, or MANUAL_REVIEW with the approved amount,
per-reason explanations, a confidence score with its contributing factors, and a
complete audit trace. Document problems stop the claim early with a specific,
actionable message and no decision is made (`status = NEEDS_RESUBMISSION`,
`decision = null`).

## 2. The core design decision: deterministic adjudication, LLM only at the edge

Every policy decision and every rupee of arithmetic is computed by deterministic
code. The LLM (Claude vision) does exactly one job: reading messy document
images into structured fields. Mapping free diagnosis text to policy conditions
is also deterministic — whole-word keyword matching (`app/condition_mapping.py`)
over keyword tables loaded from `data/clinical_mappings.json`, against the
policy's small fixed condition set. The
extractor may additionally emit a `canonical_condition` (one of the policy's
condition names) with a confidence; adjudication prefers it when present and
confident and **falls back to the keyword dictionary** otherwise, so the
decision path is never blocked on the model.

Why this split:

- **Explainability.** The trace is the logic. It records every step that
  affects the outcome — each hold, rejection, and route to review, the money
  math, the fraud score, the document-gate result, and the final confidence
  gate — plus the save-time race backstops (§5). Passing gates follow a mixed
  convention: many emit an explicit PASSED step (e.g. eligibility, waiting
  periods, limits), while several are deliberately silent on pass — emitting a
  step only when they hold or reject — to keep the trace lean (e.g. diagnosis
  certainty, registered practitioner, session cap, high-value auto-review, the
  family-floater limit). A reviewer reconstructs any decision from the output
  alone, because nothing that changes the outcome is silent.
- **Exact math in an exact order.** Network discount before co-pay
  (₹4,500 → ₹3,600 → ₹3,240) is asserted to the rupee by the test cases.
- **Testability.** The 12 official cases are exact assertions that pass on every
  run with zero LLM calls and zero mocking of decision logic.
- **Domain fit.** Adjudication must be consistent and auditable; the policy is
  already a structured JSON rulebook, and executing structured rules is what
  code is for.
- **Cost and scale.** Decisions are instant and free; the only per-claim LLM
  cost is document reading, and only when raw images are submitted.

Anything the deterministic path cannot resolve confidently routes to
MANUAL_REVIEW, the human escape hatch.

## 3. Multi-agent architecture in plain Python

An orchestrator (`app/orchestrator.py`) coordinates **seven specialized agents**,
each implementing one contract: `run(ctx: ClaimContext) -> None` over a shared,
typed `ClaimContext` (`app/agents/base.py`). The seventh agent, adjudication, is
itself an ordered rules engine (§4) and is by far the largest component.

```
ClaimSubmission  (Pydantic-validated; caps on document count, file size, and
        │         media type give a free 422 before any paid extraction call)
   Orchestrator ── pure coordination: owns the trace, wraps every agent in
        │          try/except, runs the agents in order (the business holds now
        │          live in the engine — §4/§5)
  1. IntakeAgent             policy match, member roster, category coverage,
        │                    minimum amount (coverage/minimum deferred when the
        │                    value will be derived from the documents)
  2. ExtractionAgent         per document: injected structured `content` (test
        │                    path) or Claude vision (live path); content-hash
        │                    dedupe (identical bytes read once); parallel across
        │                    documents (ThreadPoolExecutor); per-document
        │                    confidence; per-claim paid-call cap → review.
        │                    Runs BEFORE the gate so later checks use what the
        │                    documents say, not what the form claims.
  3. CategoryResolutionAgent provided category honored byte-for-byte; otherwise
        │                    derived from procedural evidence (line items,
        │                    treatment, tests — never diagnosis-only or
        │                    hospital-name); genuinely ambiguous → ask the member
        │                    (CATEGORY_NEEDED), a fixable input gap, not a queue
        │                    item. Sets `category_was_derived`.
  4. DocumentVerificationAgent  THE EARLY GATE: required types, readability,
        │                    same-patient consistency — extracted values first,
        │                    declared hints as fallback. Any issue → status
        │                    NEEDS_RESUBMISSION, decision null, specific message,
        │                    `ctx.halted = True`.
  5. ClaimDerivationAgent    real-upload path: amount from the extracted bill
        │                    total, hospital/treatment date from the documents;
        │                    unknown money or ambiguous date → MANUAL_REVIEW;
        │                    the deferred minimum-amount check runs here.
  6. ConsistencyAgent        patient-vs-roster identity (different covered
        │                    person → review normal priority; off-roster →
        │                    review high priority; no names → SKIPPED +
        │                    `no_patient_names`) and a lenient category
        │                    cross-check; sets `derived_category`.
  7. AdjudicationAgent       the deterministic rules engine (order in §4); its
        │                    terminal holds — the identity advisory and the
        │                    per-field confidence gate (§5) — were relocated
        │                    here from the orchestrator
        │
   Orchestrator post-pipeline (coordination only — no business decisions):
        │   • fallback to MANUAL_REVIEW if no decision and not halted
        │   • uniform degradation policy (any component failure ⇒ review)
        │   • review lifecycle (MANUAL_REVIEW ⇒ review_status PENDING_REVIEW)
        │   • confidence score
        ▼
ClaimResult (decision, amounts, line items, breakdown, fraud signals + score,
             review priority, derived category, component failures, confidence
             + factors, full trace)
```

No agent framework (LangGraph, CrewAI) is used, deliberately: these are
deterministic fixed stages, not autonomous reasoners, so a framework's
orchestration machinery would add dependency risk and opacity while its core
value sits unused. The multi-agent property lives in the design — specialized
agents, typed contracts, one coordinator.

State shared on `ClaimContext`: `submission`, `policy`, `result`,
`extracted_documents`, `extraction_confidence`, `halted`, `no_patient_names`,
`category_was_derived`, and `deciding_fields` (the fields the deciding check read,
consumed by the confidence gate — §5).

## 4. Adjudication check order (the order is itself a rule)

This is the exact order in `AdjudicationAgent.run`. The first check that
resolves the claim (rejects or routes to review) reports the decisive reason.

1. **Amount sanity & reconciliation** — per document/line: amounts finite and
   within the sum-insured ceiling; bill total reconciled against the itemized
   sum (exact within ₹1; ≤20% divergence accepted as discounts/taxes; beyond →
   review). Not a fraud stop (see §5).
2. **Eligibility** — treatment date within the policy period; a treatment date
   after the submission date is impossible → review.
3. **Submission deadline** — if a real `received_date` is present, a claim
   received more than `submission_rules.deadline_days_from_treatment` days after
   treatment is REJECTED (`SUBMISSION_DEADLINE_PASSED`). Measured against
   `received_date` only, never the API's auto-stamped `submission_date` (§10).
4. **Diagnosis certainty** — if the extractor's `canonical_condition` carried a
   below-threshold confidence on a decision-critical diagnosis, hold for review
   rather than decide on an uncertain mapping.
5. **Exclusions (line-item level)** — before waiting periods, because an
   excluded condition is permanently not covered (a bariatric/obesity claim
   rejects EXCLUDED_CONDITION, not WAITING_PERIOD). Some excluded + some covered
   → PARTIAL with a per-item reason. Whitelist categories (dental/vision) route
   non-covered, non-excluded lines to review.
6. **Waiting periods** — initial 30-day, then condition-specific from join date
   (dependents inherit the primary's). Rejections state the exact eligible date.
   Matched on the primary diagnosis / canonical condition only.
7. **Pre-authorization** — global scan: any MRI/CT/PET line above its threshold
   in any category. Structured path with no pre-auth → REJECTED
   (PRE_AUTH_MISSING); live path or a present-but-unverifiable reference →
   MANUAL_REVIEW (no internal pre-auth registry yet, §5).
8. **Registered practitioner** — categories with
   `requires_registered_practitioner` (alternative medicine) must carry a
   non-empty practitioner registration, else review (present-only check).
9. **Session cap** — categories with `max_sessions_per_year`: if THIS claim
   alone names more sessions than the cap, review. Cross-year accumulation is
   deferred (§10); pass-through when no count is parseable.
10. **High-value auto-review** — a claim above
    `fraud_thresholds.auto_manual_review_above` is held before the limit math so
    a genuinely large claim is reviewed, not auto-rejected by the per-claim cap.
11. **Limits** — per-claim cap = max(per_claim_limit, category sub_limit) and
    annual OPD limit, applied to the **payable (post-exclusion) eligible** amount
    (see assumptions.md A2/A3); then the **family-floater combined limit**
    (`family_ytd_amount` + eligible > `family_floater.combined_limit` →
    FAMILY_LIMIT_EXCEEDED). On the derived multi-category path, limits and money
    run per category group; a provided category folds all lines into one group.
12. **Pharmacy drug-type certainty** — pharmacy only: a payable medicine line
    with unknown/low-confidence branded-vs-generic status is held (the per-line
    co-pay cannot be computed).
13. **Money math** — network discount first, then co-pay, with the full
    breakdown. Pharmacy applies co-pay **per line** (0 for generic,
    `branded_drug_copay_percent` for branded); every other category keeps a
    group-level co-pay.
14. **Fraud signals** — binary gates (same-day, monthly, high-value, byte
    duplicate) plus a transparent weighted **fraud score** (same-day frequency
    0.30, monthly 0.20, amount-vs-history 0.25, near-duplicate 0.25; routes at
    `fraud_score_manual_review_threshold`). Each signal's contribution is in the
    trace. Signals on a payable claim → MANUAL_REVIEW, never auto-reject. This
    component is individually wrapped; `simulate_component_failure` fails exactly
    it to demonstrate degradation.
15. **Finalize** — APPROVED / PARTIAL from the line items; fraud signals route a
    payable claim to MANUAL_REVIEW; records the deciding fields for the gate.

Steps 1–15 are the resolving gates and the money sequence (money math → fraud →
finalize runs only when no gate resolved the claim). Two terminal **holds**,
relocated here from the orchestrator, then always run — after a gate rejection
*or* after finalize — so a gate-rejected claim is still confidence-checked:

16. **Identity name hold** — a live-upload payable claim whose patient name could
    not be read is held at or above `no_name_high_value`, else passes with an
    advisory note; silent on the structured cases (§5).
17. **Confidence gate** — holds APPROVED/PARTIAL/REJECTED when a field the
    deciding check read was a low-confidence read; runs after finalize (it reads
    the deciding fields) and after the identity hold (§5).

## 5. Failure handling and holds

**Two containment levels.**
- *Orchestrator level.* Every agent runs inside try/except. An unexpected
  exception records a `ComponentFailure` (component, error, impact), appends a
  DEGRADED trace step, and the pipeline continues. A claim request never 500s
  because a pipeline component broke. (The one exception is a storage I/O error
  on persistence — the decision is computed but the write surfaces as a 500;
  §10.)
- *Component level inside adjudication.* The fraud checker is individually
  contained so its failure cannot take down a decision the rules already
  justified.

**Uniform degradation policy.** Any component failure anywhere lowers confidence
by a fixed penalty and sets `manual_review_recommended`. Extraction failures
additionally zero that document's extraction confidence, which propagates
multiplicatively. Confidence is a deterministic formula
(`base 0.95 × extraction_confidence − 0.25 if any component failed`, clamped to
[0.05, 0.95], with factors listed), never an LLM judgment.

**Per-field confidence gate (shipped).** After the decision is reached, the
engine holds it (APPROVED, PARTIAL, or REJECTED) when a field the **deciding
check actually read** was a low-confidence read. The gate is a terminal step in
adjudication (`ConfidenceGateStep`, after finalize), not the orchestrator. Each
terminal check records the fields it read in `ctx.deciding_fields` (a rejection
on a misread date is held the same as a misread payment); the gate checks the
per-field confidence of exactly those fields, falling back to the document-level
extraction confidence when a per-field value is absent. So an unrelated fuzzy
field no longer holds a confidently-read decision. Threshold 0.8
(`EngineConfig.confidence_threshold`). The gate acts on the live path only:
structured/test content carries no per-field or document confidence, so it is
silent and the 12 official cases are unaffected. (This replaces the earlier
document-level interim gate.)

**Identity hold on high-value unread-name claims (live path).** A real uploaded
claim whose patient name could not be read carries no identity verification.
At or above `EngineConfig.no_name_high_value` (₹2,500, half the per-claim cap)
the otherwise-payable claim is held; below it, it passes with an advisory trace
note. Structured cases provide fields directly, so a missing name there is a
fixture choice, never held on this basis. This rule is a terminal step in the
adjudication engine (`IdentityNameHoldStep`), relocated from the orchestrator;
it runs after the engine reaches its decision, before the confidence gate.

**Pre-authorization, fail-closed.** A present pre-auth reference no longer
auto-satisfies the requirement — without an internal registry it cannot be
verified, so it routes to review; on the live path a missing reference also
routes to review (it may simply not have extracted); on the structured path
absence is definitive REJECTED. The real fix is an internal pre-auth registry.

**Save-time race backstops (traced).** Two domain re-checks run inside the
`BEGIN IMMEDIATE` write transaction in `storage.save`, closing the
read-merge-write race for two concurrent same-family/same-member claims:
- *Same-day backstop:* if a concurrent insert pushed the member over the
  same-day limit since the engine read its snapshot, an otherwise-payable claim
  is overridden to MANUAL_REVIEW.
- *Family-floater backstop:* re-computes the family's approved year-to-date
  inside the transaction and re-applies the combined limit; a crossing claim is
  overridden to REJECTED (FAMILY_LIMIT_EXCEEDED).

Both **append a trace step** (`persistence` stage, `same_day_backstop` /
`family_floater_backstop`) recording the override and the
from/to decision, so the persisted decision is fully reconstructable from the
trace alone, not only from `reasons[]`. These fire only on the concurrency
edge; the 12 official cases never reach `save()` and are unaffected.

**Amount sanity and duplicate detection.** See §4.1 (sanity/reconciliation is a
misread/absurd-amount signal, not an anti-fraud guarantee — a *matched*
inflation is bounded by the per-claim cap, not by this check) and §4.14
(byte-identical full-document-set hash flags a duplicate resubmission of an
already-DECIDED claim).

## 6. The LLM integration, bounded

`ClaudeVisionExtractor` is the only network-touching component and sits behind a
one-method protocol (`DocumentExtractor.extract(document) -> dict`), so it is
injected, mockable, and replaceable; the whole system runs without an API key
(the 12 cases carry structured content). Properties:

- **Structured output enforced and validated**: a forced tool call against a
  JSON schema (no free-text parsing), then the raw tool result is validated and
  normalized by a Pydantic model (`ExtractedDocument`, `app/models/extraction.py`)
  inside the extractor before the engine sees it — confidences clamped to
  [0, 1], numeric fields (line-item amounts, total) coerced with a bad value
  turned into None rather than crashing a rule, the enum fields (document_type,
  readability, drug_type) checked against their allowed values, and unknown
  fields dropped. The structured-content path (the 12 cases) bypasses the live
  extractor, so it is unaffected.
- **Honest uncertainty**: null for illegible fields, an overall
  `extraction_confidence`, and per-field confidences (amounts, canonical
  condition, treatment date, patient name, hospital, category) that feed the
  gate and the confidence formula.
- **Bounded failure**: hard timeout, bounded retries (4xx≠429 not retried), and
  typed `ExtractionError`; the agent converts failures into degradation.
- **Determinism**: `temperature=0`; provider-side nondeterminism remains, and
  the scheduled real fix is dual-pass extraction with reconciliation plus
  content-hash caching across claims.

Extractor fields (the contract the engine consumes): `document_type` (enum),
`patient_name`, `doctor_name`, `doctor_registration`, `hospital_name`, `date`,
`diagnosis`, `primary_diagnosis`, `comorbidities`, `canonical_condition` (+
`canonical_condition_confidence`), `treatment`, `medicines[]`,
`line_items[{description, amount, drug_type, drug_type_confidence}]`, `total`,
`amount_confidence`, `treatment_date_confidence`, `patient_name_confidence`,
`hospital_confidence`, `category_confidence`, `pre_auth_number`, `readability`,
`extraction_confidence`. The condition vocabulary the model maps
`canonical_condition` to is read from `policy_terms.json` at extractor
construction.

## 7. Storage and API

- **SQLite** (`app/storage.py`) stores every processed claim with its full
  result JSON and serves member history (merged into new submissions so fraud
  checks see prior claims) and the **family-floater year-to-date** (summed
  across the family for the policy year and injected into the submission — the
  engine never reads storage itself). `save()` also hosts the two race backstops
  (§5) under `BEGIN IMMEDIATE`, and a review lifecycle (`resolve`). Public
  surface (10 methods): `save`, `save_documents`, `get`, `get_document`,
  `list_recent`, `list_held`, `resolve`, `find_decided_by_hash`,
  `member_history`, `family_ytd_approved`.
- **FastAPI** (`app/api.py`) is a thin shell over the framework-free engine
  (the engine never imports FastAPI). Endpoints: `POST /api/claims`,
  `GET /api/claims`, `GET /api/claims/{ref}`, `GET /api/review-queue`,
  `GET /api/claims/{ref}/documents/{file_id}`,
  `POST /api/claims/{ref}/resolve`, `GET /api/policy`, `GET /api/health`.
  Pydantic request/response models double as the published contracts and give
  422s on malformed input. The built React frontend is served from the same
  service — one URL, one deploy.

## 8. Tradeoffs considered and rejected

- **LLM-adjudicated decisions** — rejected for auditability, exact math, test
  determinism, and consistency.
- **Agent frameworks** — rejected; fixed deterministic stages get no value from
  dynamic orchestration and lose transparency.
- **LLM diagnosis mapping as primary** — rejected; the policy's condition set is
  small and fixed. `canonical_condition` is an *assist* with a deterministic
  keyword fallback and a confidence gate, not the source of truth.
- **Postgres now** — rejected as premature at this scale; see §9 for the (now
  non-trivial) swap.

## 9. Scaling to 10x (75,000 → 750,000 claims/year)

~2,000–3,000 claims/day, peak tens per minute. The architecture holds; the
deployment changes:

1. **Separate the extraction path.** Document extraction is the only slow, costly
   stage. Move it to a queue (claim accepted → workers extract → adjudication
   fires on completion); adjudication stays synchronous and instant. This also
   removes the current limit that the **synchronous** `POST /api/claims` holds a
   worker thread for the whole (already parallelized but still blocking)
   extraction.
2. **Postgres behind the storage layer** for concurrent writers plus read
   replicas. Note this is **not a mechanical drop-in today**: `save()` embeds two
   race-safe domain backstops and `BEGIN IMMEDIATE` transaction semantics, so the
   port must reproduce that behavior (or, better, the backstops move into a
   dedicated commit stage and storage becomes a pure repository — a clean-up
   worth doing as part of the swap). The single-writer lock held across
   SELECT-then-INSERT is the real throughput ceiling.
3. **Stateless API replicas** behind a load balancer; the engine holds no state.
   Policy is read once per process and cacheable.
4. **Idempotency keys on submission** so retries never double-process.
5. **Extraction cost control**: cache extractions by document content hash (the
   per-claim dedupe already exists), batch where possible, route typed PDFs to
   cheaper OCR and reserve vision for messy images.
6. **Fraud checks on indexed history**: same-day/monthly counts and the family
   rollup become indexed SQL aggregates; thresholds already live in policy JSON.
7. **Observability at fleet scale**: traces are structured objects; ship them to
   a log store keyed by claim reference; add metrics on decision mix, confidence
   distribution, degradation rate, and extraction-confidence drift (the
   early-warning signal for document-quality or model regressions).

Keeping adjudication deterministic is what makes 10x cheap: the expensive part
scales with documents, not decisions.

## 10. Known limitations (deliberate, documented)

- **`ytd_claims_amount` is trusted from the submission** (defaults to 0). The
  family-floater year-to-date, by contrast, *is* computed server-side from
  stored approved claims; the same fix for the per-member annual YTD is the
  obvious next step (compute it server-side when absent).
- **The submission deadline needs a real `received_date`.** It is enforced
  against `received_date` (which a production intake supplies), never against the
  API's auto-stamped `submission_date` — stamping "today" would retroactively
  fail historical/demo data. Absent a `received_date`, the check is skipped
  (assumptions.md A11).
- **`save()` makes two decision overrides (the save-time race backstops).** They
  run only in `save()` (the API/storage path): a caller invoking the engine
  directly without `save()` does not get the concurrency re-check. This is by
  design (the race exists only at persistence), but it means the storage layer
  can override the engine's decision — same-day → MANUAL_REVIEW, family-floater →
  REJECTED (FAMILY_LIMIT_EXCEEDED) — so the engine alone is not the sole source
  of the *final* persisted decision. The override is fully traced (§5); the named
  clean fix (§9) is to move the backstops into a dedicated commit stage so
  storage becomes a pure repository.
- **`simulate_component_failure` is a public API field**, needed for TC011; in
  production it would be gated to non-prod.
- **No authentication/authorization/rate limiting.** An API-gateway concern;
  input caps bound per-request extraction cost in the meantime. CORS is open
  (`*`) for the demo.
- **Document content is untrusted model input** (prompt injection). Blast radius
  is bounded: extraction is a forced tool call, and every downstream decision is
  deterministic — a manipulated extracted value has the same power as a false
  value printed on a paper bill, which the fraud/consistency checks treat alike.
- **Provider identity is not authenticated** (a bill with no provider merely
  gets no network discount). The fix is a validation finding that lowers
  confidence and routes to a human.
- **Dependents without roster rows (DEP003–DEP006)** have no name/DOB in
  `policy_terms.json`, so their claims cannot be name-verified and route to
  manual review; the fix is upstream data completeness, never fabricated rows.
