"""Extraction agent.

Turns each document into structured fields. Two paths behind one interface:

- Injected content (`document.content` present): used by the test harness so
  the decision logic is tested deterministically with zero LLM calls.
- Live extraction (`document.file_data` present): a vision LLM extractor is
  invoked per document. The extractor is injected as a dependency so tests
  never make network calls and the orchestrator can degrade gracefully if it
  fails.

Extraction failures lower confidence and are recorded; they do not crash
the pipeline.
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Protocol

from app.agents.base import Agent, ClaimContext
from app.config import EngineConfig
from app.models.claim import SubmittedDocument
from app.models.decision import ComponentFailure, Decision, StepStatus

# Cap on concurrent per-document extraction calls (each is blocking network
# I/O). The pool is sized to min(documents, this) and joined before extraction
# returns, so the agent itself stays synchronous.
_MAX_EXTRACTION_WORKERS = 8


class DocumentExtractor(Protocol):
    """Contract for any live document extractor (e.g. Claude vision)."""

    def extract(self, document: SubmittedDocument) -> dict[str, Any]: ...


class ExtractionAgent(Agent):
    name = "extraction"

    def __init__(self, extractor: Optional[DocumentExtractor] = None,
                 config: Optional[EngineConfig] = None):
        self.extractor = extractor
        self.config = config or EngineConfig()

    def run(self, ctx: ClaimContext) -> None:
        docs = ctx.submission.documents
        cap = self.config.max_extraction_calls_per_claim

        # Pass 1 (cheap, sequential): plan the work. Collect the UNIQUE document
        # payloads needing a paid extraction, in first-seen order. Content docs
        # and content-hash duplicates cost nothing.
        digest_rep: dict[str, SubmittedDocument] = {}
        unique_digests: list[str] = []
        for doc in docs:
            if (doc.content is None and doc.file_data is not None
                    and self.extractor is not None):
                digest = hashlib.sha256(doc.file_data.encode()).hexdigest()
                if digest not in digest_rep:
                    digest_rep[digest] = doc
                    unique_digests.append(digest)

        # Cost-cap slice: pay for at most `cap` unique payloads. If more are
        # present, we extract the first `cap` (in parallel) and route the claim
        # to review for the remainder — i.e. we read all N-up-to-cap rather than
        # stopping mid-stream at the (cap+1)-th document.
        to_extract = unique_digests[:cap]
        over_cap = set(unique_digests[cap:])

        # Pass 2: extract the chosen unique payloads IN PARALLEL. Each call is
        # blocking network I/O; the engine itself stays synchronous (the pool is
        # joined before this method returns). _extract_one never raises.
        extracted: dict[str, dict] = {}
        if to_extract:
            with ThreadPoolExecutor(
                    max_workers=min(len(to_extract), _MAX_EXTRACTION_WORKERS)) as pool:
                for digest, result in zip(
                        to_extract,
                        pool.map(lambda dg: self._extract_one(digest_rep[dg]),
                                 to_extract)):
                    extracted[digest] = result

        # Per-claim token accounting: sum usage across the unique PAID calls
        # (dedupe reuse and the structured path cost nothing). Set before any
        # early return below so a cost-capped claim still reports what it spent.
        ctx.result.token_usage = {
            "input_tokens": sum(r["usage"]["input_tokens"]
                                for r in extracted.values()
                                if r.get("ok") and r.get("usage")),
            "output_tokens": sum(r["usage"]["output_tokens"]
                                 for r in extracted.values()
                                 if r.get("ok") and r.get("usage")),
            "calls": sum(1 for r in extracted.values()
                         if r.get("ok") and r.get("usage")),
        }

        # Truncation guard: a forced tool call normally stops for "tool_use"; a
        # "max_tokens" stop means the output was cut off and tail line items may
        # be missing, so the bill total cannot be trusted. Do NOT auto-decide on
        # an incomplete extraction — route the claim to review (the same path the
        # cost cap below uses); never retry with a larger limit.
        truncated_ids = [
            digest_rep[d].file_id for d, r in extracted.items()
            if r.get("ok") and r.get("truncated")
        ]
        if truncated_ids:
            names = ", ".join(truncated_ids)
            ctx.result.component_failures.append(ComponentFailure(
                component="extraction.truncated",
                error="Extraction response hit the output token limit "
                      "(stop_reason max_tokens) and was truncated.",
                impact=("The extracted fields are incomplete (tail line items "
                        "may be missing), so the claim total cannot be trusted."),
            ))
            ctx.result.decision = Decision.MANUAL_REVIEW
            ctx.result.manual_review_recommended = True
            ctx.result.reasons.append(
                f"The extraction for document(s) {names} hit the model's output "
                "limit and was truncated, so line items may be missing. A "
                "reviewer will extract the document(s) manually before the claim "
                "is decided.")
            ctx.halted = True
            ctx.trace(self.name, "truncated", StepStatus.FAILED,
                      f"Extraction truncated (stop_reason max_tokens) for {names}; "
                      "the output limit was hit and tail line items may be "
                      "missing. Routed to MANUAL_REVIEW for manual extraction.",
                      {"truncated_files": truncated_ids})
            return

        # Pass 3 (sequential assembly, in document order): every file_id gets its
        # own extracted_documents entry; an exact duplicate reuses the single
        # extraction with no additional read.
        per_doc_confidence: list[float] = []
        emitted: set[str] = set()
        for doc in docs:
            doc_label = doc.actual_type.value if doc.actual_type else "UNKNOWN"

            if doc.content is not None:
                ctx.extracted_documents.append({
                    "file_id": doc.file_id, "type": doc_label,
                    "fields": doc.content,
                })
                per_doc_confidence.append(1.0)
                ctx.trace(self.name, f"extract:{doc.file_id}", StepStatus.PASSED,
                          f"Structured content available for {doc_label} "
                          f"({doc.file_id}); fields: "
                          f"{', '.join(sorted(doc.content.keys()))}.")
                continue

            if doc.file_data is not None and self.extractor is not None:
                digest = hashlib.sha256(doc.file_data.encode()).hexdigest()
                if digest in over_cap:
                    continue  # beyond the cost cap; routed to review after loop
                res = extracted.get(digest)
                if res is None:
                    continue  # defensive; should not occur
                source = digest_rep[digest]
                if digest in emitted:
                    if res["ok"]:
                        ctx.extracted_documents.append({
                            "file_id": doc.file_id, "type": doc_label,
                            "fields": dict(res["fields"]),
                        })
                        per_doc_confidence.append(res["confidence"])
                        ctx.trace(self.name, f"extract:{doc.file_id}",
                                  StepStatus.INFO,
                                  f"{doc_label} ({doc.file_id}) is an exact "
                                  f"duplicate of {source.file_id} (same content "
                                  "hash); reused its extraction, no additional "
                                  "read.")
                    else:
                        per_doc_confidence.append(0.0)
                    continue
                emitted.add(digest)
                if res["ok"]:
                    ctx.extracted_documents.append({
                        "file_id": doc.file_id, "type": doc_label,
                        "fields": res["fields"],
                    })
                    per_doc_confidence.append(res["confidence"])
                    ctx.trace(self.name, f"extract:{doc.file_id}",
                              StepStatus.PASSED,
                              f"Vision extraction completed for {doc_label} "
                              f"({doc.file_id}) with confidence "
                              f"{res['confidence']:.2f}.")
                else:
                    per_doc_confidence.append(0.0)
                    ctx.result.component_failures.append(ComponentFailure(
                        component=f"extraction:{doc.file_id}",
                        error=res["error"],
                        impact="Document fields unavailable; downstream checks "
                               "that depend on them were skipped or degraded.",
                    ))
                    ctx.trace(self.name, f"extract:{doc.file_id}",
                              StepStatus.DEGRADED,
                              f"Vision extraction failed for {doc_label} "
                              f"({doc.file_id}): {res['error']}. Pipeline "
                              "continued without these fields.")
                continue

            # No content and no extractable file.
            per_doc_confidence.append(0.5)
            ctx.trace(self.name, f"extract:{doc.file_id}", StepStatus.SKIPPED,
                      f"No structured content or file data for {doc_label} "
                      f"({doc.file_id}); nothing to extract.")

        # Per-claim AI cost cap (paid calls only; dedupe reuse is free). More
        # unique payloads than the cap -> the extras were not read; route to
        # review, never a silent drop.
        if over_cap:
            remaining = sum(
                1 for doc in docs
                if doc.content is None and doc.file_data is not None
                and self.extractor is not None
                and hashlib.sha256(doc.file_data.encode()).hexdigest() in over_cap)
            ctx.result.component_failures.append(ComponentFailure(
                component="extraction.cost_cap",
                error=f"Per-claim extraction call cap reached ({cap}).",
                impact=(f"{remaining} document(s) were not processed; this claim "
                        "has too many documents to process automatically."),
            ))
            ctx.result.decision = Decision.MANUAL_REVIEW
            ctx.result.manual_review_recommended = True
            ctx.result.reasons.append(
                "This claim has too many documents to process automatically; a "
                "reviewer will handle it.")
            ctx.halted = True
            ctx.trace(self.name, "cost_cap", StepStatus.FAILED,
                      f"Extraction call cap ({cap}) reached; {remaining} "
                      "document(s) unprocessed. Routed to MANUAL_REVIEW.")
            return

        if per_doc_confidence:
            ctx.extraction_confidence = sum(per_doc_confidence) / len(per_doc_confidence)
        ctx.trace(self.name, "summary", StepStatus.INFO,
                  f"Extracted {len(ctx.extracted_documents)} of "
                  f"{len(ctx.submission.documents)} documents; extraction confidence "
                  f"{ctx.extraction_confidence:.2f}.")

    def _extract_one(self, doc: SubmittedDocument) -> dict:
        """Run one blocking extraction; never raises — an error becomes a result
        the caller turns into degradation (so a failure in one parallel task
        cannot crash the pool or the pipeline)."""
        try:
            fields = self.extractor.extract(doc)
            # Pull the transient signals out of the fields (token usage and the
            # truncation flag) so they are handled per claim but never persisted
            # in the extracted record.
            is_dict = isinstance(fields, dict)
            usage = fields.pop("_usage", None) if is_dict else None
            truncated = bool(fields.pop("_truncated", False)) if is_dict else False
            return {"ok": True, "fields": fields, "usage": usage,
                    "truncated": truncated,
                    "confidence": float(fields.get("_extraction_confidence", 0.9))}
        except Exception as e:  # noqa: BLE001 — degrade, never crash
            return {"ok": False, "error": str(e)}
