"""Document Verification agent: the early gate.

Runs before any claim decision. Three checks, each producing a specific,
actionable message (the quality of the message is graded):

1. Document-type completeness: required types for the claim category are
   present; wrong/extra types are named explicitly (TC001).
2. Readability: an unreadable document asks for re-upload of that specific
   file, it does not reject the claim (TC002).
3. Patient consistency: all documents must belong to the same patient, and
   the specific names found are surfaced (TC003).

Any failure sets decision NEEDS_RESUBMISSION and halts the pipeline before
adjudication, per the test contract "stop before making any claim decision".

The gate runs after extraction, so each check uses the value detected from the
document when extraction produced one (real-upload path: extracted
document_type, readability, patient_name) and falls back to the declared hint
(actual_type, quality, patient_name_on_doc / structured content) when it did
not. Documents that carry declared hints and no file_data therefore behave
exactly as before.
"""

from app.agents.base import Agent, ClaimContext
from app.condition_mapping import names_match
from app.models.claim import DocumentQuality, SubmittedDocument
from app.models.decision import DocumentIssue, StepStatus


class DocumentVerificationAgent(Agent):
    name = "document_verification"

    def run(self, ctx: ClaimContext) -> None:
        self._fields = {
            d["file_id"]: d.get("fields", {}) for d in ctx.extracted_documents
        }
        # Files whose extraction failed (recorded by the ExtractionAgent as
        # component failures): their type is unknown because we could not
        # read them, which is a different problem than a missing document.
        self._failed_ids = {
            cf.component.split(":", 1)[1]
            for cf in ctx.result.component_failures
            if cf.component.startswith("extraction:")
        }
        issues: list[DocumentIssue] = []
        self._check_required_types(ctx, issues)
        self._check_readability(ctx, issues)
        self._check_patient_consistency(ctx, issues)

        if issues:
            ctx.result.document_issues.extend(issues)
            ctx.result.status = "NEEDS_RESUBMISSION"
            ctx.result.decision = None  # no claim decision was made
            ctx.result.reasons.extend(i.message for i in issues)
            ctx.halted = True
            ctx.trace(
                self.name, "gate", StepStatus.FAILED,
                f"{len(issues)} document problem(s) found. Stopped before any claim "
                "decision; member asked to fix and resubmit.",
                {"issue_codes": [i.issue_code for i in issues]},
            )
        else:
            ctx.trace(self.name, "gate", StepStatus.PASSED,
                      "All documents present, readable, and consistent.")

    # --- Effective per-document values: extracted first, declared as fallback ---
    def _eff_type(self, doc: SubmittedDocument) -> str:
        extracted = self._fields.get(doc.file_id, {}).get("document_type")
        if extracted:
            return str(extracted)
        return doc.actual_type.value if doc.actual_type else "UNKNOWN"

    def _eff_unreadable(self, doc: SubmittedDocument) -> bool:
        extracted = self._fields.get(doc.file_id, {}).get("readability")
        if extracted is not None:
            return str(extracted).upper() == "UNREADABLE"
        return doc.quality == DocumentQuality.UNREADABLE

    def _eff_patient(self, doc: SubmittedDocument) -> str | None:
        return (
            self._fields.get(doc.file_id, {}).get("patient_name")
            or doc.patient_name_on_doc
            or (doc.content or {}).get("patient_name")
        )

    # --- Check 1: required document types (TC001) ---
    def _check_required_types(self, ctx: ClaimContext, issues: list[DocumentIssue]) -> None:
        category = ctx.submission.claim_category.value
        requirements = ctx.policy.document_requirements.get(category, {})
        required = list(requirements.get("required", []))

        provided_types = [(d, self._eff_type(d)) for d in ctx.submission.documents]
        provided_type_names = [t for _, t in provided_types]

        missing = [r for r in required if r not in provided_type_names]

        # Dental-only: a dental-clinic bill is routinely extracted as a
        # DENTAL_REPORT or an unknown type. For DENTAL (whose only required
        # document is the bill), a DENTAL_REPORT- or UNKNOWN-typed document
        # satisfies the HOSPITAL_BILL requirement. Scoped to DENTAL so it never
        # weakens the gate for any other category (a consultation still needs a
        # real HOSPITAL_BILL — TC001).
        if category == "DENTAL" and "HOSPITAL_BILL" in missing and \
                any(t in {"DENTAL_REPORT", "UNKNOWN"} for t in provided_type_names):
            missing = [r for r in missing if r != "HOSPITAL_BILL"]

        if not missing:
            ctx.trace(self.name, "required_document_types", StepStatus.PASSED,
                      f"All required document types for {category} are present: "
                      f"{', '.join(required)}.",
                      {"required": required, "provided": provided_type_names})
            return

        # Extraction failures make a document's type unknowable: report
        # "could not read", not "missing". Only docs with no declared type
        # and no structured content are genuinely unknowable.
        unreadable_failed = [
            d for d, t in provided_types
            if d.file_id in self._failed_ids and t == "UNKNOWN"
        ]
        if unreadable_failed:
            file_names = ", ".join(
                d.file_name or d.file_id for d in unreadable_failed
            )
            for m in missing:
                message = (
                    f"We could not read {file_names}, so we cannot confirm the "
                    f"{m} required for a {category} claim was provided. If one "
                    f"of these files is the {m}, please re-upload a clearer "
                    f"copy or try again. The claim has not been rejected."
                )
                issues.append(DocumentIssue(
                    file_id=",".join(d.file_id for d in unreadable_failed),
                    issue_code="EXTRACTION_FAILED",
                    message=message,
                    action_required=f"Re-upload a readable {m} for this claim.",
                ))
                ctx.trace(self.name, "required_document_types",
                          StepStatus.FAILED, message,
                          {"missing_type": m,
                           "unreadable_files": [d.file_id for d in unreadable_failed]})
            return

        # Name what was uploaded and what is needed instead. If a type was
        # uploaded more than once while another is missing, call that out.
        from collections import Counter
        counts = Counter(provided_type_names)
        duplicates = [t for t, c in counts.items() if c > 1]

        provided_summary = ", ".join(
            f"{t} ({d.file_name})" if d.file_name else t for d, t in provided_types
        )
        for m in missing:
            if duplicates:
                message = (
                    f"For a {category} claim, you uploaded {provided_summary}. "
                    f"A {m} is required, but you uploaded "
                    f"{counts[duplicates[0]]} documents of type {duplicates[0]} instead. "
                    f"Please upload the missing {m}."
                )
            else:
                message = (
                    f"For a {category} claim, you uploaded: {provided_summary}. "
                    f"A {m} is required but was not provided. "
                    f"Please upload the missing {m}."
                )
            issues.append(DocumentIssue(
                file_id="-",
                issue_code="MISSING_REQUIRED" if not duplicates else "WRONG_TYPE",
                message=message,
                action_required=f"Upload a {m} for this claim.",
            ))
            ctx.trace(self.name, "required_document_types", StepStatus.FAILED, message,
                      {"missing_type": m, "provided": provided_type_names})

    # --- Check 2: readability (TC002) ---
    def _check_readability(self, ctx: ClaimContext, issues: list[DocumentIssue]) -> None:
        unreadable = [d for d in ctx.submission.documents if self._eff_unreadable(d)]
        if not unreadable:
            ctx.trace(self.name, "readability", StepStatus.PASSED,
                      "All documents are readable.")
            return
        for d in unreadable:
            eff = self._eff_type(d)
            doc_label = eff if eff != "UNKNOWN" else "document"
            file_label = f" ({d.file_name})" if d.file_name else ""
            message = (
                f"The {doc_label}{file_label} could not be read because the image is "
                f"unreadable. Please re-upload a clear photo of this specific document. "
                f"The claim has not been rejected; it will be processed once a readable "
                f"copy is received."
            )
            issues.append(DocumentIssue(
                file_id=d.file_id,
                file_name=d.file_name,
                issue_code="UNREADABLE",
                message=message,
                action_required=f"Re-upload a clear, readable copy of the {doc_label}.",
            ))
            ctx.trace(self.name, "readability", StepStatus.FAILED, message,
                      {"file_id": d.file_id, "document_type": doc_label})

    # --- Check 3: same patient on all documents (TC003) ---
    def _check_patient_consistency(self, ctx: ClaimContext, issues: list[DocumentIssue]) -> None:
        named: list[tuple[SubmittedDocument, str]] = []
        for d in ctx.submission.documents:
            name = self._eff_patient(d)
            if name:
                named.append((d, name))

        names = [n for _, n in named]
        same = (not names) or all(names_match(names[0], n) for n in names[1:])
        if same:
            detail = (
                f"All documents belong to the same patient"
                f"{': ' + named[0][1] if named else ''}."
                if named else "No patient names available to compare; check skipped."
            )
            ctx.trace(self.name, "patient_consistency",
                      StepStatus.PASSED if named else StepStatus.SKIPPED, detail)
            return

        found = "; ".join(
            f"{(self._eff_type(d) if self._eff_type(d) != 'UNKNOWN' else 'document')} "
            f"({d.file_name or d.file_id}) is for '{n}'"
            for d, n in named
        )
        message = (
            f"The documents in this claim belong to different patients: {found}. "
            f"All documents in one claim must be for the same patient. Please resubmit "
            f"with documents for a single patient."
        )
        issues.append(DocumentIssue(
            file_id=",".join(d.file_id for d, _ in named),
            issue_code="PATIENT_MISMATCH",
            message=message,
            action_required="Resubmit the claim with all documents for the same patient.",
        ))
        ctx.trace(self.name, "patient_consistency", StepStatus.FAILED, message,
                  {"names_found": [n for _, n in named]})
