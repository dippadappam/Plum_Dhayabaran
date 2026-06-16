"""Base agent contract.

Every agent implements run(context) -> None, reading and writing a shared
ClaimContext. Agents raise typed AgentError for expected failures; the
orchestrator wraps every agent for graceful degradation, so an unexpected
exception degrades the pipeline instead of crashing it.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from app.models.claim import ClaimSubmission
from app.models.decision import (
    ClaimResult,
    StepStatus,
    TraceStep,
)
from app.models.policy import Policy


class AgentError(Exception):
    """Expected, typed agent failure. `code` is machine-readable."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(message)


@dataclass
class ClaimContext:
    """Shared state passed through the pipeline. Agents read what previous
    stages produced and append their own outputs and trace steps."""

    submission: ClaimSubmission
    policy: Policy
    result: ClaimResult
    extracted_documents: list[dict[str, Any]] = field(default_factory=list)
    extraction_confidence: float = 1.0
    # The member's ORIGINAL stated claim amount, captured before the derivation
    # agent may overwrite submission.claimed_amount with the bill total on the
    # real-upload path. The claimed-vs-bill cross-check reads this so it compares
    # the member's number, not the bill against itself. None when none was stated.
    original_claimed_amount: Optional[float] = None
    halted: bool = False  # set when a stage stops the pipeline (document gate)
    # Set by the consistency agent when no document carried a patient name:
    # the identity check could not run. The orchestrator turns this into a
    # manual-review advisory on otherwise-payable decisions.
    no_patient_names: bool = False
    # Set by the CategoryResolutionAgent when it DERIVED the category (no
    # category was provided in the submission). Per-line categorization (Batch
    # 6b) runs only on this derived path; a provided category folds every line
    # into that one category group, exactly as before 6b.
    category_was_derived: bool = False
    # The set of logical extracted fields the DECIDING check actually read
    # (Fix 7). Each terminal check records what it read at the point it resolves
    # the claim; the confidence gate then holds only when one of THESE fields was
    # a low-confidence read, instead of holding on any unrelated fuzzy field.
    deciding_fields: set[str] = field(default_factory=set)

    def trace(
        self,
        stage: str,
        check: str,
        status: StepStatus,
        detail: str,
        data: Optional[dict[str, Any]] = None,
    ) -> None:
        self.result.trace.append(
            TraceStep(stage=stage, check=check, status=status, detail=detail, data=data or {})
        )


class Agent(ABC):
    """Single-responsibility pipeline stage."""

    name: str = "agent"

    @abstractmethod
    def run(self, ctx: ClaimContext) -> None: ...
