"""Task-linked trace context, structured evaluation, and feedback types."""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterator, Protocol
from uuid import UUID

from skysun.knowledge import KnowledgeSearchResult
from skysun.knowledge_grounding import GroundingStatus
from skysun.user_identity import owner_id
from skysun.verification import VerificationResult, VerificationStatus


class EvaluationStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


class FeedbackRating(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


@dataclass(frozen=True)
class TraceRecord:
    trace_id: UUID
    task_id: UUID
    user_id: UUID
    stage: str
    status: str
    started_at: datetime
    finished_at: datetime | None
    duration_ms: float | None
    metadata: dict


@dataclass(frozen=True)
class EvaluationResult:
    task_id: UUID
    user_id: UUID
    score: int
    status: EvaluationStatus
    reasons: tuple[str, ...]
    grounding_correct: bool | None
    complete: bool | None
    unsupported_claims: tuple[str, ...]
    created_at: datetime | None = None


@dataclass(frozen=True)
class FeedbackRecord:
    feedback_id: UUID
    task_id: UUID
    user_id: UUID
    rating: FeedbackRating
    comment: str | None
    response_hash: str
    created_at: datetime
    updated_at: datetime


class ClaimReviewer(Protocol):
    def review(self, goal: str, answer: str,
               sources: tuple[KnowledgeSearchResult, ...]) -> tuple[bool, tuple[str, ...]]: ...


class ObservabilityStore(Protocol):
    def start_trace(self, task_id: UUID, stage: str, metadata: dict | None = None) -> UUID: ...
    def finish_trace(self, trace_id: UUID, status: str, metadata: dict | None = None) -> None: ...
    def save_evaluation(self, result: EvaluationResult) -> EvaluationResult: ...


@dataclass
class Span:
    status: str = "SUCCESS"
    metadata: dict = field(default_factory=dict)


_active: ContextVar[tuple[ObservabilityStore, UUID] | None] = ContextVar(
    "skysun_observability", default=None
)


@contextmanager
def trace_context(store: ObservabilityStore, task_id: UUID) -> Iterator[None]:
    token = _active.set((store, task_id))
    try:
        yield
    finally:
        _active.reset(token)


@contextmanager
def span(stage: str, metadata: dict | None = None) -> Iterator[Span]:
    active = _active.get()
    trace_id = active[0].start_trace(active[1], stage, metadata) if active else None
    handle = Span()
    try:
        yield handle
    except Exception as error:
        handle.status = "FAILED"
        handle.metadata["error_code"] = type(error).__name__
        raise
    finally:
        if active and trace_id:
            active[0].finish_trace(trace_id, handle.status, handle.metadata)


def trace_event(stage: str, status: str = "SUCCESS", metadata: dict | None = None) -> None:
    with span(stage, metadata) as handle:
        handle.status = status


class ResponseEvaluator:
    """Bounded evidence checks; uncertain model reviews become REVIEW, never PASS."""

    def __init__(self, reviewer: ClaimReviewer | None = None) -> None:
        self._reviewer = reviewer

    def evaluate(
        self, task_id: UUID, goal: str, answer: str,
        *, verification: VerificationResult | None = None,
        grounding_status: GroundingStatus | None = None,
        sources: tuple[KnowledgeSearchResult, ...] = (),
        tools_used: tuple[str, ...] = (),
        tool_failures: tuple[str, ...] = (),
    ) -> EvaluationResult:
        reasons: list[str] = []
        searched = "search_knowledge" in tools_used
        grounded = (grounding_status is GroundingStatus.SUFFICIENT and bool(sources)) if searched else None
        complete = (verification.status is VerificationStatus.SUCCESS) if verification else None
        if tool_failures or (verification and verification.status is VerificationStatus.FAILED):
            reasons.append("tool_or_workflow_failure")
            return EvaluationResult(task_id, owner_id(), 0, EvaluationStatus.FAIL,
                                    tuple(reasons), grounded, complete, ())
        if searched and not grounded:
            reasons.append("knowledge_evidence_insufficient")
            return EvaluationResult(task_id, owner_id(), 20, EvaluationStatus.FAIL,
                                    tuple(reasons), False, complete, ())
        if verification and verification.status is VerificationStatus.NEEDS_REVIEW:
            reasons.append("workflow_verification_needs_review")
            return EvaluationResult(task_id, owner_id(), 50, EvaluationStatus.REVIEW,
                                    tuple(reasons), grounded, False, ())
        if not answer.strip():
            return EvaluationResult(task_id, owner_id(), 0, EvaluationStatus.FAIL,
                                    ("empty_answer",), grounded, False, ())
        if sources and self._reviewer:
            try:
                complete_review, unsupported = self._reviewer.review(goal, answer, sources)
                complete = bool(complete_review) if complete is None else complete and complete_review
                if unsupported:
                    return EvaluationResult(task_id, owner_id(), 30, EvaluationStatus.FAIL,
                                            ("unsupported_claims_detected",), grounded, complete,
                                            tuple(unsupported)[:5])
                if not complete:
                    return EvaluationResult(task_id, owner_id(), 60, EvaluationStatus.REVIEW,
                                            ("answer_incomplete",), grounded, False, ())
                return EvaluationResult(task_id, owner_id(), 100, EvaluationStatus.PASS,
                                        ("grounded_and_complete",), grounded, True, ())
            except Exception:
                reasons.append("claim_review_unavailable")
        else:
            reasons.append("claim_review_unavailable")
        return EvaluationResult(task_id, owner_id(), 70, EvaluationStatus.REVIEW,
                                tuple(reasons), grounded, complete, ())
