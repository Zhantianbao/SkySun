"""Task-linked trace context, structured evaluation, and feedback types."""
# 本文件负责运行观测与评估中的 observability 相关流程。

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Iterator, Protocol
from uuid import UUID

from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.user.user_identity import owner_id
from skysun.workflow.verification import VerificationResult, VerificationStatus


# 以下集中定义 EvaluationStatus：数据结构或组件相关的数据和操作。
class EvaluationStatus(str, Enum):
    PASS = "PASS"
    REVIEW = "REVIEW"
    FAIL = "FAIL"


# 以下集中定义 FeedbackRating：数据结构或组件相关的数据和操作。
class FeedbackRating(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"


# 以下集中定义 TraceRecord：数据结构或组件相关的数据和操作。
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


# 以下集中定义 EvaluationResult：数据结构或组件相关的数据和操作。
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


# 以下集中定义 FeedbackRecord：数据结构或组件相关的数据和操作。
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


# 以下集中定义 ClaimReviewer：数据结构或组件相关的数据和操作。
class ClaimReviewer(Protocol):
    # 调用位置：该协议由 ClaimReviewer 的实现类调用；入口涉及工作流 Agent 和管理接口。
    def review(self, goal: str, answer: str,
               sources: tuple[KnowledgeSearchResult, ...]) -> tuple[bool, tuple[str, ...]]: """检查回复中的主张是否受到证据支持。"""; ...


# 以下集中定义 ObservabilityStore：存储层相关的数据和操作。
class ObservabilityStore(Protocol):
    # 调用位置：轨迹上下文管理器调用观测存储的实现方法。
    def start_trace(self, task_id: UUID, stage: str, metadata: dict | None = None) -> UUID: """开始记录任务的执行轨迹。"""; ...
    # 调用位置：轨迹上下文管理器在阶段结束时调用。
    def finish_trace(self, trace_id: UUID, status: str, metadata: dict | None = None) -> None: """结束轨迹并保存最终状态。"""; ...
    # 调用位置：工作流智能代理在回答评估后调用。
    def save_evaluation(self, result: EvaluationResult) -> EvaluationResult: """保存回复质量的评估结果。"""; ...


# 以下集中定义 Span：数据结构或组件相关的数据和操作。
@dataclass
class Span:
    status: str = "SUCCESS"
    metadata: dict = field(default_factory=dict)


_active: ContextVar[tuple[ObservabilityStore, UUID] | None] = ContextVar(
    "skysun_observability", default=None
)


# 调用位置：src/skysun/workflow/workflow_agent.py 的 WorkflowAgent._reply_direct_observed、src/skysun/workflow/workflow_agent.py 的 WorkflowAgent.execute_workflow、src/skysun/workflow/workflow_agent.py 的 WorkflowAgent.resume_task 等处调用。
@contextmanager
def trace_context(store: ObservabilityStore, task_id: UUID) -> Iterator[None]:
    """建立当前任务的轨迹上下文。"""
    token = _active.set((store, task_id))
    try:
        yield
    finally:
        _active.reset(token)


# 调用位置：src/skysun/evaluation/observability.py 的 trace_event、src/skysun/tool/tool_reliability.py 的 run_tool、src/skysun/workflow/workflow_agent.py 的 WorkflowAgent._reply_direct_observed 等处调用。
@contextmanager
def span(stage: str, metadata: dict | None = None) -> Iterator[Span]:
    """记录一个执行阶段的时间和状态。"""
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


# 调用位置：src/skysun/tool/tool_reliability.py 的 configure_validation.handle、src/skysun/workflow/workflow_agent.py 的 WorkflowAgent._reply_direct_observed、src/skysun/workflow/workflow_agent.py 的 WorkflowAgent.execute_workflow 等处调用。
def trace_event(stage: str, status: str = "SUCCESS", metadata: dict | None = None) -> None:
    """在当前轨迹中追加事件。"""
    with span(stage, metadata) as handle:
        handle.status = status


# 以下集中定义 ResponseEvaluator：回答评估层相关的数据和操作。
class ResponseEvaluator:
    """Bounded evidence checks; uncertain model reviews become REVIEW, never PASS."""

    # 调用位置：创建 ResponseEvaluator 实例时调用。
    def __init__(self, reviewer: ClaimReviewer | None = None) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._reviewer = reviewer

    # 调用位置：workflow/workflow_agent.py 的直接回复与计划执行流程调用。
    def evaluate(
        self, task_id: UUID, goal: str, answer: str,
        *, verification: VerificationResult | None = None,
        grounding_status: GroundingStatus | None = None,
        sources: tuple[KnowledgeSearchResult, ...] = (),
        tools_used: tuple[str, ...] = (),
        tool_failures: tuple[str, ...] = (),
    ) -> EvaluationResult:
        """根据执行证据判断回答的质量或可信度。"""
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
