"""Workflow result checks informed by tool outcomes and knowledge grounding."""
# 本文件负责任务工作流中的 verification 相关流程。

from dataclasses import dataclass
from enum import Enum

from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.workflow.planner import TaskPlan
from skysun.workflow.workflow import StepState, WorkflowRun, WorkflowStatus


# 以下集中定义 VerificationStatus：数据结构或组件相关的数据和操作。
class VerificationStatus(str, Enum):
    SUCCESS = "SUCCESS"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    FAILED = "FAILED"


# 以下集中定义 VerificationResult：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class VerificationResult:
    status: VerificationStatus
    reason: str
    affected_steps: tuple[str, ...] = ()
    next_action: str | None = None


# 以下集中定义 WorkflowVerifier：结果校验层相关的数据和操作。
class WorkflowVerifier:
    # 调用位置：通过 WorkflowVerifier 实例在工作流 Agent 与任务接口中调用。
    def verify(self, plan: TaskPlan, run: WorkflowRun) -> VerificationResult:
        """检查工作流结果和证据是否满足计划要求。"""
        if run.goal != plan.goal or len(run.steps) != len(plan.steps):
            return VerificationResult(VerificationStatus.FAILED, "Workflow result does not match its plan.")
        failed = tuple(
            step.step_id for step, record in zip(plan.steps, run.steps, strict=True)
            if step.required and record.state in (StepState.FAILED, StepState.SKIPPED)
        )
        if run.status is WorkflowStatus.FAILED or failed:
            return VerificationResult(
                VerificationStatus.FAILED, "A required workflow step failed.", failed,
                "Resolve the failed step before answering.",
            )
        review: list[str] = []
        for step, record in zip(plan.steps, run.steps, strict=True):
            if record.step_id != step.step_id or record.task_id != run.task_id:
                return VerificationResult(VerificationStatus.FAILED, "Workflow step metadata is invalid.")
            if record.state is not StepState.SUCCESS or not record.output.strip() or record.tool_status == "failed":
                review.append(step.step_id)
                continue
            required_tools = set(step.capabilities) - {"reason"}
            if not required_tools <= set(record.tools_used):
                review.append(step.step_id)
                continue
            if "search_knowledge" in required_tools and (
                record.grounding_status is not GroundingStatus.SUFFICIENT or not record.sources
            ):
                review.append(step.step_id)
        if review:
            return VerificationResult(
                VerificationStatus.NEEDS_REVIEW, "Some step results lack required tools or evidence.",
                tuple(review), "Obtain the missing result or evidence before answering.",
            )
        return VerificationResult(VerificationStatus.SUCCESS, "Required steps and evidence are complete.")
