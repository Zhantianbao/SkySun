"""Ordered workflow execution with explicit step states and no extra tool retries."""

from dataclasses import dataclass
from enum import Enum
from typing import Callable
from uuid import UUID, uuid4

from skysun.knowledge import KnowledgeSearchResult
from skysun.knowledge_grounding import GroundingStatus
from skysun.planner import PlanStep, TaskPlan


class StepState(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class WorkflowStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"


@dataclass(frozen=True)
class StepOutcome:
    output: str
    tools_used: tuple[str, ...] = ()
    tool_failures: tuple[str, ...] = ()
    grounding_status: GroundingStatus | None = None
    sources: tuple[KnowledgeSearchResult, ...] = ()


@dataclass(frozen=True)
class StepRecord:
    task_id: UUID
    step_id: str
    state: StepState
    output: str = ""
    tools_used: tuple[str, ...] = ()
    tool_status: str = "unused"
    grounding_status: GroundingStatus | None = None
    sources: tuple[KnowledgeSearchResult, ...] = ()
    error_code: str | None = None
    tool_failures: tuple[str, ...] = ()


@dataclass(frozen=True)
class StepEvent:
    step_id: str
    state: StepState


@dataclass(frozen=True)
class WorkflowRun:
    task_id: UUID
    goal: str
    status: WorkflowStatus
    steps: tuple[StepRecord, ...]
    events: tuple[StepEvent, ...]


class WorkflowExecutor:
    def execute(
        self, plan: TaskPlan,
        runner: Callable[[PlanStep, tuple[StepRecord, ...]], StepOutcome],
        *,
        task_id: UUID | None = None,
        completed: tuple[StepRecord, ...] = (),
        on_event: Callable[[StepEvent], None] | None = None,
        on_record: Callable[[StepRecord], None] | None = None,
        before_step: Callable[[PlanStep], None] | None = None,
    ) -> WorkflowRun:
        task_id = task_id or uuid4()
        records: list[StepRecord] = []
        events = [StepEvent(step.step_id, StepState.PENDING) for step in plan.steps]
        completed_by_id = {record.step_id: record for record in completed}
        if any(record.task_id != task_id or record.state is not StepState.SUCCESS for record in completed):
            raise ValueError("Only successful records for this task can be resumed")
        if not set(completed_by_id) <= {step.step_id for step in plan.steps}:
            raise ValueError("Resumed records do not match the plan")

        def emit(event: StepEvent) -> None:
            events.append(event)
            if on_event:
                on_event(event)

        def save(record: StepRecord) -> None:
            records.append(record)
            if on_record:
                on_record(record)

        stopped = False
        for step in plan.steps:
            if step.step_id in completed_by_id:
                records.append(completed_by_id[step.step_id])
                continue
            dependencies_ok = all(
                any(item.step_id == dependency and item.state is StepState.SUCCESS for item in records)
                for dependency in step.depends_on
            )
            if stopped or not dependencies_ok:
                save(StepRecord(task_id, step.step_id, StepState.SKIPPED))
                emit(StepEvent(step.step_id, StepState.SKIPPED))
                if step.required:
                    stopped = True
                continue
            if before_step:
                before_step(step)
            emit(StepEvent(step.step_id, StepState.RUNNING))
            try:
                outcome = runner(step, tuple(records))
                if not isinstance(outcome, StepOutcome):
                    raise TypeError("Step runner returned an invalid outcome")
                failed = bool(outcome.tool_failures) or not outcome.output.strip()
                state = StepState.FAILED if failed else StepState.SUCCESS
                record = StepRecord(
                    task_id, step.step_id, state, outcome.output,
                    outcome.tools_used,
                    "failed" if outcome.tool_failures else "success" if outcome.tools_used else "unused",
                    outcome.grounding_status, outcome.sources,
                    "tool_failure" if outcome.tool_failures else "empty_result" if failed else None,
                    outcome.tool_failures,
                )
            except Exception as error:
                state = StepState.FAILED
                record = StepRecord(task_id, step.step_id, state, error_code=type(error).__name__)
            save(record)
            emit(StepEvent(step.step_id, state))
            if state is StepState.FAILED and step.required:
                stopped = True
        required_failed = any(
            step.required and record.state is not StepState.SUCCESS
            for step, record in zip(plan.steps, records, strict=True)
        )
        status = (WorkflowStatus.FAILED if required_failed else
                  WorkflowStatus.PARTIAL if any(item.state is not StepState.SUCCESS for item in records)
                  else WorkflowStatus.SUCCESS)
        return WorkflowRun(task_id, plan.goal, status, tuple(records), tuple(events))
