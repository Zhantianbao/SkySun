"""Persistent workflow task state and recovery contract."""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol
from uuid import UUID

from skysun.memory.conversation import ConversationMessage
from skysun.workflow.planner import TaskPlan
from skysun.workflow.workflow import StepRecord
from skysun.user.user_identity import LEGACY_USER_ID


class TaskStatus(str, Enum):
    CREATED = "CREATED"
    PLANNING = "PLANNING"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


_TRANSITIONS = {
    TaskStatus.CREATED: {TaskStatus.PLANNING, TaskStatus.CANCELLED, TaskStatus.FAILED},
    TaskStatus.PLANNING: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.WAITING, TaskStatus.WAITING_APPROVAL, TaskStatus.COMPLETED,
                         TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.WAITING: {TaskStatus.RUNNING, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.WAITING_APPROVAL: {TaskStatus.WAITING, TaskStatus.CANCELLED},
    TaskStatus.COMPLETED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}


def validate_transition(current: TaskStatus, target: TaskStatus) -> None:
    if target not in _TRANSITIONS[current]:
        raise ValueError(f"Invalid task transition: {current.value} to {target.value}")


@dataclass(frozen=True)
class TaskSnapshot:
    task_id: UUID
    goal: str
    status: TaskStatus
    conversation_id: UUID | None
    created_at: datetime
    updated_at: datetime
    plan: TaskPlan | None = None
    history: tuple[ConversationMessage, ...] = ()
    steps: tuple[StepRecord, ...] = ()
    answer: str | None = None
    user_id: UUID = LEGACY_USER_ID


@dataclass(frozen=True)
class TaskEvent:
    event_type: str
    step_id: str | None = None
    tool_name: str | None = None
    created_at: datetime | None = None
    user_id: UUID = LEGACY_USER_ID


class TaskStore(Protocol):
    def create(self, goal: str, conversation_id: UUID | None,
               history: tuple[ConversationMessage, ...]) -> TaskSnapshot: ...

    def get(self, task_id: UUID) -> TaskSnapshot: ...

    def list_unfinished(self) -> tuple[TaskSnapshot, ...]: ...

    def transition(self, task_id: UUID, status: TaskStatus,
                   event_type: str | None = None, step_id: str | None = None) -> None: ...

    def save_plan(self, task_id: UUID, plan: TaskPlan) -> None: ...

    def start_step(self, task_id: UUID, step_id: str) -> None: ...

    def finish_step(self, record: StepRecord) -> None: ...

    def complete(self, task_id: UUID, status: TaskStatus, answer: str) -> None: ...

    def events(self, task_id: UUID) -> tuple[TaskEvent, ...]: ...
