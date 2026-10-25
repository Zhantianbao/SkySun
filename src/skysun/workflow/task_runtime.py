"""Persistent workflow task state and recovery contract."""
# 本文件负责任务工作流中的 task、runtime 相关流程。

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Protocol
from uuid import UUID

from skysun.memory.conversation import ConversationMessage
from skysun.workflow.planner import TaskPlan
from skysun.workflow.workflow import StepRecord
from skysun.user.user_identity import LEGACY_USER_ID


# 以下集中定义 TaskStatus：数据结构或组件相关的数据和操作。
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


# 调用位置：src/skysun/workflow/postgres_task.py 的 PostgresTaskStore.transition、src/skysun/workflow/postgres_task.py 的 PostgresTaskStore.complete 调用。
def validate_transition(current: TaskStatus, target: TaskStatus) -> None:
    """校验任务状态能否进行指定转换。"""
    if target not in _TRANSITIONS[current]:
        raise ValueError(f"Invalid task transition: {current.value} to {target.value}")


# 以下集中定义 TaskSnapshot：数据结构或组件相关的数据和操作。
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


# 以下集中定义 TaskEvent：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class TaskEvent:
    event_type: str
    step_id: str | None = None
    tool_name: str | None = None
    created_at: datetime | None = None
    user_id: UUID = LEGACY_USER_ID


# 以下集中定义 TaskStore：存储层相关的数据和操作。
class TaskStore(Protocol):
    # 调用位置：该协议由 TaskStore 的实现类调用；入口涉及工作流 Agent 与任务接口。
    def create(self, goal: str, conversation_id: UUID | None,
               history: tuple[ConversationMessage, ...]) -> TaskSnapshot: """创建任务并返回任务快照。"""; ...

    # 调用位置：业务服务通过存储接口读取当前用户数据。
    def get(self, task_id: UUID) -> TaskSnapshot: """按任务标识读取任务快照。"""; ...

    # 调用位置：应用启动恢复任务时调用。
    def list_unfinished(self) -> tuple[TaskSnapshot, ...]: """列出尚未完成的任务。"""; ...

    # 调用位置：该协议由 TaskStore 的实现类调用；入口涉及工作流 Agent 与任务接口。
    def transition(self, task_id: UUID, status: TaskStatus,
                   event_type: str | None = None, step_id: str | None = None) -> None: """更新任务状态并记录事件。"""; ...

    # 调用位置：工作流智能代理完成规划后调用。
    def save_plan(self, task_id: UUID, plan: TaskPlan) -> None: """保存任务计划和步骤。"""; ...

    # 调用位置：工作流事件回调记录步骤开始时调用。
    def start_step(self, task_id: UUID, step_id: str) -> None: """记录步骤开始执行。"""; ...

    # 调用位置：工作流执行器完成步骤后调用。
    def finish_step(self, record: StepRecord) -> None: """保存步骤的执行结果。"""; ...

    # 调用位置：工作流智能代理结束任务时调用。
    def complete(self, task_id: UUID, status: TaskStatus, answer: str) -> None: """结束任务并保存最终回答。"""; ...

    # 调用位置：任务接口和工作流恢复流程调用。
    def events(self, task_id: UUID) -> tuple[TaskEvent, ...]: """列出任务或设备的事件记录。"""; ...
