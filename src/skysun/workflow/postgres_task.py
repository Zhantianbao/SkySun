"""PostgreSQL persistence for Agent tasks, workflow steps, and bounded events."""
# 本文件负责任务工作流中的 postgres、task 相关流程。

from dataclasses import asdict
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Jsonb

from skysun.memory.conversation import ConversationMessage
from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.workflow.planner import TaskPlan
from skysun.workflow.task_runtime import TaskEvent, TaskSnapshot, TaskStatus, validate_transition
from skysun.workflow.workflow import StepRecord, StepState
from skysun.user.postgres_users import ensure_user_schema
from skysun.user.user_identity import LEGACY_USER_ID, owner_id, scoped_user_id


# 调用位置：src/skysun/workflow/postgres_task.py 的 PostgresTaskStore.finish_step 调用。
def _record_payload(record: StepRecord) -> dict:
    """把任务记录整理为可持久化的数据。"""
    return {
        "output": record.output,
        "tools_used": list(record.tools_used),
        "tool_status": record.tool_status,
        "grounding_status": record.grounding_status.value if record.grounding_status else None,
        "sources": [
            {**asdict(source), "chunk_id": str(source.chunk_id) if source.chunk_id else None}
            for source in record.sources
        ],
        "error_code": record.error_code,
        "tool_failures": list(record.tool_failures),
    }


# 调用位置：src/skysun/workflow/postgres_task.py 的 PostgresTaskStore.get 调用。
def _step(task_id: UUID, row: tuple) -> StepRecord:
    """把数据库步骤记录转换为领域对象。"""
    step_id, state, payload = row
    sources = tuple(
        KnowledgeSearchResult(**{**source, "chunk_id": UUID(source["chunk_id"])
                                 if source.get("chunk_id") else None})
        for source in payload.get("sources", ())
    )
    grounding = payload.get("grounding_status")
    return StepRecord(
        task_id, step_id, StepState(state), payload.get("output", ""),
        tuple(payload.get("tools_used", ())), payload.get("tool_status", "unused"),
        GroundingStatus(grounding) if grounding else None, sources,
        payload.get("error_code"), tuple(payload.get("tool_failures", ())),
    )


# 以下集中定义 PostgresTaskStore：存储层相关的数据和操作。
class PostgresTaskStore:
    # 调用位置：创建 PostgresTaskStore 实例时调用。
    def __init__(self, database_url: str) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._database_url = database_url

    # 调用位置：PostgresTaskStore 的 initialize_schema 与数据库读写方法调用。
    def _connect(self) -> psycopg.Connection:
        """建立数据库连接，并统一连接参数。"""
        try:
            return psycopg.connect(self._database_url, connect_timeout=5)
        except psycopg.Error:
            raise RuntimeError("Cannot connect to SkySun task PostgreSQL") from None

    # 调用位置：core/main.py 的 create_app 初始化存储时调用；相关测试也会直接调用。
    def initialize_schema(self) -> None:
        """创建或更新当前存储所需的数据表和索引。"""
        with self._connect() as conn:
            ensure_user_schema(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_tasks (
                    task_id UUID PRIMARY KEY, goal TEXT NOT NULL,
                    status TEXT NOT NULL, conversation_id UUID,
                    history JSONB NOT NULL DEFAULT '[]', plan JSONB,
                    answer TEXT, created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_task_steps (
                    task_id UUID NOT NULL REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
                    step_id TEXT NOT NULL, state TEXT NOT NULL,
                    result JSONB NOT NULL DEFAULT '{}',
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    PRIMARY KEY (task_id, step_id)
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_task_events (
                    id BIGSERIAL PRIMARY KEY,
                    task_id UUID NOT NULL REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
                    event_type TEXT NOT NULL, step_id TEXT, tool_name TEXT,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_tasks_unfinished_idx
                ON agent_tasks (updated_at) WHERE status IN
                ('CREATED', 'PLANNING', 'RUNNING', 'WAITING', 'WAITING_APPROVAL')
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_task_events_task_idx
                ON agent_task_events (task_id, id)
            """)
            conn.execute(f"""
                ALTER TABLE agent_tasks ADD COLUMN IF NOT EXISTS user_id UUID NOT NULL
                DEFAULT '{LEGACY_USER_ID}' REFERENCES skysun_users(user_id)
            """)
            conn.execute(f"""
                ALTER TABLE agent_task_events ADD COLUMN IF NOT EXISTS user_id UUID NOT NULL
                DEFAULT '{LEGACY_USER_ID}' REFERENCES skysun_users(user_id)
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS agent_tasks_user_idx ON agent_tasks (user_id, task_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS agent_task_events_user_idx ON agent_task_events (user_id, task_id, id)")

    # 调用位置：PostgresTaskStore 内的 transition、save_plan、start_step 调用。
    @staticmethod
    def _assert_owned(conn: psycopg.Connection, task_id: UUID) -> UUID:
        """确认任务属于当前用户。"""
        row = conn.execute("SELECT user_id FROM agent_tasks WHERE task_id = %s", (task_id,)).fetchone()
        if row is None or (scoped_user_id() is not None and row[0] != scoped_user_id()):
            raise KeyError(task_id)
        return row[0]

    # 调用位置：workflow/workflow_agent.py 创建任务时调用。
    def create(self, goal: str, conversation_id: UUID | None,
               history: tuple[ConversationMessage, ...]) -> TaskSnapshot:
        """保存新任务及其会话关联。"""
        if not goal.strip():
            raise ValueError("Task goal is required")
        task_id = uuid4()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO agent_tasks (task_id, goal, status, conversation_id, history, user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (task_id, goal, TaskStatus.CREATED.value, conversation_id,
                  Jsonb([asdict(item) for item in history]), owner_id()))
            conn.execute("INSERT INTO agent_task_events (task_id, user_id, event_type) VALUES (%s, %s, %s)",
                         (task_id, owner_id(), "task_created"))
        return self.get(task_id)

    # 调用位置：PostgresTaskStore 内的 create、list_unfinished、list_tasks 调用。
    def get(self, task_id: UUID) -> TaskSnapshot:
        """按任务标识读取当前用户可见的任务。"""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT goal, status, conversation_id, created_at, updated_at, plan, history, answer, user_id
                FROM agent_tasks WHERE task_id = %s
            """, (task_id,)).fetchone()
            if row is None or (scoped_user_id() is not None and row[-1] != scoped_user_id()):
                raise KeyError(task_id)
            steps = conn.execute("""
                SELECT step_id, state, result FROM agent_task_steps
                WHERE task_id = %s
            """, (task_id,)).fetchall()
        goal, status, conversation_id, created_at, updated_at, plan, history, answer, user_id = row
        parsed_plan = TaskPlan.model_validate(plan) if plan else None
        step_order = {step.step_id: index for index, step in enumerate(parsed_plan.steps)} if parsed_plan else {}
        steps.sort(key=lambda item: step_order.get(item[0], len(step_order)))
        return TaskSnapshot(task_id, goal, TaskStatus(status), conversation_id,
                            created_at, updated_at,
                            parsed_plan,
                            tuple(ConversationMessage(**item) for item in history),
                            tuple(_step(task_id, item) for item in steps), answer, user_id)

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def list_unfinished(self) -> tuple[TaskSnapshot, ...]:
        """列出启动后需要恢复的未完成任务。"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT task_id FROM agent_tasks WHERE status IN
                ('CREATED', 'PLANNING', 'RUNNING', 'WAITING', 'WAITING_APPROVAL')
                AND (%s::uuid IS NULL OR user_id = %s)
                ORDER BY created_at
            """, (scoped_user_id(), scoped_user_id())).fetchall()
        return tuple(self.get(row[0]) for row in rows)

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def list_tasks(self, limit: int = 50) -> tuple[TaskSnapshot, ...]:
        """按时间倒序列出当前用户可见的近期任务。"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT task_id FROM agent_tasks WHERE user_id = %s
                ORDER BY created_at DESC, task_id DESC LIMIT %s
            """, (owner_id(), limit)).fetchall()
        return tuple(self.get(row[0]) for row in rows)

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def latest_for_conversation(self, conversation_id: UUID) -> TaskSnapshot | None:
        """读取指定会话最近关联的任务。"""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT task_id FROM agent_tasks
                WHERE user_id = %s AND conversation_id = %s
                ORDER BY created_at DESC, task_id DESC LIMIT 1
            """, (owner_id(), conversation_id)).fetchone()
        return self.get(row[0]) if row else None

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def transition(self, task_id: UUID, status: TaskStatus,
                   event_type: str | None = None, step_id: str | None = None) -> None:
        """更新任务状态并记录状态事件。"""
        with self._connect() as conn:
            self._assert_owned(conn, task_id)
            row = conn.execute("SELECT status FROM agent_tasks WHERE task_id = %s FOR UPDATE",
                               (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            validate_transition(TaskStatus(row[0]), status)
            conn.execute("UPDATE agent_tasks SET status = %s, updated_at = now() WHERE task_id = %s",
                         (status.value, task_id))
            if event_type:
                conn.execute("""
                    INSERT INTO agent_task_events (task_id, user_id, event_type, step_id)
                    VALUES (%s, %s, %s, %s)
                """, (task_id, owner_id(), event_type, step_id))

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def save_plan(self, task_id: UUID, plan: TaskPlan) -> None:
        """保存任务计划及各步骤。"""
        with self._connect() as conn:
            self._assert_owned(conn, task_id)
            row = conn.execute("SELECT status FROM agent_tasks WHERE task_id = %s FOR UPDATE",
                               (task_id,)).fetchone()
            if row is None or row[0] != TaskStatus.PLANNING.value:
                raise ValueError("Task is not planning")
            conn.execute("UPDATE agent_tasks SET plan = %s, updated_at = now() WHERE task_id = %s",
                         (Jsonb(plan.model_dump(mode="json")), task_id))
            for step in plan.steps:
                conn.execute("""
                    INSERT INTO agent_task_steps (task_id, step_id, state)
                    VALUES (%s, %s, %s)
                """, (task_id, step.step_id, StepState.PENDING.value))
            conn.execute("INSERT INTO agent_task_events (task_id, user_id, event_type) VALUES (%s, %s, %s)",
                         (task_id, owner_id(), "plan_generated"))

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def start_step(self, task_id: UUID, step_id: str) -> None:
        """记录步骤开始执行。"""
        with self._connect() as conn:
            self._assert_owned(conn, task_id)
            row = conn.execute("""
                SELECT state FROM agent_task_steps WHERE task_id = %s AND step_id = %s FOR UPDATE
            """, (task_id, step_id)).fetchone()
            if row is None or row[0] not in (StepState.PENDING.value, StepState.RUNNING.value):
                raise ValueError("Step cannot start")
            conn.execute("""
                UPDATE agent_task_steps SET state = %s, updated_at = now()
                WHERE task_id = %s AND step_id = %s
            """, (StepState.RUNNING.value, task_id, step_id))
            conn.execute("""
                INSERT INTO agent_task_events (task_id, user_id, event_type, step_id)
                VALUES (%s, %s, %s, %s)
            """, (task_id, owner_id(), "step_started", step_id))

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def finish_step(self, record: StepRecord) -> None:
        """保存步骤结果和执行状态。"""
        with self._connect() as conn:
            self._assert_owned(conn, record.task_id)
            row = conn.execute("""
                SELECT state FROM agent_task_steps WHERE task_id = %s AND step_id = %s FOR UPDATE
            """, (record.task_id, record.step_id)).fetchone()
            if row is None or row[0] not in (StepState.RUNNING.value, StepState.PENDING.value):
                raise ValueError("Step cannot finish")
            conn.execute("""
                UPDATE agent_task_steps SET state = %s, result = %s, updated_at = now()
                WHERE task_id = %s AND step_id = %s
            """, (record.state.value, Jsonb(_record_payload(record)),
                  record.task_id, record.step_id))
            for tool in record.tools_used:
                conn.execute("""
                    INSERT INTO agent_task_events (task_id, user_id, event_type, step_id, tool_name)
                    VALUES (%s, %s, %s, %s, %s)
                """, (record.task_id, owner_id(), "tool_called", record.step_id, tool))
                if tool in record.tool_failures:
                    conn.execute("""
                        INSERT INTO agent_task_events (task_id, user_id, event_type, step_id, tool_name)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (record.task_id, owner_id(), "tool_failed", record.step_id, tool))
            conn.execute("""
                INSERT INTO agent_task_events (task_id, user_id, event_type, step_id)
                VALUES (%s, %s, %s, %s)
            """, (record.task_id, owner_id(), "step_completed", record.step_id))

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def complete(self, task_id: UUID, status: TaskStatus, answer: str) -> None:
        """结束任务并保存最终回复。"""
        if status not in (TaskStatus.COMPLETED, TaskStatus.FAILED):
            raise ValueError("Task completion requires a terminal status")
        with self._connect() as conn:
            self._assert_owned(conn, task_id)
            row = conn.execute("SELECT status FROM agent_tasks WHERE task_id = %s FOR UPDATE",
                               (task_id,)).fetchone()
            if row is None:
                raise KeyError(task_id)
            validate_transition(TaskStatus(row[0]), status)
            conn.execute("""
                UPDATE agent_tasks SET status = %s, answer = %s, updated_at = now()
                WHERE task_id = %s
            """, (status.value, answer, task_id))
            conn.execute("INSERT INTO agent_task_events (task_id, user_id, event_type) VALUES (%s, %s, %s)",
                         (task_id, owner_id(),
                          "task_completed" if status is TaskStatus.COMPLETED else "task_failed"))

    # 调用位置：通过 PostgresTaskStore 实例在工作流 Agent 与任务接口中调用。
    def events(self, task_id: UUID) -> tuple[TaskEvent, ...]:
        """列出设备或任务事件。"""
        with self._connect() as conn:
            self._assert_owned(conn, task_id)
            rows = conn.execute("""
                SELECT event_type, step_id, tool_name, created_at, user_id FROM agent_task_events
                WHERE task_id = %s ORDER BY id
            """, (task_id,)).fetchall()
        return tuple(TaskEvent(*row) for row in rows)
