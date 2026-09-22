"""PostgreSQL persistence for Agent tasks, workflow steps, and bounded events."""

from dataclasses import asdict
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Jsonb

from skysun.conversation import ConversationMessage
from skysun.knowledge import KnowledgeSearchResult
from skysun.knowledge_grounding import GroundingStatus
from skysun.planner import TaskPlan
from skysun.task_runtime import TaskEvent, TaskSnapshot, TaskStatus, validate_transition
from skysun.workflow import StepRecord, StepState
from skysun.postgres_users import ensure_user_schema
from skysun.user_identity import LEGACY_USER_ID, owner_id, scoped_user_id


def _record_payload(record: StepRecord) -> dict:
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


def _step(task_id: UUID, row: tuple) -> StepRecord:
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


class PostgresTaskStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self._database_url, connect_timeout=5)
        except psycopg.Error:
            raise RuntimeError("Cannot connect to SkySun task PostgreSQL") from None

    def initialize_schema(self) -> None:
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

    @staticmethod
    def _assert_owned(conn: psycopg.Connection, task_id: UUID) -> UUID:
        row = conn.execute("SELECT user_id FROM agent_tasks WHERE task_id = %s", (task_id,)).fetchone()
        if row is None or (scoped_user_id() is not None and row[0] != scoped_user_id()):
            raise KeyError(task_id)
        return row[0]

    def create(self, goal: str, conversation_id: UUID | None,
               history: tuple[ConversationMessage, ...]) -> TaskSnapshot:
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

    def get(self, task_id: UUID) -> TaskSnapshot:
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

    def list_unfinished(self) -> tuple[TaskSnapshot, ...]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT task_id FROM agent_tasks WHERE status IN
                ('CREATED', 'PLANNING', 'RUNNING', 'WAITING', 'WAITING_APPROVAL')
                AND (%s::uuid IS NULL OR user_id = %s)
                ORDER BY created_at
            """, (scoped_user_id(), scoped_user_id())).fetchall()
        return tuple(self.get(row[0]) for row in rows)

    def transition(self, task_id: UUID, status: TaskStatus,
                   event_type: str | None = None, step_id: str | None = None) -> None:
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

    def save_plan(self, task_id: UUID, plan: TaskPlan) -> None:
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

    def start_step(self, task_id: UUID, step_id: str) -> None:
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

    def finish_step(self, record: StepRecord) -> None:
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

    def complete(self, task_id: UUID, status: TaskStatus, answer: str) -> None:
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

    def events(self, task_id: UUID) -> tuple[TaskEvent, ...]:
        with self._connect() as conn:
            self._assert_owned(conn, task_id)
            rows = conn.execute("""
                SELECT event_type, step_id, tool_name, created_at, user_id FROM agent_task_events
                WHERE task_id = %s ORDER BY id
            """, (task_id,)).fetchall()
        return tuple(TaskEvent(*row) for row in rows)
