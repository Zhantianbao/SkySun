"""PostgreSQL traces, response evaluations, feedback, and derived metrics."""

import hashlib
from uuid import UUID, uuid4

import psycopg
from psycopg.types.json import Jsonb

from skysun.evaluation.observability import (
    EvaluationResult, EvaluationStatus, FeedbackRating, FeedbackRecord, TraceRecord,
)
from skysun.user.postgres_users import ensure_user_schema
from skysun.user.user_identity import owner_id, require_admin, scoped_user_id


_STAGES = {
    "request", "request_received", "planning", "workflow_execution", "workflow_step",
    "direct_execution", "tool_call", "memory_retrieval", "knowledge_retrieval",
    "verification", "response_evaluation", "final_response",
}
_META_KEYS = {"step_id", "tool_name", "grounding_status", "source_count", "error_code",
              "verification_status", "workflow_status", "tool_count", "result_status"}


def _metadata(value: dict | None) -> dict:
    result = {}
    for key, item in (value or {}).items():
        if key not in _META_KEYS or not isinstance(item, (str, int, float, bool)):
            raise ValueError("Unsupported trace metadata")
        result[key] = item[:100] if isinstance(item, str) else item
    return result


class PostgresObservabilityStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self._database_url, connect_timeout=5)
        except psycopg.Error:
            raise RuntimeError("Cannot connect to SkySun observability PostgreSQL") from None

    @staticmethod
    def _owner(conn: psycopg.Connection, task_id: UUID) -> UUID:
        row = conn.execute("SELECT user_id FROM agent_tasks WHERE task_id = %s", (task_id,)).fetchone()
        if row is None or (scoped_user_id() is not None and row[0] != scoped_user_id()):
            raise KeyError(task_id)
        return row[0]

    def initialize_schema(self) -> None:
        with self._connect() as conn:
            ensure_user_schema(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_traces (
                    trace_id UUID PRIMARY KEY,
                    task_id UUID NOT NULL REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES skysun_users(user_id),
                    stage TEXT NOT NULL, status TEXT NOT NULL,
                    started_at TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
                    finished_at TIMESTAMPTZ,
                    duration_ms DOUBLE PRECISION,
                    metadata JSONB NOT NULL DEFAULT '{}'
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_traces_task_idx
                ON agent_traces (user_id, task_id, started_at)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_evaluations (
                    task_id UUID PRIMARY KEY REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES skysun_users(user_id),
                    score INTEGER NOT NULL CHECK (score BETWEEN 0 AND 100),
                    status TEXT NOT NULL CHECK (status IN ('PASS', 'REVIEW', 'FAIL')),
                    reasons JSONB NOT NULL, grounding_correct BOOLEAN,
                    complete BOOLEAN, unsupported_claims JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS agent_evaluations_user_idx
                ON agent_evaluations (user_id, created_at)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS agent_feedback (
                    feedback_id UUID PRIMARY KEY,
                    task_id UUID NOT NULL REFERENCES agent_tasks(task_id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES skysun_users(user_id),
                    rating TEXT NOT NULL CHECK (rating IN ('POSITIVE', 'NEGATIVE')),
                    comment TEXT,
                    response_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (user_id, task_id)
                )
            """)

    def start_trace(self, task_id: UUID, stage: str, metadata: dict | None = None) -> UUID:
        if stage not in _STAGES:
            raise ValueError("Unknown trace stage")
        trace_id = uuid4()
        with self._connect() as conn:
            user_id = self._owner(conn, task_id)
            conn.execute("""
                INSERT INTO agent_traces (trace_id, task_id, user_id, stage, status, metadata)
                VALUES (%s, %s, %s, %s, 'RUNNING', %s)
            """, (trace_id, task_id, user_id, stage, Jsonb(_metadata(metadata))))
        return trace_id

    def finish_trace(self, trace_id: UUID, status: str, metadata: dict | None = None) -> None:
        if status not in ("SUCCESS", "FAILED", "REVIEW"):
            raise ValueError("Unknown trace status")
        with self._connect() as conn:
            row = conn.execute("""
                SELECT task_id, metadata, finished_at FROM agent_traces
                WHERE trace_id = %s FOR UPDATE
            """, (trace_id,)).fetchone()
            if row is None or row[2] is not None:
                raise KeyError(trace_id)
            self._owner(conn, row[0])
            details = {**row[1], **_metadata(metadata)}
            conn.execute("""
                UPDATE agent_traces SET status = %s, finished_at = clock_timestamp(),
                    duration_ms = EXTRACT(EPOCH FROM (clock_timestamp() - started_at)) * 1000,
                    metadata = %s
                WHERE trace_id = %s
            """, (status, Jsonb(details), trace_id))

    def list_traces(self, task_id: UUID | None = None, limit: int = 100) -> tuple[TraceRecord, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("Trace limit must be between 1 and 200")
        with self._connect() as conn:
            if task_id:
                self._owner(conn, task_id)
            rows = conn.execute("""
                SELECT trace_id, task_id, user_id, stage, status, started_at,
                       finished_at, duration_ms, metadata
                FROM agent_traces
                WHERE (%s::uuid IS NULL OR task_id = %s)
                  AND (%s::uuid IS NULL OR user_id = %s)
                ORDER BY started_at DESC, trace_id DESC LIMIT %s
            """, (task_id, task_id, scoped_user_id(), scoped_user_id(), limit)).fetchall()
        return tuple(TraceRecord(*row) for row in rows)

    def save_evaluation(self, result: EvaluationResult) -> EvaluationResult:
        if not 0 <= result.score <= 100:
            raise ValueError("Invalid evaluation score")
        with self._connect() as conn:
            user_id = self._owner(conn, result.task_id)
            if result.user_id != user_id:
                raise ValueError("Evaluation owner mismatch")
            row = conn.execute("""
                INSERT INTO agent_evaluations
                    (task_id, user_id, score, status, reasons, grounding_correct,
                     complete, unsupported_claims)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (task_id) DO UPDATE SET
                    score = EXCLUDED.score, status = EXCLUDED.status,
                    reasons = EXCLUDED.reasons,
                    grounding_correct = EXCLUDED.grounding_correct,
                    complete = EXCLUDED.complete,
                    unsupported_claims = EXCLUDED.unsupported_claims,
                    created_at = now()
                RETURNING created_at
            """, (result.task_id, user_id, result.score, result.status.value,
                  Jsonb(list(result.reasons)), result.grounding_correct, result.complete,
                  Jsonb(list(result.unsupported_claims)))).fetchone()
        return EvaluationResult(result.task_id, user_id, result.score, result.status,
                                result.reasons, result.grounding_correct, result.complete,
                                result.unsupported_claims, row[0])

    def list_evaluations(self, limit: int = 100) -> tuple[EvaluationResult, ...]:
        if not 1 <= limit <= 200:
            raise ValueError("Evaluation limit must be between 1 and 200")
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT task_id, user_id, score, status, reasons, grounding_correct,
                       complete, unsupported_claims, created_at
                FROM agent_evaluations
                WHERE %s::uuid IS NULL OR user_id = %s
                ORDER BY created_at DESC LIMIT %s
            """, (scoped_user_id(), scoped_user_id(), limit)).fetchall()
        return tuple(EvaluationResult(row[0], row[1], row[2], EvaluationStatus(row[3]),
                                      tuple(row[4]), row[5], row[6], tuple(row[7]), row[8])
                     for row in rows)

    def save_feedback(self, task_id: UUID, rating: FeedbackRating,
                      comment: str | None = None) -> FeedbackRecord:
        rating = FeedbackRating(rating)
        comment = comment.strip() if comment is not None else None
        if comment and len(comment) > 1000:
            raise ValueError("Feedback comment is too long")
        with self._connect() as conn:
            user_id = self._owner(conn, task_id)
            task = conn.execute("SELECT status, answer FROM agent_tasks WHERE task_id = %s",
                                (task_id,)).fetchone()
            if task[0] not in ("COMPLETED", "FAILED") or not task[1]:
                raise ValueError("Feedback requires a final task response")
            response_hash = hashlib.sha256(task[1].encode()).hexdigest()
            row = conn.execute("""
                INSERT INTO agent_feedback
                    (feedback_id, task_id, user_id, rating, comment, response_hash)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, task_id) DO UPDATE SET
                    rating = EXCLUDED.rating, comment = EXCLUDED.comment,
                    response_hash = EXCLUDED.response_hash, updated_at = now()
                RETURNING feedback_id, task_id, user_id, rating, comment,
                          response_hash, created_at, updated_at
            """, (uuid4(), task_id, user_id, rating.value, comment or None, response_hash)).fetchone()
        return FeedbackRecord(row[0], row[1], row[2], FeedbackRating(row[3]),
                              row[4], row[5], row[6], row[7])

    def metrics(self) -> dict:
        require_admin()
        with self._connect() as conn:
            task = conn.execute("""
                SELECT count(*), count(*) FILTER (WHERE status = 'COMPLETED'),
                       count(*) FILTER (WHERE status IN ('COMPLETED', 'FAILED')),
                       count(DISTINCT user_id)
                FROM agent_tasks
            """).fetchone()
            traces = conn.execute("""
                SELECT avg(duration_ms) FILTER (WHERE stage = 'request'),
                       avg(duration_ms) FILTER (WHERE stage = 'tool_call'),
                       avg(duration_ms) FILTER (WHERE stage = 'workflow_execution'),
                       count(*) FILTER (WHERE stage = 'tool_call' AND status = 'FAILED'),
                       count(*) FILTER (WHERE stage = 'tool_call'),
                       count(*) FILTER (WHERE stage = 'verification' AND status = 'FAILED')
                FROM agent_traces WHERE finished_at IS NOT NULL
            """).fetchone()
            feedback = conn.execute("""
                SELECT count(*) FILTER (WHERE rating = 'POSITIVE'),
                       count(*) FILTER (WHERE rating = 'NEGATIVE') FROM agent_feedback
            """).fetchone()
            activity = conn.execute("""
                SELECT user_id, count(*) FROM agent_tasks
                GROUP BY user_id ORDER BY count(*) DESC LIMIT 20
            """).fetchall()
        return {
            "task_count": task[0], "task_success_rate": task[1] / task[2] if task[2] else None,
            "active_user_count": task[3],
            "average_latency_ms": float(traces[0]) if traces[0] is not None else None,
            "average_tool_duration_ms": float(traces[1]) if traces[1] is not None else None,
            "average_workflow_duration_ms": float(traces[2]) if traces[2] is not None else None,
            "tool_failure_rate": traces[3] / traces[4] if traces[4] else None,
            "verification_failures": traces[5],
            "positive_feedback_count": feedback[0], "negative_feedback_count": feedback[1],
            "user_activity": [{"user_id": str(row[0]), "task_count": row[1]} for row in activity],
        }
