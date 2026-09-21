"""PostgreSQL implementation of SkySun's global single-user memory store."""

from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID, uuid4

import psycopg

from skysun.memory import PersonalMemory, normalize_key

_COLUMNS = "id, memory_key, content, category, source_conversation_id, created_at, updated_at"


def _memory(row: tuple) -> PersonalMemory:
    return PersonalMemory(*row)


class PostgresMemoryStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self._database_url, connect_timeout=5)
        except psycopg.Error:
            raise RuntimeError(
                "Cannot connect to SkySun PostgreSQL; check DATABASE_URL and server availability"
            ) from None

    @contextmanager
    def _transaction(self) -> Iterator[psycopg.Connection]:
        try:
            with self._connect() as conn:
                yield conn
        except psycopg.Error:
            raise RuntimeError("Personal memory PostgreSQL operation failed") from None

    def initialize_schema(self) -> None:
        """Create the table idempotently without changing existing data."""
        with self._transaction() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS personal_memories (
                    id UUID PRIMARY KEY,
                    memory_key TEXT NOT NULL UNIQUE,
                    content TEXT NOT NULL CHECK (content <> ''),
                    category TEXT,
                    source_conversation_id UUID,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )

    def remember(
        self,
        key: str,
        content: str,
        category: str | None = None,
        source_conversation_id: UUID | None = None,
    ) -> PersonalMemory:
        key = normalize_key(key)
        content = content.strip()
        if not content:
            raise ValueError("Memory content must not be empty")
        category = (category.strip() or None) if category is not None else None
        with self._transaction() as conn:
            row = conn.execute(
                f"""
                INSERT INTO personal_memories (id, memory_key, content, category, source_conversation_id)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (memory_key) DO UPDATE SET
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    source_conversation_id = EXCLUDED.source_conversation_id,
                    updated_at = now()
                RETURNING {_COLUMNS}
                """,
                (uuid4(), key, content, category, source_conversation_id),
            ).fetchone()
        return _memory(row)

    def get(self, key: str) -> PersonalMemory | None:
        with self._transaction() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM personal_memories WHERE memory_key = %s",
                (normalize_key(key),),
            ).fetchone()
        return _memory(row) if row else None

    def list_memories(self, limit: int = 20) -> tuple[PersonalMemory, ...]:
        if not 1 <= limit <= 20:
            raise ValueError("Memory recall limit must be between 1 and 20")
        with self._transaction() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM personal_memories ORDER BY updated_at DESC, memory_key LIMIT %s",
                (limit,),
            ).fetchall()
        return tuple(_memory(row) for row in rows)

    def forget(self, key: str) -> bool:
        with self._transaction() as conn:
            row = conn.execute(
                "DELETE FROM personal_memories WHERE memory_key = %s RETURNING id",
                (normalize_key(key),),
            ).fetchone()
        return row is not None
