"""PostgreSQL implementation of SkySun's conversation storage boundary."""

from uuid import UUID, uuid4

import psycopg

from skysun.conversation import ConversationMessage


class PostgresConversationStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self._database_url, connect_timeout=5)
        except psycopg.Error:
            raise RuntimeError(
                "Cannot connect to SkySun PostgreSQL; check DATABASE_URL and server availability"
            ) from None

    def initialize_schema(self) -> None:
        """Create the v0.3 tables if absent, preserving all existing rows."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id UUID PRIMARY KEY,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id BIGSERIAL PRIMARY KEY,
                    conversation_id UUID NOT NULL REFERENCES conversations(id),
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL CHECK (content <> ''),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS conversation_messages_conversation_id_id_idx
                ON conversation_messages (conversation_id, id)
                """
            )

    def create(self) -> str:
        conversation_id = uuid4()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id) VALUES (%s)",
                (conversation_id,),
            )
        return str(conversation_id)

    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        identifier = UUID(conversation_id)
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM conversations WHERE id = %s",
                (identifier,),
            ).fetchone()
            if exists is None:
                raise KeyError(conversation_id)
            rows = conn.execute(
                """
                SELECT role, content
                FROM conversation_messages
                WHERE conversation_id = %s
                ORDER BY id
                """,
                (identifier,),
            ).fetchall()
        return tuple(ConversationMessage(role, content) for role, content in rows)

    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None:
        """Commit both messages together or roll both back on any error."""
        identifier = UUID(conversation_id)
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversation_messages (conversation_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (identifier, "user", user),
            )
            conn.execute(
                """
                INSERT INTO conversation_messages (conversation_id, role, content)
                VALUES (%s, %s, %s)
                """,
                (identifier, "assistant", assistant),
            )
