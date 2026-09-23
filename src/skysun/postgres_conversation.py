"""PostgreSQL implementation of SkySun's conversation storage boundary."""

from uuid import UUID, uuid4

import psycopg

from skysun.conversation import ConversationMessage, ConversationSummary
from skysun.postgres_users import ensure_user_schema
from skysun.user_identity import LEGACY_USER_ID, owner_id


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
            ensure_user_schema(conn)
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
            conn.execute(f"""
                ALTER TABLE conversations ADD COLUMN IF NOT EXISTS user_id UUID NOT NULL
                DEFAULT '{LEGACY_USER_ID}' REFERENCES skysun_users(user_id)
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS conversations_user_idx ON conversations (user_id, id)")
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS conversation_messages_conversation_id_id_idx
                ON conversation_messages (conversation_id, id)
                """
            )

    def create(self, conversation_id: str | None = None) -> str:
        conversation_id = UUID(conversation_id) if conversation_id else uuid4()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, user_id) VALUES (%s, %s)",
                (conversation_id, owner_id()),
            )
        return str(conversation_id)

    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        identifier = UUID(conversation_id)
        with self._connect() as conn:
            exists = conn.execute(
                "SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
                (identifier, owner_id()),
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

    def list_conversations(self, limit: int = 50) -> tuple[ConversationSummary, ...]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT c.id, COALESCE(first_message.content, '新对话'),
                       c.created_at, COALESCE(last_message.created_at, c.created_at)
                FROM conversations AS c
                LEFT JOIN LATERAL (
                    SELECT content FROM conversation_messages
                    WHERE conversation_id = c.id AND role = 'user'
                    ORDER BY id ASC LIMIT 1
                ) AS first_message ON true
                LEFT JOIN LATERAL (
                    SELECT created_at FROM conversation_messages
                    WHERE conversation_id = c.id ORDER BY id DESC LIMIT 1
                ) AS last_message ON true
                WHERE c.user_id = %s
                ORDER BY COALESCE(last_message.created_at, c.created_at) DESC, c.id DESC
                LIMIT %s
            """, (owner_id(), limit)).fetchall()
        return tuple(ConversationSummary(str(row[0]), row[1][:80], row[2], row[3])
                     for row in rows)

    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None:
        """Commit both messages together or roll both back on any error."""
        identifier = UUID(conversation_id)
        with self._connect() as conn:
            if conn.execute("SELECT 1 FROM conversations WHERE id = %s AND user_id = %s",
                            (identifier, owner_id())).fetchone() is None:
                raise KeyError(conversation_id)
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
