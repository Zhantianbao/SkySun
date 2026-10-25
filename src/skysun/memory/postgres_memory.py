"""PostgreSQL personal memories scoped to users, with separate global memories."""
# 本文件负责会话与记忆中的 postgres、memory 相关流程。

import re
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import UUID, uuid4

import psycopg

from skysun.memory.memory import GlobalMemory, MemoryChange, PersonalMemory, normalize_key, secret_like_memory
from skysun.tool.errors import RetryableToolError
from skysun.user.postgres_users import ensure_user_schema
from skysun.user.user_identity import LEGACY_USER_ID, owner_id, require_admin

_COLUMNS = "id, memory_key, content, category, source_conversation_id, created_at, updated_at, user_id"


# 调用位置：src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore._connect、src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore._transaction、tests/test_postgres_memory.py 的 test_only_connection_failures_are_retryable 等处调用。
def _is_transient(error: psycopg.OperationalError) -> bool:
    """判断数据库错误是否属于可重试的临时连接故障，身份验证失败不会重试。"""
    return error.sqlstate is None or error.sqlstate.startswith("08") or error.sqlstate in {
        "57P01", "57P02", "57P03",
    }


# 调用位置：src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore.remember、src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore.get、src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore.list_memories 等处调用。
def _memory(row: tuple) -> PersonalMemory:
    """将数据库行转换为个人记忆对象。"""
    return PersonalMemory(*row)


# 以下集中定义 PostgresMemoryStore：存储层相关的数据和操作。
class PostgresMemoryStore:
    # 调用位置：创建 PostgresMemoryStore 实例时调用。
    def __init__(self, database_url: str) -> None:
        """保存数据库连接地址和对象依赖。"""
        self._database_url = database_url

    # 调用位置：PostgresMemoryStore 的 initialize_schema 与数据库读写方法调用。
    def _connect(self) -> psycopg.Connection:
        """建立数据库连接并处理连接异常。"""
        try:
            return psycopg.connect(self._database_url, connect_timeout=5)
        except psycopg.OperationalError as error:
            message = "Cannot connect to SkySun PostgreSQL; check DATABASE_URL and server availability"
            if _is_transient(error):
                raise RetryableToolError(message) from None
            raise RuntimeError(message) from None
        except psycopg.Error:
            raise RuntimeError(
                "Cannot connect to SkySun PostgreSQL; check DATABASE_URL and server availability"
            ) from None

    # 调用位置：PostgresMemoryStore 的建表与数据读写方法调用。
    @contextmanager
    def _transaction(self) -> Iterator[psycopg.Connection]:
        """在事务范围内执行数据库操作。"""
        try:
            with self._connect() as conn:
                yield conn
        except psycopg.OperationalError as error:
            if _is_transient(error):
                raise RetryableToolError("Personal memory PostgreSQL operation failed") from None
            raise RuntimeError("Personal memory PostgreSQL operation failed") from None
        except psycopg.Error:
            raise RuntimeError("Personal memory PostgreSQL operation failed") from None

    # 调用位置：core/main.py 的 create_app 初始化存储时调用；相关测试也会直接调用。
    def initialize_schema(self) -> None:
        """以可重复执行的方式创建记忆表，并保留已有数据。"""
        with self._transaction() as conn:
            ensure_user_schema(conn)
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
            conn.execute(f"""
                ALTER TABLE personal_memories ADD COLUMN IF NOT EXISTS user_id UUID NOT NULL
                DEFAULT '{LEGACY_USER_ID}' REFERENCES skysun_users(user_id)
            """)
            conn.execute("ALTER TABLE personal_memories DROP CONSTRAINT IF EXISTS personal_memories_memory_key_key")
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS personal_memories_user_key_idx
                ON personal_memories (user_id, memory_key)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS global_memories (
                    memory_key TEXT PRIMARY KEY, content TEXT NOT NULL CHECK (content <> ''),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

    # 调用位置：通过 PostgresMemoryStore 实例在会话服务、记忆工具与整理服务中调用。
    def remember(
        self,
        key: str,
        content: str,
        category: str | None = None,
        source_conversation_id: UUID | None = None,
    ) -> PersonalMemory:
        """保存或更新一条记忆。"""
        key = normalize_key(key)
        content = content.strip()
        if not content:
            raise ValueError("Memory content must not be empty")
        if secret_like_memory(key, content):
            raise ValueError("Secret-like memory rejected")
        category = (category.strip() or None) if category is not None else None
        with self._transaction() as conn:
            row = conn.execute(
                f"""
                INSERT INTO personal_memories (id, memory_key, content, category, source_conversation_id, user_id)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (user_id, memory_key) DO UPDATE SET
                    content = EXCLUDED.content,
                    category = EXCLUDED.category,
                    source_conversation_id = EXCLUDED.source_conversation_id,
                    updated_at = now()
                RETURNING {_COLUMNS}
                """,
                (uuid4(), key, content, category, source_conversation_id, owner_id()),
            ).fetchone()
        return _memory(row)

    # 调用位置：memory/memory_curator.py 的记忆整理流程调用。
    def get(self, key: str) -> PersonalMemory | None:
        """按记忆键读取当前用户的个人记忆。"""
        with self._transaction() as conn:
            row = conn.execute(
                f"SELECT {_COLUMNS} FROM personal_memories WHERE memory_key = %s AND user_id = %s",
                (normalize_key(key), owner_id()),
            ).fetchone()
        return _memory(row) if row else None

    # 调用位置：PostgresMemoryStore 内的 search_memories 调用。
    def list_memories(self, limit: int = 20) -> tuple[PersonalMemory, ...]:
        """列出当前用户的记忆。"""
        if not 1 <= limit <= 20:
            raise ValueError("Memory recall limit must be between 1 and 20")
        with self._transaction() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM personal_memories WHERE user_id = %s ORDER BY updated_at DESC, memory_key LIMIT %s",
                (owner_id(), limit),
            ).fetchall()
        return tuple(_memory(row) for row in rows)

    # 调用位置：通过 PostgresMemoryStore 实例在会话服务、记忆工具与整理服务中调用。
    def forget(self, key: str) -> bool:
        """删除指定记忆。"""
        with self._transaction() as conn:
            row = conn.execute(
                "DELETE FROM personal_memories WHERE memory_key = %s AND user_id = %s RETURNING id",
                (normalize_key(key), owner_id()),
            ).fetchone()
        return row is not None

    # 调用位置：通过 PostgresMemoryStore 实例在会话服务、记忆工具与整理服务中调用。
    def search_memories(self, query: str, limit: int = 20) -> tuple[PersonalMemory, ...]:
        """限制词项检索的结果数量；没有匹配词时返回近期记忆。"""
        if not 1 <= limit <= 20:
            raise ValueError("Memory recall limit must be between 1 and 20")
        stop_words = {
            "what", "when", "where", "which", "does", "have", "that", "this",
            "with", "from", "about", "prefer", "preferred", "remember",
            "memory", "user", "the", "for", "and",
        }
        tokens = [
            token
            for token in dict.fromkeys(re.findall(r"[a-z0-9]{3,}", query.lower()))
            if token not in stop_words
        ][:8]
        if not tokens:
            return self.list_memories(limit)
        clauses = " OR ".join(
            "(memory_key ILIKE %s OR content ILIKE %s OR category ILIKE %s)"
            for _ in tokens
        )
        params = tuple(value for token in tokens for value in (f"%{token}%",) * 3)
        with self._transaction() as conn:
            rows = conn.execute(
                f"SELECT {_COLUMNS} FROM personal_memories WHERE user_id = %s AND ({clauses}) "
                "ORDER BY updated_at DESC, memory_key LIMIT %s",
                (owner_id(), *params, limit),
            ).fetchall()
        return tuple(_memory(row) for row in rows) if rows else self.list_memories(limit)

    # 调用位置：通过 PostgresMemoryStore 实例在会话服务、记忆工具与整理服务中调用。
    def apply_changes(
        self, changes: tuple[MemoryChange, ...], source_conversation_id: UUID
    ) -> None:
        """在单个事务中原子地应用已校验的记忆整理计划。"""
        if len(changes) > 3:
            raise ValueError("Too many memory changes")
        prepared: list[MemoryChange] = []
        for change in changes:
            key = normalize_key(change.key)
            if change.action == "upsert":
                content = (change.content or "").strip()
                if not content:
                    raise ValueError("Memory content is required")
                if secret_like_memory(key, content):
                    raise ValueError("Secret-like memory rejected")
                prepared.append(MemoryChange("upsert", key, content, change.category))
            elif change.action == "delete":
                prepared.append(MemoryChange("delete", key))
            else:
                raise ValueError("Unsupported memory change")
        with self._transaction() as conn:
            for change in prepared:
                if change.action == "upsert":
                    conn.execute(
                        """
                        INSERT INTO personal_memories
                            (id, memory_key, content, category, source_conversation_id, user_id)
                        VALUES (%s, %s, %s, %s, %s, %s)
                        ON CONFLICT (user_id, memory_key) DO UPDATE SET
                            content = EXCLUDED.content,
                            category = EXCLUDED.category,
                            source_conversation_id = EXCLUDED.source_conversation_id,
                            updated_at = now()
                        """,
                        (uuid4(), change.key, change.content, change.category, source_conversation_id, owner_id()),
                    )
                else:
                    conn.execute(
                        "DELETE FROM personal_memories WHERE memory_key = %s AND user_id = %s",
                        (change.key, owner_id()),
                    )

    # 调用位置：通过 PostgresMemoryStore 实例在会话服务、记忆工具与整理服务中调用。
    def remember_global(self, key: str, content: str) -> GlobalMemory:
        """保存所有用户可见的全局记忆。"""
        require_admin()
        key = normalize_key(key)
        content = content.strip()
        if not content or secret_like_memory(key, content):
            raise ValueError("Invalid global memory")
        with self._transaction() as conn:
            row = conn.execute("""
                INSERT INTO global_memories (memory_key, content) VALUES (%s, %s)
                ON CONFLICT (memory_key) DO UPDATE SET
                    content = EXCLUDED.content, updated_at = now()
                RETURNING memory_key, content, created_at, updated_at
            """, (key, content)).fetchone()
        return GlobalMemory(*row)

    # 调用位置：通过 PostgresMemoryStore 实例在会话服务、记忆工具与整理服务中调用。
    def list_global_memories(self, limit: int = 20) -> tuple[GlobalMemory, ...]:
        """列出全局记忆。"""
        require_admin()
        if not 1 <= limit <= 20:
            raise ValueError("Global memory limit must be between 1 and 20")
        with self._transaction() as conn:
            rows = conn.execute("""
                SELECT memory_key, content, created_at, updated_at
                FROM global_memories ORDER BY updated_at DESC LIMIT %s
            """, (limit,)).fetchall()
        return tuple(GlobalMemory(*row) for row in rows)
