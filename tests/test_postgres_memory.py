"""Personal memory integration against the dedicated local test database."""
# 本文件负责测试用例中的 test、postgres、memory 相关流程。

import os
from collections.abc import Iterator
from uuid import UUID, uuid4

import psycopg
import pytest

from skysun.memory.conversation import ConversationMessage
from skysun.memory.memory import MemoryChange, normalize_key
from skysun.memory.memory_curator import MemoryCurationService, MemoryDecision, MemoryPlan
from skysun.memory.postgres_conversation import PostgresConversationStore
from skysun.memory.postgres_memory import PostgresMemoryStore


# 调用位置：pytest 在依赖此 fixture 的测试中调用。
@pytest.fixture
def stores() -> Iterator[tuple[PostgresMemoryStore, PostgresConversationStore]]:
    """为测试提供 stores 所需的输入或模拟行为。"""
    database_url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(database_url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("PostgreSQL integration tests require a *_test database")
    memory = PostgresMemoryStore(database_url)
    conversations = PostgresConversationStore(database_url)
    memory.initialize_schema()
    conversations.initialize_schema()
    yield memory, conversations


# 调用位置：pytest 执行该测试用例时调用。
def test_memory_create_update_recall_forget_and_repeat_schema(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
    """验证记忆可以创建、更新、检索和删除，重复建表也保持安全。"""
    memory, _ = stores
    key = f"v04_test_planet_{uuid4().hex[:8]}"
    try:
        first = memory.remember(key, "Kepler-186f", "preference")
        assert memory.get(key) == first
        assert first.created_at == first.updated_at
        memory.initialize_schema()
        second = memory.remember(key, "TRAPPIST-1e", "preference")
        assert second.id == first.id
        assert second.created_at == first.created_at
        assert second.content == "TRAPPIST-1e"
        assert sum(item.key == key for item in memory.list_memories()) == 1
        assert memory.forget(key)
        assert memory.get(key) is None
        assert not memory.forget(key)
    finally:
        memory.forget(key)


# 调用位置：pytest 执行该测试用例时调用。
def test_memory_global_while_conversation_history_isolated(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
    """验证全局记忆可共享，而会话历史仍按用户隔离。"""
    memory, conversations = stores
    first_id = conversations.create()
    second_id = conversations.create()
    key = f"v04_test_goal_{uuid4().hex[:8]}"
    try:
        conversations.append_exchange(first_id, "v04-test: remember", "v04-test: saved")
        stored = memory.remember(key, "Learn Python", source_conversation_id=UUID(first_id))
        conversations.append_exchange(second_id, "v04-test: what goal?", "v04-test: Python")
        assert stored.source_conversation_id == UUID(first_id)
        assert memory.get(key).content == "Learn Python"
        assert conversations.get_history(second_id) == (
            ConversationMessage("user", "v04-test: what goal?"),
            ConversationMessage("assistant", "v04-test: Python"),
        )
        assert conversations.get_history(first_id)[0].content == "v04-test: remember"
    finally:
        memory.forget(key)
        with psycopg.connect(os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")) as conn:
            for conversation_id in (first_id, second_id):
                conn.execute("DELETE FROM conversation_messages WHERE conversation_id = %s", (UUID(conversation_id),))
                conn.execute("DELETE FROM conversations WHERE id = %s", (UUID(conversation_id),))


# 调用位置：pytest 执行该测试用例时调用。
def test_memory_validation_and_safe_connection_error() -> None:
    """验证记忆输入校验正确且连接错误不泄露敏感信息。"""
    assert normalize_key("Favorite-Test Planet") == "favorite_test_planet"
    with pytest.raises(ValueError):
        normalize_key("bad/key")
    memory = PostgresMemoryStore("postgresql://name:example-password@127.0.0.1:1/unavailable")
    with pytest.raises(RuntimeError, match="Cannot connect to SkySun PostgreSQL") as exc:
        memory.initialize_schema()
    assert "example-password" not in str(exc.value)


# 调用位置：pytest 执行该测试用例时调用。
def test_memory_sql_error_omits_database_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证记忆存储的 SQL 错误不会泄露数据库详情。"""
    # 以下集中定义 BadConnection：数据结构或组件相关的数据和操作。
    class BadConnection:
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def __enter__(self):
            """为测试准备 __enter__ 所需的替身或输入。"""
            return self

        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def __exit__(self, *args):
            """为测试准备 __exit__ 所需的替身或输入。"""
            return False

        # 调用位置：tests/test_postgres_memory.py 的 test_memory_global_while_conversation_history_isolated 等处调用。
        def execute(self, *args):
            """执行 Agent 或工作流并返回结构化结果。"""
            raise psycopg.Error("sensitive database detail")

    memory = PostgresMemoryStore("unused")
    monkeypatch.setattr(memory, "_connect", lambda: BadConnection())
    with pytest.raises(RuntimeError, match="Personal memory PostgreSQL operation failed") as exc:
        memory.remember("test_key", "test value")
    assert "sensitive database detail" not in str(exc.value)


# 调用位置：pytest 执行该测试用例时调用。
def test_only_connection_failures_are_retryable() -> None:
    """验证只有临时数据库连接故障会被重试。"""
    from types import SimpleNamespace
    from skysun.memory.postgres_memory import _is_transient

    assert _is_transient(SimpleNamespace(sqlstate=None))
    assert _is_transient(SimpleNamespace(sqlstate="08006"))
    assert _is_transient(SimpleNamespace(sqlstate="57P01"))
    assert not _is_transient(SimpleNamespace(sqlstate="28P01"))


# 调用位置：pytest 执行该测试用例时调用。
def test_curator_batch_is_atomic_and_lexical_recall_is_bounded(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
    """验证记忆整理批次具有原子性，词项检索有数量上限。"""
    memory, _ = stores
    prefix = f"v04_test_batch_{uuid4().hex[:8]}"
    source = uuid4()
    first_key = f"{prefix}_editor"
    second_key = f"{prefix}_goal"
    try:
        memory.apply_changes((
            MemoryChange("upsert", first_key, "PyCharm", "preference"),
            MemoryChange("upsert", second_key, "AI engineer", "goal"),
        ), source)
        assert memory.get(first_key).source_conversation_id == source
        assert memory.search_memories("PyCharm editor", limit=1)[0].key == first_key
        assert len(memory.search_memories("editor", limit=1)) == 1

        with pytest.raises(RuntimeError, match="Personal memory PostgreSQL operation failed"):
            memory.apply_changes((
                MemoryChange("upsert", first_key, "VS Code", "preference"),
                MemoryChange("upsert", second_key, "invalid\x00value", "goal"),
            ), source)
        assert memory.get(first_key).content == "PyCharm"
        assert memory.get(second_key).content == "AI engineer"
    finally:
        memory.forget(first_key)
        memory.forget(second_key)


# 调用位置：pytest 执行该测试用例时调用。
def test_obvious_secret_label_is_rejected_by_storage(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
    """验证存储层拒绝明显带密钥标签的记忆。"""
    memory, _ = stores
    with pytest.raises(ValueError, match="Secret-like memory rejected"):
        memory.remember("api_key", "FAKE-TEST-TOKEN")
    assert memory.get("api_key") is None


# 调用位置：pytest 执行该测试用例时调用。
def test_accepted_candidate_persists_with_source_and_duplicate_is_skipped(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
    """验证被接受候选项持久化包含来源与重复的被跳过。"""
    memory, _ = stores
    key = f"v08_test_preference_{uuid4().hex[:8]}"
    source = uuid4()

    # 以下集中定义 Curator：数据结构或组件相关的数据和操作。
    class Curator:
        # 调用位置：src/skysun/memory/memory_curator.py 的 MemoryCurationService.process、tests/test_memory_curator.py 的 test_langchain_curator_uses_typed_json_output_without_network、tests/test_memory_extraction.py 的 test_valid_structured_output_is_accepted_without_retry 等处调用。
        def decide(self, context):
            """根据对话和已有记忆决定新增、更新或删除内容。"""
            return MemoryPlan(decisions=[MemoryDecision(
                action="upsert", key=key, content="Prefers concise answers",
                category="preference", importance="high",
            )])

    service = MemoryCurationService(Curator(), memory)
    try:
        service.process("Please keep answers concise.", (), str(source))
        first = memory.get(key)
        assert first.content == "Prefers concise answers"
        assert first.source_conversation_id == source
        service.process("Keep answers concise.", (), str(uuid4()))
        assert memory.get(key).id == first.id
        assert memory.get(key).updated_at == first.updated_at
    finally:
        memory.forget(key)
