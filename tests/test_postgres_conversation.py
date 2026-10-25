"""Integration tests use only the dedicated local *_test database."""
# 本文件负责测试用例中的 test、postgres、conversation 相关流程。

import os
from collections.abc import Callable, Iterator
from uuid import UUID, uuid4

import psycopg
import pytest

from skysun.memory.conversation import ConversationMessage, ConversationService
from skysun.memory.postgres_conversation import PostgresConversationStore


# 调用位置：pytest 在依赖此 fixture 的测试中调用。
@pytest.fixture
def postgres_store() -> Iterator[tuple[PostgresConversationStore, Callable[[], str]]]:
    """为测试提供 postgres_store 所需的输入或模拟行为。"""
    database_url = os.environ.get(
        "SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test"
    )
    with psycopg.connect(database_url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("PostgreSQL integration tests require a *_test database")

    store = PostgresConversationStore(database_url)
    store.initialize_schema()
    created_ids: list[str] = []

    # 调用位置：tests/test_postgres_conversation.py 的 test_postgres_creation_order_isolation_and_idempotent_schema、tests/test_postgres_conversation.py 的 test_postgres_rolls_back_both_messages_on_second_insert_failure、tests/test_postgres_conversation.py 的 test_agent_failure_does_not_persist_partial_turn 等处调用。
    def create() -> str:
        """创建新的记录或任务。"""
        conversation_id = store.create()
        created_ids.append(conversation_id)
        return conversation_id

    try:
        yield store, create
    finally:
        with psycopg.connect(database_url) as conn:
            for conversation_id in created_ids:
                identifier = UUID(conversation_id)
                conn.execute(
                    "DELETE FROM conversation_messages WHERE conversation_id = %s",
                    (identifier,),
                )
                conn.execute(
                    "DELETE FROM conversations WHERE id = %s",
                    (identifier,),
                )


# 调用位置：pytest 执行该测试用例时调用。
def test_postgres_creation_order_isolation_and_idempotent_schema(
    postgres_store: tuple[PostgresConversationStore, Callable[[], str]],
) -> None:
    """验证数据库会话创建顺序、用户隔离和重复建表的幂等性。"""
    store, create = postgres_store
    first_id = create()
    second_id = create()
    store.append_exchange(first_id, "first", "answer one")
    store.append_exchange(first_id, "second", "answer two")
    store.append_exchange(second_id, "other", "separate")
    store.initialize_schema()

    assert store.get_history(first_id) == (
        ConversationMessage("user", "first"),
        ConversationMessage("assistant", "answer one"),
        ConversationMessage("user", "second"),
        ConversationMessage("assistant", "answer two"),
    )
    assert store.get_history(second_id) == (
        ConversationMessage("user", "other"),
        ConversationMessage("assistant", "separate"),
    )
    with pytest.raises(KeyError):
        store.get_history(str(uuid4()))


# 调用位置：pytest 执行该测试用例时调用。
def test_postgres_rolls_back_both_messages_on_second_insert_failure(
    postgres_store: tuple[PostgresConversationStore, Callable[[], str]],
) -> None:
    """验证第二条消息写入失败时，两条消息都回滚。"""
    store, create = postgres_store
    conversation_id = create()

    with pytest.raises(psycopg.errors.CheckViolation):
        store.append_exchange(conversation_id, "should roll back", "")

    assert store.get_history(conversation_id) == ()


# 调用位置：pytest 执行该测试用例时调用。
def test_agent_failure_does_not_persist_partial_turn(
    postgres_store: tuple[PostgresConversationStore, Callable[[], str]],
) -> None:
    """验证 Agent 失败时不会只保存半轮对话。"""
    store, create = postgres_store
    conversation_id = create()

    # 以下集中定义 FailingAgent：Agent 执行层相关的数据和操作。
    class FailingAgent:
        # 调用位置：FailingAgent 的调用方通过该方法使用。
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None) -> str:
            """生成并返回本次对话的最终回复。"""
            raise RuntimeError("model unavailable")

    with pytest.raises(RuntimeError, match="model unavailable"):
        ConversationService(FailingAgent(), store).chat("hello", conversation_id)

    assert store.get_history(conversation_id) == ()


# 调用位置：pytest 执行该测试用例时调用。
def test_bad_database_url_error_omits_password() -> None:
    """验证数据库地址错误不会在异常中泄露密码。"""
    store = PostgresConversationStore(
        "postgresql://name:example-password@127.0.0.1:1/unavailable"
    )
    with pytest.raises(RuntimeError, match="Cannot connect to SkySun PostgreSQL") as exc:
        store.initialize_schema()
    assert "example-password" not in str(exc.value)
