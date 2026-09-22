"""Personal memory integration against the dedicated local test database."""

import os
from collections.abc import Iterator
from uuid import UUID, uuid4

import psycopg
import pytest

from skysun.conversation import ConversationMessage
from skysun.memory import MemoryChange, normalize_key
from skysun.memory_curator import MemoryCurationService, MemoryDecision, MemoryPlan
from skysun.postgres_conversation import PostgresConversationStore
from skysun.postgres_memory import PostgresMemoryStore


@pytest.fixture
def stores() -> Iterator[tuple[PostgresMemoryStore, PostgresConversationStore]]:
    database_url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(database_url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("PostgreSQL integration tests require a *_test database")
    memory = PostgresMemoryStore(database_url)
    conversations = PostgresConversationStore(database_url)
    memory.initialize_schema()
    conversations.initialize_schema()
    yield memory, conversations


def test_memory_create_update_recall_forget_and_repeat_schema(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
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


def test_memory_global_while_conversation_history_isolated(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
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


def test_memory_validation_and_safe_connection_error() -> None:
    assert normalize_key("Favorite-Test Planet") == "favorite_test_planet"
    with pytest.raises(ValueError):
        normalize_key("bad/key")
    memory = PostgresMemoryStore("postgresql://name:example-password@127.0.0.1:1/unavailable")
    with pytest.raises(RuntimeError, match="Cannot connect to SkySun PostgreSQL") as exc:
        memory.initialize_schema()
    assert "example-password" not in str(exc.value)


def test_memory_sql_error_omits_database_detail(monkeypatch: pytest.MonkeyPatch) -> None:
    class BadConnection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, *args):
            raise psycopg.Error("sensitive database detail")

    memory = PostgresMemoryStore("unused")
    monkeypatch.setattr(memory, "_connect", lambda: BadConnection())
    with pytest.raises(RuntimeError, match="Personal memory PostgreSQL operation failed") as exc:
        memory.remember("test_key", "test value")
    assert "sensitive database detail" not in str(exc.value)


def test_only_connection_failures_are_retryable() -> None:
    from types import SimpleNamespace
    from skysun.postgres_memory import _is_transient

    assert _is_transient(SimpleNamespace(sqlstate=None))
    assert _is_transient(SimpleNamespace(sqlstate="08006"))
    assert _is_transient(SimpleNamespace(sqlstate="57P01"))
    assert not _is_transient(SimpleNamespace(sqlstate="28P01"))


def test_curator_batch_is_atomic_and_lexical_recall_is_bounded(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
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


def test_obvious_secret_label_is_rejected_by_storage(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
    memory, _ = stores
    with pytest.raises(ValueError, match="Secret-like memory rejected"):
        memory.remember("api_key", "FAKE-TEST-TOKEN")
    assert memory.get("api_key") is None


def test_accepted_candidate_persists_with_source_and_duplicate_is_skipped(
    stores: tuple[PostgresMemoryStore, PostgresConversationStore],
) -> None:
    memory, _ = stores
    key = f"v08_test_preference_{uuid4().hex[:8]}"
    source = uuid4()

    class Curator:
        def decide(self, context):
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
