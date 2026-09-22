"""Memory curation policy, application, and failure isolation without real API calls."""

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from skysun.api import build_api
from skysun.conversation import ConversationService, InMemoryConversationStore
from skysun.memory import MemoryChange, PersonalMemory
from skysun.memory_curator import (
    MemoryCurationService, MemoryDecision, MemoryPlan, create_candidates,
)


class RecordingStore:
    def __init__(self) -> None:
        self.items: dict[str, PersonalMemory] = {}
        self.batches: list[tuple[tuple[MemoryChange, ...], UUID]] = []

    def list_memories(self, limit: int = 20) -> tuple[PersonalMemory, ...]:
        return tuple(self.items.values())[:limit]

    def get(self, key: str) -> PersonalMemory | None:
        return self.items.get(key)

    def apply_changes(self, changes: tuple[MemoryChange, ...], source_conversation_id: UUID) -> None:
        self.batches.append((changes, source_conversation_id))
        now = datetime.now(timezone.utc)
        for change in changes:
            if change.action == "delete":
                self.items.pop(change.key, None)
            else:
                old = self.items.get(change.key)
                self.items[change.key] = PersonalMemory(
                    old.id if old else uuid4(), change.key, change.content,
                    change.category, source_conversation_id,
                    old.created_at if old else now, now,
                )


class ScriptedCurator:
    def __init__(self, plans: list[MemoryPlan | Exception]) -> None:
        self.plans = plans
        self.contexts = []

    def decide(self, context):
        self.contexts.append(context)
        result = self.plans.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class ReplyAgent:
    def reply(self, message, history=(), conversation_id=None):
        return f"Reply: {message}"


def test_automatic_formation_update_multiple_and_forget_with_source() -> None:
    store = RecordingStore()
    curator = ScriptedCurator([
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="Preferred Editor", content="VS Code", category="preference"),
            MemoryDecision(action="upsert", key="career_goal", content="AI application engineer", category="goal"),
        ]),
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="preferred_editor", content="PyCharm", category="preference"),
        ]),
        MemoryPlan(decisions=[
            MemoryDecision(action="delete", key="preferred_editor"),
        ]),
    ])
    conversations = InMemoryConversationStore()
    service = ConversationService(
        ReplyAgent(), conversations, MemoryCurationService(curator, store)
    )
    first_id, _ = service.chat("My editor is VS Code and my goal is AI work.")
    first_memory_id = store.items["preferred_editor"].id
    assert len(store.items) == 2
    assert store.items["preferred_editor"].source_conversation_id == UUID(first_id)
    assert len(store.batches[0][0]) == 2
    second_id, _ = service.chat("I've switched to PyCharm.")
    assert store.items["preferred_editor"].id == first_memory_id
    assert store.items["preferred_editor"].content == "PyCharm"
    assert store.items["preferred_editor"].source_conversation_id == UUID(second_id)
    assert len(store.items) == 2
    service.chat("Forget my editor.")
    assert "preferred_editor" not in store.items
    assert "career_goal" in store.items
    assert curator.contexts[1].current_memories[0].key == "preferred_editor"
    assert conversations.get_history(first_id)[0].content.startswith("My editor")


def test_transient_and_malformed_or_secret_decisions_do_not_write_memory() -> None:
    store = RecordingStore()
    curator = ScriptedCurator([
        MemoryPlan(),
        MemoryPlan(decisions=[MemoryDecision(action="upsert", key="bad/key", content="x")]),
        MemoryPlan(decisions=[MemoryDecision(action="upsert", key="api_key", content="private value")]),
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="preferred_editor", content="VS Code"),
            MemoryDecision(action="upsert", key="missing_content"),
        ]),
    ])
    service = ConversationService(
        ReplyAgent(), InMemoryConversationStore(), MemoryCurationService(curator, store)
    )
    for message in (
        "I'm drinking coffee right now.",
        "malformed",
        "Remember my API key is a private value.",
        "mixed valid and invalid",
    ):
        _, reply = service.chat(message)
        assert reply == f"Reply: {message}"
    assert store.items == {}
    assert store.batches == []


def test_curator_failure_keeps_http_reply_and_persisted_conversation() -> None:
    store = RecordingStore()
    curator = ScriptedCurator([RuntimeError("secret internal failure")])
    conversations = InMemoryConversationStore()
    client = TestClient(build_api(ConversationService(
        ReplyAgent(), conversations, MemoryCurationService(curator, store)
    )))
    response = client.post("/chat", json={"message": "My editor is PyCharm."})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Reply: My editor is PyCharm."
    assert [turn.role for turn in conversations.get_history(body["conversation_id"])] == [
        "user", "assistant"
    ]
    assert store.items == {}


def test_bounded_decisions_and_missing_fields() -> None:
    with pytest.raises(ValueError):
        MemoryPlan(decisions=[MemoryDecision(action="ignore")] * 4)
    from skysun.memory_curator import validate_plan
    with pytest.raises(ValueError):
        validate_plan(MemoryPlan(decisions=[MemoryDecision(action="delete")]))



def test_langchain_curator_uses_typed_json_output_without_network() -> None:
    from skysun.langchain_memory_curator import LangChainMemoryCurator
    from skysun.memory_curator import CurationContext

    class FakeRunnable:
        def invoke(self, messages):
            assert "current_user_message" in messages[1][1]
            return {"decisions": [{"action": "upsert", "key": "preferred_editor", "content": "PyCharm"}]}

    class FakeModel:
        def with_structured_output(self, schema, method, include_raw):
            assert schema is MemoryPlan
            assert method == "json_mode"
            assert include_raw is True
            return FakeRunnable()

    plan = LangChainMemoryCurator(FakeModel()).decide(
        CurationContext("My editor is PyCharm.", (), (), uuid4())
    )
    assert plan.decisions[0].key == "preferred_editor"


def test_candidate_creation_keeps_importance_and_conversation_source() -> None:
    source = uuid4()
    plan = MemoryPlan(decisions=[
        MemoryDecision(action="upsert", key="Preferred Editor", content="  PyCharm  ",
                       category="preference", importance="high"),
        MemoryDecision(action="ignore"),
    ])
    candidates = create_candidates(plan, source)
    assert len(candidates) == 1
    assert (candidates[0].key, candidates[0].content, candidates[0].category) == (
        "preferred_editor", "PyCharm", "preference"
    )
    assert candidates[0].importance == "high"
    assert candidates[0].source_conversation_id == source


def test_curator_rejects_low_value_and_duplicate_candidates() -> None:
    store = RecordingStore()
    curator = ScriptedCurator([
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="preferred_editor", content="PyCharm",
                           category="preference", importance="high"),
            MemoryDecision(action="upsert", key="today_snack", content="A biscuit",
                           importance="low"),
        ]),
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="editor_choice", content=" pycharm ",
                           category="tool", importance="high"),
        ]),
    ])
    service = MemoryCurationService(curator, store)
    service.process("I prefer PyCharm.", (), str(uuid4()))
    service.process("I still prefer PyCharm.", (), str(uuid4()))
    assert tuple(store.items) == ("preferred_editor",)
    assert len(store.batches) == 1
