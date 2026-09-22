"""Fake-model verification that LangChain executes SkySun memory tools."""

from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from skysun.conversation import ConversationService, InMemoryConversationStore
from skysun.langchain_agent import LangChainAgent
from skysun.memory import PersonalMemory, normalize_key


class FakeMemoryStore:
    def __init__(self) -> None:
        self.items: dict[str, PersonalMemory] = {}

    def remember(self, key: str, content: str, category: str | None = None, source_conversation_id=None) -> PersonalMemory:
        key = normalize_key(key)
        now = datetime.now(timezone.utc)
        old = self.items.get(key)
        item = PersonalMemory(old.id if old else uuid4(), key, content, category, source_conversation_id, old.created_at if old else now, now)
        self.items[key] = item
        return item

    def get(self, key: str) -> PersonalMemory | None:
        return self.items.get(normalize_key(key))

    def list_memories(self, limit: int = 20) -> tuple[PersonalMemory, ...]:
        return tuple(self.items.values())[:limit]

    def forget(self, key: str) -> bool:
        return self.items.pop(normalize_key(key), None) is not None

    def search_memories(self, query: str, limit: int = 20) -> tuple[PersonalMemory, ...]:
        return self.list_memories(limit)


class FakeMemoryCallingModel(BaseChatModel):
    _tool_names: list[str] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-memory-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeMemoryCallingModel":
        self._tool_names = [tool.name for tool in tools]
        return self

    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs: Any) -> ChatResult:
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_results:
            answer = AIMessage(content=str(tool_results[-1].content))
        else:
            request = str(messages[-1].content)
            if request == "remember":
                name, args = "remember_memory", {"key": "favorite_test_planet", "content": "Kepler-186f"}
            elif request == "update":
                name, args = "remember_memory", {"key": "favorite_test_planet", "content": "TRAPPIST-1e"}
            elif request == "recall":
                name, args = "recall_memories", {}
            else:
                name, args = "forget_memory", {"key": "favorite_test_planet"}
            answer = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])
        return ChatResult(generations=[ChatGeneration(message=answer)])


def test_memory_tools_through_agent_and_separate_conversations() -> None:
    model = FakeMemoryCallingModel()
    memory = FakeMemoryStore()
    service = ConversationService(LangChainAgent(model, memory), InMemoryConversationStore())

    first_id, answer = service.chat("remember")
    assert "Saved" in answer
    assert memory.get("favorite_test_planet").content == "Kepler-186f"
    from uuid import UUID
    assert memory.get("favorite_test_planet").source_conversation_id == UUID(first_id)
    second_id, answer = service.chat("recall")
    assert second_id != first_id
    assert "Kepler-186f" in answer
    service.chat("update", second_id)
    assert len(memory.items) == 1
    assert memory.get("favorite_test_planet").content == "TRAPPIST-1e"
    third_id, answer = service.chat("recall")
    assert third_id not in {first_id, second_id}
    assert "TRAPPIST-1e" in answer
    service.chat("forget")
    _, answer = service.chat("recall")
    assert answer == "No personal memories are stored."
    assert {"count_words", "remember_memory", "recall_memories", "forget_memory"} <= set(model._tool_names)


def test_preference_from_prior_conversation_is_labeled_as_personal_memory() -> None:
    memory = FakeMemoryStore()
    memory.remember("response_style", "Prefers concise answers", "preference")
    model = FakeMemoryCallingModel()
    service = ConversationService(LangChainAgent(model, memory), InMemoryConversationStore())
    _, answer = service.chat("recall")
    assert "Confirmed personal memories" in answer
    assert "response_style: Prefers concise answers" in answer
    assert "document knowledge" in answer
