from typing import Any, Sequence
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr, ValidationError

from skysun.core.config import Settings
from skysun.memory.conversation import ConversationMessage
from skysun.core.langchain_agent import LangChainAgent


class FakeToolCallingModel(BaseChatModel):
    _seen: list[list[str]] = PrivateAttr(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake-tool-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingModel":
        assert any(tool.name == "count_words" for tool in tools)
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        self._seen.append([str(message.content) for message in messages])
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if tool_messages:
            answer = AIMessage(content=f"Count: {tool_messages[-1].content}")
        else:
            answer = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "count_words",
                        "args": {"text": "one two three"},
                        "id": "call_1",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=answer)])


def test_langchain_owns_tool_call_loop_and_converts_history() -> None:
    model = FakeToolCallingModel()
    agent = LangChainAgent(model)
    history = (
        ConversationMessage("user", "older question"),
        ConversationMessage("assistant", "older answer"),
    )

    assert agent.reply("Count the words", history) == "Count: 3"
    assert "older question" in model._seen[0]
    assert "older answer" in model._seen[0]
    assert agent.reply("Count again") == "Count: 3"
    assert "older question" not in model._seen[2]


def test_settings_require_key_and_use_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql:///skysun_test")
    with pytest.raises(ValidationError, match="deepseek_api_key"):
        Settings(_env_file=None)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    settings = Settings(_env_file=None)
    assert settings.skysun_model == "deepseek-flash"
    assert settings.database_url.get_secret_value() == "postgresql:///skysun_test"
    assert settings.deepseek_api_key.get_secret_value() == "test-key"

    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)


@pytest.mark.parametrize("token", ["", "   "])
def test_settings_reject_empty_admin_password(token):
    with pytest.raises(ValidationError, match="skysun_admin_token"):
        Settings(_env_file=None, deepseek_api_key="test", database_url="postgresql:///skysun_test", skysun_admin_token=token)


def test_composition_root_uses_agent_without_network(monkeypatch: pytest.MonkeyPatch, create_account) -> None:
    from skysun.core import main

    monkeypatch.setattr(
        main,
        "load_settings",
        lambda: Settings(
            deepseek_api_key="test-key",
            database_url="postgresql:///skysun_test",
            _env_file=None,
        ),
    )
    monkeypatch.setattr(main, "create_deepseek", lambda settings: FakeToolCallingModel())

    from skysun.memory.conversation import InMemoryConversationStore

    from skysun.memory.memory_curator import MemoryPlan

    class NoOpCurator:
        def decide(self, context):
            return MemoryPlan()

    class EmptyKnowledgeStore:
        def search(self, query_embedding, limit=4, max_cosine_distance=0.35):
            return ()

    class FakeEmbeddings:
        dimension = 1024

        def embed_query(self, text):
            return (1.0,) + (0.0,) * 1023

    client = TestClient(main.create_app(
        InMemoryConversationStore(), curator=NoOpCurator(),
        knowledge_store=EmptyKnowledgeStore(), embeddings=FakeEmbeddings(),
    ))
    user, token = create_account(main.PostgresUserStore("postgresql:///skysun_test"))
    try:
        response = client.post("/chat", json={"message": "Count the words"},
                               headers={"Authorization": f"Bearer {token}"})
    finally:
        with psycopg.connect("postgresql:///skysun_test") as conn:
            conn.execute("DELETE FROM agent_tasks WHERE user_id = %s", (user.user_id,))
            conn.execute("DELETE FROM skysun_users WHERE user_id = %s", (user.user_id,))

    assert response.status_code == 200
    assert response.json()["reply"] == "Count: 3"
    UUID(response.json()["conversation_id"])
