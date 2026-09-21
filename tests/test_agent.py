from typing import Any, Sequence

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ValidationError

from skysun.config import Settings
from skysun.langchain_agent import LangChainAgent


class FakeToolCallingModel(BaseChatModel):
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


def test_langchain_owns_tool_call_loop() -> None:
    agent = LangChainAgent(FakeToolCallingModel())

    assert agent.reply("Count the words") == "Count: 3"
    assert agent.reply("Count again") == "Count: 3"


def test_settings_require_key_and_use_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="deepseek_api_key"):
        Settings(_env_file=None)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    settings = Settings(_env_file=None)
    assert settings.skysun_model == "deepseek-flash"
    assert settings.deepseek_api_key.get_secret_value() == "test-key"


def test_composition_root_uses_agent_without_network(monkeypatch: pytest.MonkeyPatch) -> None:
    from skysun import main

    monkeypatch.setattr(
        main,
        "load_settings",
        lambda: Settings(deepseek_api_key="test-key", _env_file=None),
    )
    monkeypatch.setattr(main, "create_deepseek", lambda settings: FakeToolCallingModel())

    client = TestClient(main.create_app())
    response = client.post("/chat", json={"message": "Count the words"})

    assert response.status_code == 200
    assert response.json() == {"reply": "Count: 3"}
