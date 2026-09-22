"""Tool failures remain visible to the Agent without exposing exception text."""

import json
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from skysun.langchain_agent import LangChainAgent
from skysun.tool_reliability import RetryableToolError, configure_validation, run_tool


def test_successful_execution_and_argument_validation() -> None:
    @tool("echo")
    def echo(value: str) -> str:
        """Echo a value."""
        return run_tool("echo", lambda: value)

    registered = configure_validation(echo)
    assert registered.invoke({"value": "ready"}) == "ready"
    failure = json.loads(registered.invoke({"wrong": "value"}))
    assert failure["ok"] is False
    assert failure["tool"] == "echo"
    assert failure["code"] == "invalid_arguments"


def test_tool_exception_is_structured_and_sanitized() -> None:
    def fails():
        raise RuntimeError("private database detail")

    failure = json.loads(run_tool("search", fails))
    assert failure["code"] == "execution_failed"
    assert failure["attempts"] == 1
    assert "private database detail" not in json.dumps(failure)


def test_transient_read_failure_retries_once() -> None:
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RetryableToolError("temporary")
        return "recovered"

    assert run_tool("recall", flaky, retries=1) == "recovered"
    assert attempts == 2


def test_retry_exhaustion_reports_failure() -> None:
    attempts = 0

    def fails():
        nonlocal attempts
        attempts += 1
        raise TimeoutError("temporary")

    failure = json.loads(run_tool("recall", fails, retries=1))
    assert failure["code"] == "temporary_failure"
    assert failure["attempts"] == 2
    assert failure["retryable"] is True
    assert attempts == 2


def test_agent_reuses_failed_memory_read_within_one_turn() -> None:
    class RepeatingModel(BaseChatModel):
        @property
        def _llm_type(self) -> str:
            return "repeating-memory-read"

        def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "RepeatingModel":
            return self

        def _generate(self, messages: list[BaseMessage], stop=None, **kwargs: Any) -> ChatResult:
            results = [item for item in messages if isinstance(item, ToolMessage)]
            if len(results) < 2:
                answer = AIMessage(content="", tool_calls=[{
                    "name": "recall_memories", "args": {"query": "style"},
                    "id": f"read_{len(results)}",
                }])
            else:
                answer = AIMessage(content="You prefer bullet points.")
            return ChatResult(generations=[ChatGeneration(message=answer)])

    class FailingStore:
        def __init__(self):
            self.calls = 0

        def search_memories(self, query, limit=20):
            self.calls += 1
            raise RetryableToolError("unavailable")

    store = FailingStore()
    answer = LangChainAgent(RepeatingModel(), store).reply("What is my style?")
    assert store.calls == 2
    assert "couldn't access personal memory" in answer
    assert "bullet" not in answer
