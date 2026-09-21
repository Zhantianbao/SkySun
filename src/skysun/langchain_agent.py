"""LangChain adapter for SkySun's stateless Agent interface."""

from typing import Any

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel

from skysun.identity import SYSTEM_PROMPT
from skysun.tools import count_words


class LangChainAgent:
    def __init__(self, model: BaseChatModel) -> None:
        self._agent = create_agent(
            model=model,
            tools=[count_words],
            system_prompt=SYSTEM_PROMPT,
            middleware=[ModelCallLimitMiddleware(run_limit=5, exit_behavior="error")],
        )

    def reply(self, message: str) -> str:
        """Run a fresh agent invocation and return only its final text."""
        result: dict[str, Any] = self._agent.invoke(
            {"messages": [{"role": "user", "content": message}]}
        )
        messages = result["messages"]
        if not messages:
            raise RuntimeError("Agent returned no messages")
        content = messages[-1].content
        if not isinstance(content, str) or not content:
            raise RuntimeError("Agent returned no text answer")
        return content
