"""LangChain adapter for SkySun's conversation-aware Agent interface."""

from typing import Any, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel

from skysun.conversation import ConversationMessage
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

    def reply(
        self, message: str, history: Sequence[ConversationMessage] = ()
    ) -> str:
        """Convert SkySun messages and run one stateless LangChain invocation."""
        messages = [
            {"role": turn.role, "content": turn.content}
            for turn in history
        ]
        messages.append({"role": "user", "content": message})
        result: dict[str, Any] = self._agent.invoke({"messages": messages})
        final_messages = result["messages"]
        if not final_messages:
            raise RuntimeError("Agent returned no messages")
        content = final_messages[-1].content
        if not isinstance(content, str) or not content:
            raise RuntimeError("Agent returned no text answer")
        return content
