"""LangChain adapter for SkySun's conversation-aware Agent interface."""

from typing import Any, Sequence

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from skysun.conversation import ConversationMessage
from skysun.identity import SYSTEM_PROMPT
from skysun.memory import MemoryStore
from skysun.tools import count_words, forget_memory, recall_memories, remember_memory


class LangChainAgent:
    def __init__(self, model: BaseChatModel, memory_store: MemoryStore | None = None) -> None:
        tools = [count_words]
        if memory_store is not None:
            @tool("remember_memory")
            def remember_memory_tool(key: str, content: str, category: str = "") -> str:
                """Save a durable user fact, preference, or goal. Use a stable snake_case semantic key (e.g. favorite_test_planet); reuse the same key to update. Never save passwords, API keys, tokens, private keys, or other credentials. Do not save passing trivia."""
                return remember_memory(memory_store, key, content, category or None)

            @tool("recall_memories")
            def recall_memories_tool() -> str:
                """Retrieve up to 20 stored personal memories when the user's question depends on information from other conversations. Do not invent memories absent from the result."""
                return recall_memories(memory_store)

            @tool("forget_memory")
            def forget_memory_tool(key: str) -> str:
                """Delete a stored personal memory by its semantic key when the user explicitly asks to forget it."""
                return forget_memory(memory_store, key)

            tools.extend([remember_memory_tool, recall_memories_tool, forget_memory_tool])
        self._agent = create_agent(
            model=model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[ModelCallLimitMiddleware(run_limit=5, exit_behavior="error")],
        )

    def reply(
        self, message: str, history: Sequence[ConversationMessage] = ()
    ) -> str:
        """Convert SkySun messages and run one LangChain invocation."""
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
