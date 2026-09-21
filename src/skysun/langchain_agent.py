"""LangChain adapter for SkySun's conversation-aware Agent interface."""

from typing import Any, Sequence
from uuid import UUID

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
        self._model = model
        self._memory_store = memory_store

    def reply(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
        conversation_id: str | None = None,
    ) -> str:
        """Run one Agent invocation with tools scoped to this conversation."""
        tools = [count_words]
        if self._memory_store is not None:
            memory_store = self._memory_store
            source_id = UUID(conversation_id) if conversation_id else None

            @tool("remember_memory")
            def remember_memory_tool(key: str, content: str, category: str = "") -> str:
                """Save a durable user fact, preference, or goal. Reuse a stable snake_case semantic key to update. Never save passwords, API keys, tokens, private keys, or credentials."""
                return remember_memory(memory_store, key, content, category or None, source_id)

            @tool("recall_memories")
            def recall_memories_tool(query: str = "") -> str:
                """Retrieve up to 20 personal memories for user-specific questions. Pass relevant keywords in query for basic lexical lookup; empty query lists recent memories. Never invent absent memories."""
                return recall_memories(memory_store, query)

            @tool("forget_memory")
            def forget_memory_tool(key: str) -> str:
                """Delete a stored personal memory by its semantic key when the user explicitly asks to forget it."""
                return forget_memory(memory_store, key)

            tools.extend([remember_memory_tool, recall_memories_tool, forget_memory_tool])
        agent = create_agent(
            model=self._model,
            tools=tools,
            system_prompt=SYSTEM_PROMPT,
            middleware=[ModelCallLimitMiddleware(run_limit=5, exit_behavior="error")],
        )
        messages = [{"role": turn.role, "content": turn.content} for turn in history]
        messages.append({"role": "user", "content": message})
        result: dict[str, Any] = agent.invoke({"messages": messages})
        final_messages = result["messages"]
        if not final_messages:
            raise RuntimeError("Agent returned no messages")
        content = final_messages[-1].content
        if not isinstance(content, str) or not content:
            raise RuntimeError("Agent returned no text answer")
        return content
