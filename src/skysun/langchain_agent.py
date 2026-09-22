"""LangChain adapter for SkySun's conversation-aware Agent interface."""

from typing import Any, Sequence
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from skysun.conversation import ConversationMessage
from skysun.embeddings import EmbeddingProvider
from skysun.identity import SYSTEM_PROMPT
from skysun.knowledge import KnowledgeStore
from skysun.knowledge_grounding import (
    EvidenceEvaluator, GroundedKnowledge, GroundedKnowledgeService, GroundingStatus, QueryRewriter,
)
from skysun.langchain_knowledge_grounding import LangChainEvidenceEvaluator, LangChainQueryRewriter
from skysun.memory import MemoryStore
from skysun.tools import (
    count_words,
    forget_memory,
    recall_memories,
    remember_memory,
    search_knowledge,
    format_knowledge_evidence,
)


class LangChainAgent:
    def __init__(
        self,
        model: BaseChatModel,
        memory_store: MemoryStore | None = None,
        knowledge_store: KnowledgeStore | None = None,
        embeddings: EmbeddingProvider | None = None,
        evidence_evaluator: EvidenceEvaluator | None = None,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        if (knowledge_store is None) != (embeddings is None):
            raise ValueError("Knowledge store and embeddings must be supplied together")
        self._model = model
        self._memory_store = memory_store
        self._knowledge_store = knowledge_store
        self._embeddings = embeddings
        self._evidence_evaluator = evidence_evaluator
        self._query_rewriter = query_rewriter

    def reply(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
        conversation_id: str | None = None,
    ) -> str:
        """Run one Agent invocation with tools scoped to this conversation."""
        tools = [count_words]
        knowledge_outcome: GroundedKnowledge | None = None
        if self._memory_store is not None:
            memory_store = self._memory_store
            source_id = UUID(conversation_id) if conversation_id else None

            @tool("remember_memory")
            def remember_memory_tool(key: str, content: str, category: str = "") -> str:
                """Save a durable user fact, preference, or goal. Reuse a stable snake_case semantic key to update. Never save passwords, API keys, tokens, private keys, or credentials."""
                return remember_memory(memory_store, key, content, category or None, source_id)

            @tool("recall_memories")
            def recall_memories_tool(query: str = "") -> str:
                """Retrieve up to 20 personal memories for user-specific questions. Pass relevant keywords for basic lexical lookup; empty query lists recent memories. Never invent absent memories."""
                return recall_memories(memory_store, query)

            @tool("forget_memory")
            def forget_memory_tool(key: str) -> str:
                """Delete a stored personal memory by its semantic key when the user explicitly asks to forget it."""
                return forget_memory(memory_store, key)

            tools.extend([remember_memory_tool, recall_memories_tool, forget_memory_tool])
        if self._knowledge_store is not None and self._embeddings is not None:
            knowledge_store = self._knowledge_store
            embeddings = self._embeddings

            @tool("search_knowledge")
            def search_knowledge_tool(query: str, limit: int = 4) -> str:
                """Find validated evidence for a question about stored documents. Call once per user question. The application controls any retry. Do not use for personal memory or general facts."""
                nonlocal knowledge_outcome
                if knowledge_outcome is None:
                    evaluator = self._evidence_evaluator or LangChainEvidenceEvaluator(self._model)
                    rewriter = self._query_rewriter or LangChainQueryRewriter(self._model)
                    service = GroundedKnowledgeService(knowledge_store, embeddings, evaluator, rewriter)
                    knowledge_outcome = search_knowledge(service, message, limit)
                return format_knowledge_evidence(knowledge_outcome)

            tools.append(search_knowledge_tool)
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
        if knowledge_outcome is not None:
            if knowledge_outcome.status is not GroundingStatus.SUFFICIENT:
                return "The current knowledge base does not provide enough evidence to answer."
            sources = "\n".join(
                f"- {chunk.source_name} ({chunk.source_path}), chunk {chunk.chunk_index}"
                for chunk in knowledge_outcome.supporting_chunks
            )
            return f"{content}\n\nSources:\n{sources}"
        return content
