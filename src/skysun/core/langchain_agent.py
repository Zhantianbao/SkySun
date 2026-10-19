"""LangChain adapter for SkySun's conversation-aware Agent interface."""

from dataclasses import dataclass
from typing import Any, Sequence
from uuid import UUID

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from skysun.memory.conversation import ConversationMessage
from skysun.rag.embeddings import EmbeddingProvider
from skysun.core.identity import SYSTEM_PROMPT
from skysun.rag.knowledge import KnowledgeSearchResult, KnowledgeStore
from skysun.rag.knowledge_grounding import (
    EvidenceEvaluator, GroundedKnowledge, GroundedKnowledgeService, GroundingStatus, QueryRewriter,
)
from skysun.rag.langchain_knowledge_grounding import LangChainEvidenceEvaluator, LangChainQueryRewriter
from skysun.memory.memory import MemoryStore
from skysun.tool.tool_reliability import configure_validation, is_tool_failure, run_tool
from skysun.tool.tools import (
    count_words,
    forget_memory,
    recall_memories,
    remember_memory,
    search_knowledge,
    format_knowledge_evidence,
)


@dataclass(frozen=True)
class AgentExecution:
    answer: str
    tools_used: tuple[str, ...]
    tool_failures: tuple[str, ...]
    grounding_status: GroundingStatus | None
    sources: tuple[KnowledgeSearchResult, ...]


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
        return self.execute(message, history, conversation_id).answer

    def execute(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
        conversation_id: str | None = None,
        knowledge_question: str | None = None,
    ) -> AgentExecution:
        """Run one Agent invocation with tools scoped to this conversation."""
        tool_failures: set[str] = set()
        tools_used: set[str] = set()

        def record_tool_result(name: str, result: str) -> str:
            tools_used.add(name)
            if is_tool_failure(result):
                tool_failures.add(name)
            else:
                tool_failures.discard(name)
            return result

        def record_validation_failure(name: str, result: str) -> None:
            tools_used.add(name)
            tool_failures.add(name)

        @tool("count_words")
        def count_words_tool(text: str) -> str:
            """Count whitespace-separated words in text."""
            return record_tool_result("count_words", run_tool(
                "count_words", lambda: count_words(text)
            ))

        tools = [configure_validation(
            count_words_tool, lambda result: record_validation_failure("count_words", result)
        )]
        knowledge_outcome: GroundedKnowledge | None = None
        memory_read_failure: str | None = None
        if self._memory_store is not None:
            memory_store = self._memory_store
            source_id = UUID(conversation_id) if conversation_id else None

            @tool("remember_memory")
            def remember_memory_tool(key: str, content: str, category: str = "") -> str:
                """Save a durable user fact, preference, or goal. Reuse a stable snake_case semantic key to update. Never save passwords, API keys, tokens, private keys, or credentials."""
                return record_tool_result("remember_memory", run_tool("remember_memory", lambda: remember_memory(
                    memory_store, key, content, category or None, source_id
                )))

            @tool("recall_memories")
            def recall_memories_tool(query: str = "") -> str:
                """Retrieve up to 20 personal memories for user-specific questions. Pass relevant keywords for basic lexical lookup; empty query lists recent memories. Never invent absent memories."""
                nonlocal memory_read_failure
                if memory_read_failure is not None:
                    return memory_read_failure
                result = run_tool("recall_memories", lambda: recall_memories(memory_store, query), retries=1)
                if is_tool_failure(result):
                    memory_read_failure = result
                return record_tool_result("recall_memories", result)

            @tool("forget_memory")
            def forget_memory_tool(key: str) -> str:
                """Delete a stored personal memory by its semantic key when the user explicitly asks to forget it."""
                return record_tool_result("forget_memory", run_tool(
                    "forget_memory", lambda: forget_memory(memory_store, key)
                ))

            tools.extend(configure_validation(
                item, lambda result, name=item.name: record_validation_failure(name, result)
            ) for item in (
                remember_memory_tool, recall_memories_tool, forget_memory_tool
            ))
        if self._knowledge_store is not None and self._embeddings is not None:
            knowledge_store = self._knowledge_store
            embeddings = self._embeddings

            @tool("search_knowledge")
            def search_knowledge_tool(query: str, limit: int = 4) -> str:
                """Find validated evidence for a question about stored documents. Call once per user question. The application controls any retry. Do not use for personal memory or general facts."""
                nonlocal knowledge_outcome
                def execute() -> str:
                    nonlocal knowledge_outcome
                    if knowledge_outcome is None:
                        evaluator = self._evidence_evaluator or LangChainEvidenceEvaluator(self._model)
                        rewriter = self._query_rewriter or LangChainQueryRewriter(self._model)
                        service = GroundedKnowledgeService(knowledge_store, embeddings, evaluator, rewriter)
                        knowledge_outcome = search_knowledge(service, knowledge_question or message, limit)
                    return format_knowledge_evidence(knowledge_outcome)

                return record_tool_result("search_knowledge", run_tool("search_knowledge", execute))

            tools.append(configure_validation(
                search_knowledge_tool, lambda result: record_validation_failure("search_knowledge", result)
            ))
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
        if "recall_memories" in tool_failures:
            content = "I couldn't access personal memory, so I can't verify that answer right now."
        elif {"remember_memory", "forget_memory"} & tool_failures:
            content = "I couldn't update personal memory right now."
        elif tool_failures:
            content = "I couldn't complete the requested tool operation right now."
        elif knowledge_outcome is not None:
            if knowledge_outcome.status is not GroundingStatus.SUFFICIENT:
                content = "The current knowledge base does not provide enough evidence to answer."
            else:
                sources = "\n".join(
                    f"- {chunk.source_name} ({chunk.source_path}), chunk {chunk.chunk_index}"
                    for chunk in knowledge_outcome.supporting_chunks
                )
                content = f"{content}\n\nSources:\n{sources}"
        return AgentExecution(
            content, tuple(sorted(tools_used)), tuple(sorted(tool_failures)),
            knowledge_outcome.status if knowledge_outcome is not None else None,
            knowledge_outcome.supporting_chunks if knowledge_outcome is not None else (),
        )
