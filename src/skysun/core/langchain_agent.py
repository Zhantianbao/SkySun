"""LangChain adapter for SkySun's conversation-aware Agent interface."""
# 本文件负责核心服务中的 langchain、agent 相关流程。

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


# 以下集中定义 AgentExecution：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class AgentExecution:
    answer: str
    tools_used: tuple[str, ...]
    tool_failures: tuple[str, ...]
    grounding_status: GroundingStatus | None
    sources: tuple[KnowledgeSearchResult, ...]


# 以下集中定义 LangChainAgent：Agent 执行层相关的数据和操作。
class LangChainAgent:
    # 调用位置：创建 LangChainAgent 实例时调用。
    def __init__(
        self,
        model: BaseChatModel,
        memory_store: MemoryStore | None = None,
        knowledge_store: KnowledgeStore | None = None,
        embeddings: EmbeddingProvider | None = None,
        evidence_evaluator: EvidenceEvaluator | None = None,
        query_rewriter: QueryRewriter | None = None,
    ) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        if (knowledge_store is None) != (embeddings is None):
            raise ValueError("Knowledge store and embeddings must be supplied together")
        self._model = model
        self._memory_store = memory_store
        self._knowledge_store = knowledge_store
        self._embeddings = embeddings
        self._evidence_evaluator = evidence_evaluator
        self._query_rewriter = query_rewriter

    # 调用位置：ConversationService.chat 通过 Agent 接口调用。
    def reply(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
        conversation_id: str | None = None,
    ) -> str:
        """生成并返回本次对话的最终回复。"""
        return self.execute(message, history, conversation_id).answer

    # 调用位置：LangChainAgent 内的 reply 调用。
    def execute(
        self,
        message: str,
        history: Sequence[ConversationMessage] = (),
        conversation_id: str | None = None,
        knowledge_question: str | None = None,
    ) -> AgentExecution:
        """在当前会话的用户范围内调用智能代理和工具，收集回答与执行信息。"""
        tool_failures: set[str] = set()
        tools_used: set[str] = set()

        # 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.count_words_tool、src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.remember_memory_tool、src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.recall_memories_tool 等处调用。
        def record_tool_result(name: str, result: str) -> str:
            """记录工具使用情况和结果，供回复评估使用。"""
            tools_used.add(name)
            if is_tool_failure(result):
                tool_failures.add(name)
            else:
                tool_failures.discard(name)
            return result

        # 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute 调用。
        def record_validation_failure(name: str, result: str) -> None:
            """记录参数校验失败的工具调用。"""
            tools_used.add(name)
            tool_failures.add(name)

        # 调用位置：LangChainAgent 的调用方通过该方法使用。
        @tool("count_words")
        def count_words_tool(text: str) -> str:
            """统计文本中由空白字符分隔的词数。"""
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

            # 调用位置：LangChain 的智能代理在需要保存个人记忆时调用此工具。
            @tool("remember_memory")
            def remember_memory_tool(key: str, content: str, category: str = "") -> str:
                """用户明确要求记住时，保存长期有效的事实、偏好或目标；更新时复用稳定的语义键，不保存密码、接口密钥、令牌、私钥或其他凭据。"""
                return record_tool_result("remember_memory", run_tool("remember_memory", lambda: remember_memory(
                    memory_store, key, content, category or None, source_id
                )))

            # 调用位置：LangChain 的智能代理在回答用户个人信息时调用此工具。
            @tool("recall_memories")
            def recall_memories_tool(query: str = "") -> str:
                """回答涉及用户个人情况的问题时，最多检索二十条个人记忆；用相关关键词查询，空查询列出近期记忆，不编造不存在的记忆。"""
                nonlocal memory_read_failure
                if memory_read_failure is not None:
                    return memory_read_failure
                result = run_tool("recall_memories", lambda: recall_memories(memory_store, query), retries=1)
                if is_tool_failure(result):
                    memory_read_failure = result
                return record_tool_result("recall_memories", result)

            # 调用位置：LangChain 的智能代理在用户要求遗忘时调用此工具。
            @tool("forget_memory")
            def forget_memory_tool(key: str) -> str:
                """仅在用户明确要求遗忘时，按语义键删除个人记忆。"""
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

            # 调用位置：LangChain 的智能代理在回答私有文档问题时调用此工具。
            @tool("search_knowledge")
            def search_knowledge_tool(query: str, limit: int = 4) -> str:
                """针对已存文档检索经过核对的证据；每个问题调用一次，重试由应用控制，不用于个人记忆或通用事实。"""
                nonlocal knowledge_outcome
                # 调用位置：作为回调交给工具执行器调用。
                def execute() -> str:
                    """执行一次知识检索，并缓存已经核对的证据。"""
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
