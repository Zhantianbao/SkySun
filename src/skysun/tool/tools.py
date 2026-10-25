"""Small SkySun tool business functions; LangChain registration lives in adapters."""
# 本文件负责工具调用中的 tools 相关流程。

from uuid import UUID

from skysun.rag.knowledge_grounding import GroundedKnowledge, GroundedKnowledgeService, GroundingStatus
from skysun.memory.memory import MemoryStore


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.count_words_tool、tests/test_tools.py 的 test_count_words_uses_whitespace 调用。
def count_words(text: str) -> int:
    """统计文本中由空白字符分隔的词数。"""
    return len(text.split())


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.remember_memory_tool 调用。
def remember_memory(
    store: MemoryStore, key: str, content: str, category: str | None = None,
    source_conversation_id: UUID | None = None,
) -> str:
    """按语义键保存或更新一条长期有效的个人记忆。"""
    memory = store.remember(key, content, category, source_conversation_id)
    return f"Saved personal memory {memory.key}."


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.recall_memories_tool 调用。
def recall_memories(store: MemoryStore, query: str = "") -> str:
    """通过词项检索最多二十条当前用户的个人记忆。"""
    memories = store.search_memories(query, limit=20) if query else store.list_memories(limit=20)
    if not memories:
        return "No personal memories are stored."
    return "Confirmed personal memories (not document knowledge):\n" + "\n".join(
        f"{memory.key}: {memory.content}" for memory in memories
    )


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.forget_memory_tool 调用。
def forget_memory(store: MemoryStore, key: str) -> str:
    """删除指定的个人记忆，若不存在则返回相应结果。"""
    deleted = store.forget(key)
    return "Personal memory forgotten." if deleted else "No memory found for that key."


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.search_knowledge_tool.execute 调用。
def search_knowledge(service: GroundedKnowledgeService, question: str, limit: int = 4) -> GroundedKnowledge:
    """返回经过核对的知识证据，由智能代理适配层负责格式化。"""
    return service.search(question, limit)


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.search_knowledge_tool.execute 调用。
def format_knowledge_evidence(outcome: GroundedKnowledge) -> str:
    """将知识检索结果整理为可供回答引用的证据。"""
    if outcome.status is not GroundingStatus.SUFFICIENT:
        return "The current knowledge base does not provide enough evidence to answer."
    return "\n\n".join(chunk.content for chunk in outcome.supporting_chunks)
