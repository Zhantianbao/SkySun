"""Small SkySun tool business functions; LangChain registration lives in adapters."""

from uuid import UUID

from skysun.rag.knowledge_grounding import GroundedKnowledge, GroundedKnowledgeService, GroundingStatus
from skysun.memory.memory import MemoryStore


def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())


def remember_memory(
    store: MemoryStore, key: str, content: str, category: str | None = None,
    source_conversation_id: UUID | None = None,
) -> str:
    """Store or update one durable personal fact under a semantic key."""
    memory = store.remember(key, content, category, source_conversation_id)
    return f"Saved personal memory {memory.key}."


def recall_memories(store: MemoryStore, query: str = "") -> str:
    """Retrieve up to 20 current personal memories using basic lexical search."""
    memories = store.search_memories(query, limit=20) if query else store.list_memories(limit=20)
    if not memories:
        return "No personal memories are stored."
    return "Confirmed personal memories (not document knowledge):\n" + "\n".join(
        f"{memory.key}: {memory.content}" for memory in memories
    )


def forget_memory(store: MemoryStore, key: str) -> str:
    """Remove one personal memory if it exists."""
    deleted = store.forget(key)
    return "Personal memory forgotten." if deleted else "No memory found for that key."


def search_knowledge(service: GroundedKnowledgeService, question: str, limit: int = 4) -> GroundedKnowledge:
    """Return validated evidence; formatting belongs to the Agent adapter."""
    return service.search(question, limit)


def format_knowledge_evidence(outcome: GroundedKnowledge) -> str:
    if outcome.status is not GroundingStatus.SUFFICIENT:
        return "The current knowledge base does not provide enough evidence to answer."
    return "\n\n".join(chunk.content for chunk in outcome.supporting_chunks)
