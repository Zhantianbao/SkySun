"""Small SkySun tool business functions; LangChain registration lives in the adapter."""

from uuid import UUID

from skysun.memory import MemoryStore


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
    return "\n".join(f"{memory.key}: {memory.content}" for memory in memories)


def forget_memory(store: MemoryStore, key: str) -> str:
    """Remove one personal memory if it exists."""
    deleted = store.forget(key)
    return "Personal memory forgotten." if deleted else "No memory found for that key."
