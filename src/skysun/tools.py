"""Small SkySun tool business functions; LangChain registration lives in the adapter."""

from skysun.memory import MemoryStore


def count_words(text: str) -> int:
    """Count whitespace-separated words in text."""
    return len(text.split())


def remember_memory(
    store: MemoryStore, key: str, content: str, category: str | None = None
) -> str:
    """Store or update one durable personal fact under a semantic key."""
    memory = store.remember(key, content, category)
    return f"Saved personal memory {memory.key}."


def recall_memories(store: MemoryStore) -> str:
    """List at most 20 current personal memories for the Agent to inspect."""
    memories = store.list_memories(limit=20)
    if not memories:
        return "No personal memories are stored."
    return "\n".join(f"{memory.key}: {memory.content}" for memory in memories)


def forget_memory(store: MemoryStore, key: str) -> str:
    """Remove one personal memory if it exists."""
    deleted = store.forget(key)
    return "Personal memory forgotten." if deleted else "No memory found for that key."
