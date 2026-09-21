"""SkySun-owned personal memory domain, independent of conversations and LangChain."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol
from uuid import UUID

_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,79}")


def normalize_key(key: str) -> str:
    """Normalize a stable semantic key to lowercase snake_case."""
    normalized = re.sub(r"[\s-]+", "_", key.strip().lower())
    if not _KEY_PATTERN.fullmatch(normalized):
        raise ValueError("Memory key must be snake_case, start with a letter, and be at most 80 characters")
    return normalized


@dataclass(frozen=True)
class PersonalMemory:
    id: UUID
    key: str
    content: str
    category: str | None
    source_conversation_id: UUID | None
    created_at: datetime
    updated_at: datetime


class MemoryStore(Protocol):
    def remember(
        self,
        key: str,
        content: str,
        category: str | None = None,
        source_conversation_id: UUID | None = None,
    ) -> PersonalMemory: ...

    def get(self, key: str) -> PersonalMemory | None: ...

    def list_memories(self, limit: int = 20) -> tuple[PersonalMemory, ...]: ...

    def forget(self, key: str) -> bool: ...
