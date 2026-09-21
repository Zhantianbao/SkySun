"""SkySun-owned personal memory domain, independent of conversations and LangChain."""

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,79}")
_SECRET_TERMS = re.compile(
    r"\b(?:password|passphrase|api[\s_-]?key|access[\s_-]?token|"
    r"private[\s_-]?key|auth(?:entication)?[\s_-]?credential|"
    r"database[\s_-]?password)\b",
    re.IGNORECASE,
)


def secret_like_memory(key: str, content: str = "") -> bool:
    """Reject obvious credential labels; this is a narrow safeguard, not a classifier."""
    return bool(_SECRET_TERMS.search(key) or _SECRET_TERMS.search(content))


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


@dataclass(frozen=True)
class MemoryChange:
    action: Literal["upsert", "delete"]
    key: str
    content: str | None = None
    category: str | None = None


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

    def search_memories(self, query: str, limit: int = 20) -> tuple[PersonalMemory, ...]: ...

    def apply_changes(self, changes: tuple[MemoryChange, ...], source_conversation_id: UUID) -> None: ...
