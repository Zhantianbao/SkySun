"""SkySun-owned structured memory curation and application rules."""

import logging
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from skysun.conversation import ConversationMessage
from skysun.memory import MemoryChange, MemoryStore, PersonalMemory, normalize_key, secret_like_memory

logger = logging.getLogger(__name__)


class MemoryDecision(BaseModel):
    """One proposed change derived from the current user's message."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["ignore", "upsert", "delete"]
    key: str | None = Field(
        default=None, description="Stable snake_case semantic key; reuse an existing key for updates."
    )
    content: str | None = Field(
        default=None, description="Durable user-sourced fact or preference, needed for upsert."
    )
    category: str | None = Field(
        default=None, description="Optional category such as preference, goal, constraint, or project."
    )


class MemoryPlan(BaseModel):
    """At most three structured decisions for one successful user turn."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[MemoryDecision] = Field(default_factory=list, max_length=3)


@dataclass(frozen=True)
class CurationContext:
    user_message: str
    recent_history: tuple[ConversationMessage, ...]
    current_memories: tuple[PersonalMemory, ...]
    conversation_id: UUID


class MemoryCurator(Protocol):
    def decide(self, context: CurationContext) -> MemoryPlan: ...


def validate_plan(plan: MemoryPlan) -> tuple[MemoryChange, ...]:
    """Validate every action before any database write; reject malformed plans."""
    if len(plan.decisions) > 3:
        raise ValueError("Too many memory decisions")
    changes: list[MemoryChange] = []
    for decision in plan.decisions:
        if decision.action == "ignore":
            continue
        if not decision.key:
            raise ValueError("Memory key is required")
        key = normalize_key(decision.key)
        if decision.action == "upsert":
            content = (decision.content or "").strip()
            if not content:
                raise ValueError("Memory content is required")
            if secret_like_memory(key, content):
                raise ValueError("Secret-like memory rejected")
            category = (decision.category or "").strip() or None
            changes.append(MemoryChange("upsert", key, content, category))
        else:
            changes.append(MemoryChange("delete", key))
    return tuple(changes)


class MemoryCurationService:
    """Best-effort curation after conversation persistence.

    The caller handles exceptions so the successful chat turn still returns.
    One plan is validated before a single transactional memory-store call.
    """

    def __init__(self, curator: MemoryCurator, store: MemoryStore) -> None:
        self._curator = curator
        self._store = store

    def process(
        self,
        user_message: str,
        history: tuple[ConversationMessage, ...],
        conversation_id: str,
    ) -> None:
        context = CurationContext(
            user_message=user_message,
            recent_history=history[-6:],
            current_memories=self._store.list_memories(limit=20),
            conversation_id=UUID(conversation_id),
        )
        logger.info("Memory curator invoked")
        plan = self._curator.decide(context)
        if not isinstance(plan, MemoryPlan):
            plan = MemoryPlan.model_validate(plan)
        changes = validate_plan(plan)
        logger.info("Memory curator validated %d changes", len(changes))
        if changes:
            self._store.apply_changes(changes, context.conversation_id)
