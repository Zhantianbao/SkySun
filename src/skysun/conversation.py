"""SkySun-owned conversation types and temporary in-memory storage.

InMemoryConversationStore loses history on restart. PostgreSQL storage is the
normal runtime. Neither store is long-term personal memory.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

from skysun.agent import Agent
from skysun.user_identity import owner_id

if TYPE_CHECKING:
    from skysun.memory_curator import MemoryCurationService

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime


class ConversationNotFound(Exception):
    """The requested conversation does not exist in this store."""


class ConversationStore(Protocol):
    def create(self, conversation_id: str | None = None) -> str: ...
    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]: ...
    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None: ...
    def list_conversations(self, limit: int = 50) -> tuple[ConversationSummary, ...]: ...


class InMemoryConversationStore:
    """Process-local storage useful for tests and lightweight development."""

    def __init__(self) -> None:
        self._messages: dict[str, list[ConversationMessage]] = {}
        self._owners: dict[str, object] = {}
        self._created: dict[str, datetime] = {}
        self._updated: dict[str, datetime] = {}
        self._lock = RLock()

    def create(self, conversation_id: str | None = None) -> str:
        with self._lock:
            conversation_id = conversation_id or str(uuid4())
            if conversation_id in self._messages:
                raise ValueError("Conversation already exists")
            self._messages[conversation_id] = []
            self._owners[conversation_id] = owner_id()
            self._created[conversation_id] = datetime.now(timezone.utc)
            self._updated[conversation_id] = self._created[conversation_id]
            return conversation_id

    def list_conversations(self, limit: int = 50) -> tuple[ConversationSummary, ...]:
        with self._lock:
            rows = (
                ConversationSummary(identifier,
                                    next((message.content[:80] for message in messages
                                          if message.role == "user"), "新对话"),
                                    self._created[identifier], self._updated[identifier])
                for identifier, messages in self._messages.items()
                if self._owners[identifier] == owner_id()
            )
            return tuple(sorted(rows, key=lambda item: item.updated_at, reverse=True)[:limit])

    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        with self._lock:
            if self._owners[conversation_id] != owner_id():
                raise KeyError(conversation_id)
            return tuple(self._messages[conversation_id])

    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None:
        with self._lock:
            if self._owners[conversation_id] != owner_id():
                raise KeyError(conversation_id)
            history = self._messages[conversation_id]
            history.extend(
                (
                    ConversationMessage("user", user),
                    ConversationMessage("assistant", assistant),
                )
            )
            self._updated[conversation_id] = datetime.now(timezone.utc)


class ConversationService:
    """Store only completed turns.

    Agent failures leave history unchanged. Separate conversations can run
    concurrently. Clients should send turns to one conversation sequentially:
    simultaneous turns may both read the same prior history.
    """

    def __init__(
        self, agent: Agent, store: ConversationStore,
        memory_curation: "MemoryCurationService | None" = None,
    ) -> None:
        self._agent = agent
        self._store = store
        self._memory_curation = memory_curation

    def chat(self, message: str, conversation_id: str | None = None) -> tuple[str, str]:
        if conversation_id is None:
            history: tuple[ConversationMessage, ...] = ()
        else:
            try:
                history = self._store.get_history(conversation_id)
            except KeyError as exc:
                raise ConversationNotFound from exc
        actual_id = conversation_id or str(uuid4())
        reply = self._agent.reply(message, history, actual_id)
        if conversation_id is None:
            self._store.create(actual_id)
        self._store.append_exchange(actual_id, message, reply)
        if self._memory_curation is not None:
            try:
                self._memory_curation.process(message, history, actual_id)
            except Exception as exc:
                logger.warning("Memory curation failed: %s", type(exc).__name__)
        return actual_id, reply

    def history(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        try:
            return self._store.get_history(conversation_id)
        except KeyError as exc:
            raise ConversationNotFound from exc

    def list_conversations(self, limit: int = 50) -> tuple[ConversationSummary, ...]:
        return self._store.list_conversations(limit)
