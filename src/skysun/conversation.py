"""Temporary process-local conversation context for SkySun v0.2.

History is lost on restart. This is working context, not long-term memory.
"""

from dataclasses import dataclass
from threading import RLock
from typing import Literal, Protocol
from uuid import uuid4

from skysun.agent import Agent


@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


class ConversationNotFound(Exception):
    """The requested conversation does not exist in this process."""


class ConversationStore(Protocol):
    def create(self) -> str: ...
    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]: ...
    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None: ...


class InMemoryConversationStore:
    """Process-local storage; no disk persistence or cross-process sharing."""

    def __init__(self) -> None:
        self._messages: dict[str, list[ConversationMessage]] = {}

    def create(self) -> str:
        conversation_id = str(uuid4())
        self._messages[conversation_id] = []
        return conversation_id

    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        return tuple(self._messages[conversation_id])

    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None:
        history = self._messages[conversation_id]
        history.extend(
            (
                ConversationMessage("user", user),
                ConversationMessage("assistant", assistant),
            )
        )


class ConversationService:
    """Own one store and serialize turns, including calls to the Agent.

    Only successful user/assistant exchanges are recorded. A failed Agent call
    leaves existing history unchanged. The single lock also serializes distinct
    conversations while the process runs; that is sufficient for v0.2.
    """

    def __init__(self, agent: Agent, store: ConversationStore) -> None:
        self._agent = agent
        self._store = store
        self._lock = RLock()

    def chat(self, message: str, conversation_id: str | None = None) -> tuple[str, str]:
        with self._lock:
            if conversation_id is None:
                history: tuple[ConversationMessage, ...] = ()
            else:
                try:
                    history = self._store.get_history(conversation_id)
                except KeyError as exc:
                    raise ConversationNotFound from exc
            reply = self._agent.reply(message, history)
            actual_id = self._store.create() if conversation_id is None else conversation_id
            self._store.append_exchange(actual_id, message, reply)
            return actual_id, reply
