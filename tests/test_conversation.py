from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from skysun.api import build_api
from skysun.conversation import (
    ConversationMessage,
    ConversationService,
    InMemoryConversationStore,
)


class RecordingAgent:
    def __init__(self) -> None:
        self.histories: list[tuple[ConversationMessage, ...]] = []

    def reply(
        self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None
    ) -> str:
        self.histories.append(tuple(history))
        return f"Answered: {message}"


def test_store_and_service_keep_conversations_isolated() -> None:
    store = InMemoryConversationStore()
    agent = RecordingAgent()
    service = ConversationService(agent, store)

    first_id, first_reply = service.chat("I like Python")
    second_id, second_reply = service.chat("I like Rust")
    continued_id, reply = service.chat("What do I like?", first_id)

    assert UUID(first_id) != UUID(second_id)
    assert continued_id == first_id
    assert first_reply == "Answered: I like Python"
    assert second_reply == "Answered: I like Rust"
    assert reply == "Answered: What do I like?"
    assert agent.histories == [
        (),
        (),
        (
            ConversationMessage("user", "I like Python"),
            ConversationMessage("assistant", "Answered: I like Python"),
        ),
    ]
    assert store.get_history(second_id) == (
        ConversationMessage("user", "I like Rust"),
        ConversationMessage("assistant", "Answered: I like Rust"),
    )


def test_api_continues_and_creates_separate_conversations() -> None:
    agent = RecordingAgent()
    client = TestClient(
        build_api(ConversationService(agent, InMemoryConversationStore()),
                  allow_legacy_unauthenticated=True)
    )

    first = client.post("/chat", json={"message": "I like Python"}).json()
    continued = client.post(
        "/chat",
        json={"message": "What do I like?", "conversation_id": first["conversation_id"]},
    ).json()
    other = client.post("/chat", json={"message": "What do I like?"}).json()

    assert continued["conversation_id"] == first["conversation_id"]
    assert other["conversation_id"] != first["conversation_id"]
    assert agent.histories[1][0].content == "I like Python"
    assert agent.histories[2] == ()


def test_failed_turn_does_not_change_history() -> None:
    store = InMemoryConversationStore()
    conversation_id = store.create()

    class FailingAgent:
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None) -> str:
            raise RuntimeError("provider unavailable")

    service = ConversationService(FailingAgent(), store)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        service.chat("Hello", conversation_id)

    assert store.get_history(conversation_id) == ()



def test_separate_conversations_can_run_concurrently() -> None:
    barrier = Barrier(2)

    class ConcurrentAgent:
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None) -> str:
            barrier.wait(timeout=5)
            return message

    service = ConversationService(ConcurrentAgent(), InMemoryConversationStore())
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.chat, "first")
        second = executor.submit(service.chat, "second")
        assert first.result()[1] == "first"
        assert second.result()[1] == "second"
