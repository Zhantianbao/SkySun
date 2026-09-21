from uuid import UUID, uuid4

from fastapi.testclient import TestClient

from skysun.api import build_api
from skysun.conversation import ConversationMessage, ConversationService, InMemoryConversationStore


class StubAgent:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[ConversationMessage, ...]]] = []

    def reply(
        self, message: str, history: tuple[ConversationMessage, ...] = ()
    ) -> str:
        self.calls.append((message, tuple(history)))
        return "Hello from SkySun"


def make_client(agent: StubAgent) -> TestClient:
    return TestClient(build_api(ConversationService(agent, InMemoryConversationStore())))


def test_chat_contract_and_agent_boundary() -> None:
    agent = StubAgent()
    response = make_client(agent).post("/chat", json={"message": "  hello  "})

    assert response.status_code == 200
    assert response.json()["reply"] == "Hello from SkySun"
    UUID(response.json()["conversation_id"])
    assert agent.calls == [("hello", ())]


def test_chat_rejects_invalid_input() -> None:
    client = make_client(StubAgent())

    assert client.post("/chat", json={"message": "   "}).status_code == 422
    assert client.post(
        "/chat", json={"message": "hello", "conversation_id": ""}
    ).status_code == 422


def test_unknown_conversation_is_not_created() -> None:
    client = make_client(StubAgent())
    response = client.post(
        "/chat", json={"message": "hello", "conversation_id": str(uuid4())}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}


def test_chat_hides_agent_error() -> None:
    class FailingAgent:
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = ()) -> str:
            raise RuntimeError("private provider failure")

    client = TestClient(
        build_api(ConversationService(FailingAgent(), InMemoryConversationStore()))
    )
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent response unavailable"}


def test_agent_key_error_is_not_mistaken_for_missing_conversation() -> None:
    class FailingAgent:
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = ()) -> str:
            raise KeyError("provider data missing")

    client = TestClient(
        build_api(ConversationService(FailingAgent(), InMemoryConversationStore()))
    )
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
