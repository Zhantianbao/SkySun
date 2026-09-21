from fastapi.testclient import TestClient

from skysun.api import build_api


class StubAgent:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def reply(self, message: str) -> str:
        self.messages.append(message)
        return "Hello from SkySun"


def test_chat_contract_and_agent_boundary() -> None:
    agent = StubAgent()
    client = TestClient(build_api(agent))

    response = client.post("/chat", json={"message": "  hello  "})

    assert response.status_code == 200
    assert response.json() == {"reply": "Hello from SkySun"}
    assert agent.messages == ["hello"]


def test_chat_rejects_empty_message() -> None:
    client = TestClient(build_api(StubAgent()))

    response = client.post("/chat", json={"message": "   "})

    assert response.status_code == 422


def test_chat_hides_agent_error() -> None:
    class FailingAgent:
        def reply(self, message: str) -> str:
            raise RuntimeError("private provider failure")

    client = TestClient(build_api(FailingAgent()))
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent response unavailable"}
