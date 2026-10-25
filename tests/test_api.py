# 本文件负责测试用例中的 test、api 相关流程。
from uuid import UUID, uuid4

from fastapi.testclient import TestClient
import pytest

from skysun.core.api import build_api
from skysun.memory.conversation import ConversationMessage, ConversationService, InMemoryConversationStore


# 以下集中定义 StubAgent：Agent 执行层相关的数据和操作。
class StubAgent:
    # 调用位置：创建 StubAgent 实例时调用。
    def __init__(self) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.calls: list[tuple[str, tuple[ConversationMessage, ...]]] = []

    # 调用位置：StubAgent 的调用方通过该方法使用。
    def reply(
        self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None
    ) -> str:
        """生成并返回本次对话的最终回复。"""
        self.calls.append((message, tuple(history)))
        return "Hello from SkySun"


# 调用位置：tests/test_api.py 的 test_chat_contract_and_agent_boundary、tests/test_api.py 的 test_chat_rejects_invalid_input、tests/test_api.py 的 test_unknown_conversation_is_not_created 调用。
def make_client(agent: StubAgent) -> TestClient:
    """为测试提供 make_client 所需的输入或模拟行为。"""
    return TestClient(build_api(ConversationService(agent, InMemoryConversationStore()),
                                allow_legacy_unauthenticated=True))


# 调用位置：pytest 执行该测试用例时调用。
def test_api_requires_identity_store_by_default() -> None:
    """验证接口要求身份存储由默认。"""
    with pytest.raises(RuntimeError, match="authenticated user store"):
        build_api(ConversationService(StubAgent(), InMemoryConversationStore()))


# 调用位置：pytest 执行该测试用例时调用。
def test_chat_contract_and_agent_boundary() -> None:
    """验证聊天接口遵守约定的请求和响应格式，并正确调用 Agent。"""
    agent = StubAgent()
    response = make_client(agent).post("/chat", json={"message": "  hello  "})

    assert response.status_code == 200
    assert response.json()["reply"] == "Hello from SkySun"
    UUID(response.json()["conversation_id"])
    assert agent.calls == [("hello", ())]


# 调用位置：pytest 执行该测试用例时调用。
def test_chat_rejects_invalid_input() -> None:
    """验证聊天接口拒绝无效输入。"""
    client = make_client(StubAgent())

    assert client.post("/chat", json={"message": "   "}).status_code == 422
    assert client.post(
        "/chat", json={"message": "hello", "conversation_id": ""}
    ).status_code == 422


# 调用位置：pytest 执行该测试用例时调用。
def test_unknown_conversation_is_not_created() -> None:
    """验证未知会话不创建。"""
    client = make_client(StubAgent())
    response = client.post(
        "/chat", json={"message": "hello", "conversation_id": str(uuid4())}
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Conversation not found"}


# 调用位置：pytest 执行该测试用例时调用。
def test_chat_hides_agent_error() -> None:
    """验证Agent 内部错误不会直接暴露给聊天用户。"""
    # 以下集中定义 FailingAgent：Agent 执行层相关的数据和操作。
    class FailingAgent:
        # 调用位置：FailingAgent 的调用方通过该方法使用。
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None) -> str:
            """生成并返回本次对话的最终回复。"""
            raise RuntimeError("private provider failure")

    client = TestClient(
        build_api(ConversationService(FailingAgent(), InMemoryConversationStore()),
                  allow_legacy_unauthenticated=True)
    )
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
    assert response.json() == {"detail": "Agent response unavailable"}


# 调用位置：pytest 执行该测试用例时调用。
def test_agent_key_error_is_not_mistaken_for_missing_conversation() -> None:
    """验证Agent 的 KeyError 不会被误判为会话不存在。"""
    # 以下集中定义 FailingAgent：Agent 执行层相关的数据和操作。
    class FailingAgent:
        # 调用位置：FailingAgent 的调用方通过该方法使用。
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None) -> str:
            """生成并返回本次对话的最终回复。"""
            raise KeyError("provider data missing")

    client = TestClient(
        build_api(ConversationService(FailingAgent(), InMemoryConversationStore()),
                  allow_legacy_unauthenticated=True)
    )
    response = client.post("/chat", json={"message": "hello"})

    assert response.status_code == 502
