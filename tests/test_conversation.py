# 本文件负责测试用例中的 test、conversation 相关流程。
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

from skysun.core.api import build_api
from skysun.memory.conversation import (
    ConversationMessage,
    ConversationService,
    InMemoryConversationStore,
)


# 以下集中定义 RecordingAgent：Agent 执行层相关的数据和操作。
class RecordingAgent:
    # 调用位置：创建 RecordingAgent 实例时调用。
    def __init__(self) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.histories: list[tuple[ConversationMessage, ...]] = []

    # 调用位置：RecordingAgent 的调用方通过该方法使用。
    def reply(
        self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None
    ) -> str:
        """生成并返回本次对话的最终回复。"""
        self.histories.append(tuple(history))
        return f"Answered: {message}"


# 调用位置：pytest 执行该测试用例时调用。
def test_store_and_service_keep_conversations_isolated() -> None:
    """验证会话存储和服务都保持不同会话隔离。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_api_continues_and_creates_separate_conversations() -> None:
    """验证接口既能继续已有会话，也能创建独立的新会话。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_failed_turn_does_not_change_history() -> None:
    """验证失败的对话轮次不会改写已保存的历史。"""
    store = InMemoryConversationStore()
    conversation_id = store.create()

    # 以下集中定义 FailingAgent：Agent 执行层相关的数据和操作。
    class FailingAgent:
        # 调用位置：FailingAgent 的调用方通过该方法使用。
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None) -> str:
            """生成并返回本次对话的最终回复。"""
            raise RuntimeError("provider unavailable")

    service = ConversationService(FailingAgent(), store)
    with pytest.raises(RuntimeError, match="provider unavailable"):
        service.chat("Hello", conversation_id)

    assert store.get_history(conversation_id) == ()



# 调用位置：pytest 执行该测试用例时调用。
def test_separate_conversations_can_run_concurrently() -> None:
    """验证独立会话可以并发执行且互不串扰。"""
    barrier = Barrier(2)

    # 以下集中定义 ConcurrentAgent：Agent 执行层相关的数据和操作。
    class ConcurrentAgent:
        # 调用位置：ConcurrentAgent 的调用方通过该方法使用。
        def reply(self, message: str, history: tuple[ConversationMessage, ...] = (), conversation_id: str | None = None) -> str:
            """生成并返回本次对话的最终回复。"""
            barrier.wait(timeout=5)
            return message

    service = ConversationService(ConcurrentAgent(), InMemoryConversationStore())
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(service.chat, "first")
        second = executor.submit(service.chat, "second")
        assert first.result()[1] == "first"
        assert second.result()[1] == "second"
