"""Fake-model verification that LangChain executes SkySun memory tools."""
# 本文件负责测试用例中的 test、memory、agent 相关流程。

from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from skysun.memory.conversation import ConversationService, InMemoryConversationStore
from skysun.core.langchain_agent import LangChainAgent
from skysun.memory.memory import PersonalMemory, normalize_key


# 以下集中定义 FakeMemoryStore：存储层相关的数据和操作。
class FakeMemoryStore:
    # 调用位置：创建 FakeMemoryStore 实例时调用。
    def __init__(self) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.items: dict[str, PersonalMemory] = {}

    # 调用位置：src/skysun/tool/tools.py 的 remember_memory、tests/test_memory_agent.py 的 test_preference_from_prior_conversation_is_labeled_as_personal_memory、tests/test_multi_user.py 的 test_legacy_rows_keep_their_dedicated_owner 等处调用。
    def remember(self, key: str, content: str, category: str | None = None, source_conversation_id=None) -> PersonalMemory:
        """保存或更新一条记忆。"""
        key = normalize_key(key)
        now = datetime.now(timezone.utc)
        old = self.items.get(key)
        item = PersonalMemory(old.id if old else uuid4(), key, content, category, source_conversation_id, old.created_at if old else now, now)
        self.items[key] = item
        return item

    # 调用位置：tests/test_memory_agent.py 的 FakeMemoryStore.remember、tests/test_memory_agent.py 的 test_memory_tools_through_agent_and_separate_conversations 等处调用。
    def get(self, key: str) -> PersonalMemory | None:
        """按标识读取当前用户可见的记录。"""
        return self.items.get(normalize_key(key))

    # 调用位置：tests/test_memory_agent.py 的 FakeMemoryStore.search_memories 等处调用。
    def list_memories(self, limit: int = 20) -> tuple[PersonalMemory, ...]:
        """列出当前用户的记忆。"""
        return tuple(self.items.values())[:limit]

    # 调用位置：src/skysun/tool/tools.py 的 forget_memory、tests/test_multi_user.py 的 test_legacy_rows_keep_their_dedicated_owner、tests/test_postgres_memory.py 的 test_memory_create_update_recall_forget_and_repeat_schema 等处调用。
    def forget(self, key: str) -> bool:
        """删除指定记忆。"""
        return self.items.pop(normalize_key(key), None) is not None

    # 调用位置：src/skysun/tool/tools.py 的 recall_memories、tests/test_multi_user.py 的 test_personal_memory_same_key_isolated_and_global_memory_separate、tests/test_postgres_memory.py 的 test_curator_batch_is_atomic_and_lexical_recall_is_bounded 等处调用。
    def search_memories(self, query: str, limit: int = 20) -> tuple[PersonalMemory, ...]:
        """检索与查询相关的记忆。"""
        return self.list_memories(limit)


# 以下集中定义 FakeMemoryCallingModel：数据结构或组件相关的数据和操作。
class FakeMemoryCallingModel(BaseChatModel):
    _tool_names: list[str] = PrivateAttr(default_factory=list)

    # 调用位置：FakeMemoryCallingModel 的调用方通过该方法使用。
    @property
    def _llm_type(self) -> str:
        """为测试准备 _llm_type 所需的替身或输入。"""
        return "fake-memory-calling"

    # 调用位置：FakeMemoryCallingModel 的调用方通过该方法使用。
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeMemoryCallingModel":
        """为测试提供 bind_tools 所需的输入或模拟行为。"""
        self._tool_names = [tool.name for tool in tools]
        return self

    # 调用位置：LangChain 模型接口在生成回复时调用。
    def _generate(self, messages: list[BaseMessage], stop=None, **kwargs: Any) -> ChatResult:
        """为测试准备 _generate 所需的替身或输入。"""
        tool_results = [m for m in messages if isinstance(m, ToolMessage)]
        if tool_results:
            answer = AIMessage(content=str(tool_results[-1].content))
        else:
            request = str(messages[-1].content)
            if request == "remember":
                name, args = "remember_memory", {"key": "favorite_test_planet", "content": "Kepler-186f"}
            elif request == "update":
                name, args = "remember_memory", {"key": "favorite_test_planet", "content": "TRAPPIST-1e"}
            elif request == "recall":
                name, args = "recall_memories", {}
            else:
                name, args = "forget_memory", {"key": "favorite_test_planet"}
            answer = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": "call_1"}])
        return ChatResult(generations=[ChatGeneration(message=answer)])


# 调用位置：pytest 执行该测试用例时调用。
def test_memory_tools_through_agent_and_separate_conversations() -> None:
    """验证 Agent 能调用记忆工具，且不同会话可以独立使用记忆。"""
    model = FakeMemoryCallingModel()
    memory = FakeMemoryStore()
    service = ConversationService(LangChainAgent(model, memory), InMemoryConversationStore())

    first_id, answer = service.chat("remember")
    assert "Saved" in answer
    assert memory.get("favorite_test_planet").content == "Kepler-186f"
    from uuid import UUID
    assert memory.get("favorite_test_planet").source_conversation_id == UUID(first_id)
    second_id, answer = service.chat("recall")
    assert second_id != first_id
    assert "Kepler-186f" in answer
    service.chat("update", second_id)
    assert len(memory.items) == 1
    assert memory.get("favorite_test_planet").content == "TRAPPIST-1e"
    third_id, answer = service.chat("recall")
    assert third_id not in {first_id, second_id}
    assert "TRAPPIST-1e" in answer
    service.chat("forget")
    _, answer = service.chat("recall")
    assert answer == "No personal memories are stored."
    assert {"count_words", "remember_memory", "recall_memories", "forget_memory"} <= set(model._tool_names)


# 调用位置：pytest 执行该测试用例时调用。
def test_preference_from_prior_conversation_is_labeled_as_personal_memory() -> None:
    """验证先前会话中的偏好被标注为个人记忆。"""
    memory = FakeMemoryStore()
    memory.remember("response_style", "Prefers concise answers", "preference")
    model = FakeMemoryCallingModel()
    service = ConversationService(LangChainAgent(model, memory), InMemoryConversationStore())
    _, answer = service.chat("recall")
    assert "Confirmed personal memories" in answer
    assert "response_style: Prefers concise answers" in answer
    assert "document knowledge" in answer
