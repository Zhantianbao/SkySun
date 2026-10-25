"""Memory curation policy, application, and failure isolation without real API calls."""
# 本文件负责测试用例中的 test、memory、curator 相关流程。

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from skysun.core.api import build_api
from skysun.memory.conversation import ConversationService, InMemoryConversationStore
from skysun.memory.memory import MemoryChange, PersonalMemory
from skysun.memory.memory_curator import (
    MemoryCurationService, MemoryDecision, MemoryPlan, create_candidates,
)


# 以下集中定义 RecordingStore：存储层相关的数据和操作。
class RecordingStore:
    # 调用位置：创建 RecordingStore 实例时调用。
    def __init__(self) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.items: dict[str, PersonalMemory] = {}
        self.batches: list[tuple[tuple[MemoryChange, ...], UUID]] = []

    # 调用位置：RecordingStore 的调用方通过该方法使用。
    def list_memories(self, limit: int = 20) -> tuple[PersonalMemory, ...]:
        """列出当前用户的记忆。"""
        return tuple(self.items.values())[:limit]

    # 调用位置：tests/test_memory_curator.py 的 RecordingStore.apply_changes 等处调用。
    def get(self, key: str) -> PersonalMemory | None:
        """按标识读取当前用户可见的记录。"""
        return self.items.get(key)

    # 调用位置：src/skysun/memory/memory_curator.py 的 MemoryCurationService.process、tests/test_postgres_memory.py 的 test_curator_batch_is_atomic_and_lexical_recall_is_bounded 调用。
    def apply_changes(self, changes: tuple[MemoryChange, ...], source_conversation_id: UUID) -> None:
        """在一次操作中应用记忆变更。"""
        self.batches.append((changes, source_conversation_id))
        now = datetime.now(timezone.utc)
        for change in changes:
            if change.action == "delete":
                self.items.pop(change.key, None)
            else:
                old = self.items.get(change.key)
                self.items[change.key] = PersonalMemory(
                    old.id if old else uuid4(), change.key, change.content,
                    change.category, source_conversation_id,
                    old.created_at if old else now, now,
                )


# 以下集中定义 ScriptedCurator：数据结构或组件相关的数据和操作。
class ScriptedCurator:
    # 调用位置：创建 ScriptedCurator 实例时调用。
    def __init__(self, plans: list[MemoryPlan | Exception]) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.plans = plans
        self.contexts = []

    # 调用位置：src/skysun/memory/memory_curator.py 的 MemoryCurationService.process、tests/test_memory_curator.py 的 test_langchain_curator_uses_typed_json_output_without_network、tests/test_memory_extraction.py 的 test_valid_structured_output_is_accepted_without_retry 等处调用。
    def decide(self, context):
        """根据对话和已有记忆决定新增、更新或删除内容。"""
        self.contexts.append(context)
        result = self.plans.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


# 以下集中定义 ReplyAgent：Agent 执行层相关的数据和操作。
class ReplyAgent:
    # 调用位置：ReplyAgent 的调用方通过该方法使用。
    def reply(self, message, history=(), conversation_id=None):
        """生成并返回本次对话的最终回复。"""
        return f"Reply: {message}"


# 调用位置：pytest 执行该测试用例时调用。
def test_automatic_formation_update_multiple_and_forget_with_source() -> None:
    """验证自动整理会形成、更新和遗忘多条记忆，并保留来源。"""
    store = RecordingStore()
    curator = ScriptedCurator([
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="Preferred Editor", content="VS Code", category="preference"),
            MemoryDecision(action="upsert", key="career_goal", content="AI application engineer", category="goal"),
        ]),
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="preferred_editor", content="PyCharm", category="preference"),
        ]),
        MemoryPlan(decisions=[
            MemoryDecision(action="delete", key="preferred_editor"),
        ]),
    ])
    conversations = InMemoryConversationStore()
    service = ConversationService(
        ReplyAgent(), conversations, MemoryCurationService(curator, store)
    )
    first_id, _ = service.chat("My editor is VS Code and my goal is AI work.")
    first_memory_id = store.items["preferred_editor"].id
    assert len(store.items) == 2
    assert store.items["preferred_editor"].source_conversation_id == UUID(first_id)
    assert len(store.batches[0][0]) == 2
    second_id, _ = service.chat("I've switched to PyCharm.")
    assert store.items["preferred_editor"].id == first_memory_id
    assert store.items["preferred_editor"].content == "PyCharm"
    assert store.items["preferred_editor"].source_conversation_id == UUID(second_id)
    assert len(store.items) == 2
    service.chat("Forget my editor.")
    assert "preferred_editor" not in store.items
    assert "career_goal" in store.items
    assert curator.contexts[1].current_memories[0].key == "preferred_editor"
    assert conversations.get_history(first_id)[0].content.startswith("My editor")


# 调用位置：pytest 执行该测试用例时调用。
def test_transient_and_malformed_or_secret_decisions_do_not_write_memory() -> None:
    """验证临时、格式错误或含密钥的决策不会写入记忆。"""
    store = RecordingStore()
    curator = ScriptedCurator([
        MemoryPlan(),
        MemoryPlan(decisions=[MemoryDecision(action="upsert", key="bad/key", content="x")]),
        MemoryPlan(decisions=[MemoryDecision(action="upsert", key="api_key", content="private value")]),
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="preferred_editor", content="VS Code"),
            MemoryDecision(action="upsert", key="missing_content"),
        ]),
    ])
    service = ConversationService(
        ReplyAgent(), InMemoryConversationStore(), MemoryCurationService(curator, store)
    )
    for message in (
        "I'm drinking coffee right now.",
        "malformed",
        "Remember my API key is a private value.",
        "mixed valid and invalid",
    ):
        _, reply = service.chat(message)
        assert reply == f"Reply: {message}"
    assert store.items == {}
    assert store.batches == []


# 调用位置：pytest 执行该测试用例时调用。
def test_curator_failure_keeps_http_reply_and_persisted_conversation() -> None:
    """验证记忆整理失败时仍返回 HTTP 回复并保存会话。"""
    store = RecordingStore()
    curator = ScriptedCurator([RuntimeError("secret internal failure")])
    conversations = InMemoryConversationStore()
    client = TestClient(build_api(ConversationService(
        ReplyAgent(), conversations, MemoryCurationService(curator, store)
    ), allow_legacy_unauthenticated=True))
    response = client.post("/chat", json={"message": "My editor is PyCharm."})
    assert response.status_code == 200
    body = response.json()
    assert body["reply"] == "Reply: My editor is PyCharm."
    assert [turn.role for turn in conversations.get_history(body["conversation_id"])] == [
        "user", "assistant"
    ]
    assert store.items == {}


# 调用位置：pytest 执行该测试用例时调用。
def test_bounded_decisions_and_missing_fields() -> None:
    """验证有界决策与缺失字段。"""
    with pytest.raises(ValueError):
        MemoryPlan(decisions=[MemoryDecision(action="ignore")] * 4)
    from skysun.memory.memory_curator import validate_plan
    with pytest.raises(ValueError):
        validate_plan(MemoryPlan(decisions=[MemoryDecision(action="delete")]))



# 调用位置：pytest 执行该测试用例时调用。
def test_langchain_curator_uses_typed_json_output_without_network() -> None:
    """验证LangChain记忆整理器使用类型化JSON输出不网络。"""
    from skysun.memory.langchain_memory_curator import LangChainMemoryCurator
    from skysun.memory.memory_curator import CurationContext

    # 以下集中定义 FakeRunnable：数据结构或组件相关的数据和操作。
    class FakeRunnable:
        # 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute、src/skysun/evaluation/deepseek_evaluation.py 的 DeepSeekClaimReviewer.review、src/skysun/memory/langchain_memory_curator.py 的 LangChainMemoryCurator.decide 等处调用。
        def invoke(self, messages):
            """为测试提供 invoke 所需的输入或模拟行为。"""
            assert "current_user_message" in messages[1][1]
            return {"decisions": [{"action": "upsert", "key": "preferred_editor", "content": "PyCharm"}]}

    # 以下集中定义 FakeModel：数据结构或组件相关的数据和操作。
    class FakeModel:
        # 调用位置：src/skysun/evaluation/deepseek_evaluation.py 的 DeepSeekClaimReviewer.review、src/skysun/memory/langchain_memory_curator.py 的 LangChainMemoryCurator.__init__、src/skysun/rag/langchain_knowledge_grounding.py 的 LangChainEvidenceEvaluator.__init__ 等处调用。
        def with_structured_output(self, schema, method, include_raw):
            """为测试提供 with_structured_output 所需的输入或模拟行为。"""
            assert schema is MemoryPlan
            assert method == "json_mode"
            assert include_raw is True
            return FakeRunnable()

    plan = LangChainMemoryCurator(FakeModel()).decide(
        CurationContext("My editor is PyCharm.", (), (), uuid4())
    )
    assert plan.decisions[0].key == "preferred_editor"


# 调用位置：pytest 执行该测试用例时调用。
def test_candidate_creation_keeps_importance_and_conversation_source() -> None:
    """验证候选项创建保持重要度与会话来源。"""
    source = uuid4()
    plan = MemoryPlan(decisions=[
        MemoryDecision(action="upsert", key="Preferred Editor", content="  PyCharm  ",
                       category="preference", importance="high"),
        MemoryDecision(action="ignore"),
    ])
    candidates = create_candidates(plan, source)
    assert len(candidates) == 1
    assert (candidates[0].key, candidates[0].content, candidates[0].category) == (
        "preferred_editor", "PyCharm", "preference"
    )
    assert candidates[0].importance == "high"
    assert candidates[0].source_conversation_id == source


# 调用位置：pytest 执行该测试用例时调用。
def test_curator_rejects_low_value_and_duplicate_candidates() -> None:
    """验证记忆整理拒绝低价值或重复候选内容。"""
    store = RecordingStore()
    curator = ScriptedCurator([
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="preferred_editor", content="PyCharm",
                           category="preference", importance="high"),
            MemoryDecision(action="upsert", key="today_snack", content="A biscuit",
                           importance="low"),
        ]),
        MemoryPlan(decisions=[
            MemoryDecision(action="upsert", key="editor_choice", content=" pycharm ",
                           category="tool", importance="high"),
        ]),
    ])
    service = MemoryCurationService(curator, store)
    service.process("I prefer PyCharm.", (), str(uuid4()))
    service.process("I still prefer PyCharm.", (), str(uuid4()))
    assert tuple(store.items) == ("preferred_editor",)
    assert len(store.batches) == 1
