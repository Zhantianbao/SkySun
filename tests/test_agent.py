# 本文件负责测试用例中的 test、agent 相关流程。
from typing import Any, Sequence
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr, ValidationError

from skysun.core.config import Settings
from skysun.memory.conversation import ConversationMessage
from skysun.core.langchain_agent import LangChainAgent


# 以下集中定义 FakeToolCallingModel：数据结构或组件相关的数据和操作。
class FakeToolCallingModel(BaseChatModel):
    _seen: list[list[str]] = PrivateAttr(default_factory=list)

    # 调用位置：FakeToolCallingModel 的调用方通过该方法使用。
    @property
    def _llm_type(self) -> str:
        """为测试准备 _llm_type 所需的替身或输入。"""
        return "fake-tool-calling"

    # 调用位置：FakeToolCallingModel 的调用方通过该方法使用。
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeToolCallingModel":
        """为测试提供 bind_tools 所需的输入或模拟行为。"""
        assert any(tool.name == "count_words" for tool in tools)
        return self

    # 调用位置：LangChain 模型接口在生成回复时调用。
    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """为测试准备 _generate 所需的替身或输入。"""
        self._seen.append([str(message.content) for message in messages])
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if tool_messages:
            answer = AIMessage(content=f"Count: {tool_messages[-1].content}")
        else:
            answer = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "count_words",
                        "args": {"text": "one two three"},
                        "id": "call_1",
                    }
                ],
            )
        return ChatResult(generations=[ChatGeneration(message=answer)])


# 调用位置：pytest 执行该测试用例时调用。
def test_langchain_owns_tool_call_loop_and_converts_history() -> None:
    """验证 LangChain 执行工具调用循环并转换会话历史。"""
    model = FakeToolCallingModel()
    agent = LangChainAgent(model)
    history = (
        ConversationMessage("user", "older question"),
        ConversationMessage("assistant", "older answer"),
    )

    assert agent.reply("Count the words", history) == "Count: 3"
    assert "older question" in model._seen[0]
    assert "older answer" in model._seen[0]
    assert agent.reply("Count again") == "Count: 3"
    assert "older question" not in model._seen[2]


# 调用位置：pytest 执行该测试用例时调用。
def test_settings_require_key_and_use_default_model(monkeypatch: pytest.MonkeyPatch) -> None:
    """验证配置要求 API 密钥并正确使用默认模型。"""
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.setenv("DATABASE_URL", "postgresql:///skysun_test")
    with pytest.raises(ValidationError, match="deepseek_api_key"):
        Settings(_env_file=None)

    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    settings = Settings(_env_file=None)
    assert settings.skysun_model == "deepseek-flash"
    assert settings.database_url.get_secret_value() == "postgresql:///skysun_test"
    assert settings.deepseek_api_key.get_secret_value() == "test-key"

    monkeypatch.delenv("DATABASE_URL")
    with pytest.raises(ValidationError, match="database_url"):
        Settings(_env_file=None)


# 调用位置：pytest 执行该测试用例时调用。
@pytest.mark.parametrize("token", ["", "   "])
def test_settings_reject_empty_admin_password(token):
    """验证配置拒绝空管理员密码。"""
    with pytest.raises(ValidationError, match="skysun_admin_token"):
        Settings(_env_file=None, deepseek_api_key="test", database_url="postgresql:///skysun_test", skysun_admin_token=token)


# 调用位置：pytest 执行该测试用例时调用。
def test_composition_root_uses_agent_without_network(monkeypatch: pytest.MonkeyPatch, create_account) -> None:
    """验证应用组装入口使用 Agent，且测试不发起网络请求。"""
    from skysun.core import main

    monkeypatch.setattr(
        main,
        "load_settings",
        lambda: Settings(
            deepseek_api_key="test-key",
            database_url="postgresql:///skysun_test",
            _env_file=None,
        ),
    )
    monkeypatch.setattr(main, "create_deepseek", lambda settings: FakeToolCallingModel())

    from skysun.memory.conversation import InMemoryConversationStore

    from skysun.memory.memory_curator import MemoryPlan

    # 以下集中定义 NoOpCurator：数据结构或组件相关的数据和操作。
    class NoOpCurator:
        # 调用位置：src/skysun/memory/memory_curator.py 的 MemoryCurationService.process、tests/test_memory_curator.py 的 test_langchain_curator_uses_typed_json_output_without_network、tests/test_memory_extraction.py 的 test_valid_structured_output_is_accepted_without_retry 等处调用。
        def decide(self, context):
            """根据对话和已有记忆决定新增、更新或删除内容。"""
            return MemoryPlan()

    # 以下集中定义 EmptyKnowledgeStore：存储层相关的数据和操作。
    class EmptyKnowledgeStore:
        # 调用位置：EmptyKnowledgeStore 的调用方通过该方法使用。
        def search(self, query_embedding, limit=4, max_cosine_distance=0.35):
            """按问题检索相关内容。"""
            return ()

    # 以下集中定义 FakeEmbeddings：数据结构或组件相关的数据和操作。
    class FakeEmbeddings:
        dimension = 1024

        # 调用位置：src/skysun/rag/knowledge_grounding.py 的 GroundedKnowledgeService.search、tests/test_knowledge.py 的 test_qwen_adapter_uses_compatible_api_and_validates_vectors 等处调用。
        def embed_query(self, text):
            """把检索问题转换为向量。"""
            return (1.0,) + (0.0,) * 1023

    client = TestClient(main.create_app(
        InMemoryConversationStore(), curator=NoOpCurator(),
        knowledge_store=EmptyKnowledgeStore(), embeddings=FakeEmbeddings(),
    ))
    user, token = create_account(main.PostgresUserStore("postgresql:///skysun_test"))
    try:
        response = client.post("/chat", json={"message": "Count the words"},
                               headers={"Authorization": f"Bearer {token}"})
    finally:
        with psycopg.connect("postgresql:///skysun_test") as conn:
            conn.execute("DELETE FROM agent_tasks WHERE user_id = %s", (user.user_id,))
            conn.execute("DELETE FROM skysun_users WHERE user_id = %s", (user.user_id,))

    assert response.status_code == 200
    assert response.json()["reply"] == "Count: 3"
    UUID(response.json()["conversation_id"])
