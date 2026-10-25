"""Tool failures remain visible to the Agent without exposing exception text."""
# 本文件负责测试用例中的 test、tool、reliability 相关流程。

import json
from typing import Any, Sequence

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool

from skysun.core.langchain_agent import LangChainAgent
from skysun.tool.tool_reliability import RetryableToolError, configure_validation, run_tool


# 调用位置：pytest 执行该测试用例时调用。
def test_successful_execution_and_argument_validation() -> None:
    """验证成功执行与参数校验。"""
    # 调用位置：LangChain 在该测试触发工具调用时执行。
    @tool("echo")
    def echo(value: str) -> str:
        """在工具调用测试中原样返回输入值。"""
        return run_tool("echo", lambda: value)

    registered = configure_validation(echo)
    assert registered.invoke({"value": "ready"}) == "ready"
    failure = json.loads(registered.invoke({"wrong": "value"}))
    assert failure["ok"] is False
    assert failure["tool"] == "echo"
    assert failure["code"] == "invalid_arguments"


# 调用位置：pytest 执行该测试用例时调用。
def test_tool_exception_is_structured_and_sanitized() -> None:
    """验证工具异常被整理为结构化错误且不泄露敏感信息。"""
    # 调用位置：当前测试或 fixture test_tool_exception_is_structured_and_sanitized 直接调用，或通过 monkeypatch 作为替身调用。
    def fails():
        """为测试提供 fails 所需的输入或模拟行为。"""
        raise RuntimeError("private database detail")

    failure = json.loads(run_tool("search", fails))
    assert failure["code"] == "execution_failed"
    assert failure["attempts"] == 1
    assert "private database detail" not in json.dumps(failure)


# 调用位置：pytest 执行该测试用例时调用。
def test_transient_read_failure_retries_once() -> None:
    """验证临时读取失败重试一次。"""
    attempts = 0

    # 调用位置：当前测试或 fixture test_transient_read_failure_retries_once 直接调用，或通过 monkeypatch 作为替身调用。
    def flaky():
        """为测试提供 flaky 所需的输入或模拟行为。"""
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RetryableToolError("temporary")
        return "recovered"

    assert run_tool("recall", flaky, retries=1) == "recovered"
    assert attempts == 2


# 调用位置：pytest 执行该测试用例时调用。
def test_retry_exhaustion_reports_failure() -> None:
    """验证重试耗尽报告失败。"""
    attempts = 0

    # 调用位置：当前测试或 fixture test_retry_exhaustion_reports_failure 直接调用，或通过 monkeypatch 作为替身调用。
    def fails():
        """为测试提供 fails 所需的输入或模拟行为。"""
        nonlocal attempts
        attempts += 1
        raise TimeoutError("temporary")

    failure = json.loads(run_tool("recall", fails, retries=1))
    assert failure["code"] == "temporary_failure"
    assert failure["attempts"] == 2
    assert failure["retryable"] is True
    assert attempts == 2


# 调用位置：pytest 执行该测试用例时调用。
def test_agent_reuses_failed_memory_read_within_one_turn() -> None:
    """验证 Agent 在同一轮对话中复用失败的记忆读取结果。"""
    # 以下集中定义 RepeatingModel：数据结构或组件相关的数据和操作。
    class RepeatingModel(BaseChatModel):
        # 调用位置：RepeatingModel 的调用方通过该方法使用。
        @property
        def _llm_type(self) -> str:
            """为测试准备 _llm_type 所需的替身或输入。"""
            return "repeating-memory-read"

        # 调用位置：RepeatingModel 的调用方通过该方法使用。
        def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "RepeatingModel":
            """为测试提供 bind_tools 所需的输入或模拟行为。"""
            return self

        # 调用位置：LangChain 模型接口在生成回复时调用。
        def _generate(self, messages: list[BaseMessage], stop=None, **kwargs: Any) -> ChatResult:
            """为测试准备 _generate 所需的替身或输入。"""
            results = [item for item in messages if isinstance(item, ToolMessage)]
            if len(results) < 2:
                answer = AIMessage(content="", tool_calls=[{
                    "name": "recall_memories", "args": {"query": "style"},
                    "id": f"read_{len(results)}",
                }])
            else:
                answer = AIMessage(content="You prefer bullet points.")
            return ChatResult(generations=[ChatGeneration(message=answer)])

    # 以下集中定义 FailingStore：存储层相关的数据和操作。
    class FailingStore:
        # 调用位置：创建 FailingStore 实例时调用。
        def __init__(self):
            """保存依赖和初始状态，供该对象后续方法使用。"""
            self.calls = 0

        # 调用位置：src/skysun/tool/tools.py 的 recall_memories、tests/test_multi_user.py 的 test_personal_memory_same_key_isolated_and_global_memory_separate、tests/test_postgres_memory.py 的 test_curator_batch_is_atomic_and_lexical_recall_is_bounded 等处调用。
        def search_memories(self, query, limit=20):
            """检索与查询相关的记忆。"""
            self.calls += 1
            raise RetryableToolError("unavailable")

    store = FailingStore()
    answer = LangChainAgent(RepeatingModel(), store).reply("What is my style?")
    assert store.calls == 2
    assert "couldn't access personal memory" in answer
    assert "bullet" not in answer
