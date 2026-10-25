"""Safe, structured results for Agent tool execution failures."""
# 本文件负责工具调用中的 tool、reliability 相关流程。

import json
from collections.abc import Callable
from contextlib import nullcontext
from typing import TypeVar

from langchain_core.tools import BaseTool

from skysun.tool.errors import RetryableToolError
from skysun.evaluation.observability import span, trace_event

T = TypeVar("T")


# 调用位置：src/skysun/tool/tool_reliability.py 的 configure_validation.handle、src/skysun/tool/tool_reliability.py 的 _run_tool_core 等处调用。
def tool_error(name: str, code: str, attempts: int, retryable: bool = False) -> str:
    """将工具异常转为统一错误结果，避免泄露异常内容、连接地址或工具参数。"""
    return json.dumps({
        "ok": False,
        "tool": name,
        "code": code,
        "message": "The tool could not complete the request.",
        "retryable": retryable,
        "attempts": attempts,
    })


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute、tests/test_tool_reliability.py 的 test_successful_execution_and_argument_validation 等处调用。
def configure_validation(
    tool: BaseTool, on_failure: Callable[[str], None] | None = None
) -> BaseTool:
    """在工具执行前配置参数校验失败时的统一处理。"""
    # 调用位置：LangChain 在工具参数校验失败时通过回调调用。
    def handle(error: Exception) -> str:
        """将工具参数错误转换为统一结果。"""
        result = tool_error(tool.name, "invalid_arguments", 0)
        trace_event("tool_call", "FAILED", {"tool_name": tool.name, "error_code": "invalid_arguments"})
        if on_failure is not None:
            on_failure(result)
        return result

    tool.handle_validation_error = handle
    return tool


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.record_tool_result、src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.recall_memories_tool、src/skysun/tool/tool_reliability.py 的 run_tool 调用。
def is_tool_failure(result: str) -> bool:
    """只识别本项目定义的结构化工具错误结果。"""
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("ok") is False


# 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.count_words_tool、src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.remember_memory_tool、src/skysun/core/langchain_agent.py 的 LangChainAgent.execute.recall_memories_tool 等处调用。
def run_tool(name: str, operation: Callable[[], T], *, retries: int = 0) -> str:
    """执行工具并仅对明确可重试的临时故障重试；写入类工具默认不重试。"""
    if retries < 0 or retries > 1:
        raise ValueError("Tool retry count must be zero or one")
    stage = ("memory_retrieval" if name == "recall_memories" else
             "knowledge_retrieval" if name == "search_knowledge" else None)
    with span("tool_call", {"tool_name": name}) as tool_span:
        with (span(stage) if stage else nullcontext()) as retrieval_span:
            result = _run_tool_core(name, operation, retries)
            tool_span.status = "FAILED" if is_tool_failure(result) else "SUCCESS"
            if retrieval_span is not None:
                retrieval_span.status = tool_span.status
            return result


# 调用位置：src/skysun/tool/tool_reliability.py 的 run_tool 调用。
def _run_tool_core(name: str, operation: Callable[[], T], retries: int) -> str:
    """执行单次工具调用并整理结果。"""
    for attempt in range(1, retries + 2):
        try:
            return str(operation())
        except ValueError:
            return tool_error(name, "invalid_arguments", attempt)
        except (RetryableToolError, TimeoutError, ConnectionError):
            if attempt <= retries:
                continue
            return tool_error(name, "temporary_failure", attempt, retryable=True)
        except Exception:
            return tool_error(name, "execution_failed", attempt)
    raise AssertionError("Unreachable tool retry state")
