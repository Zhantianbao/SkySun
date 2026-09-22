"""Safe, structured results for Agent tool execution failures."""

import json
from collections.abc import Callable
from contextlib import nullcontext
from typing import TypeVar

from langchain_core.tools import BaseTool

from skysun.errors import RetryableToolError
from skysun.observability import span, trace_event

T = TypeVar("T")


def tool_error(name: str, code: str, attempts: int, retryable: bool = False) -> str:
    """Avoid leaking exception messages, connection strings, or tool arguments."""
    return json.dumps({
        "ok": False,
        "tool": name,
        "code": code,
        "message": "The tool could not complete the request.",
        "retryable": retryable,
        "attempts": attempts,
    })


def configure_validation(
    tool: BaseTool, on_failure: Callable[[str], None] | None = None
) -> BaseTool:
    """LangChain validates arguments before calling the wrapped function."""
    def handle(error: Exception) -> str:
        result = tool_error(tool.name, "invalid_arguments", 0)
        trace_event("tool_call", "FAILED", {"tool_name": tool.name, "error_code": "invalid_arguments"})
        if on_failure is not None:
            on_failure(result)
        return result

    tool.handle_validation_error = handle
    return tool


def is_tool_failure(result: str) -> bool:
    """Recognize only our structured error payloads."""
    try:
        payload = json.loads(result)
    except (TypeError, ValueError):
        return False
    return isinstance(payload, dict) and payload.get("ok") is False


def run_tool(name: str, operation: Callable[[], T], *, retries: int = 0) -> str:
    """Retry only explicit transient failures; write tools use the default of zero."""
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


def _run_tool_core(name: str, operation: Callable[[], T], retries: int) -> str:
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
