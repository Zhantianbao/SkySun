"""LangChain/DeepSeek adapter for SkySun's structured Memory Curator."""
# 本文件负责会话与记忆中的 langchain、memory、curator 相关流程。

import json
import logging

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel

from skysun.memory.memory_curator import CurationContext, MemoryPlan, validate_plan

logger = logging.getLogger(__name__)

_CURATOR_PROMPT = (
    "You are SkySun's private memory curator. Return a JSON object with a decisions array matching the MemoryPlan schema; "
    "return no prose and never answer the user. Consider durable, future-useful facts stated by the USER in "
    "the CURRENT message: stable preferences, long-term goals, recurring constraints, "
    "project facts, explicit remember requests, clear replacements, and explicit forget "
    "requests. Ignore transient states, passing activities, one-off tasks, speculation, "
    "hypotheticals, and assistant-generated assumptions. Be conservative: an empty "
    "decisions list means ignore. Never store passwords, API keys, access tokens, "
    "private keys, authentication credentials, or database passwords. Use existing "
    "semantic keys for updates and avoid duplicate facts. For a clear replacement, "
    "upsert the existing key with the new fact. Delete only on an explicit forget "
    "request. Return no more than three decisions. Each upsert needs a stable "
    "snake_case key and concise content. Give each upsert an importance of low, medium, "
    "or high based on long-term usefulness; use low for weak or one-off candidates. "
    "The supplied history is context only; do "
    "not independently mine old turns for new facts."
)

_REPAIR_SUFFIX = (
    " Return exactly one JSON object with a decisions array. "
    "Each decision must use action ignore, upsert, or delete; optional key, content, "
    "category, and importance fields must match the schema. No markdown or prose."
)
_WRAPPER_KEYS = frozenset({"memory_plan", "plan", "result", "output", "data"})


# 调用位置：src/skysun/memory/langchain_memory_curator.py 的 _as_plan 调用。
def _decode_json(text: str) -> object | None:
    """从纯文本、代码围栏或带简短前缀的模型输出中提取结构化数据。"""
    if len(text) > 32768:
        return None
    decoder = json.JSONDecoder()
    for index, character in enumerate(text):
        if character not in "{[":
            continue
        try:
            value, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            continue
        if "{" in text[end:] or "[" in text[end:]:
            return None
        return value
    return None


# 调用位置：src/skysun/memory/langchain_memory_curator.py 的 _parse_result 调用。
def _as_plan(value: object) -> MemoryPlan | None:
    """解析允许的模型输出包装形式，再按完整记忆规则生成整理计划。"""
    if isinstance(value, MemoryPlan):
        try:
            validate_plan(value)
            return value
        except ValueError:
            return None
    if isinstance(value, str):
        value = _decode_json(value)
    for _ in range(3):
        if isinstance(value, list):
            value = {"decisions": value}
        if not isinstance(value, dict):
            break
        if "decisions" in value:
            try:
                plan = MemoryPlan.model_validate(value)
                validate_plan(plan)
                return plan
            except ValueError:
                return None
        if len(value) != 1:
            break
        key, value = next(iter(value.items()))
        if key not in _WRAPPER_KEYS:
            break
        if isinstance(value, str):
            value = _decode_json(value)
    return None


# 调用位置：src/skysun/memory/langchain_memory_curator.py 的 LangChainMemoryCurator.decide 调用。
def _parse_result(result: object) -> MemoryPlan | None:
    """解析并校验模型返回的结构化结果。"""
    if isinstance(result, dict) and {"raw", "parsed", "parsing_error"} <= result.keys():
        parsed = _as_plan(result["parsed"])
        if parsed is not None:
            return parsed
        return _as_plan(getattr(result["raw"], "content", None))
    return _as_plan(result)


# 以下集中定义 LangChainMemoryCurator：数据结构或组件相关的数据和操作。
class LangChainMemoryCurator:
    # 调用位置：创建 LangChainMemoryCurator 实例时调用。
    def __init__(self, model: BaseChatModel) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._structured_model = model.with_structured_output(
            MemoryPlan, method="json_mode", include_raw=True
        )

    # 调用位置：通过 LangChainMemoryCurator 实例在会话服务、记忆工具与整理服务中调用。
    def decide(self, context: CurationContext) -> MemoryPlan:
        """根据对话和已有记忆决定新增、更新或删除内容。"""
        memories = [
            {"key": memory.key, "content": memory.content, "category": memory.category}
            for memory in context.current_memories
        ]
        history = [
            {"role": turn.role, "content": turn.content}
            for turn in context.recent_history
        ]
        payload = {
            "conversation_id": str(context.conversation_id),
            "current_user_message": context.user_message,
            "recent_history": history,
            "current_personal_memories": memories,
        }
        for attempt in range(2):
            try:
                result = self._structured_model.invoke([
                    ("system", _CURATOR_PROMPT + (_REPAIR_SUFFIX if attempt else "")),
                    ("user", json.dumps(payload, ensure_ascii=False)),
                ])
            except OutputParserException:
                logger.warning("Memory curator structured output parsing failed")
                continue
            plan = _parse_result(result)
            if plan is not None:
                return plan
        logger.warning("Memory curator rejected invalid structured output")
        return MemoryPlan()
