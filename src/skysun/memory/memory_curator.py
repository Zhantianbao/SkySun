"""SkySun-owned structured memory curation and application rules."""
# 本文件负责会话与记忆中的 memory、curator 相关流程。

import logging
from dataclasses import dataclass
from typing import Literal, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from skysun.memory.conversation import ConversationMessage
from skysun.memory.memory import MemoryCandidate, MemoryChange, MemoryStore, PersonalMemory, normalize_key, secret_like_memory

logger = logging.getLogger(__name__)


# 以下集中定义 MemoryDecision：数据结构或组件相关的数据和操作。
class MemoryDecision(BaseModel):
    """One proposed change derived from the current user's message."""

    model_config = ConfigDict(extra="forbid")

    action: Literal["ignore", "upsert", "delete"]
    key: str | None = Field(
        default=None, description="Stable snake_case semantic key; reuse an existing key for updates."
    )
    content: str | None = Field(
        default=None, description="Durable user-sourced fact or preference, needed for upsert."
    )
    category: str | None = Field(
        default=None, description="Optional category such as preference, goal, constraint, or project."
    )
    importance: Literal["low", "medium", "high"] = Field(
        default="medium", description="Expected long-term usefulness of this candidate."
    )


# 以下集中定义 MemoryPlan：计划数据相关的数据和操作。
class MemoryPlan(BaseModel):
    """At most three structured decisions for one successful user turn."""

    model_config = ConfigDict(extra="forbid")

    decisions: list[MemoryDecision] = Field(default_factory=list, max_length=3)


# 以下集中定义 CurationContext：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class CurationContext:
    user_message: str
    recent_history: tuple[ConversationMessage, ...]
    current_memories: tuple[PersonalMemory, ...]
    conversation_id: UUID


# 以下集中定义 MemoryCurator：数据结构或组件相关的数据和操作。
class MemoryCurator(Protocol):
    # 调用位置：记忆整理服务调用实现类生成决策。
    def decide(self, context: CurationContext) -> MemoryPlan: """根据对话和已有记忆生成整理决策。"""; ...


# 调用位置：src/skysun/memory/langchain_memory_curator.py 的 _as_plan、src/skysun/memory/memory_curator.py 的 create_candidates、src/skysun/memory/memory_curator.py 的 MemoryCurationService.process 等处调用。
def validate_plan(plan: MemoryPlan) -> tuple[MemoryChange, ...]:
    """在写入数据库前校验每项记忆动作，拒绝格式错误的计划。"""
    if len(plan.decisions) > 3:
        raise ValueError("Too many memory decisions")
    changes: list[MemoryChange] = []
    for decision in plan.decisions:
        if decision.action == "ignore":
            continue
        if not decision.key:
            raise ValueError("Memory key is required")
        key = normalize_key(decision.key)
        if decision.action == "upsert":
            content = (decision.content or "").strip()
            if not content:
                raise ValueError("Memory content is required")
            if secret_like_memory(key, content):
                raise ValueError("Secret-like memory rejected")
            category = (decision.category or "").strip() or None
            changes.append(MemoryChange("upsert", key, content, category))
        else:
            changes.append(MemoryChange("delete", key))
    return tuple(changes)


# 调用位置：src/skysun/memory/memory_curator.py 的 MemoryCurationService.process、tests/test_memory_curator.py 的 test_candidate_creation_keeps_importance_and_conversation_source 调用。
def create_candidates(plan: MemoryPlan, source_conversation_id: UUID) -> tuple[MemoryCandidate, ...]:
    """将提议保存的内容整理为候选记忆，与已确认的个人记忆区分。"""
    validate_plan(plan)
    return tuple(
        MemoryCandidate(
            key=normalize_key(decision.key or ""),
            content=(decision.content or "").strip(),
            category=(decision.category or "").strip() or None,
            importance=decision.importance,
            source_conversation_id=source_conversation_id,
        )
        for decision in plan.decisions if decision.action == "upsert"
    )


# 调用位置：src/skysun/memory/memory_curator.py 的 MemoryCurationService.process 调用。
def accept_candidate(
    candidate: MemoryCandidate, current_memories: tuple[PersonalMemory, ...]
) -> bool:
    """在保存前剔除低价值或重复的候选记忆。"""
    if candidate.importance == "low":
        return False
    normalized_content = " ".join(candidate.content.casefold().split())
    return not any(
        " ".join(memory.content.casefold().split()) == normalized_content
        for memory in current_memories
    )


# 以下集中定义 MemoryCurationService：服务层相关的数据和操作。
class MemoryCurationService:
    """Best-effort curation after conversation persistence.

    The caller handles exceptions so the successful chat turn still returns.
    One plan is validated before a single transactional memory-store call.
    """

    # 调用位置：创建 MemoryCurationService 实例时调用。
    def __init__(self, curator: MemoryCurator, store: MemoryStore) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._curator = curator
        self._store = store

    # 调用位置：通过 MemoryCurationService 实例在会话服务、记忆工具与整理服务中调用。
    def process(
        self,
        user_message: str,
        history: tuple[ConversationMessage, ...],
        conversation_id: str,
    ) -> None:
        """处理一轮对话并整理长期记忆。"""
        context = CurationContext(
            user_message=user_message,
            recent_history=history[-6:],
            current_memories=self._store.list_memories(limit=20),
            conversation_id=UUID(conversation_id),
        )
        logger.info("Memory curator invoked")
        plan = self._curator.decide(context)
        if not isinstance(plan, MemoryPlan):
            plan = MemoryPlan.model_validate(plan)
        changes = validate_plan(plan)
        candidates = create_candidates(plan, context.conversation_id)
        accepted: list[MemoryChange] = []
        seen = list(context.current_memories)
        accepted_candidates: list[MemoryCandidate] = []
        for change in changes:
            if change.action == "delete":
                accepted.append(change)
                seen = [memory for memory in seen if memory.key != change.key]
                continue
            candidate = next(item for item in candidates if item.key == change.key and item.content == change.content)
            existing = self._store.get(candidate.key)
            if existing is not None and all(item.key != existing.key for item in seen):
                seen.append(existing)
            duplicate_in_plan = any(
                item.key == candidate.key
                or " ".join(item.content.casefold().split()) == " ".join(candidate.content.casefold().split())
                for item in accepted_candidates
            )
            if not duplicate_in_plan and accept_candidate(candidate, tuple(seen)):
                accepted.append(change)
                accepted_candidates.append(candidate)
        logger.info("Memory curator accepted %d changes", len(accepted))
        if accepted:
            self._store.apply_changes(tuple(accepted), context.conversation_id)
