"""SkySun-owned personal memory domain, independent of conversations and LangChain."""
# 本文件负责会话与记忆中的 memory 相关流程。

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol
from uuid import UUID

from skysun.user.user_identity import LEGACY_USER_ID

_KEY_PATTERN = re.compile(r"[a-z][a-z0-9_]{0,79}")
_SECRET_TERMS = re.compile(
    r"\b(?:password|passphrase|api[\s_-]?key|access[\s_-]?token|"
    r"private[\s_-]?key|auth(?:entication)?[\s_-]?credential|"
    r"database[\s_-]?password)\b",
    re.IGNORECASE,
)


# 调用位置：src/skysun/memory/memory_curator.py 的 validate_plan、src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore.remember、src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore.apply_changes 等处调用。
def secret_like_memory(key: str, content: str = "") -> bool:
    """识别明显带凭据标签的记忆内容，避免保存敏感信息。"""
    return bool(_SECRET_TERMS.search(key) or _SECRET_TERMS.search(content))


# 调用位置：src/skysun/memory/memory_curator.py 的 validate_plan、src/skysun/memory/memory_curator.py 的 create_candidates、src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore.remember 等处调用。
def normalize_key(key: str) -> str:
    """将记忆语义键规范化为稳定的小写下划线形式。"""
    normalized = re.sub(r"[\s-]+", "_", key.strip().lower())
    if not _KEY_PATTERN.fullmatch(normalized):
        raise ValueError("Memory key must be snake_case, start with a letter, and be at most 80 characters")
    return normalized


# 以下集中定义 PersonalMemory：记忆数据相关的数据和操作。
@dataclass(frozen=True)
class PersonalMemory:
    id: UUID
    key: str
    content: str
    category: str | None
    source_conversation_id: UUID | None
    created_at: datetime
    updated_at: datetime
    user_id: UUID = LEGACY_USER_ID


# 以下集中定义 GlobalMemory：记忆数据相关的数据和操作。
@dataclass(frozen=True)
class GlobalMemory:
    key: str
    content: str
    created_at: datetime
    updated_at: datetime


# 以下集中定义 MemoryChange：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class MemoryChange:
    action: Literal["upsert", "delete"]
    key: str
    content: str | None = None
    category: str | None = None


# 以下集中定义 MemoryCandidate：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class MemoryCandidate:
    """A proposed durable fact; it is not confirmed memory until accepted and saved."""

    key: str
    content: str
    category: str | None
    importance: Literal["low", "medium", "high"]
    source_conversation_id: UUID


# 以下集中定义 MemoryStore：存储层相关的数据和操作。
class MemoryStore(Protocol):
    # 调用位置：该协议由 MemoryStore 的实现类调用；入口涉及会话服务、记忆工具与整理服务。
    def remember(
        self,
        key: str,
        content: str,
        category: str | None = None,
        source_conversation_id: UUID | None = None,
    ) -> PersonalMemory: """保存或更新个人记忆。"""; ...

    # 调用位置：业务服务通过存储接口读取当前用户数据。
    def get(self, key: str) -> PersonalMemory | None: """按记忆键读取当前用户的个人记忆。"""; ...

    # 调用位置：记忆页面和记忆整理服务调用。
    def list_memories(self, limit: int = 20) -> tuple[PersonalMemory, ...]: """列出当前用户的个人记忆。"""; ...

    # 调用位置：记忆工具和记忆整理服务调用。
    def forget(self, key: str) -> bool: """删除指定的个人记忆。"""; ...

    # 调用位置：记忆工具检索相关记忆时调用。
    def search_memories(self, query: str, limit: int = 20) -> tuple[PersonalMemory, ...]: """查询相关的个人记忆。"""; ...

    # 调用位置：记忆整理服务在校验后批量写入时调用。
    def apply_changes(self, changes: tuple[MemoryChange, ...], source_conversation_id: UUID) -> None: """一次性应用已校验的记忆变更。"""; ...
