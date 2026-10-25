"""Request-scoped identity and roles; direct legacy calls retain their own owner."""
# 本文件负责用户身份中的 user、identity 相关流程。

import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Iterator
from uuid import UUID


ADMIN_USER_ID = UUID("00000000-0000-0000-0000-000000000001")

# Historical records and the built-in administrator keep the same owner ID.
LEGACY_USER_ID = ADMIN_USER_ID


# 以下集中定义 Role：数据结构或组件相关的数据和操作。
class Role(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"


# 以下集中定义 User：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class User:
    user_id: UUID
    username: str
    role: Role
    created_at: datetime
    updated_at: datetime


# 以下集中定义 Actor：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class Actor:
    user_id: UUID
    role: Role


# 以下集中定义 AuthorizationError：数据结构或组件相关的数据和操作。
class AuthorizationError(PermissionError):
    """The current identity cannot access the requested resource."""


_SYSTEM = object()
_actor: ContextVar[Actor | object | None] = ContextVar("skysun_actor", default=None)


# 调用位置：src/skysun/evaluation/observability.py 的 ResponseEvaluator.evaluate、src/skysun/memory/conversation.py 的 InMemoryConversationStore.create、src/skysun/memory/conversation.py 的 InMemoryConversationStore.list_conversations 等处调用。
def owner_id() -> UUID:
    """取得当前用户的私有数据所有者标识。"""
    actor = _actor.get()
    return actor.user_id if isinstance(actor, Actor) else LEGACY_USER_ID


# 调用位置：src/skysun/evaluation/postgres_observability.py 的 PostgresObservabilityStore._owner、src/skysun/evaluation/postgres_observability.py 的 PostgresObservabilityStore.list_traces、src/skysun/evaluation/postgres_observability.py 的 PostgresObservabilityStore.list_evaluations 等处调用。
def scoped_user_id() -> UUID | None:
    """取得当前用户标识；系统级恢复任务返回空值以遍历所有用户。"""
    return None if _actor.get() is _SYSTEM else owner_id()


# 调用位置：src/skysun/rag/postgres_knowledge.py 的 PostgresKnowledgeStore.get_by_path、src/skysun/rag/postgres_knowledge.py 的 PostgresKnowledgeStore.list_documents、src/skysun/rag/postgres_knowledge.py 的 PostgresKnowledgeStore.save_document 等处调用。
def private_knowledge_owner() -> UUID | None:
    """已登录用户入库的知识归其本人所有；历史本地入库保持全局可见。"""
    actor = _actor.get()
    return actor.user_id if isinstance(actor, Actor) else None


# 调用位置：src/skysun/evaluation/postgres_observability.py 的 PostgresObservabilityStore.metrics、src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore.remember_global、src/skysun/memory/postgres_memory.py 的 PostgresMemoryStore.list_global_memories 调用。
def require_admin() -> None:
    """检查当前身份是否具有管理员权限。"""
    actor = _actor.get()
    if actor is not _SYSTEM and not (isinstance(actor, Actor) and actor.role is Role.ADMIN):
        raise AuthorizationError("Administrator access required")


# 调用位置：src/skysun/core/api.py 的 build_api.chat、src/skysun/core/api.py 的 build_api.conversation_messages、src/skysun/core/api.py 的 build_api.conversation_list 等处调用。
@contextmanager
def actor_scope(actor: Actor) -> Iterator[None]:
    """在当前操作范围内设置用户身份。"""
    token = _actor.set(actor)
    try:
        yield
    finally:
        _actor.reset(token)


# 调用位置：src/skysun/core/api.py 的 build_api.summary、src/skysun/core/api.py 的 build_api.traces、src/skysun/core/api.py 的 build_api.evaluations 等处调用。
@contextmanager
def system_scope() -> Iterator[None]:
    """在系统任务范围内设置系统身份。"""
    token = _actor.set(_SYSTEM)
    try:
        yield
    finally:
        _actor.reset(token)

# 验证姓名是否为 2～4 个常用汉字。
# 调用位置：src/skysun/core/api.py 的 build_api.login、src/skysun/user/postgres_users.py 的 PostgresUserStore.create_user、tests/test_user_identity.py 的 test_chinese_names_and_surrounding_spaces 等处调用。
def normalize_username(username: str) -> str:
    # 去掉首尾空格。
    """清理并校验登录姓名。"""
    value = username.strip()

    if not re.fullmatch(r"[\u4e00-\u9fff]{2,4}", value):
        raise ValueError("姓名只能包含中文, 长度为2到4个汉字")
    return value


# 调用位置：src/skysun/core/api.py 的 build_api.login、src/skysun/user/postgres_users.py 的 PostgresUserStore.bootstrap_admin、src/skysun/user/postgres_users.py 的 PostgresUserStore.create_user 等处调用。
def normalize_token(token: str) -> str:
    """清理并校验登录令牌。"""
    value = token.strip()
    if not value:
        raise ValueError("邀请码不能为空")
    return value
