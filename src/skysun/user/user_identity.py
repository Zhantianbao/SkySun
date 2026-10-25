"""Request-scoped identity and roles; direct legacy calls retain their own owner."""

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


class Role(str, Enum):
    USER = "USER"
    ADMIN = "ADMIN"


@dataclass(frozen=True)
class User:
    user_id: UUID
    username: str
    role: Role
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class Actor:
    user_id: UUID
    role: Role


class AuthorizationError(PermissionError):
    """The current identity cannot access the requested resource."""


_SYSTEM = object()
_actor: ContextVar[Actor | object | None] = ContextVar("skysun_actor", default=None)


def owner_id() -> UUID:
    actor = _actor.get()
    return actor.user_id if isinstance(actor, Actor) else LEGACY_USER_ID


def scoped_user_id() -> UUID | None:
    """None is reserved for internal recovery across all owners."""
    return None if _actor.get() is _SYSTEM else owner_id()


def private_knowledge_owner() -> UUID | None:
    """Legacy local ingestion remains global; authenticated ingestion is private."""
    actor = _actor.get()
    return actor.user_id if isinstance(actor, Actor) else None


def require_admin() -> None:
    actor = _actor.get()
    if actor is not _SYSTEM and not (isinstance(actor, Actor) and actor.role is Role.ADMIN):
        raise AuthorizationError("Administrator access required")


@contextmanager
def actor_scope(actor: Actor) -> Iterator[None]:
    token = _actor.set(actor)
    try:
        yield
    finally:
        _actor.reset(token)


@contextmanager
def system_scope() -> Iterator[None]:
    token = _actor.set(_SYSTEM)
    try:
        yield
    finally:
        _actor.reset(token)

# 验证姓名是否为 2～4 个常用汉字。
def normalize_username(username: str) -> str:
    # 去掉首尾空格。
    value = username.strip()

    if not re.fullmatch(r"[\u4e00-\u9fff]{2,4}", value):
        raise ValueError("姓名只能包含中文, 长度为2到4个汉字")
    return value


def normalize_token(token: str) -> str:
    value = token.strip()
    if not value:
        raise ValueError("邀请码不能为空")
    return value
