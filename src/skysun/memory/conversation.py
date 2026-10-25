"""SkySun-owned conversation types and temporary in-memory storage.

InMemoryConversationStore loses history on restart. PostgreSQL storage is the
normal runtime. Neither store is long-term personal memory.
"""
# 本文件负责会话与记忆中的 conversation 相关流程。

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from typing import TYPE_CHECKING, Literal, Protocol
from uuid import uuid4

from skysun.core.agent import Agent
from skysun.user.user_identity import owner_id

if TYPE_CHECKING:
    from skysun.memory.memory_curator import MemoryCurationService

logger = logging.getLogger(__name__)


# 以下集中定义 ConversationMessage：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class ConversationMessage:
    role: Literal["user", "assistant"]
    content: str


# 以下集中定义 ConversationSummary：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    title: str
    created_at: datetime
    updated_at: datetime


# 以下集中定义 ConversationNotFound：数据结构或组件相关的数据和操作。
class ConversationNotFound(Exception):
    """The requested conversation does not exist in this store."""


# 以下集中定义 ConversationStore：存储层相关的数据和操作。
class ConversationStore(Protocol):
    # 调用位置：会话服务或测试夹具通过此接口创建数据。
    def create(self, conversation_id: str | None = None) -> str: """创建新会话并返回会话标识。"""; ...
    # 调用位置：会话服务读取历史消息时调用。
    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]: """读取指定会话的消息历史。"""; ...
    # 调用位置：会话服务保存一轮对话时调用。
    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None: """保存用户消息和助手回复。"""; ...
    # 调用位置：会话列表接口通过会话服务调用。
    def list_conversations(self, limit: int = 50) -> tuple[ConversationSummary, ...]: """列出当前用户的会话。"""; ...


# 以下集中定义 InMemoryConversationStore：存储层相关的数据和操作。
class InMemoryConversationStore:
    """Process-local storage useful for tests and lightweight development."""

    # 调用位置：创建 InMemoryConversationStore 实例时调用。
    def __init__(self) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._messages: dict[str, list[ConversationMessage]] = {}
        self._owners: dict[str, object] = {}
        self._created: dict[str, datetime] = {}
        self._updated: dict[str, datetime] = {}
        self._lock = RLock()

    # 调用位置：通过 InMemoryConversationStore 实例在会话服务、记忆工具与整理服务中调用。
    def create(self, conversation_id: str | None = None) -> str:
        """创建新会话并返回会话标识。"""
        with self._lock:
            conversation_id = conversation_id or str(uuid4())
            if conversation_id in self._messages:
                raise ValueError("Conversation already exists")
            self._messages[conversation_id] = []
            self._owners[conversation_id] = owner_id()
            self._created[conversation_id] = datetime.now(timezone.utc)
            self._updated[conversation_id] = self._created[conversation_id]
            return conversation_id

    # 调用位置：通过 InMemoryConversationStore 实例在会话服务、记忆工具与整理服务中调用。
    def list_conversations(self, limit: int = 50) -> tuple[ConversationSummary, ...]:
        """按更新时间列出会话。"""
        with self._lock:
            rows = (
                ConversationSummary(identifier,
                                    next((message.content[:80] for message in messages
                                          if message.role == "user"), "新对话"),
                                    self._created[identifier], self._updated[identifier])
                for identifier, messages in self._messages.items()
                if self._owners[identifier] == owner_id()
            )
            return tuple(sorted(rows, key=lambda item: item.updated_at, reverse=True)[:limit])

    # 调用位置：通过 InMemoryConversationStore 实例在会话服务、记忆工具与整理服务中调用。
    def get_history(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        """读取指定会话的历史消息。"""
        with self._lock:
            if self._owners[conversation_id] != owner_id():
                raise KeyError(conversation_id)
            return tuple(self._messages[conversation_id])

    # 调用位置：通过 InMemoryConversationStore 实例在会话服务、记忆工具与整理服务中调用。
    def append_exchange(self, conversation_id: str, user: str, assistant: str) -> None:
        """把用户消息和 Agent 回复一起保存。"""
        with self._lock:
            if self._owners[conversation_id] != owner_id():
                raise KeyError(conversation_id)
            history = self._messages[conversation_id]
            history.extend(
                (
                    ConversationMessage("user", user),
                    ConversationMessage("assistant", assistant),
                )
            )
            self._updated[conversation_id] = datetime.now(timezone.utc)


# 以下集中定义 ConversationService：服务层相关的数据和操作。
class ConversationService:
    """Store only completed turns.

    Agent failures leave history unchanged. Separate conversations can run
    concurrently. Clients should send turns to one conversation sequentially:
    simultaneous turns may both read the same prior history.
    """

    # 调用位置：创建 ConversationService 实例时调用。
    def __init__(
        self, agent: Agent, store: ConversationStore,
        memory_curation: "MemoryCurationService | None" = None,
    ) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._agent = agent
        self._store = store
        self._memory_curation = memory_curation

    # 调用位置：通过 ConversationService 实例在会话服务、记忆工具与整理服务中调用。
    def chat(self, message: str, conversation_id: str | None = None) -> tuple[str, str]:
        """在当前用户范围内处理聊天并返回回复及任务状态。"""
        if conversation_id is None:
            history: tuple[ConversationMessage, ...] = ()
        else:
            try:
                history = self._store.get_history(conversation_id)
            except KeyError as exc:
                raise ConversationNotFound from exc
        actual_id = conversation_id or str(uuid4())
        reply = self._agent.reply(message, history, actual_id)
        if conversation_id is None:
            self._store.create(actual_id)
        self._store.append_exchange(actual_id, message, reply)
        if self._memory_curation is not None:
            try:
                self._memory_curation.process(message, history, actual_id)
            except Exception as exc:
                logger.warning("Memory curation failed: %s", type(exc).__name__)
        return actual_id, reply

    # 调用位置：通过 ConversationService 实例在会话服务、记忆工具与整理服务中调用。
    def history(self, conversation_id: str) -> tuple[ConversationMessage, ...]:
        """读取指定会话的历史消息。"""
        try:
            return self._store.get_history(conversation_id)
        except KeyError as exc:
            raise ConversationNotFound from exc

    # 调用位置：通过 ConversationService 实例在会话服务、记忆工具与整理服务中调用。
    def list_conversations(self, limit: int = 50) -> tuple[ConversationSummary, ...]:
        """按更新时间列出会话。"""
        return self._store.list_conversations(limit)
