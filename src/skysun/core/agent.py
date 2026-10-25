"""SkySun's framework-independent agent boundary."""
# 本文件负责核心服务中的 agent 相关流程。

from typing import Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from skysun.memory.conversation import ConversationMessage


# 以下集中定义 Agent：Agent 执行层相关的数据和操作。
class Agent(Protocol):
    # 调用位置：该协议由 Agent 的实现类调用；入口涉及会话服务和应用入口。
    def reply(
        self, message: str, history: Sequence["ConversationMessage"] = (),
        conversation_id: str | None = None,
    ) -> str:
        """结合会话历史和可用能力，返回本轮对话的最终回答。"""
