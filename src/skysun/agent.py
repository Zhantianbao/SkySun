"""SkySun's framework-independent agent boundary."""

from typing import Protocol, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from skysun.conversation import ConversationMessage


class Agent(Protocol):
    def reply(self, message: str, history: Sequence["ConversationMessage"] = ()) -> str:
        """Return a final answer using only the supplied conversation history."""
