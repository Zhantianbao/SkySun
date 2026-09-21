"""SkySun's framework-independent agent boundary."""

from typing import Protocol


class Agent(Protocol):
    def reply(self, message: str) -> str:
        """Return one answer for a single, stateless chat request."""
