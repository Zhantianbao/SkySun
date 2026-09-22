"""Validated, bounded plans for multi-step Agent requests."""

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    description: str = Field(min_length=1, max_length=500)
    capabilities: tuple[str, ...] = Field(min_length=1, max_length=4)
    depends_on: tuple[str, ...] = ()
    required: bool = True


class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=2000)
    steps: tuple[PlanStep, ...] = Field(min_length=1, max_length=5)

    @model_validator(mode="after")
    def validate_order(self) -> "TaskPlan":
        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError("Plan step IDs must be unique")
            if len(set(step.depends_on)) != len(step.depends_on) or not set(step.depends_on) <= seen:
                raise ValueError("Plan dependencies must refer to earlier steps")
            seen.add(step.step_id)
        return self


class TaskPlanner(Protocol):
    def plan(self, goal: str, capabilities: tuple[str, ...]) -> TaskPlan: ...


_ACTION = r"(?:analy[sz]e|review|retrieve|search|find|read|compare|summari[sz]e|calculate|count|explain|draft|write|verify)"


def needs_planning(message: str) -> bool:
    """Use the direct Agent path for ordinary questions and one-step requests."""
    text = message.strip().lower()
    if re.search(r"\b(?:first|then|after that|afterwards|finally)\b", text):
        return True
    return bool(re.search(rf"\b{_ACTION}\b.*\band\b.*\b{_ACTION}\b", text))
