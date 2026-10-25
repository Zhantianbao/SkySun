"""Validated, bounded plans for multi-step Agent requests."""
# 本文件负责任务工作流中的 planner 相关流程。

import re
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field, model_validator


# 以下集中定义 PlanStep：数据结构或组件相关的数据和操作。
class PlanStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_id: str = Field(pattern=r"^[a-z][a-z0-9_]{0,39}$")
    description: str = Field(min_length=1, max_length=500)
    capabilities: tuple[str, ...] = Field(min_length=1, max_length=4)
    depends_on: tuple[str, ...] = ()
    required: bool = True


# 以下集中定义 TaskPlan：计划数据相关的数据和操作。
class TaskPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal: str = Field(min_length=1, max_length=2000)
    steps: tuple[PlanStep, ...] = Field(min_length=1, max_length=5)

    # 调用位置：通过 TaskPlan 实例在工作流 Agent 与任务接口中调用。
    @model_validator(mode="after")
    def validate_order(self) -> "TaskPlan":
        """校验计划步骤的顺序和依赖。"""
        seen: set[str] = set()
        for step in self.steps:
            if step.step_id in seen:
                raise ValueError("Plan step IDs must be unique")
            if len(set(step.depends_on)) != len(step.depends_on) or not set(step.depends_on) <= seen:
                raise ValueError("Plan dependencies must refer to earlier steps")
            seen.add(step.step_id)
        return self


# 以下集中定义 TaskPlanner：计划生成层相关的数据和操作。
class TaskPlanner(Protocol):
    # 调用位置：工作流智能代理生成任务步骤时调用。
    def plan(self, goal: str, capabilities: tuple[str, ...]) -> TaskPlan: """根据目标和能力生成任务计划。"""; ...


_ACTION = r"(?:analy[sz]e|review|retrieve|search|find|read|compare|summari[sz]e|calculate|count|explain|draft|write|verify)"


# 调用位置：src/skysun/workflow/workflow_agent.py 的 WorkflowAgent.reply、tests/test_planner.py 的 test_simple_plan_and_direct_request_routing、tests/test_planner.py 的 test_multistep_plan_preserves_order_and_dependencies 等处调用。
def needs_planning(message: str) -> bool:
    """判断请求是否需要多步骤计划；普通问题和单步请求直接执行。"""
    text = message.strip().lower()
    if re.search(r"\b(?:first|then|after that|afterwards|finally)\b", text):
        return True
    return bool(re.search(rf"\b{_ACTION}\b.*\band\b.*\b{_ACTION}\b", text))
