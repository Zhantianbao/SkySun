"""DeepSeek adapter for SkySun task plans."""
# 本文件负责任务工作流中的 langchain、planner 相关流程。

import json

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel

from skysun.workflow.planner import TaskPlan


# 以下集中定义 PlanningError：数据结构或组件相关的数据和操作。
class PlanningError(RuntimeError):
    """No safe structured plan could be produced."""


# 以下集中定义 LangChainTaskPlanner：计划生成层相关的数据和操作。
class LangChainTaskPlanner:
    # 调用位置：创建 LangChainTaskPlanner 实例时调用。
    def __init__(self, model: BaseChatModel) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._model = model

    # 调用位置：通过 LangChainTaskPlanner 实例在工作流 Agent 与任务接口中调用。
    def plan(self, goal: str, capabilities: tuple[str, ...]) -> TaskPlan:
        """根据用户目标生成可执行的步骤计划。"""
        if not goal.strip() or not capabilities:
            raise PlanningError("A goal and available capabilities are required")
        structured = self._model.with_structured_output(
            TaskPlan, method="json_mode", include_raw=True
        )
        prompt = (
            "Create a bounded, ordered execution plan for the user goal. Return a JSON "
            "object with goal and steps matching TaskPlan. Use 1 to 5 steps. Give each "
            "step a unique snake_case step_id, concrete description, capabilities array, "
            "depends_on array referencing only earlier steps, and required boolean. "
            "Use only available capabilities. Include a final answer step using reason. "
            "Do not invent facts, perform actions, or include credentials."
        )
        payload = {"goal": goal, "available_capabilities": capabilities}
        for attempt in range(2):
            try:
                result = structured.invoke([
                    ("system", prompt + (" Return only the JSON object, without markdown." if attempt else "")),
                    ("user", json.dumps(payload, ensure_ascii=False)),
                ])
            except OutputParserException:
                continue
            parsed = result.get("parsed") if isinstance(result, dict) and "parsed" in result else result
            if parsed is None and isinstance(result, dict):
                raw = getattr(result.get("raw"), "content", None)
                if isinstance(raw, str):
                    try:
                        parsed = json.loads(raw)
                    except json.JSONDecodeError:
                        parsed = None
            try:
                plan = TaskPlan.model_validate(parsed)
            except ValueError:
                continue
            if set(capability for step in plan.steps for capability in step.capabilities) <= set(capabilities):
                return plan
        raise PlanningError("The planner could not produce a valid plan")
