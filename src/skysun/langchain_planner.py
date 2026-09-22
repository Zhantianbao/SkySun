"""DeepSeek adapter for SkySun task plans."""

import json

from langchain_core.exceptions import OutputParserException
from langchain_core.language_models.chat_models import BaseChatModel

from skysun.planner import TaskPlan


class PlanningError(RuntimeError):
    """No safe structured plan could be produced."""


class LangChainTaskPlanner:
    def __init__(self, model: BaseChatModel) -> None:
        self._model = model

    def plan(self, goal: str, capabilities: tuple[str, ...]) -> TaskPlan:
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
