"""Plan generation and validation without network calls."""

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
import pytest

from skysun.langchain_planner import LangChainTaskPlanner, PlanningError
from skysun.planner import PlanStep, TaskPlan, needs_planning


def plan_data(steps):
    return {"goal": "Count words and summarize", "steps": steps}


class FakeModel:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def with_structured_output(self, schema, method, include_raw):
        assert schema is TaskPlan
        assert method == "json_mode"
        assert include_raw is True
        return self

    def invoke(self, messages):
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


def test_simple_plan_and_direct_request_routing() -> None:
    data = plan_data([{
        "step_id": "count", "description": "Count the supplied words",
        "capabilities": ["count_words"], "depends_on": [], "required": True,
    }])
    model = FakeModel([{"parsed": data, "raw": AIMessage(content=""), "parsing_error": None}])
    plan = LangChainTaskPlanner(model).plan(data["goal"], ("count_words", "reason"))
    assert len(plan.steps) == 1
    assert plan.steps[0].capabilities == ("count_words",)
    assert not needs_planning("Count these words")


def test_multistep_plan_preserves_order_and_dependencies() -> None:
    plan = TaskPlan.model_validate(plan_data([
        {"step_id": "retrieve", "description": "Find the note", "capabilities": ["search_knowledge"]},
        {"step_id": "summarize", "description": "Summarize the note", "capabilities": ["reason"],
         "depends_on": ["retrieve"]},
    ]))
    assert plan.steps[1].depends_on == ("retrieve",)
    assert needs_planning("Analyze a document and summarize the important points")
    assert not needs_planning("What are cats and dogs?")


def test_invalid_plan_rejects_dependencies_and_unknown_capabilities() -> None:
    with pytest.raises(ValueError):
        TaskPlan(goal="bad", steps=(PlanStep(
            step_id="first", description="Read", capabilities=("reason",), depends_on=("later",)
        ),))
    bad = plan_data([{
        "step_id": "do_it", "description": "Use unknown tool", "capabilities": ["invented_tool"]
    }])
    model = FakeModel([bad, bad])
    with pytest.raises(PlanningError):
        LangChainTaskPlanner(model).plan("Count words and summarize", ("count_words", "reason"))
    assert model.calls == 2


def test_parser_failure_retries_then_fails_safely() -> None:
    model = FakeModel([OutputParserException("bad"), {"parsed": None, "raw": AIMessage(content="not JSON")}])
    with pytest.raises(PlanningError):
        LangChainTaskPlanner(model).plan("Count words and summarize", ("count_words", "reason"))
    assert model.calls == 2
