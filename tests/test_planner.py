"""Plan generation and validation without network calls."""
# 本文件负责测试用例中的 test、planner 相关流程。

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage
import pytest

from skysun.workflow.langchain_planner import LangChainTaskPlanner, PlanningError
from skysun.workflow.planner import PlanStep, TaskPlan, needs_planning


# 调用位置：tests/test_planner.py 的 test_simple_plan_and_direct_request_routing、tests/test_planner.py 的 test_multistep_plan_preserves_order_and_dependencies、tests/test_planner.py 的 test_invalid_plan_rejects_dependencies_and_unknown_capabilities 调用。
def plan_data(steps):
    """为测试提供 plan_data 所需的输入或模拟行为。"""
    return {"goal": "Count words and summarize", "steps": steps}


# 以下集中定义 FakeModel：数据结构或组件相关的数据和操作。
class FakeModel:
    # 调用位置：创建 FakeModel 实例时调用。
    def __init__(self, responses):
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.responses = iter(responses)
        self.calls = 0

    # 调用位置：src/skysun/evaluation/deepseek_evaluation.py 的 DeepSeekClaimReviewer.review、src/skysun/memory/langchain_memory_curator.py 的 LangChainMemoryCurator.__init__、src/skysun/rag/langchain_knowledge_grounding.py 的 LangChainEvidenceEvaluator.__init__ 等处调用。
    def with_structured_output(self, schema, method, include_raw):
        """为测试提供 with_structured_output 所需的输入或模拟行为。"""
        assert schema is TaskPlan
        assert method == "json_mode"
        assert include_raw is True
        return self

    # 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute、src/skysun/evaluation/deepseek_evaluation.py 的 DeepSeekClaimReviewer.review、src/skysun/memory/langchain_memory_curator.py 的 LangChainMemoryCurator.decide 等处调用。
    def invoke(self, messages):
        """为测试提供 invoke 所需的输入或模拟行为。"""
        self.calls += 1
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


# 调用位置：pytest 执行该测试用例时调用。
def test_simple_plan_and_direct_request_routing() -> None:
    """验证简单计划与直接请求分别进入正确的执行路径。"""
    data = plan_data([{
        "step_id": "count", "description": "Count the supplied words",
        "capabilities": ["count_words"], "depends_on": [], "required": True,
    }])
    model = FakeModel([{"parsed": data, "raw": AIMessage(content=""), "parsing_error": None}])
    plan = LangChainTaskPlanner(model).plan(data["goal"], ("count_words", "reason"))
    assert len(plan.steps) == 1
    assert plan.steps[0].capabilities == ("count_words",)
    assert not needs_planning("Count these words")


# 调用位置：pytest 执行该测试用例时调用。
def test_multistep_plan_preserves_order_and_dependencies() -> None:
    """验证多步骤计划保留步骤顺序和依赖。"""
    plan = TaskPlan.model_validate(plan_data([
        {"step_id": "retrieve", "description": "Find the note", "capabilities": ["search_knowledge"]},
        {"step_id": "summarize", "description": "Summarize the note", "capabilities": ["reason"],
         "depends_on": ["retrieve"]},
    ]))
    assert plan.steps[1].depends_on == ("retrieve",)
    assert needs_planning("Analyze a document and summarize the important points")
    assert not needs_planning("What are cats and dogs?")


# 调用位置：pytest 执行该测试用例时调用。
def test_invalid_plan_rejects_dependencies_and_unknown_capabilities() -> None:
    """验证无效依赖和未知能力会被计划校验拒绝。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_parser_failure_retries_then_fails_safely() -> None:
    """验证解析失败后只重试一次，再安全地报告失败。"""
    model = FakeModel([OutputParserException("bad"), {"parsed": None, "raw": AIMessage(content="not JSON")}])
    with pytest.raises(PlanningError):
        LangChainTaskPlanner(model).plan("Count words and summarize", ("count_words", "reason"))
    assert model.calls == 2
