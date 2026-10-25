"""Workflow state transitions, safe termination, and evidence verification."""
# 本文件负责测试用例中的 test、workflow 相关流程。

from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.workflow.planner import TaskPlan
from skysun.workflow.verification import VerificationStatus, WorkflowVerifier
from skysun.workflow.workflow import StepOutcome, StepState, WorkflowExecutor, WorkflowStatus


# 调用位置：tests/test_workflow.py 的 test_successful_ordered_workflow_stores_results_and_events、tests/test_workflow.py 的 test_failed_required_step_stops_and_skips_remaining、tests/test_workflow.py 的 test_missing_tool_or_grounding_evidence_needs_review 等处调用。
def sample_plan():
    """为测试提供 sample_plan 所需的输入或模拟行为。"""
    return TaskPlan.model_validate({"goal": "Find and summarize", "steps": [
        {"step_id": "find", "description": "Find a note", "capabilities": ["search_knowledge"]},
        {"step_id": "summarize", "description": "Summarize it", "capabilities": ["reason"],
         "depends_on": ["find"]},
    ]})


# 调用位置：pytest 执行该测试用例时调用。
def test_successful_ordered_workflow_stores_results_and_events() -> None:
    """验证成功的有序工作流保存步骤结果和事件。"""
    plan = sample_plan()
    called = []

    # 调用位置：src/skysun/workflow/workflow.py 的 WorkflowExecutor.execute 调用。
    def runner(step, previous):
        """为测试提供 runner 所需的输入或模拟行为。"""
        called.append(step.step_id)
        if step.step_id == "find":
            return StepOutcome("Evidence found", tools_used=("search_knowledge",),
                               grounding_status=GroundingStatus.SUFFICIENT)
        assert previous[0].output == "Evidence found"
        return StepOutcome("Summary ready")

    run = WorkflowExecutor().execute(plan, runner)
    assert run.status is WorkflowStatus.SUCCESS
    assert called == ["find", "summarize"]
    assert [record.state for record in run.steps] == [StepState.SUCCESS, StepState.SUCCESS]
    assert [event.state for event in run.events if event.step_id == "find"] == [
        StepState.PENDING, StepState.RUNNING, StepState.SUCCESS
    ]
    assert run.steps[0].task_id == run.task_id
    assert run.steps[0].tool_status == "success"


# 调用位置：pytest 执行该测试用例时调用。
def test_failed_required_step_stops_and_skips_remaining() -> None:
    """验证必需步骤失败后停止执行并跳过剩余步骤。"""
    plan = sample_plan()
    called = []

    # 调用位置：src/skysun/workflow/workflow.py 的 WorkflowExecutor.execute 调用。
    def runner(step, previous):
        """为测试提供 runner 所需的输入或模拟行为。"""
        called.append(step.step_id)
        raise RuntimeError("private database detail")

    run = WorkflowExecutor().execute(plan, runner)
    assert called == ["find"]
    assert run.status is WorkflowStatus.FAILED
    assert [item.state for item in run.steps] == [StepState.FAILED, StepState.SKIPPED]
    assert run.steps[0].error_code == "RuntimeError"
    assert "private database detail" not in str(run)
    assert WorkflowVerifier().verify(plan, run).status is VerificationStatus.FAILED


# 调用位置：pytest 执行该测试用例时调用。
def test_optional_failure_allows_partial_execution_but_needs_review() -> None:
    """验证可选步骤失败后允许部分执行，但结果需要复核。"""
    plan = TaskPlan.model_validate({"goal": "Two checks", "steps": [
        {"step_id": "optional", "description": "Try optional check", "capabilities": ["reason"],
         "required": False},
        {"step_id": "final", "description": "Give answer", "capabilities": ["reason"]},
    ]})
    run = WorkflowExecutor().execute(plan, lambda step, previous: (
        StepOutcome("") if step.step_id == "optional" else StepOutcome("Answer")
    ))
    assert run.status is WorkflowStatus.PARTIAL
    assert [item.state for item in run.steps] == [StepState.FAILED, StepState.SUCCESS]
    assert WorkflowVerifier().verify(plan, run).status is VerificationStatus.NEEDS_REVIEW


# 调用位置：pytest 执行该测试用例时调用。
def test_missing_tool_or_grounding_evidence_needs_review() -> None:
    """验证缺少工具结果或知识证据时需要人工复核。"""
    plan = sample_plan()
    run = WorkflowExecutor().execute(plan, lambda step, previous: StepOutcome("Plausible text"))
    result = WorkflowVerifier().verify(plan, run)
    assert result.status is VerificationStatus.NEEDS_REVIEW
    assert result.affected_steps == ("find",)


# 调用位置：pytest 执行该测试用例时调用。
def test_valid_grounded_result_passes_verification() -> None:
    """验证有有效知识证据支持的结果通过验证。"""
    from uuid import uuid4
    plan = sample_plan()
    source = KnowledgeSearchResult("A grounded fact", "note.md", "/notes/note.md", 0, 0.1, uuid4())

    # 调用位置：src/skysun/workflow/workflow.py 的 WorkflowExecutor.execute 调用。
    def runner(step, previous):
        """为测试提供 runner 所需的输入或模拟行为。"""
        if step.step_id == "find":
            return StepOutcome("A grounded fact", tools_used=("search_knowledge",),
                               grounding_status=GroundingStatus.SUFFICIENT, sources=(source,))
        return StepOutcome("A short summary")

    run = WorkflowExecutor().execute(plan, runner)
    assert WorkflowVerifier().verify(plan, run).status is VerificationStatus.SUCCESS


# 调用位置：pytest 执行该测试用例时调用。
def test_structured_tool_failure_marks_step_failed_without_retry() -> None:
    """验证结构化工具错误将步骤标记为失败且不重复执行。"""
    plan = sample_plan()
    calls = 0

    # 调用位置：src/skysun/workflow/workflow.py 的 WorkflowExecutor.execute 调用。
    def runner(step, previous):
        """为测试提供 runner 所需的输入或模拟行为。"""
        nonlocal calls
        calls += 1
        return StepOutcome("Tool failed", tools_used=("search_knowledge",),
                           tool_failures=("search_knowledge",))

    run = WorkflowExecutor().execute(plan, runner)
    assert calls == 1
    assert run.steps[0].tool_status == "failed"
    assert run.steps[0].state is StepState.FAILED
