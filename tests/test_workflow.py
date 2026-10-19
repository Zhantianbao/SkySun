"""Workflow state transitions, safe termination, and evidence verification."""

from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.workflow.planner import TaskPlan
from skysun.workflow.verification import VerificationStatus, WorkflowVerifier
from skysun.workflow.workflow import StepOutcome, StepState, WorkflowExecutor, WorkflowStatus


def sample_plan():
    return TaskPlan.model_validate({"goal": "Find and summarize", "steps": [
        {"step_id": "find", "description": "Find a note", "capabilities": ["search_knowledge"]},
        {"step_id": "summarize", "description": "Summarize it", "capabilities": ["reason"],
         "depends_on": ["find"]},
    ]})


def test_successful_ordered_workflow_stores_results_and_events() -> None:
    plan = sample_plan()
    called = []

    def runner(step, previous):
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


def test_failed_required_step_stops_and_skips_remaining() -> None:
    plan = sample_plan()
    called = []

    def runner(step, previous):
        called.append(step.step_id)
        raise RuntimeError("private database detail")

    run = WorkflowExecutor().execute(plan, runner)
    assert called == ["find"]
    assert run.status is WorkflowStatus.FAILED
    assert [item.state for item in run.steps] == [StepState.FAILED, StepState.SKIPPED]
    assert run.steps[0].error_code == "RuntimeError"
    assert "private database detail" not in str(run)
    assert WorkflowVerifier().verify(plan, run).status is VerificationStatus.FAILED


def test_optional_failure_allows_partial_execution_but_needs_review() -> None:
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


def test_missing_tool_or_grounding_evidence_needs_review() -> None:
    plan = sample_plan()
    run = WorkflowExecutor().execute(plan, lambda step, previous: StepOutcome("Plausible text"))
    result = WorkflowVerifier().verify(plan, run)
    assert result.status is VerificationStatus.NEEDS_REVIEW
    assert result.affected_steps == ("find",)


def test_valid_grounded_result_passes_verification() -> None:
    from uuid import uuid4
    plan = sample_plan()
    source = KnowledgeSearchResult("A grounded fact", "note.md", "/notes/note.md", 0, 0.1, uuid4())

    def runner(step, previous):
        if step.step_id == "find":
            return StepOutcome("A grounded fact", tools_used=("search_knowledge",),
                               grounding_status=GroundingStatus.SUFFICIENT, sources=(source,))
        return StepOutcome("A short summary")

    run = WorkflowExecutor().execute(plan, runner)
    assert WorkflowVerifier().verify(plan, run).status is VerificationStatus.SUCCESS


def test_structured_tool_failure_marks_step_failed_without_retry() -> None:
    plan = sample_plan()
    calls = 0

    def runner(step, previous):
        nonlocal calls
        calls += 1
        return StepOutcome("Tool failed", tools_used=("search_knowledge",),
                           tool_failures=("search_knowledge",))

    run = WorkflowExecutor().execute(plan, runner)
    assert calls == 1
    assert run.steps[0].tool_status == "failed"
    assert run.steps[0].state is StepState.FAILED
