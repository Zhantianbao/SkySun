"""Persistent runtime integration tests use only the dedicated test database."""

import os
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from skysun.memory.conversation import InMemoryConversationStore
from skysun.core.langchain_agent import AgentExecution
from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.workflow.planner import TaskPlan
from skysun.workflow.postgres_task import PostgresTaskStore
from skysun.workflow.task_runtime import TaskStatus
from skysun.workflow.workflow import StepRecord, StepState
from skysun.workflow.workflow_agent import WorkflowAgent


@pytest.fixture
def task_store():
    url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("Task tests require a *_test database")
    store = PostgresTaskStore(url)
    store.initialize_schema()
    with psycopg.connect(url) as conn:
        existing = {row[0] for row in conn.execute("SELECT task_id FROM agent_tasks")}
    try:
        yield store
    finally:
        with psycopg.connect(url) as conn:
            for task_id in ({row[0] for row in conn.execute("SELECT task_id FROM agent_tasks")}
                            - existing):
                conn.execute("DELETE FROM agent_tasks WHERE task_id = %s", (task_id,))


class Planner:
    def plan(self, goal, capabilities):
        return TaskPlan.model_validate({"goal": goal, "steps": [
            {"step_id": "count", "description": "Count three words", "capabilities": ["count_words"]},
            {"step_id": "explain", "description": "Explain the count", "capabilities": ["reason"],
             "depends_on": ["count"]},
        ]})


class Agent:
    def __init__(self):
        self.calls = []

    def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
        self.calls.append(message)
        if "Current workflow step count:" in message:
            return AgentExecution("3", ("count_words",), (), None, ())
        return AgentExecution("There are three words.", (), (), None, ())


def test_task_creation_transition_and_persistence(task_store):
    task = task_store.create("Count then explain", uuid4(), ())
    assert task.status is TaskStatus.CREATED
    assert task_store.get(task.task_id).conversation_id == task.conversation_id
    task_store.transition(task.task_id, TaskStatus.PLANNING)
    assert task_store.get(task.task_id).status is TaskStatus.PLANNING
    with pytest.raises(ValueError, match="Invalid task transition"):
        task_store.transition(task.task_id, TaskStatus.COMPLETED)
    assert [event.event_type for event in task_store.events(task.task_id)] == ["task_created"]


def test_workflow_steps_and_events_persist(task_store):
    agent = WorkflowAgent(Agent(), Planner(), ("count_words", "reason"), task_store)
    result = agent.execute_workflow("Count three words and then explain")
    saved = task_store.get(result.run.task_id)
    assert saved.status is TaskStatus.COMPLETED
    assert saved.plan == result.plan
    assert [record.state for record in saved.steps] == [StepState.SUCCESS, StepState.SUCCESS]
    assert saved.answer == result.answer
    events = [event.event_type for event in task_store.events(saved.task_id)]
    assert events == ["task_created", "plan_generated", "step_started", "tool_called",
                      "step_completed", "step_started", "step_completed", "task_completed"]


def test_interrupted_workflow_resumes_without_repeating_success(task_store):
    original = task_store.start_step

    def interrupt(task_id, step_id):
        if step_id == "explain":
            raise RuntimeError("simulated process interruption")
        original(task_id, step_id)

    task_store.start_step = interrupt
    first_agent = Agent()
    workflow = WorkflowAgent(first_agent, Planner(), ("count_words", "reason"), task_store)
    with pytest.raises(RuntimeError, match="simulated process interruption"):
        workflow.execute_workflow("Count three words and then explain")
    task_store.start_step = original
    unfinished = task_store.list_unfinished()
    task = unfinished[-1]
    assert task.status is TaskStatus.RUNNING
    assert [record.state for record in task.steps] == [StepState.SUCCESS, StepState.PENDING]

    second_agent = Agent()
    restarted = WorkflowAgent(second_agent, Planner(), ("count_words", "reason"),
                              PostgresTaskStore(task_store._database_url))
    restarted.recover_unfinished()
    saved = task_store.get(task.task_id)
    assert saved.status is TaskStatus.COMPLETED
    assert len(first_agent.calls) == 1
    assert len(second_agent.calls) == 1
    assert "Current workflow step explain:" in second_agent.calls[0]


def test_approval_pauses_and_can_resume(task_store):
    workflow = WorkflowAgent(Agent(), Planner(), ("count_words", "reason"), task_store,
                             approval_required=lambda step: step.step_id == "explain")
    assert "waiting for approval" in workflow.reply("Count three words and then explain")
    task = task_store.list_unfinished()[-1]
    assert task.status is TaskStatus.WAITING_APPROVAL
    assert workflow.resume_task(task.task_id) is None
    result = workflow.approve_task(task.task_id)
    assert result.answer == "There are three words."
    assert task_store.get(task.task_id).status is TaskStatus.COMPLETED
    assert any(event.event_type == "approval_granted" for event in task_store.events(task.task_id))


def test_step_metadata_and_tool_failure_events_survive_restart(task_store):
    task = task_store.create("Find evidence", None, ())
    task_store.transition(task.task_id, TaskStatus.PLANNING)
    plan = TaskPlan.model_validate({"goal": task.goal, "steps": [
        {"step_id": "find", "description": "Find evidence", "capabilities": ["search_knowledge"]},
    ]})
    task_store.save_plan(task.task_id, plan)
    task_store.transition(task.task_id, TaskStatus.RUNNING)
    task_store.start_step(task.task_id, "find")
    source = KnowledgeSearchResult("evidence", "note.md", "/notes/note.md", 0, 0.1, uuid4())
    task_store.finish_step(StepRecord(task.task_id, "find", StepState.SUCCESS, "evidence",
                                      ("search_knowledge",), "success",
                                      GroundingStatus.SUFFICIENT, (source,)))
    restored = PostgresTaskStore(task_store._database_url).get(task.task_id)
    assert restored.steps[0].sources == (source,)
    assert restored.steps[0].grounding_status is GroundingStatus.SUFFICIENT
    failed = task_store.create("Test failed tool", None, ())
    task_store.transition(failed.task_id, TaskStatus.PLANNING)
    failure_plan = TaskPlan.model_validate({"goal": failed.goal, "steps": [
        {"step_id": "check", "description": "Check a tool", "capabilities": ["count_words"]},
    ]})
    task_store.save_plan(failed.task_id, failure_plan)
    task_store.transition(failed.task_id, TaskStatus.RUNNING)
    task_store.start_step(failed.task_id, "check")
    task_store.finish_step(StepRecord(failed.task_id, "check", StepState.FAILED,
                                      "Tool failed", ("count_words",), "failed",
                                      error_code="tool_failure", tool_failures=("count_words",)))
    assert [event.event_type for event in task_store.events(failed.task_id)][-3:] == [
        "tool_called", "tool_failed", "step_completed"
    ]


@pytest.mark.parametrize("capability", ["remember_memory", "count_words"])
def test_interrupted_inflight_step_fails_without_replay(task_store, capability):
    task = task_store.create("Remember a preference", None, ())
    task_store.transition(task.task_id, TaskStatus.PLANNING)
    plan = TaskPlan.model_validate({"goal": task.goal, "steps": [
        {"step_id": "prepare", "description": "Prepare", "capabilities": ["reason"]},
        {"step_id": "remember", "description": "Save preference", "capabilities": [capability],
         "depends_on": ["prepare"]},
    ]})
    task_store.save_plan(task.task_id, plan)
    task_store.transition(task.task_id, TaskStatus.RUNNING)
    task_store.start_step(task.task_id, "prepare")
    task_store.finish_step(StepRecord(task.task_id, "prepare", StepState.SUCCESS, "Prepared"))
    task_store.start_step(task.task_id, "remember")
    direct = Agent()
    workflow = WorkflowAgent(direct, Planner(), (capability,), task_store)
    assert workflow.resume_task(task.task_id) is None
    saved = task_store.get(task.task_id)
    assert saved.status is TaskStatus.FAILED
    assert [record.state for record in saved.steps] == [StepState.SUCCESS, StepState.FAILED]
    assert saved.steps[0].output == "Prepared"
    assert saved.steps[1].error_code == "interrupted"
    assert direct.calls == []


def test_application_startup_recovers_interrupted_planning(task_store, monkeypatch):
    from skysun.core import main
    from skysun.core.config import Settings

    task = task_store.create("Interrupted before planning", None, ())
    task_store.transition(task.task_id, TaskStatus.PLANNING)
    monkeypatch.setattr(main, "load_settings", lambda: Settings(
        deepseek_api_key="test-key", database_url=task_store._database_url, _env_file=None
    ))
    monkeypatch.setattr(main, "create_deepseek", lambda settings: object())

    class NoOpCurator:
        def decide(self, context):
            raise AssertionError("Curator should not run during startup")

    with TestClient(main.create_app(
        store=InMemoryConversationStore(), memory_store=object(),
        knowledge_store=object(), embeddings=object(), curator=NoOpCurator(),
    )):
        assert task_store.get(task.task_id).status is TaskStatus.FAILED
