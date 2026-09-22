"""Trace, metrics, evaluation, and feedback integration on the test database."""

import os
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from skysun.api import build_api
from skysun.conversation import ConversationService, InMemoryConversationStore
from skysun.knowledge import KnowledgeSearchResult
from skysun.knowledge_grounding import GroundingStatus
from skysun.langchain_agent import AgentExecution
from skysun.observability import (
    EvaluationStatus, FeedbackRating, ResponseEvaluator, span, trace_context,
)
from skysun.planner import TaskPlan
from skysun.postgres_observability import PostgresObservabilityStore
from skysun.postgres_task import PostgresTaskStore
from skysun.postgres_users import PostgresUserStore
from skysun.task_runtime import TaskStatus
from skysun.tool_reliability import run_tool
from skysun.user_identity import Actor, Role, actor_scope, system_scope
from skysun.verification import VerificationResult, VerificationStatus
from skysun.workflow_agent import WorkflowAgent


@pytest.fixture
def stores():
    url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("Observability tests require a *_test database")
    users, tasks, obs = PostgresUserStore(url), PostgresTaskStore(url), PostgresObservabilityStore(url)
    for store in (users, tasks, obs):
        store.initialize_schema()
    suffix = uuid4().hex[:10]
    admin, admin_token = users.create_user("obs_admin_" + suffix, Role.ADMIN)
    alice, alice_token = users.create_user("obs_alice_" + suffix)
    bob, bob_token = users.create_user("obs_bob_" + suffix)
    try:
        yield users, tasks, obs, ((admin, admin_token), (alice, alice_token), (bob, bob_token))
    finally:
        with psycopg.connect(url) as conn:
            ids = [admin.user_id, alice.user_id, bob.user_id]
            conn.execute("DELETE FROM agent_tasks WHERE user_id = ANY(%s)", (ids,))
            conn.execute("DELETE FROM skysun_users WHERE user_id = ANY(%s)", (ids,))


def _scope(user):
    return actor_scope(Actor(user.user_id, user.role))


def _completed_task(tasks, answer="A final answer"):
    task = tasks.create("A goal", None, ())
    tasks.transition(task.task_id, TaskStatus.PLANNING)
    tasks.transition(task.task_id, TaskStatus.RUNNING)
    tasks.complete(task.task_id, TaskStatus.COMPLETED, answer)
    return task


def test_trace_persistence_duration_and_user_isolation(stores):
    _, tasks, obs, (_, (alice, _), (bob, _)) = stores
    with _scope(alice):
        task = _completed_task(tasks)
        with trace_context(obs, task.task_id):
            with span("request_received"):
                pass
            assert run_tool("count_words", lambda: "3") == "3"
            assert "ok" in run_tool("recall_memories", lambda: "ok")
        saved = obs.list_traces(task.task_id)
        assert {trace.stage for trace in saved} >= {
            "request_received", "tool_call", "memory_retrieval"
        }
        assert all(trace.user_id == alice.user_id for trace in saved)
        assert all(trace.finished_at and trace.duration_ms >= 0 for trace in saved)
        with pytest.raises(ValueError, match="metadata"):
            obs.start_trace(task.task_id, "tool_call", {"arguments": "private text"})
    with _scope(bob):
        assert obs.list_traces() == ()
        with pytest.raises(KeyError):
            obs.list_traces(task.task_id)


def test_response_evaluation_scores_and_unsupported_claims():
    source = KnowledgeSearchResult("The marker is cobalt.", "note.md", "/note.md", 0, 0.1)

    class Reviewer:
        def __init__(self, claims=(), complete=True):
            self.claims, self.complete = claims, complete

        def review(self, goal, answer, sources):
            return self.complete, self.claims

    task_id = uuid4()
    success = VerificationResult(VerificationStatus.SUCCESS, "complete")
    passed = ResponseEvaluator(Reviewer()).evaluate(
        task_id, "What marker?", "Cobalt", verification=success,
        grounding_status=GroundingStatus.SUFFICIENT, sources=(source,),
        tools_used=("search_knowledge",),
    )
    assert passed.status is EvaluationStatus.PASS and passed.score == 100
    unsupported = ResponseEvaluator(Reviewer(("The marker is red",))).evaluate(
        task_id, "What marker?", "Cobalt and red", grounding_status=GroundingStatus.SUFFICIENT,
        sources=(source,), tools_used=("search_knowledge",),
    )
    assert unsupported.status is EvaluationStatus.FAIL
    assert unsupported.unsupported_claims == ("The marker is red",)
    assert ResponseEvaluator().evaluate(task_id, "goal", "answer").status is EvaluationStatus.REVIEW
    class BrokenReviewer:
        def review(self, goal, answer, sources):
            raise ValueError("invalid structured model output")

    recovered = ResponseEvaluator(BrokenReviewer()).evaluate(
        task_id, "goal", "answer", grounding_status=GroundingStatus.SUFFICIENT,
        sources=(source,), tools_used=("search_knowledge",),
    )
    assert recovered.status is EvaluationStatus.REVIEW
    assert recovered.reasons == ("claim_review_unavailable",)
    assert ResponseEvaluator().evaluate(task_id, "goal", "answer",
                                        tool_failures=("recall_memories",)).score == 0
    assert ResponseEvaluator().evaluate(task_id, "goal", "answer",
                                        tools_used=("search_knowledge",),
                                        grounding_status=GroundingStatus.INSUFFICIENT).status is EvaluationStatus.FAIL


class _DirectAgent:
    def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
        run_tool("count_words", lambda: "3")
        return AgentExecution("Three words.", ("count_words",), (), None, ())

    def reply(self, message, history=(), conversation_id=None):
        return self.execute(message, history, conversation_id).answer


class _Planner:
    def plan(self, goal, capabilities):
        return TaskPlan.model_validate({"goal": goal, "steps": [
            {"step_id": "count", "description": "Count words", "capabilities": ["count_words"]},
            {"step_id": "explain", "description": "Explain", "capabilities": ["reason"],
             "depends_on": ["count"]},
        ]})


def test_direct_and_workflow_traces_evaluations_and_metrics(stores):
    _, tasks, obs, ((admin, _), (alice, _), _) = stores
    workflow = WorkflowAgent(_DirectAgent(), _Planner(), ("count_words", "reason"), tasks,
                             observability=obs)
    with _scope(alice):
        assert workflow.reply("Count words") == "Three words."
        result = workflow.execute_workflow("First count words, then explain")
        assert result.run.task_id
        with psycopg.connect(tasks._database_url) as conn:
            ids = [row[0] for row in conn.execute(
                "SELECT task_id FROM agent_tasks WHERE user_id = %s", (alice.user_id,)
            )]
        assert len(ids) == 2
        stages = {item.stage for item in obs.list_traces(limit=100)}
        assert {"request_received", "request", "direct_execution", "planning",
                "workflow_execution", "workflow_step", "tool_call", "verification",
                "response_evaluation", "final_response"} <= stages
        assert len(obs.list_evaluations()) == 2
        assert all(item.user_id == alice.user_id for item in obs.list_evaluations())
    with _scope(admin):
        metrics = obs.metrics()
    assert metrics["task_count"] >= 2
    assert metrics["average_latency_ms"] is not None
    assert metrics["average_tool_duration_ms"] is not None
    assert metrics["average_workflow_duration_ms"] is not None
    assert metrics["tool_failure_rate"] is not None
    assert any(item["user_id"] == str(alice.user_id) for item in metrics["user_activity"])


def test_feedback_associates_owned_final_response(stores):
    _, tasks, obs, (_, (alice, _), (bob, _)) = stores
    with _scope(alice):
        task = _completed_task(tasks, "Final response")
        first = obs.save_feedback(task.task_id, FeedbackRating.POSITIVE, "Useful")
        second = obs.save_feedback(task.task_id, FeedbackRating.NEGATIVE, "Missing detail")
        assert second.feedback_id == first.feedback_id
        assert second.response_hash == first.response_hash
        assert second.rating is FeedbackRating.NEGATIVE
    with _scope(bob):
        with pytest.raises(KeyError):
            obs.save_feedback(task.task_id, FeedbackRating.POSITIVE)


def test_reliability_metrics_count_tool_and_verification_failures(stores):
    _, tasks, obs, ((admin, _), (alice, _), _) = stores
    with _scope(admin):
        before = obs.metrics()
    with _scope(alice):
        task = _completed_task(tasks)
        with trace_context(obs, task.task_id):
            result = run_tool("count_words", lambda: (_ for _ in ()).throw(RuntimeError("failed")))
            assert "execution_failed" in result
            with span("verification") as verification_span:
                verification_span.status = "FAILED"
    with _scope(admin):
        after = obs.metrics()
    assert after["verification_failures"] == before["verification_failures"] + 1
    assert after["tool_failure_rate"] is not None and after["tool_failure_rate"] > 0


def test_admin_observability_api_and_user_feedback(stores):
    users, tasks, obs, ((admin, admin_token), (alice, alice_token), (bob, bob_token)) = stores

    class Agent:
        def reply(self, message, history=(), conversation_id=None):
            return "ok"

    client = TestClient(build_api(ConversationService(Agent(), InMemoryConversationStore()),
                                  users=users, tasks=tasks, observability=obs))
    ah = {"Authorization": f"Bearer {alice_token}"}
    bh = {"Authorization": f"Bearer {bob_token}"}
    admin_h = {"Authorization": f"Bearer {admin_token}"}
    with _scope(alice):
        task = _completed_task(tasks)
        with trace_context(obs, task.task_id):
            with span("verification"):
                pass
        evaluation = ResponseEvaluator().evaluate(task.task_id, "goal", "answer")
        obs.save_evaluation(evaluation)
    for path in ("/admin/metrics", "/admin/traces", "/admin/evaluations"):
        assert client.get(path, headers=ah).status_code == 403
        assert client.get(path, headers=admin_h).status_code == 200
    traces = client.get("/admin/traces", params={"task_id": str(task.task_id)},
                        headers=admin_h).json()
    assert traces[0]["user_id"] == str(alice.user_id)
    assert client.post(f"/tasks/{task.task_id}/feedback",
                       json={"rating": "POSITIVE", "comment": "good"}, headers=bh).status_code == 404
    response = client.post(f"/tasks/{task.task_id}/feedback",
                           json={"rating": "POSITIVE", "comment": "good"}, headers=ah)
    assert response.status_code == 200 and response.json()["rating"] == "POSITIVE"
