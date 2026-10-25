"""Trace, metrics, evaluation, and feedback integration on the test database."""
# 本文件负责测试用例中的 test、observability 相关流程。

import os
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from skysun.core.api import build_api
from skysun.memory.conversation import ConversationService, InMemoryConversationStore
from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.core.langchain_agent import AgentExecution
from skysun.evaluation.observability import (
    EvaluationStatus, FeedbackRating, ResponseEvaluator, span, trace_context,
)
from skysun.workflow.planner import TaskPlan
from skysun.evaluation.postgres_observability import PostgresObservabilityStore
from skysun.workflow.postgres_task import PostgresTaskStore
from skysun.user.postgres_users import PostgresUserStore
from skysun.workflow.task_runtime import TaskStatus
from skysun.tool.tool_reliability import run_tool
from skysun.user.user_identity import Actor, Role, actor_scope, system_scope
from skysun.workflow.verification import VerificationResult, VerificationStatus
from skysun.workflow.workflow_agent import WorkflowAgent


# 调用位置：pytest 在依赖此 fixture 的测试中调用。
@pytest.fixture
def stores(create_account):
    """为测试提供 stores 所需的输入或模拟行为。"""
    url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("Observability tests require a *_test database")
    users, tasks, obs = PostgresUserStore(url), PostgresTaskStore(url), PostgresObservabilityStore(url)
    for store in (users, tasks, obs):
        store.initialize_schema()
    admin, admin_token = create_account(users, Role.ADMIN)
    alice, alice_token = create_account(users)
    bob, bob_token = create_account(users)
    try:
        yield users, tasks, obs, ((admin, admin_token), (alice, alice_token), (bob, bob_token))
    finally:
        with psycopg.connect(url) as conn:
            ids = [admin.user_id, alice.user_id, bob.user_id]
            conn.execute("DELETE FROM agent_tasks WHERE user_id = ANY(%s)", (ids,))
            conn.execute("DELETE FROM skysun_users WHERE user_id = ANY(%s)", (ids,))


# 调用位置：tests/test_multi_user.py 的 test_identity_authentication_and_conversation_isolation、tests/test_multi_user.py 的 test_legacy_rows_keep_their_dedicated_owner、tests/test_multi_user.py 的 test_personal_memory_same_key_isolated_and_global_memory_separate 等处调用。
def _scope(user):
    """为测试准备 _scope 所需的替身或输入。"""
    return actor_scope(Actor(user.user_id, user.role))


# 调用位置：tests/test_observability.py 的 test_trace_persistence_duration_and_user_isolation、tests/test_observability.py 的 test_feedback_associates_owned_final_response、tests/test_observability.py 的 test_reliability_metrics_count_tool_and_verification_failures 等处调用。
def _completed_task(tasks, answer="A final answer"):
    """为测试准备 _completed_task 所需的替身或输入。"""
    task = tasks.create("A goal", None, ())
    tasks.transition(task.task_id, TaskStatus.PLANNING)
    tasks.transition(task.task_id, TaskStatus.RUNNING)
    tasks.complete(task.task_id, TaskStatus.COMPLETED, answer)
    return task


# 调用位置：pytest 执行该测试用例时调用。
def test_trace_persistence_duration_and_user_isolation(stores):
    """验证执行轨迹持久化、时长统计和用户隔离。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_response_evaluation_scores_and_unsupported_claims():
    """验证回复评估给出评分并识别缺少证据的断言。"""
    source = KnowledgeSearchResult("The marker is cobalt.", "note.md", "/note.md", 0, 0.1)

    # 以下集中定义 Reviewer：数据结构或组件相关的数据和操作。
    class Reviewer:
        # 调用位置：创建 Reviewer 实例时调用。
        def __init__(self, claims=(), complete=True):
            """保存依赖和初始状态，供该对象后续方法使用。"""
            self.claims, self.complete = claims, complete

        # 调用位置：Reviewer 的调用方通过该方法使用。
        def review(self, goal, answer, sources):
            """检查回答质量并给出评估结果。"""
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
    # 以下集中定义 BrokenReviewer：数据结构或组件相关的数据和操作。
    class BrokenReviewer:
        # 调用位置：BrokenReviewer 的调用方通过该方法使用。
        def review(self, goal, answer, sources):
            """检查回答质量并给出评估结果。"""
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


# 以下集中定义 _DirectAgent：Agent 执行层相关的数据和操作。
class _DirectAgent:
    # 调用位置：tests/test_observability.py 的 stores、tests/test_observability.py 的 _DirectAgent.reply、tests/test_observability.py 的 test_direct_and_workflow_traces_evaluations_and_metrics 等处调用。
    def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
        """执行 Agent 或工作流并返回结构化结果。"""
        run_tool("count_words", lambda: "3")
        return AgentExecution("Three words.", ("count_words",), (), None, ())

    # 调用位置：tests/test_observability.py 的 test_direct_and_workflow_traces_evaluations_and_metrics 等处调用。
    def reply(self, message, history=(), conversation_id=None):
        """生成并返回本次对话的最终回复。"""
        return self.execute(message, history, conversation_id).answer


# 以下集中定义 _Planner：计划生成层相关的数据和操作。
class _Planner:
    # 调用位置：_Planner 的调用方通过该方法使用。
    def plan(self, goal, capabilities):
        """根据用户目标生成可执行的步骤计划。"""
        return TaskPlan.model_validate({"goal": goal, "steps": [
            {"step_id": "count", "description": "Count words", "capabilities": ["count_words"]},
            {"step_id": "explain", "description": "Explain", "capabilities": ["reason"],
             "depends_on": ["count"]},
        ]})


# 调用位置：pytest 执行该测试用例时调用。
def test_direct_and_workflow_traces_evaluations_and_metrics(stores):
    """验证直接执行和工作流都产生轨迹、评估与指标。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_feedback_associates_owned_final_response(stores):
    """验证反馈关联本用户的最终回复。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_reliability_metrics_count_tool_and_verification_failures(stores):
    """验证可靠性指标统计工具失败和结果验证失败。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_admin_observability_api_and_user_feedback(stores):
    """验证管理员可观测性接口与用户反馈。"""
    users, tasks, obs, ((admin, admin_token), (alice, alice_token), (bob, bob_token)) = stores

    # 以下集中定义 Agent：Agent 执行层相关的数据和操作。
    class Agent:
        # 调用位置：tests/test_observability.py 的 test_direct_and_workflow_traces_evaluations_and_metrics 等处调用。
        def reply(self, message, history=(), conversation_id=None):
            """生成并返回本次对话的最终回复。"""
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
