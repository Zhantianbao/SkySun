"""The workflow path is optional and preserves direct Agent behavior."""

from skysun.core.langchain_agent import AgentExecution
from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.workflow.planner import TaskPlan
from skysun.workflow.verification import VerificationStatus
from skysun.workflow.workflow_agent import WorkflowAgent


class FakeDirectAgent:
    def __init__(self):
        self.direct_calls = 0
        self.step_calls = []

    def reply(self, message, history=(), conversation_id=None):
        self.direct_calls += 1
        return "Direct answer"

    def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
        self.step_calls.append(message)
        if len(self.step_calls) == 1:
            return AgentExecution("3", ("count_words",), (), None, ())
        assert "count: 3" in message
        return AgentExecution("The text has three words.\n\nSources:\n- forged", (), (), None, ())


class FakePlanner:
    def __init__(self):
        self.calls = 0

    def plan(self, goal, capabilities):
        self.calls += 1
        return TaskPlan.model_validate({"goal": goal, "steps": [
            {"step_id": "count", "description": "Count the words", "capabilities": ["count_words"]},
            {"step_id": "explain", "description": "Explain the count", "capabilities": ["reason"],
             "depends_on": ["count"]},
        ]})


def test_direct_and_planned_agent_paths() -> None:
    direct, planner = FakeDirectAgent(), FakePlanner()
    agent = WorkflowAgent(direct, planner, ("count_words", "reason"))
    assert agent.reply("What time is it?") == "Direct answer"
    assert planner.calls == 0
    assert agent.reply("Count these words and then explain the count") == "The text has three words."
    assert planner.calls == 1
    assert direct.direct_calls == 1
    assert len(direct.step_calls) == 2


def test_planner_failure_returns_safe_response() -> None:
    class BrokenPlanner:
        def plan(self, goal, capabilities):
            raise RuntimeError("internal detail")

    direct = FakeDirectAgent()
    answer = WorkflowAgent(direct, BrokenPlanner(), ("reason",)).reply(
        "Analyze this note and summarize it"
    )
    assert "couldn't create a safe plan" in answer
    assert direct.step_calls == []


def test_workflow_appends_only_validated_source_metadata() -> None:
    from uuid import uuid4

    source = KnowledgeSearchResult("Cobalt shield", "note.md", "/notes/note.md", 2, 0.1, uuid4())

    class SourcePlanner:
        def plan(self, goal, capabilities):
            return TaskPlan.model_validate({"goal": goal, "steps": [
                {"step_id": "find", "description": "Find shield color", "capabilities": ["search_knowledge"]},
                {"step_id": "summarize", "description": "Summarize it", "capabilities": ["reason"],
                 "depends_on": ["find"]},
            ]})

    class SourceAgent:
        def __init__(self):
            self.calls = 0

        def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
            self.calls += 1
            if self.calls == 1:
                return AgentExecution("Cobalt", ("search_knowledge",), (),
                                      GroundingStatus.SUFFICIENT, (source,))
            return AgentExecution("The shield is cobalt.\n\nSources:\n- invented", (), (), None, ())

    result = WorkflowAgent(SourceAgent(), SourcePlanner(), ("search_knowledge", "reason")).execute_workflow(
        "Find the shield color and summarize it"
    )
    assert result.verification.status is VerificationStatus.SUCCESS
    assert "- note.md (/notes/note.md), chunk 2" in result.answer
    assert "invented" not in result.answer
