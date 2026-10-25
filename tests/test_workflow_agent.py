"""The workflow path is optional and preserves direct Agent behavior."""
# 本文件负责测试用例中的 test、workflow、agent 相关流程。

from skysun.core.langchain_agent import AgentExecution
from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.rag.knowledge_grounding import GroundingStatus
from skysun.workflow.planner import TaskPlan
from skysun.workflow.verification import VerificationStatus
from skysun.workflow.workflow_agent import WorkflowAgent


# 以下集中定义 FakeDirectAgent：Agent 执行层相关的数据和操作。
class FakeDirectAgent:
    # 调用位置：创建 FakeDirectAgent 实例时调用。
    def __init__(self):
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.direct_calls = 0
        self.step_calls = []

    # 调用位置：tests/test_workflow_agent.py 的 test_direct_and_planned_agent_paths、tests/test_workflow_agent.py 的 test_planner_failure_returns_safe_response 等处调用。
    def reply(self, message, history=(), conversation_id=None):
        """生成并返回本次对话的最终回复。"""
        self.direct_calls += 1
        return "Direct answer"

    # 调用位置：FakeDirectAgent 的调用方通过该方法使用。
    def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
        """执行 Agent 或工作流并返回结构化结果。"""
        self.step_calls.append(message)
        if len(self.step_calls) == 1:
            return AgentExecution("3", ("count_words",), (), None, ())
        assert "count: 3" in message
        return AgentExecution("The text has three words.\n\nSources:\n- forged", (), (), None, ())


# 以下集中定义 FakePlanner：计划生成层相关的数据和操作。
class FakePlanner:
    # 调用位置：创建 FakePlanner 实例时调用。
    def __init__(self):
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.calls = 0

    # 调用位置：FakePlanner 的调用方通过该方法使用。
    def plan(self, goal, capabilities):
        """根据用户目标生成可执行的步骤计划。"""
        self.calls += 1
        return TaskPlan.model_validate({"goal": goal, "steps": [
            {"step_id": "count", "description": "Count the words", "capabilities": ["count_words"]},
            {"step_id": "explain", "description": "Explain the count", "capabilities": ["reason"],
             "depends_on": ["count"]},
        ]})


# 调用位置：pytest 执行该测试用例时调用。
def test_direct_and_planned_agent_paths() -> None:
    """验证直接执行和按计划执行分别进入对应的 Agent 路径。"""
    direct, planner = FakeDirectAgent(), FakePlanner()
    agent = WorkflowAgent(direct, planner, ("count_words", "reason"))
    assert agent.reply("What time is it?") == "Direct answer"
    assert planner.calls == 0
    assert agent.reply("Count these words and then explain the count") == "The text has three words."
    assert planner.calls == 1
    assert direct.direct_calls == 1
    assert len(direct.step_calls) == 2


# 调用位置：pytest 执行该测试用例时调用。
def test_planner_failure_returns_safe_response() -> None:
    """验证规划器失败返回安全回复。"""
    # 以下集中定义 BrokenPlanner：计划生成层相关的数据和操作。
    class BrokenPlanner:
        # 调用位置：BrokenPlanner 的调用方通过该方法使用。
        def plan(self, goal, capabilities):
            """根据用户目标生成可执行的步骤计划。"""
            raise RuntimeError("internal detail")

    direct = FakeDirectAgent()
    answer = WorkflowAgent(direct, BrokenPlanner(), ("reason",)).reply(
        "Analyze this note and summarize it"
    )
    assert "couldn't create a safe plan" in answer
    assert direct.step_calls == []


# 调用位置：pytest 执行该测试用例时调用。
def test_workflow_appends_only_validated_source_metadata() -> None:
    """验证工作流只附加经过校验的来源信息。"""
    from uuid import uuid4

    source = KnowledgeSearchResult("Cobalt shield", "note.md", "/notes/note.md", 2, 0.1, uuid4())

    # 以下集中定义 SourcePlanner：计划生成层相关的数据和操作。
    class SourcePlanner:
        # 调用位置：SourcePlanner 的调用方通过该方法使用。
        def plan(self, goal, capabilities):
            """根据用户目标生成可执行的步骤计划。"""
            return TaskPlan.model_validate({"goal": goal, "steps": [
                {"step_id": "find", "description": "Find shield color", "capabilities": ["search_knowledge"]},
                {"step_id": "summarize", "description": "Summarize it", "capabilities": ["reason"],
                 "depends_on": ["find"]},
            ]})

    # 以下集中定义 SourceAgent：Agent 执行层相关的数据和操作。
    class SourceAgent:
        # 调用位置：创建 SourceAgent 实例时调用。
        def __init__(self):
            """保存依赖和初始状态，供该对象后续方法使用。"""
            self.calls = 0

        # 调用位置：SourceAgent 的调用方通过该方法使用。
        def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
            """执行 Agent 或工作流并返回结构化结果。"""
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
