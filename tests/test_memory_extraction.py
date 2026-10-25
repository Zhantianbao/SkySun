"""Recover bounded formatting variations in real-model memory extraction."""
# 本文件负责测试用例中的 test、memory、extraction 相关流程。

from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from skysun.memory.langchain_memory_curator import LangChainMemoryCurator
from skysun.memory.memory_curator import CurationContext, MemoryPlan


# 调用位置：tests/test_memory_extraction.py 的 test_valid_structured_output_is_accepted_without_retry、tests/test_memory_extraction.py 的 test_wrapped_json_is_recovered_from_raw_response、tests/test_memory_extraction.py 的 test_malformed_output_is_rejected_after_one_retry 等处调用。
def context() -> CurationContext:
    """为测试提供 context 所需的输入或模拟行为。"""
    return CurationContext("I prefer concise answers.", (), (), uuid4())


# 以下集中定义 Responses：数据结构或组件相关的数据和操作。
class Responses:
    # 调用位置：创建 Responses 实例时调用。
    def __init__(self, responses):
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.responses = iter(responses)
        self.calls = 0

    # 调用位置：src/skysun/evaluation/deepseek_evaluation.py 的 DeepSeekClaimReviewer.review、src/skysun/memory/langchain_memory_curator.py 的 LangChainMemoryCurator.__init__、src/skysun/rag/langchain_knowledge_grounding.py 的 LangChainEvidenceEvaluator.__init__ 等处调用。
    def with_structured_output(self, schema, method, include_raw):
        """为测试提供 with_structured_output 所需的输入或模拟行为。"""
        assert schema is MemoryPlan
        assert method == "json_mode"
        assert include_raw is True
        return self

    # 调用位置：src/skysun/core/langchain_agent.py 的 LangChainAgent.execute、src/skysun/evaluation/deepseek_evaluation.py 的 DeepSeekClaimReviewer.review、src/skysun/memory/langchain_memory_curator.py 的 LangChainMemoryCurator.decide 等处调用。
    def invoke(self, messages):
        """为测试提供 invoke 所需的输入或模拟行为。"""
        self.calls += 1
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item


# 调用位置：tests/test_memory_extraction.py 的 test_valid_structured_output_is_accepted_without_retry、tests/test_memory_extraction.py 的 test_wrapped_json_is_recovered_from_raw_response、tests/test_memory_extraction.py 的 test_malformed_output_is_rejected_after_one_retry 等处调用。
def envelope(parsed=None, raw="", error=None):
    """为测试提供 envelope 所需的输入或模拟行为。"""
    return {"parsed": parsed, "raw": AIMessage(content=raw), "parsing_error": error}


# 调用位置：pytest 执行该测试用例时调用。
def test_valid_structured_output_is_accepted_without_retry() -> None:
    """验证有效结构化输出可直接接受，无须重试。"""
    model = Responses([envelope(parsed=MemoryPlan(decisions=[{
        "action": "upsert", "key": "answer_style", "content": "Prefers concise answers",
        "importance": "high",
    }]))])
    plan = LangChainMemoryCurator(model).decide(context())
    assert plan.decisions[0].key == "answer_style"
    assert model.calls == 1


# 调用位置：pytest 执行该测试用例时调用。
def test_wrapped_json_is_recovered_from_raw_response() -> None:
    """验证从带包装的模型原始回复中提取 JSON。"""
    raw = 'Here is the plan:\n```json\n{"result":{"decisions":[{"action":"upsert","key":"answer_style","content":"Prefers concise answers","importance":"high"}]}}\n```'
    model = Responses([envelope(raw=raw, error=OutputParserException("schema wrapper"))])
    plan = LangChainMemoryCurator(model).decide(context())
    assert plan.decisions[0].key == "answer_style"
    assert model.calls == 1


# 调用位置：pytest 执行该测试用例时调用。
def test_malformed_output_is_rejected_after_one_retry() -> None:
    """验证模型输出格式错误时只重试一次并拒绝结果。"""
    model = Responses([
        envelope(raw='{"decisions":[{"action":"upsert","key":"api_key","content":"secret"}]}'),
        envelope(raw='not JSON'),
    ])
    assert LangChainMemoryCurator(model).decide(context()).decisions == []
    assert model.calls == 2


# 调用位置：pytest 执行该测试用例时调用。
def test_parser_exception_recovers_on_second_response() -> None:
    """验证解析器抛错后第二次模型回复可以恢复。"""
    model = Responses([
        OutputParserException("first parse failed"),
        envelope(raw='{"decisions":[{"action":"upsert","key":"answer_style","content":"Prefers concise answers"}]}'),
    ])
    plan = LangChainMemoryCurator(model).decide(context())
    assert plan.decisions[0].key == "answer_style"
    assert model.calls == 2
