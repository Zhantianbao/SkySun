"""Recover bounded formatting variations in real-model memory extraction."""

from uuid import uuid4

from langchain_core.exceptions import OutputParserException
from langchain_core.messages import AIMessage

from skysun.langchain_memory_curator import LangChainMemoryCurator
from skysun.memory_curator import CurationContext, MemoryPlan


def context() -> CurationContext:
    return CurationContext("I prefer concise answers.", (), (), uuid4())


class Responses:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0

    def with_structured_output(self, schema, method, include_raw):
        assert schema is MemoryPlan
        assert method == "json_mode"
        assert include_raw is True
        return self

    def invoke(self, messages):
        self.calls += 1
        item = next(self.responses)
        if isinstance(item, Exception):
            raise item
        return item


def envelope(parsed=None, raw="", error=None):
    return {"parsed": parsed, "raw": AIMessage(content=raw), "parsing_error": error}


def test_valid_structured_output_is_accepted_without_retry() -> None:
    model = Responses([envelope(parsed=MemoryPlan(decisions=[{
        "action": "upsert", "key": "answer_style", "content": "Prefers concise answers",
        "importance": "high",
    }]))])
    plan = LangChainMemoryCurator(model).decide(context())
    assert plan.decisions[0].key == "answer_style"
    assert model.calls == 1


def test_wrapped_json_is_recovered_from_raw_response() -> None:
    raw = 'Here is the plan:\n```json\n{"result":{"decisions":[{"action":"upsert","key":"answer_style","content":"Prefers concise answers","importance":"high"}]}}\n```'
    model = Responses([envelope(raw=raw, error=OutputParserException("schema wrapper"))])
    plan = LangChainMemoryCurator(model).decide(context())
    assert plan.decisions[0].key == "answer_style"
    assert model.calls == 1


def test_malformed_output_is_rejected_after_one_retry() -> None:
    model = Responses([
        envelope(raw='{"decisions":[{"action":"upsert","key":"api_key","content":"secret"}]}'),
        envelope(raw='not JSON'),
    ])
    assert LangChainMemoryCurator(model).decide(context()).decisions == []
    assert model.calls == 2


def test_parser_exception_recovers_on_second_response() -> None:
    model = Responses([
        OutputParserException("first parse failed"),
        envelope(raw='{"decisions":[{"action":"upsert","key":"answer_style","content":"Prefers concise answers"}]}'),
    ])
    plan = LangChainMemoryCurator(model).decide(context())
    assert plan.decisions[0].key == "answer_style"
    assert model.calls == 2
