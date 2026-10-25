"""Grounding decisions and retrieval budgets use fake model boundaries."""
# 本文件负责测试用例中的 test、knowledge、grounding 相关流程。

from uuid import uuid4

import pytest

from skysun.rag.knowledge import KnowledgeSearchResult
from skysun.rag.knowledge_grounding import (
    GroundedKnowledgeService, GroundingResult, GroundingStatus,
)


CHUNK = KnowledgeSearchResult("The craft uses a violet plasma sail.", "sail.md", "/notes/sail.md",
                              2, 0.12, uuid4())
RELATED = KnowledgeSearchResult("The craft uses PostgreSQL.", "craft.md", "/notes/craft.md",
                                0, 0.15, uuid4())


# 以下集中定义 Embeddings：数据结构或组件相关的数据和操作。
class Embeddings:
    # 调用位置：创建 Embeddings 实例时调用。
    def __init__(self):
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.queries = []

    # 调用位置：src/skysun/rag/knowledge_grounding.py 的 GroundedKnowledgeService.search、tests/test_knowledge.py 的 test_qwen_adapter_uses_compatible_api_and_validates_vectors 等处调用。
    def embed_query(self, text):
        """把检索问题转换为向量。"""
        self.queries.append(text)
        return (float(len(self.queries)),)


# 以下集中定义 Store：存储层相关的数据和操作。
class Store:
    # 调用位置：创建 Store 实例时调用。
    def __init__(self, results):
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.results = results
        self.calls = []

    # 调用位置：tests/test_knowledge_grounding.py 的 run、tests/test_knowledge_grounding.py 的 test_no_candidates_and_failed_rewrite_do_not_loop 等处调用。
    def search(self, vector, limit=4, max_cosine_distance=0.35):
        """按问题检索相关内容。"""
        self.calls.append(vector)
        return self.results[len(self.calls) - 1]


# 以下集中定义 Evaluator：回答评估层相关的数据和操作。
class Evaluator:
    # 调用位置：创建 Evaluator 实例时调用。
    def __init__(self, decisions):
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.decisions = decisions
        self.calls = []

    # 调用位置：Evaluator 的调用方通过该方法使用。
    def evaluate(self, question, candidates):
        """根据执行证据判断回答的质量或可信度。"""
        self.calls.append((question, candidates))
        decision = self.decisions[len(self.calls) - 1]
        if isinstance(decision, Exception):
            raise decision
        if decision == "support":
            return GroundingResult(status=GroundingStatus.SUFFICIENT,
                                   supporting_chunk_ids=(candidates[0][0],), reason="Explicit fact")
        return decision


# 以下集中定义 Rewriter：数据结构或组件相关的数据和操作。
class Rewriter:
    # 调用位置：创建 Rewriter 实例时调用。
    def __init__(self, query="violet plasma sail inventor"):
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.query = query
        self.calls = []

    # 调用位置：src/skysun/rag/knowledge_grounding.py 的 GroundedKnowledgeService.search 调用。
    def rewrite(self, question):
        """根据证据不足的原因改写检索问题。"""
        self.calls.append(question)
        return self.query


# 调用位置：tests/test_knowledge_grounding.py 的 test_explicit_support_preserves_only_selected_source、tests/test_knowledge_grounding.py 的 test_related_or_partial_evidence_rewrites_once_and_stops、tests/test_knowledge_grounding.py 的 test_second_attempt_can_succeed 等处调用。
def run(results, decisions, question="Who invented the violet plasma sail?"):
    """维持设备连接、发送心跳并处理动作。"""
    store, embeddings = Store(results), Embeddings()
    evaluator, rewriter = Evaluator(decisions), Rewriter()
    outcome = GroundedKnowledgeService(store, embeddings, evaluator, rewriter).search(question)
    return outcome, store, embeddings, evaluator, rewriter


# 调用位置：pytest 执行该测试用例时调用。
def test_explicit_support_preserves_only_selected_source() -> None:
    """验证明确的证据支持只保留已选中的来源。"""
    outcome, store, embeddings, evaluator, rewriter = run(
        [(CHUNK, RELATED)], ["support"], "What color is the plasma sail?"
    )
    assert outcome.status is GroundingStatus.SUFFICIENT
    assert outcome.supporting_chunks == (CHUNK,)
    assert outcome.retrieval_attempts == 1
    assert outcome.rewrite_attempts == 0
    assert embeddings.queries == ["What color is the plasma sail?"]
    assert not rewriter.calls


# 调用位置：pytest 执行该测试用例时调用。
@pytest.mark.parametrize("first_status", [GroundingStatus.INSUFFICIENT, GroundingStatus.AMBIGUOUS])
def test_related_or_partial_evidence_rewrites_once_and_stops(first_status) -> None:
    """验证证据仅相关或不完整时改写查询一次后停止。"""
    first = GroundingResult(status=first_status, supporting_chunk_ids=(), reason="No inventor")
    second = GroundingResult(status=GroundingStatus.INSUFFICIENT,
                             supporting_chunk_ids=(), reason="Still no inventor")
    outcome, store, embeddings, evaluator, rewriter = run(
        [(RELATED,), (RELATED,)], [first, second]
    )
    assert outcome.status is GroundingStatus.INSUFFICIENT
    assert outcome.supporting_chunks == ()
    assert outcome.retrieval_attempts == 2
    assert outcome.rewrite_attempts == 1
    assert len(store.calls) == 2
    assert len(rewriter.calls) == 1
    assert embeddings.queries == ["Who invented the violet plasma sail?", rewriter.query]


# 调用位置：pytest 执行该测试用例时调用。
def test_second_attempt_can_succeed() -> None:
    """验证第二次检索可以补足证据并成功。"""
    insufficient = GroundingResult(status=GroundingStatus.INSUFFICIENT,
                                   supporting_chunk_ids=(), reason="Need another passage")
    outcome, store, embeddings, evaluator, rewriter = run(
        [(RELATED,), (CHUNK,)], [insufficient, "support"]
    )
    assert outcome.status is GroundingStatus.SUFFICIENT
    assert outcome.supporting_chunks == (CHUNK,)
    assert outcome.retrieval_attempts == 2
    assert len(store.calls) == 2


# 调用位置：pytest 执行该测试用例时调用。
@pytest.mark.parametrize("decision", [
    {"status": "SUFFICIENT", "supporting_chunk_ids": ["invented-id"], "reason": "Invalid"},
    {"status": "SUFFICIENT", "supporting_chunk_ids": [], "reason": "No ID"},
    {"status": "UNKNOWN", "supporting_chunk_ids": [], "reason": "Bad status"},
    {"status": "SUFFICIENT", "reason": "Missing field"},
    {"status": "SUFFICIENT", "supporting_chunk_ids": [str(CHUNK.chunk_id)] * 2,
     "reason": "Duplicate"},
    RuntimeError("model unavailable"),
])
def test_invalid_or_failed_evaluation_never_authorizes_answer(decision) -> None:
    """验证无效或失败的证据评估不能放行回答。"""
    outcome, store, embeddings, evaluator, rewriter = run([(CHUNK,)], [decision])
    assert outcome.status is GroundingStatus.INSUFFICIENT
    assert outcome.supporting_chunks == ()
    assert outcome.retrieval_attempts == 1
    assert len(store.calls) == 1
    assert not rewriter.calls


# 调用位置：pytest 执行该测试用例时调用。
def test_no_candidates_and_failed_rewrite_do_not_loop() -> None:
    """验证没有候选片段或改写失败时不会无限循环。"""
    store, embeddings, evaluator, rewriter = Store([()]), Embeddings(), Evaluator([]), Rewriter("")
    outcome = GroundedKnowledgeService(store, embeddings, evaluator, rewriter).search("Unknown fact?")
    assert outcome.status is GroundingStatus.INSUFFICIENT
    assert outcome.retrieval_attempts == 1
    assert outcome.rewrite_attempts == 1
    assert len(store.calls) == 1
