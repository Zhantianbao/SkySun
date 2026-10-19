"""Grounding decisions and retrieval budgets use fake model boundaries."""

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


class Embeddings:
    def __init__(self):
        self.queries = []

    def embed_query(self, text):
        self.queries.append(text)
        return (float(len(self.queries)),)


class Store:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, vector, limit=4, max_cosine_distance=0.35):
        self.calls.append(vector)
        return self.results[len(self.calls) - 1]


class Evaluator:
    def __init__(self, decisions):
        self.decisions = decisions
        self.calls = []

    def evaluate(self, question, candidates):
        self.calls.append((question, candidates))
        decision = self.decisions[len(self.calls) - 1]
        if isinstance(decision, Exception):
            raise decision
        if decision == "support":
            return GroundingResult(status=GroundingStatus.SUFFICIENT,
                                   supporting_chunk_ids=(candidates[0][0],), reason="Explicit fact")
        return decision


class Rewriter:
    def __init__(self, query="violet plasma sail inventor"):
        self.query = query
        self.calls = []

    def rewrite(self, question):
        self.calls.append(question)
        return self.query


def run(results, decisions, question="Who invented the violet plasma sail?"):
    store, embeddings = Store(results), Embeddings()
    evaluator, rewriter = Evaluator(decisions), Rewriter()
    outcome = GroundedKnowledgeService(store, embeddings, evaluator, rewriter).search(question)
    return outcome, store, embeddings, evaluator, rewriter


def test_explicit_support_preserves_only_selected_source() -> None:
    outcome, store, embeddings, evaluator, rewriter = run(
        [(CHUNK, RELATED)], ["support"], "What color is the plasma sail?"
    )
    assert outcome.status is GroundingStatus.SUFFICIENT
    assert outcome.supporting_chunks == (CHUNK,)
    assert outcome.retrieval_attempts == 1
    assert outcome.rewrite_attempts == 0
    assert embeddings.queries == ["What color is the plasma sail?"]
    assert not rewriter.calls


@pytest.mark.parametrize("first_status", [GroundingStatus.INSUFFICIENT, GroundingStatus.AMBIGUOUS])
def test_related_or_partial_evidence_rewrites_once_and_stops(first_status) -> None:
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


def test_second_attempt_can_succeed() -> None:
    insufficient = GroundingResult(status=GroundingStatus.INSUFFICIENT,
                                   supporting_chunk_ids=(), reason="Need another passage")
    outcome, store, embeddings, evaluator, rewriter = run(
        [(RELATED,), (CHUNK,)], [insufficient, "support"]
    )
    assert outcome.status is GroundingStatus.SUFFICIENT
    assert outcome.supporting_chunks == (CHUNK,)
    assert outcome.retrieval_attempts == 2
    assert len(store.calls) == 2


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
    outcome, store, embeddings, evaluator, rewriter = run([(CHUNK,)], [decision])
    assert outcome.status is GroundingStatus.INSUFFICIENT
    assert outcome.supporting_chunks == ()
    assert outcome.retrieval_attempts == 1
    assert len(store.calls) == 1
    assert not rewriter.calls


def test_no_candidates_and_failed_rewrite_do_not_loop() -> None:
    store, embeddings, evaluator, rewriter = Store([()]), Embeddings(), Evaluator([]), Rewriter("")
    outcome = GroundedKnowledgeService(store, embeddings, evaluator, rewriter).search("Unknown fact?")
    assert outcome.status is GroundingStatus.INSUFFICIENT
    assert outcome.retrieval_attempts == 1
    assert outcome.rewrite_attempts == 1
    assert len(store.calls) == 1
