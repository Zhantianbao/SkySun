"""SkySun-owned evidence evaluation and bounded knowledge retrieval."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from skysun.embeddings import EmbeddingProvider
from skysun.knowledge import KnowledgeSearchResult, KnowledgeStore


class GroundingStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"


class GroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: GroundingStatus
    supporting_chunk_ids: tuple[str, ...]
    reason: str = Field(min_length=1)


class EvidenceEvaluator(Protocol):
    def evaluate(
        self, question: str, candidates: tuple[tuple[str, KnowledgeSearchResult], ...]
    ) -> GroundingResult: ...


class QueryRewriter(Protocol):
    def rewrite(self, question: str) -> str: ...


@dataclass(frozen=True)
class GroundedKnowledge:
    status: GroundingStatus
    supporting_chunks: tuple[KnowledgeSearchResult, ...]
    reason: str
    retrieval_attempts: int
    rewrite_attempts: int


class GroundedKnowledgeService:
    def __init__(
        self,
        store: KnowledgeStore,
        embeddings: EmbeddingProvider,
        evaluator: EvidenceEvaluator,
        rewriter: QueryRewriter,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._evaluator = evaluator
        self._rewriter = rewriter

    def search(self, question: str, limit: int = 4) -> GroundedKnowledge:
        if not question.strip() or not 1 <= limit <= 5:
            return GroundedKnowledge(GroundingStatus.INSUFFICIENT, (), "Invalid knowledge query.", 0, 0)
        query = question
        rewrites = 0
        for attempt in (1, 2):
            try:
                chunks = self._store.search(self._embeddings.embed_query(query), limit=limit)
                candidates = tuple((str(chunk.chunk_id) if chunk.chunk_id else f"candidate-{index}", chunk)
                                   for index, chunk in enumerate(chunks, start=1))
                if candidates:
                    raw = self._evaluator.evaluate(question, candidates)
                    result = GroundingResult.model_validate(raw)
                    available = {identifier for identifier, _ in candidates}
                    selected = result.supporting_chunk_ids
                    if (
                        len(available) != len(candidates)
                        or len(set(selected)) != len(selected)
                        or not set(selected) <= available
                        or (result.status is GroundingStatus.SUFFICIENT) != bool(selected)
                        or not result.reason.strip()
                    ):
                        raise ValueError("Invalid grounding evidence")
                    if result.status is GroundingStatus.SUFFICIENT:
                        by_id = dict(candidates)
                        return GroundedKnowledge(result.status, tuple(by_id[key] for key in selected),
                                                 result.reason, attempt, rewrites)
                    status = result.status
                    reason = result.reason
                else:
                    status = GroundingStatus.INSUFFICIENT
                    reason = "No relevant stored knowledge found."
            except Exception:
                return GroundedKnowledge(GroundingStatus.INSUFFICIENT, (),
                                         "Knowledge evidence could not be validated.", attempt, rewrites)
            if attempt == 1:
                rewrites = 1
                try:
                    query = self._rewriter.rewrite(question).strip()
                    if not query or len(query) > 1000:
                        raise ValueError("Invalid rewritten query")
                except Exception:
                    return GroundedKnowledge(GroundingStatus.INSUFFICIENT, (),
                                             "Knowledge query rewrite failed.", attempt, rewrites)
            else:
                return GroundedKnowledge(status, (), reason, attempt, rewrites)
        raise AssertionError("Unreachable retrieval state")
