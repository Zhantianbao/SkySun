"""SkySun-owned evidence evaluation and bounded knowledge retrieval."""
# 本文件负责知识检索中的 knowledge、grounding 相关流程。

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from skysun.rag.embeddings import EmbeddingProvider
from skysun.rag.knowledge import KnowledgeSearchResult, KnowledgeStore


# 以下集中定义 GroundingStatus：数据结构或组件相关的数据和操作。
class GroundingStatus(str, Enum):
    SUFFICIENT = "SUFFICIENT"
    INSUFFICIENT = "INSUFFICIENT"
    AMBIGUOUS = "AMBIGUOUS"


# 以下集中定义 GroundingResult：数据结构或组件相关的数据和操作。
class GroundingResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: GroundingStatus
    supporting_chunk_ids: tuple[str, ...]
    reason: str = Field(min_length=1)


# 以下集中定义 EvidenceEvaluator：回答评估层相关的数据和操作。
class EvidenceEvaluator(Protocol):
    # 调用位置：该协议由 EvidenceEvaluator 的实现类调用；入口涉及知识入库、检索服务与 Agent 工具。
    def evaluate(
        self, question: str, candidates: tuple[tuple[str, KnowledgeSearchResult], ...]
    ) -> GroundingResult: """评估检索证据或回复结果。"""; ...


# 以下集中定义 QueryRewriter：数据结构或组件相关的数据和操作。
class QueryRewriter(Protocol):
    # 调用位置：证据检索服务在证据不足时调用实现类的改写方法。
    def rewrite(self, question: str) -> str: """根据检索不足的原因改写查询。"""; ...


# 以下集中定义 GroundedKnowledge：知识数据相关的数据和操作。
@dataclass(frozen=True)
class GroundedKnowledge:
    status: GroundingStatus
    supporting_chunks: tuple[KnowledgeSearchResult, ...]
    reason: str
    retrieval_attempts: int
    rewrite_attempts: int


# 以下集中定义 GroundedKnowledgeService：服务层相关的数据和操作。
class GroundedKnowledgeService:
    # 调用位置：创建 GroundedKnowledgeService 实例时调用。
    def __init__(
        self,
        store: KnowledgeStore,
        embeddings: EmbeddingProvider,
        evaluator: EvidenceEvaluator,
        rewriter: QueryRewriter,
    ) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._store = store
        self._embeddings = embeddings
        self._evaluator = evaluator
        self._rewriter = rewriter

    # 调用位置：通过 GroundedKnowledgeService 实例在知识入库、检索服务与 Agent 工具中调用。
    def search(self, question: str, limit: int = 4) -> GroundedKnowledge:
        """按问题检索相关内容。"""
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
