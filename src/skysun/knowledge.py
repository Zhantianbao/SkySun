"""SkySun-owned document knowledge types and storage boundary."""

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, Sequence
from uuid import UUID


@dataclass(frozen=True)
class KnowledgeDocument:
    id: UUID
    source_path: str
    source_name: str
    content_hash: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class KnowledgeChunkInput:
    content: str
    embedding: tuple[float, ...]


@dataclass(frozen=True)
class KnowledgeSearchResult:
    content: str
    source_name: str
    source_path: str
    chunk_index: int
    cosine_distance: float


@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    source_path: str
    chunk_count: int
    status: Literal["ingested", "updated", "unchanged"]


class KnowledgeStore(Protocol):
    def get_by_path(self, source_path: str) -> KnowledgeDocument | None: ...

    def save_document(
        self,
        source_path: str,
        source_name: str,
        content_hash: str,
        chunks: Sequence[KnowledgeChunkInput],
    ) -> IngestionResult: ...

    def delete_document(self, source_path: str) -> bool: ...

    def search(
        self,
        query_embedding: tuple[float, ...],
        limit: int = 4,
        max_cosine_distance: float = 0.35,
    ) -> tuple[KnowledgeSearchResult, ...]: ...
