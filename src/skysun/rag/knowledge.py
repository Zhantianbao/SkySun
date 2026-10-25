"""SkySun-owned document knowledge types and storage boundary."""
# 本文件负责知识检索中的 knowledge 相关流程。

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol, Sequence
from uuid import UUID


# 以下集中定义 KnowledgeDocument：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class KnowledgeDocument:
    id: UUID
    source_path: str
    source_name: str
    content_hash: str
    created_at: datetime
    updated_at: datetime
    owner_id: UUID | None = None


# 以下集中定义 KnowledgeChunkInput：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class KnowledgeChunkInput:
    content: str
    embedding: tuple[float, ...]


# 以下集中定义 KnowledgeSearchResult：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class KnowledgeSearchResult:
    content: str
    source_name: str
    source_path: str
    chunk_index: int
    cosine_distance: float
    chunk_id: UUID | None = None


# 以下集中定义 IngestionResult：数据结构或组件相关的数据和操作。
@dataclass(frozen=True)
class IngestionResult:
    document_id: UUID
    source_path: str
    chunk_count: int
    status: Literal["ingested", "updated", "unchanged"]


# 以下集中定义 KnowledgeStore：存储层相关的数据和操作。
class KnowledgeStore(Protocol):
    # 调用位置：知识入库服务调用实现类的路径查询方法。
    def get_by_path(self, source_path: str) -> KnowledgeDocument | None: """按路径读取知识文档。"""; ...

    # 调用位置：该协议由 KnowledgeStore 的实现类调用；入口涉及知识入库、检索服务与 Agent 工具。
    def save_document(
        self,
        source_path: str,
        source_name: str,
        content_hash: str,
        chunks: Sequence[KnowledgeChunkInput],
    ) -> IngestionResult: """保存文档及其检索片段。"""; ...

    # 调用位置：知识存储的调用方删除文档时使用。
    def delete_document(self, source_path: str) -> bool: """删除文档及其片段。"""; ...

    # 调用位置：该协议由 KnowledgeStore 的实现类调用；入口涉及知识入库、检索服务与 Agent 工具。
    def search(
        self,
        query_embedding: tuple[float, ...],
        limit: int = 4,
        max_cosine_distance: float = 0.35,
    ) -> tuple[KnowledgeSearchResult, ...]: """按查询返回最相关的知识片段。"""; ...
