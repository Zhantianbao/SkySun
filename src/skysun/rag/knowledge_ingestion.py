"""Explicit local-file ingestion into SkySun document knowledge."""
# 本文件负责知识检索中的 knowledge、ingestion 相关流程。

import hashlib
from pathlib import Path

from skysun.rag.embeddings import EmbeddingProvider
from skysun.rag.knowledge import IngestionResult, KnowledgeChunkInput, KnowledgeStore

_SUPPORTED = {".md", ".markdown", ".txt"}


# 调用位置：src/skysun/rag/knowledge_ingestion.py 的 KnowledgeIngestionService.ingest_text、tests/test_knowledge.py 的 test_chunking_is_deterministic_and_overlaps_at_boundaries 等处调用。
def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> tuple[str, ...]:
    """优先在段落或换行附近切分文本，并保留片段重叠区域。"""
    if size < 100 or overlap < 0 or overlap >= size // 2:
        raise ValueError("Invalid chunk size or overlap")
    text = text.strip()
    if not text:
        return ()
    # 优先按段落或换行切分，再保留重叠内容，避免检索片段在边界处丢失上下文。
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind("\n\n", start + size // 2, end)
            if boundary < 0:
                boundary = text.rfind("\n", start + size // 2, end)
            if boundary >= 0:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(start + 1, end - overlap)
    return tuple(chunks)


# 以下集中定义 KnowledgeIngestionService：服务层相关的数据和操作。
class KnowledgeIngestionService:
    # 调用位置：创建 KnowledgeIngestionService 实例时调用。
    def __init__(
        self,
        store: KnowledgeStore,
        embeddings: EmbeddingProvider,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._store = store
        self._embeddings = embeddings
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    # 调用位置：通过 KnowledgeIngestionService 实例在知识入库、检索服务与 Agent 工具中调用。
    def ingest(self, source: str | Path) -> IngestionResult:
        """读取文件并写入知识库。"""
        path = Path(source).expanduser()
        if path.suffix.lower() not in _SUPPORTED:
            raise ValueError("Supported knowledge files are .md, .markdown, and .txt")
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("Knowledge source must be a regular file")
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        return self.ingest_text(str(path), path.name, text)

    # 调用位置：KnowledgeIngestionService 内的 ingest 调用。
    def ingest_text(self, source_path: str, source_name: str, text: str) -> IngestionResult:
        """校验来自本地文件或网页登录上传的文本，分段生成向量后写入知识库。"""
        text = text.replace("\r\n", "\n").strip()
        if not text:
            raise ValueError("Knowledge source is empty")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self._store.get_by_path(source_path)
        if existing is not None and existing.content_hash == content_hash:
            return IngestionResult(existing.id, source_path, 0, "unchanged")
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        vectors = self._embeddings.embed_passages(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding provider returned the wrong number of vectors")
        prepared = tuple(
            KnowledgeChunkInput(content, vector)
            for content, vector in zip(chunks, vectors, strict=True)
        )
        return self._store.save_document(
            source_path, source_name, content_hash, prepared
        )
