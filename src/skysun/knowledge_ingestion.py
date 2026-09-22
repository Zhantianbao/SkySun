"""Explicit local-file ingestion into SkySun document knowledge."""

import hashlib
from pathlib import Path

from skysun.embeddings import EmbeddingProvider
from skysun.knowledge import IngestionResult, KnowledgeChunkInput, KnowledgeStore

_SUPPORTED = {".md", ".markdown", ".txt"}


def chunk_text(text: str, size: int = 1000, overlap: int = 150) -> tuple[str, ...]:
    """Split deterministically near paragraph/newline boundaries with overlap."""
    if size < 100 or overlap < 0 or overlap >= size // 2:
        raise ValueError("Invalid chunk size or overlap")
    text = text.strip()
    if not text:
        return ()
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


class KnowledgeIngestionService:
    def __init__(
        self,
        store: KnowledgeStore,
        embeddings: EmbeddingProvider,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

    def ingest(self, source: str | Path) -> IngestionResult:
        path = Path(source).expanduser()
        if path.suffix.lower() not in _SUPPORTED:
            raise ValueError("Supported knowledge files are .md, .markdown, and .txt")
        path = path.resolve(strict=True)
        if not path.is_file():
            raise ValueError("Knowledge source must be a regular file")
        text = path.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
        if not text:
            raise ValueError("Knowledge source is empty")
        content_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        existing = self._store.get_by_path(str(path))
        if existing is not None and existing.content_hash == content_hash:
            return IngestionResult(existing.id, str(path), 0, "unchanged")
        chunks = chunk_text(text, self._chunk_size, self._chunk_overlap)
        vectors = self._embeddings.embed_passages(chunks)
        if len(vectors) != len(chunks):
            raise RuntimeError("Embedding provider returned the wrong number of vectors")
        prepared = tuple(
            KnowledgeChunkInput(content, vector)
            for content, vector in zip(chunks, vectors, strict=True)
        )
        return self._store.save_document(
            str(path), path.name, content_hash, prepared
        )
