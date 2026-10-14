"""Embedding boundary and Alibaba Cloud Model Studio adapter."""

import math
from typing import Protocol, Sequence

from langchain_openai import OpenAIEmbeddings

from skysun.config import Settings

MODEL_NAME = "qwen3.7-text-embedding"
EMBEDDING_DIMENSION = 1024


class EmbeddingProvider(Protocol):
    dimension: int

    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: ...

    def embed_query(self, text: str) -> tuple[float, ...]: ...


class QwenEmbeddingProvider:
    """Use Model Studio's OpenAI-compatible synchronous embedding endpoint."""

    dimension = EMBEDDING_DIMENSION

    def __init__(self, api_key: str, base_url: str) -> None:
        if not api_key.strip() or not base_url.strip():
            raise ValueError("DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL are required")
        self._client = OpenAIEmbeddings(
            model=MODEL_NAME,
            dimensions=self.dimension,
            api_key=api_key,
            base_url=base_url,
            chunk_size=20,
            check_embedding_ctx_length=False,
            model_kwargs={"encoding_format": "float"},
            request_timeout=30,
            max_retries=2,
        )

    def _vectors(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        vectors = self._client.embed_documents(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding API returned the wrong number of vectors")
        result = tuple(tuple(float(value) for value in vector) for vector in vectors)
        if any(len(vector) != self.dimension or not all(math.isfinite(value) for value in vector)
               for vector in result):
            raise RuntimeError("Embedding API returned an invalid vector")
        return result

    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        if not texts:
            return ()
        return self._vectors(list(texts))

    def embed_query(self, text: str) -> tuple[float, ...]:
        if not text.strip():
            raise ValueError("Knowledge search query must not be empty")
        return self._vectors([text])[0]


def create_embeddings(settings: Settings) -> QwenEmbeddingProvider:
    if settings.dashscope_api_key is None or settings.dashscope_base_url is None:
        raise RuntimeError("Set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL for knowledge embeddings")
    return QwenEmbeddingProvider(
        settings.dashscope_api_key.get_secret_value(), settings.dashscope_base_url
    )
