"""Embedding boundary and Alibaba Cloud Model Studio adapter."""
# 本文件负责知识检索中的 embeddings 相关流程。

import math
from typing import Protocol, Sequence

from langchain_openai import OpenAIEmbeddings

from skysun.core.config import Settings

MODEL_NAME = "qwen3.7-text-embedding"
EMBEDDING_DIMENSION = 1024


# 以下集中定义 EmbeddingProvider：数据结构或组件相关的数据和操作。
class EmbeddingProvider(Protocol):
    dimension: int

    # 调用位置：知识入库服务调用嵌入模型的片段向量方法。
    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]: """为一组知识片段生成向量。"""; ...

    # 调用位置：证据检索服务调用嵌入模型的查询向量方法。
    def embed_query(self, text: str) -> tuple[float, ...]: """为检索问题生成向量。"""; ...


# 以下集中定义 QwenEmbeddingProvider：数据结构或组件相关的数据和操作。
class QwenEmbeddingProvider:
    """Use Model Studio's OpenAI-compatible synchronous embedding endpoint."""

    dimension = EMBEDDING_DIMENSION

    # 调用位置：创建 QwenEmbeddingProvider 实例时调用。
    def __init__(self, api_key: str, base_url: str) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
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

    # 调用位置：QwenEmbeddingProvider 内的 embed_passages、embed_query 调用。
    def _vectors(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        """调用嵌入服务并校验返回的向量。"""
        vectors = self._client.embed_documents(texts)
        if len(vectors) != len(texts):
            raise RuntimeError("Embedding API returned the wrong number of vectors")
        result = tuple(tuple(float(value) for value in vector) for vector in vectors)
        if any(len(vector) != self.dimension or not all(math.isfinite(value) for value in vector)
               for vector in result):
            raise RuntimeError("Embedding API returned an invalid vector")
        return result

    # 调用位置：通过 QwenEmbeddingProvider 实例在知识入库、检索服务与 Agent 工具中调用。
    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """把多个知识片段转换为向量。"""
        if not texts:
            return ()
        return self._vectors(list(texts))

    # 调用位置：通过 QwenEmbeddingProvider 实例在知识入库、检索服务与 Agent 工具中调用。
    def embed_query(self, text: str) -> tuple[float, ...]:
        """把检索问题转换为向量。"""
        if not text.strip():
            raise ValueError("Knowledge search query must not be empty")
        return self._vectors([text])[0]


# 调用位置：src/skysun/core/main.py 的 create_app、src/skysun/rag/ingest.py 的 main 调用。
def create_embeddings(settings: Settings) -> QwenEmbeddingProvider:
    """根据配置创建文本嵌入模型。"""
    if settings.dashscope_api_key is None or settings.dashscope_base_url is None:
        raise RuntimeError("Set DASHSCOPE_API_KEY and DASHSCOPE_BASE_URL for knowledge embeddings")
    return QwenEmbeddingProvider(
        settings.dashscope_api_key.get_secret_value(), settings.dashscope_base_url
    )
