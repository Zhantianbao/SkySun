"""Offline chunking, ingestion, embedding boundary, and Agent tool tests."""
# 本文件负责测试用例中的 test、knowledge 相关流程。

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from skysun.rag.knowledge import (
    IngestionResult,
    KnowledgeDocument,
    KnowledgeSearchResult,
)
from skysun.rag.knowledge_ingestion import KnowledgeIngestionService, chunk_text
from skysun.core.langchain_agent import LangChainAgent
from skysun.rag.knowledge_grounding import GroundingResult, GroundingStatus

_VECTOR_A = (1.0,) + (0.0,) * 1023


# 以下集中定义 FakeEmbeddings：数据结构或组件相关的数据和操作。
class FakeEmbeddings:
    dimension = 1024

    # 调用位置：创建 FakeEmbeddings 实例时调用。
    def __init__(self) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.passages: list[tuple[str, ...]] = []
        self.queries: list[str] = []

    # 调用位置：src/skysun/rag/knowledge_ingestion.py 的 KnowledgeIngestionService.ingest_text、tests/test_knowledge.py 的 test_qwen_adapter_uses_compatible_api_and_validates_vectors 调用。
    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        """把多个知识片段转换为向量。"""
        self.passages.append(tuple(texts))
        return tuple(_VECTOR_A for _ in texts)

    # 调用位置：src/skysun/rag/knowledge_grounding.py 的 GroundedKnowledgeService.search、tests/test_knowledge.py 的 test_qwen_adapter_uses_compatible_api_and_validates_vectors 等处调用。
    def embed_query(self, text: str) -> tuple[float, ...]:
        """把检索问题转换为向量。"""
        self.queries.append(text)
        return _VECTOR_A


# 以下集中定义 FakeKnowledgeStore：存储层相关的数据和操作。
class FakeKnowledgeStore:
    # 调用位置：创建 FakeKnowledgeStore 实例时调用。
    def __init__(self) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self.documents: dict[str, KnowledgeDocument] = {}
        self.chunks: dict[str, tuple] = {}
        self.search_results: tuple[KnowledgeSearchResult, ...] = ()
        self.searches: list[tuple[tuple[float, ...], int]] = []

    # 调用位置：src/skysun/rag/knowledge_ingestion.py 的 KnowledgeIngestionService.ingest_text、tests/test_multi_user.py 的 test_private_knowledge_isolation_and_global_visibility、tests/test_postgres_knowledge.py 的 test_postgres_idempotent_update_vector_order_sources_and_cascade 等处调用。
    def get_by_path(self, source_path: str) -> KnowledgeDocument | None:
        """按路径查找已入库文档。"""
        return self.documents.get(source_path)

    # 调用位置：src/skysun/rag/knowledge_ingestion.py 的 KnowledgeIngestionService.ingest_text、tests/test_multi_user.py 的 test_private_knowledge_isolation_and_global_visibility、tests/test_multi_user.py 的 test_web_read_endpoints_keep_user_data_isolated 等处调用。
    def save_document(self, source_path, source_name, content_hash, chunks):
        """保存文档及其知识片段。"""
        old = self.documents.get(source_path)
        now = datetime.now(timezone.utc)
        item = KnowledgeDocument(
            old.id if old else uuid4(), source_path, source_name,
            content_hash, old.created_at if old else now, now,
        )
        self.documents[source_path] = item
        self.chunks[source_path] = tuple(chunks)
        return IngestionResult(
            item.id, source_path, len(chunks), "updated" if old else "ingested"
        )

    # 调用位置：src/skysun/rag/ingest.py 的 main、tests/test_multi_user.py 的 test_private_knowledge_isolation_and_global_visibility、tests/test_postgres_knowledge.py 的 test_postgres_idempotent_update_vector_order_sources_and_cascade 等处调用。
    def delete_document(self, source_path: str) -> bool:
        """删除指定文档及其片段。"""
        existed = self.documents.pop(source_path, None) is not None
        self.chunks.pop(source_path, None)
        return existed

    # 调用位置：FakeKnowledgeStore 的调用方通过该方法使用。
    def search(self, query_embedding, limit=4, max_cosine_distance=0.35):
        """按问题检索相关内容。"""
        self.searches.append((query_embedding, limit))
        return self.search_results[:limit]


# 调用位置：pytest 执行该测试用例时调用。
def test_chunking_is_deterministic_and_overlaps_at_boundaries() -> None:
    """验证文本切片结果稳定且相邻片段在边界处重叠。"""
    text = "A" * 580 + "\n\n" + "B" * 580 + "\n\n" + "C" * 580
    first = chunk_text(text, size=800, overlap=120)
    assert first == chunk_text(text, size=800, overlap=120)
    assert len(first) >= 3
    assert all(len(part) <= 800 for part in first)
    assert first[0][-40:] in first[1]
    assert chunk_text("") == ()
    assert chunk_text("  \n  ") == ()


# 调用位置：pytest 执行该测试用例时调用。
def test_ingestion_idempotency_update_and_errors(tmp_path: Path) -> None:
    """验证知识入库的重复写入、更新和错误处理正确。"""
    store = FakeKnowledgeStore()
    embeddings = FakeEmbeddings()
    service = KnowledgeIngestionService(store, embeddings, chunk_size=200, chunk_overlap=30)
    source = tmp_path / "sample.md"
    source.write_text("Project Helios uses a sapphire thermal shield.\n" * 9, encoding="utf-8")

    first = service.ingest(source)
    assert first.status == "ingested"
    assert first.chunk_count > 1
    assert len(embeddings.passages) == 1
    assert len(store.chunks[str(source.resolve())]) == first.chunk_count

    unchanged = service.ingest(source)
    assert unchanged.status == "unchanged"
    assert unchanged.document_id == first.document_id
    assert len(embeddings.passages) == 1

    source.write_text("Project Helios now uses a diamond shield.", encoding="utf-8")
    updated = service.ingest(source)
    assert updated.status == "updated"
    assert updated.document_id == first.document_id
    assert len(store.chunks[str(source.resolve())]) == 1
    assert "diamond" in store.chunks[str(source.resolve())][0].content

    text_source = tmp_path / "brief.txt"
    text_source.write_text("Project Orion uses ceramic shielding.", encoding="utf-8")
    assert service.ingest(text_source).status == "ingested"

    with pytest.raises(FileNotFoundError):
        service.ingest(tmp_path / "missing.md")
    unsupported = tmp_path / "bad.pdf"
    unsupported.write_text("not parsed", encoding="utf-8")
    with pytest.raises(ValueError, match="Supported knowledge files"):
        service.ingest(unsupported)
    empty = tmp_path / "empty.txt"
    empty.write_text("   ", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        service.ingest(empty)
    invalid = tmp_path / "invalid.txt"
    invalid.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        service.ingest(invalid)


# 以下集中定义 FakeKnowledgeCallingModel：数据结构或组件相关的数据和操作。
class FakeKnowledgeCallingModel(BaseChatModel):
    # 调用位置：FakeKnowledgeCallingModel 的调用方通过该方法使用。
    @property
    def _llm_type(self) -> str:
        """为测试准备 _llm_type 所需的替身或输入。"""
        return "fake-knowledge-calling"

    # 调用位置：FakeKnowledgeCallingModel 的调用方通过该方法使用。
    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeKnowledgeCallingModel":
        """为测试提供 bind_tools 所需的输入或模拟行为。"""
        assert {tool.name for tool in tools} >= {"count_words", "search_knowledge"}
        return self

    # 调用位置：LangChain 模型接口在生成回复时调用。
    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
        """为测试准备 _generate 所需的替身或输入。"""
        tool_messages = [message for message in messages if isinstance(message, ToolMessage)]
        if tool_messages:
            answer = AIMessage(content=str(tool_messages[-1].content))
        else:
            answer = AIMessage(content="", tool_calls=[{
                "name": "search_knowledge",
                "args": {"query": "Helios shield", "limit": 4},
                "id": "knowledge_call",
            }])
        return ChatResult(generations=[ChatGeneration(message=answer)])


# 调用位置：pytest 执行该测试用例时调用。
def test_agent_knowledge_tool_uses_query_embeddings_and_source_metadata() -> None:
    """验证Agent 知识工具使用查询向量并返回来源信息。"""
    store = FakeKnowledgeStore()
    embeddings = FakeEmbeddings()
    store.search_results = (
        KnowledgeSearchResult(
            "Project Helios uses a sapphire thermal shield.",
            "helios.md", "/notes/helios.md", 2, 0.12,
        ),
    )
    # 以下集中定义 Evaluator：回答评估层相关的数据和操作。
    class Evaluator:
        # 调用位置：Evaluator 的调用方通过该方法使用。
        def evaluate(self, question, candidates):
            """根据执行证据判断回答的质量或可信度。"""
            return GroundingResult(status=GroundingStatus.SUFFICIENT,
                                   supporting_chunk_ids=(candidates[0][0],), reason="Explicit answer")

    # 以下集中定义 Rewriter：数据结构或组件相关的数据和操作。
    class Rewriter:
        # 调用位置：src/skysun/rag/knowledge_grounding.py 的 GroundedKnowledgeService.search 调用。
        def rewrite(self, question):
            """根据证据不足的原因改写检索问题。"""
            return question

    agent = LangChainAgent(FakeKnowledgeCallingModel(), knowledge_store=store, embeddings=embeddings,
                           evidence_evaluator=Evaluator(), query_rewriter=Rewriter())
    answer = agent.reply("What shield does Helios use?")
    assert "sapphire" in answer
    assert "helios.md" in answer
    assert "/notes/helios.md" in answer
    assert "chunk 2" in answer
    assert embeddings.queries == ["What shield does Helios use?"]
    assert store.searches[0][1] == 4



# 调用位置：pytest 执行该测试用例时调用。
def test_qwen_adapter_uses_compatible_api_and_validates_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """验证通义嵌入适配器调用兼容接口并校验向量。"""
    from skysun.rag import embeddings

    calls = []

    # 以下集中定义 FakeOpenAIEmbeddings：数据结构或组件相关的数据和操作。
    class FakeOpenAIEmbeddings:
        # 调用位置：创建 FakeOpenAIEmbeddings 实例时调用。
        def __init__(self, **kwargs):
            """保存依赖和初始状态，供该对象后续方法使用。"""
            calls.append(kwargs)

        # 调用位置：src/skysun/rag/embeddings.py 的 QwenEmbeddingProvider._vectors 调用。
        def embed_documents(self, texts):
            """为测试提供 embed_documents 所需的输入或模拟行为。"""
            calls.append(texts)
            return [_VECTOR_A for _ in texts]

    monkeypatch.setattr(embeddings, "OpenAIEmbeddings", FakeOpenAIEmbeddings)
    provider = embeddings.QwenEmbeddingProvider("test-key", "https://example.com/compatible-mode/v1")
    assert provider.embed_passages([]) == ()
    assert provider.embed_passages(["中文材料", "English material"]) == (_VECTOR_A, _VECTOR_A)
    assert provider.embed_query("What is the material?") == _VECTOR_A
    assert calls[0]["model"] == "qwen3.7-text-embedding"
    assert calls[0]["dimensions"] == 1024
    assert calls[0]["chunk_size"] == 20
    assert calls[0]["check_embedding_ctx_length"] is False
    assert calls[0]["model_kwargs"] == {"encoding_format": "float"}
    assert calls[1:] == [["中文材料", "English material"], ["What is the material?"]]
    with pytest.raises(ValueError, match="empty"):
        provider.embed_query("  ")
    monkeypatch.setattr(provider._client, "embed_documents", lambda texts: [(1.0,)] * len(texts))
    with pytest.raises(RuntimeError, match="invalid vector"):
        provider.embed_query("question")
