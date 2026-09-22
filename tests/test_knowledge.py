"""Offline chunking, ingestion, embedding boundary, and Agent tool tests."""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from skysun.knowledge import (
    IngestionResult,
    KnowledgeDocument,
    KnowledgeSearchResult,
)
from skysun.knowledge_ingestion import KnowledgeIngestionService, chunk_text
from skysun.langchain_agent import LangChainAgent
from skysun.knowledge_grounding import GroundingResult, GroundingStatus

_VECTOR_A = (1.0,) + (0.0,) * 383


class FakeEmbeddings:
    dimension = 384

    def __init__(self) -> None:
        self.passages: list[tuple[str, ...]] = []
        self.queries: list[str] = []

    def embed_passages(self, texts: Sequence[str]) -> tuple[tuple[float, ...], ...]:
        self.passages.append(tuple(texts))
        return tuple(_VECTOR_A for _ in texts)

    def embed_query(self, text: str) -> tuple[float, ...]:
        self.queries.append(text)
        return _VECTOR_A


class FakeKnowledgeStore:
    def __init__(self) -> None:
        self.documents: dict[str, KnowledgeDocument] = {}
        self.chunks: dict[str, tuple] = {}
        self.search_results: tuple[KnowledgeSearchResult, ...] = ()
        self.searches: list[tuple[tuple[float, ...], int]] = []

    def get_by_path(self, source_path: str) -> KnowledgeDocument | None:
        return self.documents.get(source_path)

    def save_document(self, source_path, source_name, content_hash, chunks):
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

    def delete_document(self, source_path: str) -> bool:
        existed = self.documents.pop(source_path, None) is not None
        self.chunks.pop(source_path, None)
        return existed

    def search(self, query_embedding, limit=4, max_cosine_distance=0.35):
        self.searches.append((query_embedding, limit))
        return self.search_results[:limit]


def test_chunking_is_deterministic_and_overlaps_at_boundaries() -> None:
    text = "A" * 580 + "\n\n" + "B" * 580 + "\n\n" + "C" * 580
    first = chunk_text(text, size=800, overlap=120)
    assert first == chunk_text(text, size=800, overlap=120)
    assert len(first) >= 3
    assert all(len(part) <= 800 for part in first)
    assert first[0][-40:] in first[1]
    assert chunk_text("") == ()
    assert chunk_text("  \n  ") == ()


def test_ingestion_idempotency_update_and_errors(tmp_path: Path) -> None:
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


class FakeKnowledgeCallingModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "fake-knowledge-calling"

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> "FakeKnowledgeCallingModel":
        assert {tool.name for tool in tools} >= {"count_words", "search_knowledge"}
        return self

    def _generate(
        self, messages: list[BaseMessage], stop: list[str] | None = None, **kwargs: Any
    ) -> ChatResult:
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


def test_agent_knowledge_tool_uses_query_embeddings_and_source_metadata() -> None:
    store = FakeKnowledgeStore()
    embeddings = FakeEmbeddings()
    store.search_results = (
        KnowledgeSearchResult(
            "Project Helios uses a sapphire thermal shield.",
            "helios.md", "/notes/helios.md", 2, 0.12,
        ),
    )
    class Evaluator:
        def evaluate(self, question, candidates):
            return GroundingResult(status=GroundingStatus.SUFFICIENT,
                                   supporting_chunk_ids=(candidates[0][0],), reason="Explicit answer")

    class Rewriter:
        def rewrite(self, question):
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



def test_local_e5_adapter_uses_cpu_prefixes_and_normalized_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys
    from types import SimpleNamespace

    calls = []

    class FakeSentenceTransformer:
        def __init__(self, name, device):
            assert name == "intfloat/multilingual-e5-small"
            assert device == "cpu"

        def get_embedding_dimension(self):
            return 384

        def encode(self, texts, normalize_embeddings, show_progress_bar):
            calls.append((texts, normalize_embeddings, show_progress_bar))
            return [_VECTOR_A for _ in texts]

    monkeypatch.setitem(
        sys.modules, "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    from skysun.embeddings import SentenceTransformerEmbeddingProvider

    provider = SentenceTransformerEmbeddingProvider()
    assert provider.embed_passages(["中文材料", "English material"]) == (_VECTOR_A, _VECTOR_A)
    assert provider.embed_query("What is the material?") == _VECTOR_A
    assert calls == [
        (["passage: 中文材料", "passage: English material"], True, False),
        (["query: What is the material?"], True, False),
    ]
