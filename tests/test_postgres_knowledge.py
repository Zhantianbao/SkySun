"""PostgreSQL/pgvector integration uses only the dedicated *_test database."""

import os
from collections.abc import Iterator
from uuid import uuid4

import psycopg
import pytest

from skysun.rag.knowledge import KnowledgeChunkInput
from skysun.rag.postgres_knowledge import PostgresKnowledgeStore

_A = (1.0,) + (0.0,) * 1023
_B = (0.0, 1.0) + (0.0,) * 1022


@pytest.fixture
def knowledge_store() -> Iterator[PostgresKnowledgeStore]:
    database_url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(database_url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("Knowledge integration tests require a *_test database")
    store = PostgresKnowledgeStore(database_url)
    store.initialize_schema()
    yield store


def test_postgres_idempotent_update_vector_order_sources_and_cascade(
    knowledge_store: PostgresKnowledgeStore,
) -> None:
    store = knowledge_store
    path = f"/tmp/skysun_test_helios_{uuid4().hex}.md"
    other_path = f"/tmp/skysun_test_orion_{uuid4().hex}.txt"
    try:
        first = store.save_document(
            path, "helios.md", "hash-one",
            (KnowledgeChunkInput("Helios has a sapphire shield.", _A),),
        )
        assert first.status == "ingested"
        store.initialize_schema()
        again = store.save_document(
            path, "helios.md", "hash-one",
            (KnowledgeChunkInput("Helios has a sapphire shield.", _A),),
        )
        assert again.status == "unchanged"
        assert first.document_id == again.document_id

        store.save_document(
            other_path, "orion.txt", "hash-two",
            (KnowledgeChunkInput("Orion has a ceramic shield.", _B),),
        )
        all_results = store.search(_A, limit=2, max_cosine_distance=2.0)
        assert [item.source_name for item in all_results] == ["helios.md", "orion.txt"]
        assert all_results[0].source_path == path
        assert all_results[0].chunk_index == 0
        assert all_results[0].cosine_distance < all_results[1].cosine_distance
        assert [item.source_name for item in store.search(_A)] == ["helios.md"]
        assert store.search(tuple([0.0, 0.0, 1.0] + [0.0] * 1021)) == ()

        updated = store.save_document(
            path, "helios.md", "hash-three",
            (
                KnowledgeChunkInput("Helios has a new ruby shield.", _A),
                KnowledgeChunkInput("Helios has a spare sapphire shield.", _A),
            ),
        )
        assert updated.status == "updated"
        assert updated.document_id == first.document_id
        assert store.get_by_path(path).content_hash == "hash-three"
        with store._connect() as conn:
            rows = conn.execute(
                "SELECT content FROM knowledge_chunks WHERE document_id = %s ORDER BY chunk_index",
                (first.document_id,),
            ).fetchall()
        assert rows == [
            ("Helios has a new ruby shield.",),
            ("Helios has a spare sapphire shield.",),
        ]
        with pytest.raises(RuntimeError, match="PostgreSQL operation failed"):
            store.save_document(
                path, "helios.md", "hash-broken",
                (KnowledgeChunkInput("invalid\x00content", _A),),
            )
        assert store.get_by_path(path).content_hash == "hash-three"
        with store._connect() as conn:
            assert conn.execute(
                "SELECT count(*) FROM knowledge_chunks WHERE document_id = %s",
                (first.document_id,),
            ).fetchone()[0] == 2
    finally:
        store.delete_document(path)
        store.delete_document(other_path)
    with store._connect() as conn:
        remaining = conn.execute(
            "SELECT count(*) FROM knowledge_chunks WHERE document_id = %s",
            (first.document_id,),
        ).fetchone()[0]
    assert remaining == 0


def test_postgres_rejects_invalid_vectors_without_writes(
    knowledge_store: PostgresKnowledgeStore,
) -> None:
    path = f"/tmp/skysun_test_bad_{uuid4().hex}.md"
    with pytest.raises(ValueError, match="1024"):
        knowledge_store.save_document(
            path, "bad.md", "bad-hash", (KnowledgeChunkInput("bad", (1.0, 0.0)),)
        )
    assert knowledge_store.get_by_path(path) is None
