"""PostgreSQL/pgvector implementation of SkySun's document knowledge store."""

import math
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from uuid import uuid4

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector

from skysun.embeddings import EMBEDDING_DIMENSION
from skysun.postgres_users import ensure_user_schema
from skysun.user_identity import private_knowledge_owner
from skysun.knowledge import (
    IngestionResult,
    KnowledgeChunkInput,
    KnowledgeDocument,
    KnowledgeSearchResult,
)

_DOCUMENT_COLUMNS = "id, source_path, source_name, content_hash, created_at, updated_at, owner_id"


def _validate_vector(values: tuple[float, ...]) -> Vector:
    if len(values) != EMBEDDING_DIMENSION or not all(math.isfinite(v) for v in values):
        raise ValueError(f"Knowledge embeddings must contain {EMBEDDING_DIMENSION} finite values")
    return Vector(list(values))


class PostgresKnowledgeStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self, register: bool = True) -> psycopg.Connection:
        try:
            conn = psycopg.connect(self._database_url, connect_timeout=5)
            if register:
                register_vector(conn)
            return conn
        except psycopg.Error:
            raise RuntimeError("SkySun knowledge PostgreSQL connection failed") from None

    @contextmanager
    def _transaction(self, register: bool = True) -> Iterator[psycopg.Connection]:
        try:
            with self._connect(register) as conn:
                yield conn
        except psycopg.Error:
            raise RuntimeError("SkySun knowledge PostgreSQL operation failed") from None

    def initialize_schema(self) -> None:
        """Create only the v0.6 tables; an administrator must enable pgvector first."""
        with self._transaction(register=False) as conn:
            ensure_user_schema(conn)
            present = conn.execute(
                "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
            if present is None:
                raise RuntimeError(
                    "pgvector extension is not enabled in this SkySun database"
                )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_documents (
                    id UUID PRIMARY KEY,
                    source_path TEXT NOT NULL UNIQUE,
                    source_name TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS knowledge_chunks (
                    id UUID PRIMARY KEY,
                    document_id UUID NOT NULL
                        REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL CHECK (content <> ''),
                    embedding VECTOR(1024) NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (document_id, chunk_index)
                )
                """
            )
            conn.execute("""
                ALTER TABLE knowledge_documents ADD COLUMN IF NOT EXISTS owner_id UUID
                REFERENCES skysun_users(user_id)
            """)
            conn.execute("""
                ALTER TABLE knowledge_documents
                DROP CONSTRAINT IF EXISTS knowledge_documents_source_path_key
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS knowledge_documents_global_path_idx
                ON knowledge_documents (source_path) WHERE owner_id IS NULL
            """)
            conn.execute("""
                CREATE UNIQUE INDEX IF NOT EXISTS knowledge_documents_owner_path_idx
                ON knowledge_documents (owner_id, source_path) WHERE owner_id IS NOT NULL
            """)
            dimension = conn.execute("""
                SELECT format_type(atttypid, atttypmod)
                FROM pg_attribute
                WHERE attrelid = 'knowledge_chunks'::regclass AND attname = 'embedding'
            """).fetchone()[0]
            if dimension != f"vector({EMBEDDING_DIMENSION})":
                count = conn.execute("SELECT count(*) FROM knowledge_chunks").fetchone()[0]
                if count:
                    raise RuntimeError(
                        "Existing knowledge vectors use a different dimension; "
                        "this database needs a separate data migration before startup"
                    )
                if dimension != "vector(384)":
                    raise RuntimeError(f"Unsupported knowledge vector dimension: {dimension}")
                conn.execute(
                    f"ALTER TABLE knowledge_chunks ALTER COLUMN embedding "
                    f"TYPE VECTOR({EMBEDDING_DIMENSION})"
                )

    def get_by_path(self, source_path: str) -> KnowledgeDocument | None:
        with self._transaction() as conn:
            row = conn.execute(
                f"SELECT {_DOCUMENT_COLUMNS} FROM knowledge_documents "
                "WHERE source_path = %s AND owner_id IS NOT DISTINCT FROM %s",
                (source_path, private_knowledge_owner()),
            ).fetchone()
        return KnowledgeDocument(*row) if row else None

    def list_documents(self, limit: int = 100) -> tuple[KnowledgeDocument, ...]:
        """List public documents and documents owned by this user."""
        with self._transaction() as conn:
            rows = conn.execute(
                f"SELECT {_DOCUMENT_COLUMNS} FROM knowledge_documents "
                "WHERE owner_id IS NULL OR owner_id = %s "
                "ORDER BY updated_at DESC, id DESC LIMIT %s",
                (private_knowledge_owner(), limit),
            ).fetchall()
        return tuple(KnowledgeDocument(*row) for row in rows)

    def save_document(
        self,
        source_path: str,
        source_name: str,
        content_hash: str,
        chunks: Sequence[KnowledgeChunkInput],
    ) -> IngestionResult:
        """Insert or replace metadata and all chunks in one transaction."""
        if not chunks:
            raise ValueError("Knowledge document must have at least one chunk")
        vectors = tuple(_validate_vector(chunk.embedding) for chunk in chunks)
        user_id = private_knowledge_owner()
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT id, content_hash FROM knowledge_documents "
                "WHERE source_path = %s AND owner_id IS NOT DISTINCT FROM %s FOR UPDATE",
                (source_path, user_id),
            ).fetchone()
            if existing and existing[1] == content_hash:
                return IngestionResult(existing[0], source_path, 0, "unchanged")
            if existing:
                document_id = existing[0]
                conn.execute(
                    """
                    UPDATE knowledge_documents
                    SET source_name = %s, content_hash = %s, updated_at = now()
                    WHERE id = %s
                    """,
                    (source_name, content_hash, document_id),
                )
                conn.execute(
                    "DELETE FROM knowledge_chunks WHERE document_id = %s",
                    (document_id,),
                )
                status = "updated"
            else:
                document_id = uuid4()
                conn.execute(
                    """
                    INSERT INTO knowledge_documents
                    (id, source_path, source_name, content_hash, owner_id)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (document_id, source_path, source_name, content_hash, user_id),
                )
                status = "ingested"
            for index, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
                conn.execute(
                    """
                    INSERT INTO knowledge_chunks
                        (id, document_id, chunk_index, content, embedding)
                    VALUES (%s, %s, %s, %s, %s)
                    """,
                    (uuid4(), document_id, index, chunk.content, vector),
                )
        return IngestionResult(document_id, source_path, len(chunks), status)

    def delete_document(self, source_path: str) -> bool:
        """Delete an explicitly selected document and cascade to its chunks."""
        with self._transaction() as conn:
            row = conn.execute(
                "DELETE FROM knowledge_documents WHERE source_path = %s "
                "AND owner_id IS NOT DISTINCT FROM %s RETURNING id",
                (source_path, private_knowledge_owner()),
            ).fetchone()
        return row is not None

    def search(
        self,
        query_embedding: tuple[float, ...],
        limit: int = 4,
        max_cosine_distance: float = 0.35,
    ) -> tuple[KnowledgeSearchResult, ...]:
        """Return nearest chunks by ascending cosine distance (smaller is better)."""
        if not 1 <= limit <= 5:
            raise ValueError("Knowledge search limit must be between 1 and 5")
        if not 0 <= max_cosine_distance <= 2:
            raise ValueError("Invalid cosine distance threshold")
        vector = _validate_vector(query_embedding)
        with self._transaction() as conn:
            rows = conn.execute(
                """
                SELECT c.content, d.source_name, d.source_path, c.chunk_index,
                       c.embedding <=> %s AS cosine_distance, c.id
                FROM knowledge_chunks AS c
                JOIN knowledge_documents AS d ON d.id = c.document_id
                WHERE c.embedding <=> %s <= %s
                  AND (d.owner_id IS NULL OR d.owner_id = %s)
                ORDER BY cosine_distance ASC, d.source_path, c.chunk_index
                LIMIT %s
                """,
                (vector, vector, max_cosine_distance, private_knowledge_owner(), limit),
            ).fetchall()
        return tuple(KnowledgeSearchResult(*row) for row in rows)
