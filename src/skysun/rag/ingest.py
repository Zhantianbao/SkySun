"""Developer CLI for explicit local document ingestion."""

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from skysun.core.config import load_settings
from skysun.rag.embeddings import create_embeddings
from skysun.rag.knowledge_ingestion import KnowledgeIngestionService
from skysun.rag.postgres_knowledge import PostgresKnowledgeStore


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest one UTF-8 Markdown or text document")
    parser.add_argument("path", help="Local .md, .markdown, or .txt path")
    parser.add_argument(
        "--delete", action="store_true", help="Remove this source and its chunks"
    )
    args = parser.parse_args()
    try:
        settings = load_settings()
        store = PostgresKnowledgeStore(settings.database_url.get_secret_value())
        store.initialize_schema()
        if args.delete:
            source_path = str(Path(args.path).expanduser().resolve())
            deleted = store.delete_document(source_path)
            print(json.dumps({"source_path": source_path, "deleted": deleted}))
        else:
            embeddings = create_embeddings(settings)
            result = KnowledgeIngestionService(store, embeddings).ingest(args.path)
            print(json.dumps({
                "document_id": str(result.document_id),
                "source_path": result.source_path,
                "chunk_count": result.chunk_count,
                "status": result.status,
            }))
        return 0
    except (ValidationError, OSError, UnicodeError, ValueError, RuntimeError) as exc:
        parser.exit(1, f"Knowledge ingestion failed: {type(exc).__name__}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
