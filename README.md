# SkySun v0.6

SkySun is a personal FastAPI agent with separate conversation history, personal memory, and document knowledge.

## Local setup

Use Python 3.11 and the project virtual environment. Install a CPU PyTorch wheel first, then install the project:

```bash
python -m pip install 'torch==2.7.1+cpu' --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[test]'
```

PostgreSQL 14 must have the pgvector server extension installed. Enable `vector` only in the SkySun development and test databases as a database administrator. Set `DATABASE_URL` and `DEEPSEEK_API_KEY` locally using `.env.example` as a template; never commit `.env`.

Start the API:

```bash
python -m uvicorn skysun.main:create_app --factory
```

## Explicit document ingestion

Only UTF-8 `.md`, `.markdown`, and `.txt` files are supported. Files are never indexed automatically.

```bash
python -m skysun.ingest /absolute/path/to/notes.md
python -m skysun.ingest --delete /absolute/path/to/notes.md
```

Re-ingesting an unchanged file is a no-op. Re-ingesting a changed file atomically replaces its chunks. Deleting a document removes its chunks; it does not affect personal memory or conversations.

The first embedding use downloads `intfloat/multilingual-e5-small` into the normal Hugging Face cache outside the repository. SkySun loads it once per application process on CPU. E5 input uses `passage:` for chunks and `query:` for searches; vectors are normalized to 384 dimensions.

Chunking targets 1,000 characters with 150 characters of overlap and prefers newline boundaries. Search performs exact pgvector cosine-distance ordering, returns at most four chunks by default, and filters results above an initial 0.35 distance. This threshold is a starting heuristic and can miss useful passages or admit unrelated ones; a returned passage is not proof that it answers the question. There is no ANN index or reranker in v0.6. The Agent invokes document search as a tool only when a question calls for stored document knowledge. Selected retrieved passages are sent to the configured DeepSeek model for answering; embeddings are computed locally.


## Current WSL pgvector installation

This environment uses pgvector 0.8.6 built from its tagged source against Ubuntu PostgreSQL 14.24 headers. The only installed system files are `vector.so`, `vector.control`, and `vector--0.8.6.sql` in PostgreSQL 14's standard extension directories. The extension is enabled only in `skysun` and `skysun_test`. No PostgreSQL configuration or package repository was changed. The PGDG binary package was not used because its package metadata conflicts with Ubuntu's PostgreSQL 14 JIT provider.
