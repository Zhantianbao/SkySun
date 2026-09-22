# SkySun v0.9

SkySun is a personal FastAPI agent with separate conversation history, personal memory, and document knowledge.

## Versions

- [v0.1.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.1.0) — Agent core, chat API, and word-count tool.
- [v0.2.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.2.0) — Conversation history within a running process.
- [v0.3.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.3.0) — PostgreSQL-backed conversation persistence.
- [v0.4.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.4.0) — Durable personal memory and memory tools.
- [v0.5.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.5.0) — Automatic memory curation.
- [v0.6.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.6.0) — Persistent document knowledge and vector RAG.
- [v0.7.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.7.0) — Grounded Knowledge Reasoning.
- [v0.8.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.8.0) — Agent Memory & Tool Reliability.
- [v0.9.0](https://github.com/Zhantianbao/SkySun/releases/tag/v0.9.0) — Agent Workflow & Task Planning (this version).

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

Chunking targets 1,000 characters with 150 characters of overlap and prefers newline boundaries. Search performs exact pgvector cosine-distance ordering, returns at most four chunks by default, and filters results above an initial 0.35 distance. This threshold is only a retrieval heuristic: semantic similarity does not prove an answer. There is no ANN index or reranker.

## Grounded Knowledge Reasoning

When the Agent searches stored documents, SkySun asks an evidence evaluator whether the retrieved passages support the whole question. `SUFFICIENT` allows an answer with citations derived from the selected chunks' real filename, path, and index. `INSUFFICIENT` or `AMBIGUOUS` allows one query rewrite and one more retrieval; after at most two retrieval attempts, SkySun stops and reports that evidence is lacking. Invalid evaluator output, unknown or duplicate chunk identifiers, and evaluator errors fail closed without authorizing an answer or citation. The Agent remains responsible for choosing when document search is needed; general and personal-memory answers use their existing paths. The evaluator is model-based and may still make mistakes, and the distance threshold may omit useful evidence.

## Agent Memory & Tool Reliability

Raw conversation turns remain in conversation history. After a successful turn, the memory curator proposes separate candidates with content, category, importance, and source conversation. The application rejects low-value or duplicate candidates before persisting accepted facts as confirmed personal memory. The curator accepts limited JSON formatting variations, retries one parser failure, and discards output that still cannot be validated. Existing explicit remember and forget commands still use the same memory API. Candidate proposals are transient; confirmed memories persist in PostgreSQL. Memory retrieval labels its results as personal memory, separate from document knowledge and citations.

Agent tools now return structured, sanitized errors for invalid arguments and execution failures. Read-only memory retrieval retries once for explicitly transient failures; write tools do not retry automatically. Repeated reads in one turn reuse the same failure. When a tool failure remains unresolved, the Agent returns an explicit safe failure response.

## Agent Workflow & Task Planning

Simple requests still use the direct Agent path. Requests with explicit multi-step instructions use a bounded DeepSeek-generated plan of at most five ordered steps. Each step records its ID, required capabilities, dependencies, result, tool status, and state (`PENDING`, `RUNNING`, `SUCCESS`, `FAILED`, or `SKIPPED`). Required-step failure stops later steps; the workflow layer uses the existing Agent tools and does not add another retry loop.

Before returning a workflow answer, SkySun verifies that required steps completed, named tools were actually used, and document-search steps have sufficient grounded evidence and validated sources. Missing evidence or unresolved failures produce a safe response. Task and step IDs and workflow status are available in execution metadata and concise logs; the HTTP API remains unchanged. Planning is selected by a conservative multi-step phrase heuristic, so some complex requests may still use the direct Agent path.


## Current WSL pgvector installation

This environment uses pgvector 0.8.6 built from its tagged source against Ubuntu PostgreSQL 14.24 headers. The only installed system files are `vector.so`, `vector.control`, and `vector--0.8.6.sql` in PostgreSQL 14's standard extension directories. The extension is enabled only in `skysun` and `skysun_test`. No PostgreSQL configuration or package repository was changed. The PGDG binary package was not used because its package metadata conflicts with Ubuntu's PostgreSQL 14 JIT provider.
