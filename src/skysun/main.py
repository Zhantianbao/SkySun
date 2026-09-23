"""Application composition root."""

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import FileResponse

from skysun.api import build_api
from skysun.config import load_settings
from skysun.conversation import ConversationService, ConversationStore
from skysun.deepseek import create_deepseek
from skysun.deepseek_evaluation import DeepSeekClaimReviewer
from skysun.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from skysun.knowledge import KnowledgeStore
from skysun.knowledge_ingestion import KnowledgeIngestionService
from skysun.langchain_agent import LangChainAgent
from skysun.langchain_memory_curator import LangChainMemoryCurator
from skysun.langchain_planner import LangChainTaskPlanner
from skysun.memory import MemoryStore
from skysun.memory_curator import MemoryCurator, MemoryCurationService
from skysun.postgres_conversation import PostgresConversationStore
from skysun.postgres_knowledge import PostgresKnowledgeStore
from skysun.postgres_memory import PostgresMemoryStore
from skysun.postgres_task import PostgresTaskStore
from skysun.postgres_observability import PostgresObservabilityStore
from skysun.observability import ResponseEvaluator
from skysun.postgres_users import PostgresUserStore
from skysun.workflow_agent import WorkflowAgent


class WebStaticFiles(StaticFiles):
    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or scope.get("skysun_api") or "." in Path(path).name:
                raise
            return FileResponse(Path(self.directory) / "index.html")
        return response


def create_app(
    store: ConversationStore | None = None,
    memory_store: MemoryStore | None = None,
    curator: MemoryCurator | None = None,
    knowledge_store: KnowledgeStore | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> FastAPI:
    """Build the app; normal runtime initializes all PostgreSQL stores."""
    settings = load_settings()
    database_url = settings.database_url.get_secret_value()
    users = PostgresUserStore(database_url)
    users.initialize_schema()
    if settings.skysun_admin_token is not None:
        users.bootstrap_admin(settings.skysun_admin_token.get_secret_value())
    if store is None:
        postgres_store = PostgresConversationStore(database_url)
        postgres_store.initialize_schema()
        store = postgres_store
    if memory_store is None:
        postgres_memory = PostgresMemoryStore(database_url)
        postgres_memory.initialize_schema()
        memory_store = postgres_memory
    if knowledge_store is None:
        postgres_knowledge = PostgresKnowledgeStore(database_url)
        postgres_knowledge.initialize_schema()
        knowledge_store = postgres_knowledge
    if embeddings is None:
        embeddings = SentenceTransformerEmbeddingProvider()
    task_store = PostgresTaskStore(database_url)
    task_store.initialize_schema()
    observability = PostgresObservabilityStore(database_url)
    observability.initialize_schema()
    model = create_deepseek(settings)
    direct_agent = LangChainAgent(model, memory_store, knowledge_store, embeddings)
    agent = WorkflowAgent(
        direct_agent, LangChainTaskPlanner(model),
        ("reason", "count_words", "recall_memories", "remember_memory", "forget_memory", "search_knowledge"),
        task_store=task_store,
        observability=observability,
        response_evaluator=ResponseEvaluator(DeepSeekClaimReviewer(model)),
    )
    curator = curator or LangChainMemoryCurator(model)
    curation = MemoryCurationService(curator, memory_store)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        agent.recover_unfinished()
        yield

    app = build_api(ConversationService(agent, store, curation), lifespan=lifespan,
                    users=users, tasks=task_store, observability=observability,
                    memory=memory_store, knowledge=knowledge_store,
                    ingestion=KnowledgeIngestionService(knowledge_store, embeddings))
    web_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    if web_dist.is_dir():
        app.mount("/", WebStaticFiles(directory=web_dist, html=True), name="web")
    return app
