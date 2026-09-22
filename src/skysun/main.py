"""Application composition root."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from skysun.api import build_api
from skysun.config import load_settings
from skysun.conversation import ConversationService, ConversationStore
from skysun.deepseek import create_deepseek
from skysun.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from skysun.knowledge import KnowledgeStore
from skysun.langchain_agent import LangChainAgent
from skysun.langchain_memory_curator import LangChainMemoryCurator
from skysun.langchain_planner import LangChainTaskPlanner
from skysun.memory import MemoryStore
from skysun.memory_curator import MemoryCurator, MemoryCurationService
from skysun.postgres_conversation import PostgresConversationStore
from skysun.postgres_knowledge import PostgresKnowledgeStore
from skysun.postgres_memory import PostgresMemoryStore
from skysun.postgres_task import PostgresTaskStore
from skysun.postgres_users import PostgresUserStore
from skysun.workflow_agent import WorkflowAgent


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
    model = create_deepseek(settings)
    direct_agent = LangChainAgent(model, memory_store, knowledge_store, embeddings)
    agent = WorkflowAgent(
        direct_agent, LangChainTaskPlanner(model),
        ("reason", "count_words", "recall_memories", "remember_memory", "forget_memory", "search_knowledge"),
        task_store=task_store,
    )
    curator = curator or LangChainMemoryCurator(model)
    curation = MemoryCurationService(curator, memory_store)
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        agent.recover_unfinished()
        yield

    return build_api(ConversationService(agent, store, curation), lifespan=lifespan,
                     users=users, tasks=task_store)
