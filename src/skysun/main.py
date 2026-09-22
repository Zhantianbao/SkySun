"""Application composition root."""

from fastapi import FastAPI

from skysun.api import build_api
from skysun.config import load_settings
from skysun.conversation import ConversationService, ConversationStore
from skysun.deepseek import create_deepseek
from skysun.embeddings import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from skysun.knowledge import KnowledgeStore
from skysun.langchain_agent import LangChainAgent
from skysun.langchain_memory_curator import LangChainMemoryCurator
from skysun.memory import MemoryStore
from skysun.memory_curator import MemoryCurator, MemoryCurationService
from skysun.postgres_conversation import PostgresConversationStore
from skysun.postgres_knowledge import PostgresKnowledgeStore
from skysun.postgres_memory import PostgresMemoryStore


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
    model = create_deepseek(settings)
    agent = LangChainAgent(model, memory_store, knowledge_store, embeddings)
    curator = curator or LangChainMemoryCurator(model)
    curation = MemoryCurationService(curator, memory_store)
    return build_api(ConversationService(agent, store, curation))
