"""Application composition root."""

from fastapi import FastAPI

from skysun.api import build_api
from skysun.config import load_settings
from skysun.conversation import ConversationService, ConversationStore
from skysun.deepseek import create_deepseek
from skysun.langchain_agent import LangChainAgent
from skysun.langchain_memory_curator import LangChainMemoryCurator
from skysun.memory import MemoryStore
from skysun.memory_curator import MemoryCurator, MemoryCurationService
from skysun.postgres_conversation import PostgresConversationStore
from skysun.postgres_memory import PostgresMemoryStore


def create_app(
    store: ConversationStore | None = None,
    memory_store: MemoryStore | None = None,
    curator: MemoryCurator | None = None,
) -> FastAPI:
    """Build the app; normal runtime initializes both PostgreSQL stores."""
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
    model = create_deepseek(settings)
    agent = LangChainAgent(model, memory_store)
    curator = curator or LangChainMemoryCurator(model)
    curation = MemoryCurationService(curator, memory_store)
    return build_api(ConversationService(agent, store, curation))
