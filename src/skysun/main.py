"""Application composition root."""

from fastapi import FastAPI

from skysun.api import build_api
from skysun.config import load_settings
from skysun.conversation import ConversationService, ConversationStore
from skysun.deepseek import create_deepseek
from skysun.langchain_agent import LangChainAgent
from skysun.postgres_conversation import PostgresConversationStore


def create_app(store: ConversationStore | None = None) -> FastAPI:
    """Build the app; default runtime uses PostgreSQL with idempotent schema setup."""
    settings = load_settings()
    if store is None:
        postgres_store = PostgresConversationStore(
            settings.database_url.get_secret_value()
        )
        postgres_store.initialize_schema()
        store = postgres_store
    model = create_deepseek(settings)
    agent = LangChainAgent(model)
    return build_api(ConversationService(agent, store))
