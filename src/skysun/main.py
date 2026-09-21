"""Application composition root."""

from fastapi import FastAPI

from skysun.api import build_api
from skysun.config import load_settings
from skysun.conversation import ConversationService, InMemoryConversationStore
from skysun.deepseek import create_deepseek
from skysun.langchain_agent import LangChainAgent


def create_app() -> FastAPI:
    """Build one Agent and one process-local store for this app instance."""
    settings = load_settings()
    model = create_deepseek(settings)
    agent = LangChainAgent(model)
    store = InMemoryConversationStore()
    return build_api(ConversationService(agent, store))
