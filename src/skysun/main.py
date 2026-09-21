"""Application composition root."""

from fastapi import FastAPI

from skysun.api import build_api
from skysun.config import load_settings
from skysun.deepseek import create_deepseek
from skysun.langchain_agent import LangChainAgent


def create_app() -> FastAPI:
    """Build the app at server startup; missing configuration fails clearly."""
    settings = load_settings()
    model = create_deepseek(settings)
    return build_api(LangChainAgent(model))
