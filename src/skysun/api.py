"""SkySun's HTTP contract, independent of LangChain types."""

import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from skysun.agent import Agent

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1)


class ChatResponse(BaseModel):
    reply: str


def build_api(agent: Agent) -> FastAPI:
    app = FastAPI(title="SkySun", version="0.1.0")

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            return ChatResponse(reply=agent.reply(request.message))
        except Exception as exc:
            logger.exception("SkySun agent failed")
            raise HTTPException(
                status_code=502,
                detail="Agent response unavailable",
            ) from exc

    return app
