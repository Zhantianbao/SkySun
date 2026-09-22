"""SkySun's HTTP contract, independent of LangChain types."""

import logging
from uuid import UUID

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from skysun.conversation import ConversationNotFound, ConversationService

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


def build_api(conversations: ConversationService) -> FastAPI:
    app = FastAPI(title="SkySun", version="0.6.0")

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        try:
            conversation_id, reply = conversations.chat(
                request.message,
                str(request.conversation_id) if request.conversation_id else None,
            )
            return ChatResponse(conversation_id=conversation_id, reply=reply)
        except ConversationNotFound as exc:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found",
            ) from exc
        except Exception as exc:
            logger.exception("SkySun agent failed")
            raise HTTPException(
                status_code=502,
                detail="Agent response unavailable",
            ) from exc

    return app
