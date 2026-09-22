"""SkySun's HTTP contract, independent of LangChain types."""

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from uuid import UUID

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from skysun.conversation import ConversationNotFound, ConversationService
from skysun.postgres_users import PostgresUserStore
from skysun.postgres_observability import PostgresObservabilityStore
from skysun.observability import FeedbackRating
from skysun.task_runtime import TaskStore
from skysun.user_identity import Actor, LEGACY_USER_ID, Role, actor_scope, system_scope

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    role: Role = Role.USER


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=1000)


def build_api(
    conversations: ConversationService,
    lifespan: Callable[[FastAPI], AbstractAsyncContextManager[None]] | None = None,
    users: PostgresUserStore | None = None,
    tasks: TaskStore | None = None,
    observability: PostgresObservabilityStore | None = None,
    allow_legacy_unauthenticated: bool = False,
) -> FastAPI:
    if users is None and not allow_legacy_unauthenticated:
        raise RuntimeError("An authenticated user store is required")
    app = FastAPI(title="SkySun", version="1.2.0", lifespan=lifespan)

    def identify(authorization: str | None = Header(default=None)) -> Actor:
        if users is None:
            return Actor(LEGACY_USER_ID, Role.USER)
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Authentication required",
                                headers={"WWW-Authenticate": "Bearer"})
        token = authorization[7:]
        user = users.authenticate(token)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid authentication",
                                headers={"WWW-Authenticate": "Bearer"})
        return Actor(user.user_id, user.role)

    def admin(actor: Actor) -> None:
        if actor.role is not Role.ADMIN:
            raise HTTPException(status_code=403, detail="Administrator access required")

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, actor: Actor = Depends(identify)) -> ChatResponse:
        try:
            with actor_scope(actor):
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

    @app.get("/me")
    def me(actor: Actor = Depends(identify)) -> dict:
        if users is None:
            return {"user_id": str(actor.user_id), "username": "legacy_admin", "role": actor.role.value}
        user = users.get(actor.user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return {"user_id": str(user.user_id), "username": user.username, "role": user.role.value}

    @app.post("/admin/users", status_code=201)
    def create_user(request: UserCreate, actor: Actor = Depends(identify)) -> dict:
        admin(actor)
        if users is None:
            raise HTTPException(status_code=503, detail="User store unavailable")
        try:
            user, token = users.create_user(request.username, request.role)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            if getattr(exc, "sqlstate", None) == "23505":
                raise HTTPException(status_code=409, detail="Username already exists") from exc
            raise
        return {"user_id": str(user.user_id), "username": user.username,
                "role": user.role.value, "token": token}

    @app.get("/admin/summary")
    def summary(actor: Actor = Depends(identify)) -> dict:
        admin(actor)
        if users is None or tasks is None:
            raise HTTPException(status_code=503, detail="System stores unavailable")
        with system_scope():
            unfinished = tasks.list_unfinished()
        return {"user_count": len(users.list_users()), "unfinished_task_count": len(unfinished)}

    @app.get("/admin/metrics")
    def metrics(actor: Actor = Depends(identify)) -> dict:
        admin(actor)
        if observability is None:
            raise HTTPException(status_code=503, detail="Observability unavailable")
        with actor_scope(actor):
            return observability.metrics()

    @app.get("/admin/traces")
    def traces(limit: int = Query(default=100, ge=1, le=200),
               task_id: UUID | None = None, actor: Actor = Depends(identify)) -> list[dict]:
        admin(actor)
        if observability is None:
            raise HTTPException(status_code=503, detail="Observability unavailable")
        with system_scope():
            rows = observability.list_traces(task_id, limit)
        return [{"trace_id": str(row.trace_id), "task_id": str(row.task_id),
                 "user_id": str(row.user_id), "stage": row.stage, "status": row.status,
                 "started_at": row.started_at.isoformat(),
                 "finished_at": row.finished_at.isoformat() if row.finished_at else None,
                 "duration_ms": row.duration_ms, "metadata": row.metadata} for row in rows]

    @app.get("/admin/evaluations")
    def evaluations(limit: int = Query(default=100, ge=1, le=200),
                    actor: Actor = Depends(identify)) -> list[dict]:
        admin(actor)
        if observability is None:
            raise HTTPException(status_code=503, detail="Observability unavailable")
        with system_scope():
            rows = observability.list_evaluations(limit)
        return [{"task_id": str(row.task_id), "user_id": str(row.user_id),
                 "score": row.score, "status": row.status.value,
                 "reasons": row.reasons, "grounding_correct": row.grounding_correct,
                 "complete": row.complete, "unsupported_claims": row.unsupported_claims,
                 "created_at": row.created_at.isoformat() if row.created_at else None}
                for row in rows]

    @app.get("/tasks/{task_id}")
    def task_detail(task_id: UUID, actor: Actor = Depends(identify)) -> dict:
        if tasks is None:
            raise HTTPException(status_code=503, detail="Task store unavailable")
        try:
            with actor_scope(actor):
                task = tasks.get(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Task not found") from exc
        return {"task_id": str(task.task_id), "user_id": str(task.user_id),
                "goal": task.goal, "status": task.status.value,
                "steps": [{"step_id": step.step_id, "state": step.state.value}
                          for step in task.steps]}

    @app.get("/tasks/{task_id}/events")
    def task_events(task_id: UUID, actor: Actor = Depends(identify)) -> list[dict]:
        if tasks is None:
            raise HTTPException(status_code=503, detail="Task store unavailable")
        try:
            with actor_scope(actor):
                events = tasks.events(task_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Task not found") from exc
        return [{"event_type": event.event_type, "step_id": event.step_id,
                 "tool_name": event.tool_name, "user_id": str(event.user_id),
                 "created_at": event.created_at.isoformat() if event.created_at else None}
                for event in events]

    @app.post("/tasks/{task_id}/feedback")
    def feedback(task_id: UUID, request: FeedbackRequest,
                 actor: Actor = Depends(identify)) -> dict:
        if observability is None:
            raise HTTPException(status_code=503, detail="Observability unavailable")
        try:
            with actor_scope(actor):
                saved = observability.save_feedback(task_id, request.rating, request.comment)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Task not found") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"feedback_id": str(saved.feedback_id), "task_id": str(saved.task_id),
                "rating": saved.rating.value, "created_at": saved.created_at.isoformat()}

    return app
