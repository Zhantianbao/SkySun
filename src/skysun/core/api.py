"""SkySun's HTTP contract, independent of LangChain types."""

import logging
from collections.abc import Callable
from contextlib import AbstractAsyncContextManager
from uuid import UUID

from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, SecretStr

from skysun.memory.conversation import ConversationNotFound, ConversationService
from skysun.device.device_api import device_router
from skysun.device.devices import PostgresDeviceStore
from skysun.device.development import PostgresDevelopmentStore
from skysun.rag.knowledge import KnowledgeStore
from skysun.rag.knowledge_ingestion import KnowledgeIngestionService
from skysun.memory.memory import MemoryStore
from skysun.user.postgres_users import PostgresUserStore
from skysun.evaluation.postgres_observability import PostgresObservabilityStore
from skysun.evaluation.observability import FeedbackRating
from skysun.workflow.task_runtime import TaskStore
from skysun.user.user_identity import (
    Actor, LEGACY_USER_ID, Role, actor_scope, system_scope, normalize_username, normalize_token,
)

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1)
    conversation_id: UUID | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    task_id: str | None = None
    task_status: str | None = None


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    token: SecretStr
    role: Role = Role.USER


class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    token: SecretStr


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
    memory: MemoryStore | None = None,
    knowledge: KnowledgeStore | None = None,
    ingestion: KnowledgeIngestionService | None = None,
    devices: PostgresDeviceStore | None = None,
    development: PostgresDevelopmentStore | None = None,
) -> FastAPI:
    if users is None and not allow_legacy_unauthenticated:
        raise RuntimeError("An authenticated user store is required")
    app = FastAPI(title="SkySun", version="1.7.0", lifespan=lifespan)

    @app.middleware("http")
    async def api_prefix(request: Request, call_next):
        """Keep legacy URLs while the web client uses same-origin /api URLs."""
        if request.scope["path"].startswith("/api/"):
            request.scope["skysun_api"] = True
            request.scope["path"] = request.scope["path"][4:]
            request.scope["raw_path"] = request.scope["path"].encode()
        return await call_next(request)

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

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.post("/login")
    def login(request: LoginRequest) -> dict:
        try:
            username = normalize_username(request.username)
            token = normalize_token(request.token.get_secret_value())
        except ValueError:
            raise HTTPException(status_code=401, detail="姓名或密码错误") from None
        user = users.authenticate(token) if users is not None else None
        if user is None or user.username != username:
            raise HTTPException(status_code=401, detail="姓名或密码错误")
        return {"user_id": str(user.user_id), "username": user.username, "role": user.role.value}

    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, actor: Actor = Depends(identify)) -> ChatResponse:
        try:
            with actor_scope(actor):
                conversation_id, reply = conversations.chat(
                    request.message,
                    str(request.conversation_id) if request.conversation_id else None,
                )
                task = tasks.latest_for_conversation(UUID(conversation_id)) if tasks else None
            return ChatResponse(conversation_id=conversation_id, reply=reply,
                                task_id=str(task.task_id) if task else None,
                                task_status=task.status.value if task else None)
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

    # 用户系统请求令牌验证。
    @app.get("/me")
    def me(actor: Actor = Depends(identify)) -> dict:
        if users is None:
            return {"user_id": str(actor.user_id), "username": "legacy_admin", "role": actor.role.value}
        user = users.get(actor.user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return {"user_id": str(user.user_id), "username": user.username, "role": user.role.value}

    @app.get("/conversations/{conversation_id}/messages")
    def conversation_messages(conversation_id: UUID,
                              actor: Actor = Depends(identify)) -> list[dict]:
        try:
            with actor_scope(actor):
                messages = conversations.history(str(conversation_id))
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        return [{"role": item.role, "content": item.content} for item in messages]

    @app.get("/conversations")
    def conversation_list(limit: int = Query(default=50, ge=1, le=100),
                          actor: Actor = Depends(identify)) -> list[dict]:
        with actor_scope(actor):
            rows = conversations.list_conversations(limit)
        return [{"conversation_id": row.conversation_id, "title": row.title,
                 "created_at": row.created_at.isoformat(),
                 "updated_at": row.updated_at.isoformat()} for row in rows]

    @app.get("/memory")
    def personal_memories(limit: int = Query(default=20, ge=1, le=20),
                          actor: Actor = Depends(identify)) -> list[dict]:
        if memory is None:
            raise HTTPException(status_code=503, detail="Memory store unavailable")
        with actor_scope(actor):
            rows = memory.list_memories(limit)
        return [{"id": str(row.id), "key": row.key, "content": row.content,
                 "category": row.category,
                 "source_conversation_id": str(row.source_conversation_id)
                 if row.source_conversation_id else None,
                 "created_at": row.created_at.isoformat(),
                 "updated_at": row.updated_at.isoformat()} for row in rows]

    @app.get("/knowledge/documents")
    def knowledge_documents(limit: int = Query(default=100, ge=1, le=200),
                            actor: Actor = Depends(identify)) -> list[dict]:
        if knowledge is None:
            raise HTTPException(status_code=503, detail="Knowledge store unavailable")
        with actor_scope(actor):
            rows = knowledge.list_documents(limit)
        return [{"id": str(row.id), "source_name": row.source_name,
                 "source_path": row.source_path,
                 "scope": "global" if row.owner_id is None else "personal",
                 "updated_at": row.updated_at.isoformat()} for row in rows]

    @app.post("/knowledge/documents", status_code=201)
    def upload_document(file: UploadFile = File(...),
                        actor: Actor = Depends(identify)) -> dict:
        if ingestion is None:
            raise HTTPException(status_code=503, detail="Knowledge ingestion unavailable")
        filename = (file.filename or "").replace("\\", "/").split("/")[-1]
        if not filename or len(filename) > 120 or any(ord(char) < 32 or ord(char) == 127
                                                       for char in filename) or not filename.lower().endswith(
                (".md", ".markdown", ".txt")):
            raise HTTPException(status_code=422, detail="Upload a .md, .markdown, or .txt file")
        try:
            content = file.file.read(2 * 1024 * 1024 + 1)
        finally:
            file.file.close()
        if len(content) > 2 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Document exceeds 2 MiB limit")
        try:
            text = content.decode("utf-8")
            source_path = f"upload://{actor.user_id}/{filename}"
            with actor_scope(actor):
                result = ingestion.ingest_text(source_path, filename, text)
        except UnicodeDecodeError as exc:
            raise HTTPException(status_code=422, detail="Document must be UTF-8 text") from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"document_id": str(result.document_id), "source_path": result.source_path,
                "chunk_count": result.chunk_count, "status": result.status}

    @app.get("/tasks")
    def task_list(limit: int = Query(default=50, ge=1, le=200),
                  actor: Actor = Depends(identify)) -> list[dict]:
        if tasks is None:
            raise HTTPException(status_code=503, detail="Task store unavailable")
        with actor_scope(actor):
            rows = tasks.list_tasks(limit)
        return [{"task_id": str(row.task_id), "conversation_id": str(row.conversation_id)
                 if row.conversation_id else None, "goal": row.goal,
                 "status": row.status.value, "created_at": row.created_at.isoformat(),
                 "updated_at": row.updated_at.isoformat(),
                 "steps": [{"step_id": step.step_id, "state": step.state.value}
                           for step in row.steps]} for row in rows]

    @app.post("/admin/users", status_code=201)
    def create_user(request: UserCreate, actor: Actor = Depends(identify)) -> dict:
        admin(actor)
        if users is None:
            raise HTTPException(status_code=503, detail="User store unavailable")
        try:
            user, token = users.create_user(
                request.username, request.role, token=request.token.get_secret_value(),
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            if getattr(exc, "sqlstate", None) == "23505":
                raise HTTPException(status_code=409, detail="姓名或邀请码已被使用") from exc
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

    @app.get("/tasks/{task_id}/activity")
    def task_activity(task_id: UUID, actor: Actor = Depends(identify)) -> list[dict]:
        if observability is None:
            raise HTTPException(status_code=503, detail="Observability unavailable")
        try:
            with actor_scope(actor):
                rows = observability.list_traces(task_id, 100)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="Task not found") from exc
        return [{"stage": row.stage, "status": row.status,
                 "started_at": row.started_at.isoformat(),
                 "duration_ms": row.duration_ms} for row in rows]

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

    if devices is not None:
        app.include_router(device_router(devices, identify, development))
    return app
