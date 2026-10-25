"""SkySun's HTTP contract, independent of LangChain types."""
# 本文件负责核心服务中的 api 相关流程。

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


# 以下集中定义 ChatRequest：请求模型相关的数据和操作。
class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    message: str = Field(min_length=1)
    conversation_id: UUID | None = None


# 以下集中定义 ChatResponse：响应模型相关的数据和操作。
class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    task_id: str | None = None
    task_status: str | None = None


# 以下集中定义 UserCreate：数据结构或组件相关的数据和操作。
class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    token: SecretStr
    role: Role = Role.USER


# 以下集中定义 LoginRequest：请求模型相关的数据和操作。
class LoginRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str
    token: SecretStr


# 以下集中定义 FeedbackRequest：请求模型相关的数据和操作。
class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    rating: FeedbackRating
    comment: str | None = Field(default=None, max_length=1000)


# 调用位置：src/skysun/core/main.py 的 create_app、tests/test_api.py 的 make_client、tests/test_api.py 的 test_api_requires_identity_store_by_default 等处调用。
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
    """注册 HTTP 路由、鉴权依赖和请求处理函数。"""
    if users is None and not allow_legacy_unauthenticated:
        raise RuntimeError("An authenticated user store is required")
    app = FastAPI(title="SkySun", version="1.7.1", lifespan=lifespan)

    # 调用位置：FastAPI 的 MIDDLEWARE http 请求触发。
    @app.middleware("http")
    async def api_prefix(request: Request, call_next):
        """将前端使用的同源接口前缀转换为现有路由路径。"""
        if request.scope["path"].startswith("/api/"):
            request.scope["skysun_api"] = True
            request.scope["path"] = request.scope["path"][4:]
            request.scope["raw_path"] = request.scope["path"].encode()
        return await call_next(request)

    # 调用位置：下方所有声明 Depends(identify) 的受保护路由由 FastAPI 调用。
    def identify(authorization: str | None = Header(default=None)) -> Actor:
        """从身份令牌识别当前用户，拒绝未登录或无效令牌。"""
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

    # 调用位置：src/skysun/core/api.py 的 build_api.create_user、src/skysun/core/api.py 的 build_api.summary、src/skysun/core/api.py 的 build_api.metrics 等处调用。
    def admin(actor: Actor) -> None:
        """检查当前用户是否具有管理员权限。"""
        if actor.role is not Role.ADMIN:
            raise HTTPException(status_code=403, detail="Administrator access required")

    # 以下路由处理健康检查、登录和聊天；聊天请求先识别身份，再在用户范围内运行会话服务。
    # 调用位置：FastAPI 的 GET /health 请求触发。
    @app.get("/health")
    def health() -> dict:
        """返回服务健康状态。"""
        return {"status": "ok"}

    # 调用位置：FastAPI 的 POST /login 请求触发。
    @app.post("/login")
    def login(request: LoginRequest) -> dict:
        """校验姓名和令牌并返回登录用户信息。"""
        try:
            username = normalize_username(request.username)
            token = normalize_token(request.token.get_secret_value())
        except ValueError:
            raise HTTPException(status_code=401, detail="姓名或密码错误") from None
        user = users.authenticate(token) if users is not None else None
        if user is None or user.username != username:
            raise HTTPException(status_code=401, detail="姓名或密码错误")
        return {"user_id": str(user.user_id), "username": user.username, "role": user.role.value}

    # 调用位置：FastAPI 的 POST /chat 请求触发。
    @app.post("/chat", response_model=ChatResponse)
    def chat(request: ChatRequest, actor: Actor = Depends(identify)) -> ChatResponse:
        """在当前用户范围内处理聊天并返回回复及任务状态。"""
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
    # 调用位置：FastAPI 的 GET /me 请求触发。
    @app.get("/me")
    def me(actor: Actor = Depends(identify)) -> dict:
        """读取当前登录用户的信息。"""
        if users is None:
            return {"user_id": str(actor.user_id), "username": "legacy_admin", "role": actor.role.value}
        user = users.get(actor.user_id)
        if user is None:
            raise HTTPException(status_code=401, detail="Invalid authentication")
        return {"user_id": str(user.user_id), "username": user.username, "role": user.role.value}

    # 以下读取会话、记忆和知识数据，统一通过当前身份限定可见范围。
    # 调用位置：FastAPI 的 GET /conversations/{conversation_id}/messages 请求触发。
    @app.get("/conversations/{conversation_id}/messages")
    def conversation_messages(conversation_id: UUID,
                              actor: Actor = Depends(identify)) -> list[dict]:
        """读取指定会话的消息历史。"""
        try:
            with actor_scope(actor):
                messages = conversations.history(str(conversation_id))
        except ConversationNotFound as exc:
            raise HTTPException(status_code=404, detail="Conversation not found") from exc
        return [{"role": item.role, "content": item.content} for item in messages]

    # 调用位置：FastAPI 的 GET /conversations 请求触发。
    @app.get("/conversations")
    def conversation_list(limit: int = Query(default=50, ge=1, le=100),
                          actor: Actor = Depends(identify)) -> list[dict]:
        """列出当前用户的会话。"""
        with actor_scope(actor):
            rows = conversations.list_conversations(limit)
        return [{"conversation_id": row.conversation_id, "title": row.title,
                 "created_at": row.created_at.isoformat(),
                 "updated_at": row.updated_at.isoformat()} for row in rows]

    # 调用位置：FastAPI 的 GET /memory 请求触发。
    @app.get("/memory")
    def personal_memories(limit: int = Query(default=20, ge=1, le=20),
                          actor: Actor = Depends(identify)) -> list[dict]:
        """列出当前用户的长期记忆。"""
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

    # 调用位置：FastAPI 的 GET /knowledge/documents 请求触发。
    @app.get("/knowledge/documents")
    def knowledge_documents(limit: int = Query(default=100, ge=1, le=200),
                            actor: Actor = Depends(identify)) -> list[dict]:
        """列出可见的知识文档。"""
        if knowledge is None:
            raise HTTPException(status_code=503, detail="Knowledge store unavailable")
        with actor_scope(actor):
            rows = knowledge.list_documents(limit)
        return [{"id": str(row.id), "source_name": row.source_name,
                 "source_path": row.source_path,
                 "scope": "global" if row.owner_id is None else "personal",
                 "updated_at": row.updated_at.isoformat()} for row in rows]

    # 上传流程校验文件，再交给入库服务完成分段、向量化和保存。
    # 调用位置：FastAPI 的 POST /knowledge/documents 请求触发。
    @app.post("/knowledge/documents", status_code=201)
    def upload_document(file: UploadFile = File(...),
                        actor: Actor = Depends(identify)) -> dict:
        """接收文档并交给知识入库服务。"""
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

    # 调用位置：FastAPI 的 GET /tasks 请求触发。
    @app.get("/tasks")
    def task_list(limit: int = Query(default=50, ge=1, le=200),
                  actor: Actor = Depends(identify)) -> list[dict]:
        """列出当前用户的任务。"""
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

    # 以下管理接口先核对管理员角色，再提供用户创建与系统观测数据。
    # 调用位置：FastAPI 的 POST /admin/users 请求触发。
    @app.post("/admin/users", status_code=201)
    def create_user(request: UserCreate, actor: Actor = Depends(identify)) -> dict:
        """创建新用户并保存令牌摘要。"""
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

    # 调用位置：FastAPI 的 GET /admin/summary 请求触发。
    @app.get("/admin/summary")
    def summary(actor: Actor = Depends(identify)) -> dict:
        """返回管理端的汇总信息。"""
        admin(actor)
        if users is None or tasks is None:
            raise HTTPException(status_code=503, detail="System stores unavailable")
        with system_scope():
            unfinished = tasks.list_unfinished()
        return {"user_count": len(users.list_users()), "unfinished_task_count": len(unfinished)}

    # 调用位置：FastAPI 的 GET /admin/metrics 请求触发。
    @app.get("/admin/metrics")
    def metrics(actor: Actor = Depends(identify)) -> dict:
        """汇总任务和观测指标。"""
        admin(actor)
        if observability is None:
            raise HTTPException(status_code=503, detail="Observability unavailable")
        with actor_scope(actor):
            return observability.metrics()

    # 调用位置：FastAPI 的 GET /admin/traces 请求触发。
    @app.get("/admin/traces")
    def traces(limit: int = Query(default=100, ge=1, le=200),
               task_id: UUID | None = None, actor: Actor = Depends(identify)) -> list[dict]:
        """列出最近的执行轨迹。"""
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

    # 调用位置：FastAPI 的 GET /admin/evaluations 请求触发。
    @app.get("/admin/evaluations")
    def evaluations(limit: int = Query(default=100, ge=1, le=200),
                    actor: Actor = Depends(identify)) -> list[dict]:
        """列出回答评估记录。"""
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

    # 以下任务详情、活动和反馈接口均先按当前用户读取任务。
    # 调用位置：FastAPI 的 GET /tasks/{task_id} 请求触发。
    @app.get("/tasks/{task_id}")
    def task_detail(task_id: UUID, actor: Actor = Depends(identify)) -> dict:
        """读取指定任务的详情。"""
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

    # 调用位置：FastAPI 的 GET /tasks/{task_id}/events 请求触发。
    @app.get("/tasks/{task_id}/events")
    def task_events(task_id: UUID, actor: Actor = Depends(identify)) -> list[dict]:
        """读取指定任务的事件记录。"""
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

    # 调用位置：FastAPI 的 GET /tasks/{task_id}/activity 请求触发。
    @app.get("/tasks/{task_id}/activity")
    def task_activity(task_id: UUID, actor: Actor = Depends(identify)) -> list[dict]:
        """合并任务步骤与事件供页面展示。"""
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

    # 调用位置：FastAPI 的 POST /tasks/{task_id}/feedback 请求触发。
    @app.post("/tasks/{task_id}/feedback")
    def feedback(task_id: UUID, request: FeedbackRequest,
                 actor: Actor = Depends(identify)) -> dict:
        """保存当前用户对回答的反馈。"""
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
