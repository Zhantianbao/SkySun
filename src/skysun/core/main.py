"""Application composition root."""
# 本文件负责核心服务中的 main 相关流程。

import asyncio
from contextlib import suppress
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.responses import FileResponse

from skysun.core.api import build_api
from skysun.core.config import load_settings
from skysun.memory.conversation import ConversationService, ConversationStore
from skysun.core.deepseek import create_deepseek
from skysun.evaluation.deepseek_evaluation import DeepSeekClaimReviewer
from skysun.device.devices import PostgresDeviceStore
from skysun.device.development import PostgresDevelopmentStore
from skysun.rag.embeddings import EmbeddingProvider, create_embeddings
from skysun.rag.knowledge import KnowledgeStore
from skysun.rag.knowledge_ingestion import KnowledgeIngestionService
from skysun.core.langchain_agent import LangChainAgent
from skysun.memory.langchain_memory_curator import LangChainMemoryCurator
from skysun.workflow.langchain_planner import LangChainTaskPlanner
from skysun.memory.memory import MemoryStore
from skysun.memory.memory_curator import MemoryCurator, MemoryCurationService
from skysun.memory.postgres_conversation import PostgresConversationStore
from skysun.rag.postgres_knowledge import PostgresKnowledgeStore
from skysun.memory.postgres_memory import PostgresMemoryStore
from skysun.workflow.postgres_task import PostgresTaskStore
from skysun.evaluation.postgres_observability import PostgresObservabilityStore
from skysun.evaluation.observability import ResponseEvaluator
from skysun.user.postgres_users import PostgresUserStore
from skysun.workflow.workflow_agent import WorkflowAgent


# 以下集中定义 WebStaticFiles：数据结构或组件相关的数据和操作。
class WebStaticFiles(StaticFiles):
    # 调用位置：Starlette StaticFiles 处理静态资源请求时调用；create_app 挂载本类。
    async def get_response(self, path: str, scope):
        """返回静态文件；前端页面路径缺失时回退到 index.html。"""
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code != 404 or scope.get("skysun_api") or "." in Path(path).name:
                raise
            return FileResponse(Path(self.directory) / "index.html")
        return response


# 调用位置：Dockerfile.backend 的 uvicorn --factory 启动入口调用；应用组装测试也直接调用。
def create_app(
    store: ConversationStore | None = None,
    memory_store: MemoryStore | None = None,
    curator: MemoryCurator | None = None,
    knowledge_store: KnowledgeStore | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> FastAPI:
    """读取配置并初始化数据库存储，组装智能代理、会话服务和后端接口。"""
    settings = load_settings()

    #初始化数据库
    database_url = settings.database_url.get_secret_value()
    usersStore = PostgresUserStore(database_url)
    usersStore.initialize_schema()
    if settings.skysun_admin_token is not None:
        usersStore.bootstrap_admin(settings.skysun_admin_token.get_secret_value())
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
        embeddings = create_embeddings(settings)
    task_store = PostgresTaskStore(database_url)
    task_store.initialize_schema()
    observability = PostgresObservabilityStore(database_url)
    observability.initialize_schema()
    devices = PostgresDeviceStore(database_url)
    devices.initialize_schema()
    development = PostgresDevelopmentStore(database_url)
    development.initialize_schema()
    # 持久化存储准备好后，组装直接 Agent、工作流、记忆整理和评估链。
    model = create_deepseek(settings)
    direct_agent = LangChainAgent(model, memory_store, knowledge_store, embeddings)
    agent = WorkflowAgent(
        direct_agent, LangChainTaskPlanner(model),
        ("reason", "count_words", "recall_memories", "remember_memory", "forget_memory", "search_knowledge"),
        task_store=task_store,
        observability=observability,
        response_evaluator=ResponseEvaluator(DeepSeekClaimReviewer(model)),
    )
    curator = curator or LangChainMemoryCurator(model)
    curation = MemoryCurationService(curator, memory_store)
    # 调用位置：FastAPI 启动和关闭时调用。
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """在应用启动时初始化存储，并在关闭时清理资源。"""
        agent.recover_unfinished()
        development.cleanup()
        # 调用位置：src/skysun/core/main.py 的 create_app.lifespan 调用。
        async def prune_expired():
            """定期清理过期的设备数据。"""
            while True:
                await asyncio.sleep(3600)
                await asyncio.to_thread(development.cleanup)
        cleanup_task = asyncio.create_task(prune_expired())
        try:
            yield
        finally:
            cleanup_task.cancel()
            with suppress(asyncio.CancelledError):
                await cleanup_task

    # 最后把会话服务和其他存储注入 HTTP 接口，并按需挂载前端构建产物。
    app = build_api(ConversationService(agent, store, curation), lifespan=lifespan,
                    users=usersStore, tasks=task_store, observability=observability,
                    memory=memory_store, knowledge=knowledge_store,
                    ingestion=KnowledgeIngestionService(knowledge_store, embeddings),
                    devices=devices, development=development)
    web_dist = Path(__file__).resolve().parents[3] / "frontend" / "dist"
    if web_dist.is_dir():
        app.mount("/", WebStaticFiles(directory=web_dist, html=True), name="web")
    return app
