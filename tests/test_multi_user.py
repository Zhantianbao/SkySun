"""Multi-user ownership and authorization against the dedicated test database."""
# 本文件负责测试用例中的 test、multi、user 相关流程。

import os
import secrets
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from skysun.core.api import build_api
from skysun.memory.conversation import ConversationService
from skysun.rag.knowledge import KnowledgeChunkInput
from skysun.rag.knowledge_ingestion import KnowledgeIngestionService
from skysun.memory.memory_curator import MemoryCurationService, MemoryDecision, MemoryPlan
from skysun.workflow.planner import TaskPlan
from skysun.memory.postgres_conversation import PostgresConversationStore
from skysun.rag.postgres_knowledge import PostgresKnowledgeStore
from skysun.memory.postgres_memory import PostgresMemoryStore
from skysun.evaluation.postgres_observability import PostgresObservabilityStore
from skysun.workflow.postgres_task import PostgresTaskStore
from skysun.user.postgres_users import PostgresUserStore, ensure_user_schema
from skysun.workflow.task_runtime import TaskStatus
from skysun.user.user_identity import Actor, AuthorizationError, LEGACY_USER_ID, Role, actor_scope
from skysun.workflow.workflow import StepRecord, StepState
from skysun.workflow.workflow_agent import WorkflowAgent


# 调用位置：pytest 在依赖此 fixture 的测试中调用。
@pytest.fixture
def platform(create_account):
    """为测试提供 platform 所需的输入或模拟行为。"""
    url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("Multi-user tests require a *_test database")
    users = PostgresUserStore(url)
    users.initialize_schema()
    conversations = PostgresConversationStore(url)
    memory = PostgresMemoryStore(url)
    tasks = PostgresTaskStore(url)
    knowledge = PostgresKnowledgeStore(url)
    for store in (conversations, memory, tasks, knowledge):
        store.initialize_schema()
    admin, admin_token = create_account(users, Role.ADMIN)
    alice, alice_token = create_account(users)
    bob, bob_token = create_account(users)
    try:
        yield users, conversations, memory, tasks, knowledge, (
            (admin, admin_token), (alice, alice_token), (bob, bob_token)
        )
    finally:
        ids = (admin.user_id, alice.user_id, bob.user_id)
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM agent_tasks WHERE user_id = ANY(%s)", (list(ids),))
            conn.execute("""
                DELETE FROM conversation_messages WHERE conversation_id IN
                (SELECT id FROM conversations WHERE user_id = ANY(%s))
            """, (list(ids),))
            conn.execute("DELETE FROM conversations WHERE user_id = ANY(%s)", (list(ids),))
            conn.execute("DELETE FROM personal_memories WHERE user_id = ANY(%s)", (list(ids),))
            conn.execute("DELETE FROM knowledge_documents WHERE owner_id = ANY(%s)", (list(ids),))
            conn.execute("DELETE FROM skysun_users WHERE user_id = ANY(%s)", (list(ids),))


# 调用位置：tests/test_multi_user.py 的 test_identity_authentication_and_conversation_isolation、tests/test_multi_user.py 的 test_legacy_rows_keep_their_dedicated_owner、tests/test_multi_user.py 的 test_personal_memory_same_key_isolated_and_global_memory_separate 等处调用。
def _scope(user):
    """为测试准备 _scope 所需的替身或输入。"""
    return actor_scope(Actor(user.user_id, user.role))


# 调用位置：pytest 执行该测试用例时调用。
def test_old_admin_name_is_migrated_without_changing_identity_or_token(platform):
    """验证旧管理员姓名迁移后保留固定身份和原令牌。"""
    users, *_ = platform
    with psycopg.connect(users._database_url) as conn:
        conn.execute("""
            CREATE TEMP TABLE skysun_users (
                user_id UUID PRIMARY KEY, username TEXT NOT NULL UNIQUE,
                role TEXT NOT NULL, token_hash TEXT UNIQUE,
                created_at TIMESTAMPTZ DEFAULT now(), updated_at TIMESTAMPTZ DEFAULT now()
            )
        """)
        conn.execute("""
            INSERT INTO skysun_users (user_id, username, role, token_hash)
            VALUES (%s, 'legacy_admin', 'ADMIN', 'existing-digest')
        """, (LEGACY_USER_ID,))
        ensure_user_schema(conn)
        ensure_user_schema(conn)
        assert conn.execute("SELECT user_id, username, token_hash FROM skysun_users").fetchone() == (
            LEGACY_USER_ID, "詹天宝", "existing-digest",
        )
        conn.execute("UPDATE skysun_users SET username = '管理员'")
        ensure_user_schema(conn)
        assert conn.execute("SELECT username FROM skysun_users").fetchone()[0] == "管理员"


# 调用位置：pytest 执行该测试用例时调用。
@pytest.mark.parametrize("token_length", [8, 513])
def test_creation_api_requires_token_and_rejects_empty_or_duplicate_credentials(platform, new_username, token_length):
    """验证创建用户接口要求令牌，并拒绝空值或重复凭据。"""
    users, conversations, _, tasks, _, accounts = platform
    (admin, admin_token), (alice, alice_token), _ = accounts

    # 以下集中定义 Agent：Agent 执行层相关的数据和操作。
    class Agent:
        # 调用位置：Agent 的调用方通过该方法使用。
        def reply(self, message, history=(), conversation_id=None):
            """生成并返回本次对话的最终回复。"""
            return "Hello"

    client = TestClient(build_api(ConversationService(Agent(), conversations), users=users, tasks=tasks))
    headers = {"Authorization": f"Bearer {admin_token}"}
    for payload in ({"username": new_username()}, {"username": new_username(), "token": "   "}):
        assert client.post("/admin/users", json=payload, headers=headers).status_code == 422
    for payload in (
        {"username": new_username(), "token": alice_token},
        {"username": alice.username, "token": secrets.token_urlsafe(32)},
    ):
        assert client.post("/admin/users", json=payload, headers=headers).status_code == 409

    token = secrets.token_urlsafe(400)[:token_length]
    # Neither creation nor authentication may introduce a token-length policy.
    response = client.post("/admin/users", json={"username": new_username(), "token": token}, headers=headers)
    assert response.status_code == 201
    created_id = UUID(response.json()["user_id"])
    try:
        assert response.json()["token"] == token
        assert users.authenticate(token).user_id == created_id
    finally:
        with psycopg.connect(users._database_url) as conn:
            conn.execute("DELETE FROM skysun_users WHERE user_id = %s", (created_id,))


# 调用位置：pytest 执行该测试用例时调用。
def test_admin_bootstrap_stores_only_token_hash(platform):
    """验证管理员初始化只保存令牌哈希。"""
    users, _, _, _, _, _ = platform
    with psycopg.connect(users._database_url) as conn:
        original = conn.execute("SELECT token_hash FROM skysun_users WHERE user_id = %s",
                                (LEGACY_USER_ID,)).fetchone()[0]
    token = secrets.token_urlsafe(32)
    try:
        users.bootstrap_admin(token)
        assert users.authenticate(token).role is Role.ADMIN
        with psycopg.connect(users._database_url) as conn:
            stored = conn.execute("SELECT token_hash FROM skysun_users WHERE user_id = %s",
                                  (LEGACY_USER_ID,)).fetchone()[0]
        assert stored != token
        assert len(stored) == 64
    finally:
        with psycopg.connect(users._database_url) as conn:
            conn.execute("UPDATE skysun_users SET token_hash = %s WHERE user_id = %s",
                         (original, LEGACY_USER_ID))


# 调用位置：pytest 执行该测试用例时调用。
def test_login_checks_both_credentials_on_the_server(platform):
    """验证登录时由服务端同时核对姓名和令牌。"""
    users, conversations, _, tasks, _, accounts = platform
    (admin, token), (alice, alice_token), _ = accounts

    # 以下集中定义 Agent：Agent 执行层相关的数据和操作。
    class Agent:
        # 调用位置：Agent 的调用方通过该方法使用。
        def reply(self, message, history=(), conversation_id=None):
            """生成并返回本次对话的最终回复。"""
            return "Hello"

    client = TestClient(build_api(ConversationService(Agent(), conversations), users=users, tasks=tasks))
    for user, secret in ((admin, token), (alice, alice_token)):
        response = client.post("/api/login", json={"username": f" {user.username} ", "token": f" {secret} "})
        assert response.status_code == 200
        assert response.json() == {"user_id": str(user.user_id), "username": user.username, "role": user.role.value}
    for name, secret in ((alice.username, token), (admin.username, "wrong"), ("", token), ("alice", token), (admin.username, " ")):
        response = client.post("/api/login", json={"username": name, "token": secret})
        assert response.status_code == 401
        assert response.json() == {"detail": "姓名或密码错误"}
    assert client.post("/api/login", json={"username": admin.username}).status_code == 422


# 调用位置：pytest 执行该测试用例时调用。
def test_bootstrap_restores_builtin_admin_and_rotates_password(platform):
    """验证初始化可恢复内置管理员并轮换密码。"""
    users, *_ = platform
    with psycopg.connect(users._database_url) as conn:
        original = conn.execute("SELECT username, role, token_hash FROM skysun_users WHERE user_id = %s", (LEGACY_USER_ID,)).fetchone()
    first, second = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
    try:
        with psycopg.connect(users._database_url) as conn:
            conn.execute("UPDATE skysun_users SET username = '旧管理员', role = 'USER' WHERE user_id = %s", (LEGACY_USER_ID,))
        users.bootstrap_admin(first)
        user = users.authenticate(first)
        assert (user.user_id, user.username, user.role) == (LEGACY_USER_ID, "詹天宝", Role.ADMIN)
        users.bootstrap_admin(second)
        users.bootstrap_admin(second)
        assert users.authenticate(first) is None
        assert users.authenticate(second).user_id == LEGACY_USER_ID
    finally:
        with psycopg.connect(users._database_url) as conn:
            conn.execute("UPDATE skysun_users SET username = %s, role = %s, token_hash = %s WHERE user_id = %s", (*original, LEGACY_USER_ID))


# 调用位置：pytest 执行该测试用例时调用。
def test_identity_authentication_and_conversation_isolation(platform):
    """验证身份验证正确且用户会话相互隔离。"""
    users, conversations, _, _, _, accounts = platform
    _, (alice, alice_token), (bob, bob_token) = accounts
    assert users.authenticate(alice_token).user_id == alice.user_id
    assert users.authenticate(bob_token).user_id == bob.user_id
    assert users.authenticate("wrong") is None
    with _scope(alice):
        conversation_id = conversations.create()
        conversations.append_exchange(conversation_id, "private", "answer")
        assert conversations.get_history(conversation_id)[0].content == "private"
    with _scope(bob):
        with pytest.raises(KeyError):
            conversations.get_history(conversation_id)
        with pytest.raises(KeyError):
            conversations.append_exchange(conversation_id, "intrusion", "no")


# 调用位置：pytest 执行该测试用例时调用。
def test_legacy_rows_keep_their_dedicated_owner(platform):
    """验证历史记录继续归属于固定的原所有者。"""
    _, conversations, memory, _, _, accounts = platform
    _, (alice, _), _ = accounts
    legacy_conversation = conversations.create()
    key = "legacy_" + uuid4().hex[:10]
    try:
        memory.remember(key, "Legacy-only preference")
        with _scope(alice):
            with pytest.raises(KeyError):
                conversations.get_history(legacy_conversation)
            assert memory.get(key) is None
        assert memory.get(key).content == "Legacy-only preference"
    finally:
        memory.forget(key)
        with psycopg.connect(conversations._database_url) as conn:
            conn.execute("DELETE FROM conversations WHERE id = %s", (UUID(legacy_conversation),))


# 调用位置：pytest 执行该测试用例时调用。
def test_personal_memory_same_key_isolated_and_global_memory_separate(platform):
    """验证相同键名的个人记忆按用户隔离，全局记忆独立保存。"""
    _, _, memory, _, _, accounts = platform
    (admin, _), (alice, _), (bob, _) = accounts
    with _scope(alice):
        first = memory.remember("answer_style", "Alice prefers concise Chinese", "preference")
        assert memory.search_memories("Chinese")[0].id == first.id
    with _scope(bob):
        second = memory.remember("answer_style", "Bob prefers detailed English", "preference")
        assert second.id != first.id
        assert memory.get("answer_style").content == "Bob prefers detailed English"
        assert all(item.user_id == bob.user_id for item in memory.list_memories())
        with pytest.raises(AuthorizationError):
            memory.list_global_memories()
    with _scope(alice):
        assert memory.get("answer_style").content == "Alice prefers concise Chinese"
        assert memory.search_memories("English")[0].id == first.id  # own fallback only
    key = "global_" + uuid4().hex[:10]
    with _scope(admin):
        memory.remember_global(key, "SkySun system guidance")
        assert any(item.key == key for item in memory.list_global_memories())
        with memory._transaction() as conn:
            conn.execute("DELETE FROM global_memories WHERE memory_key = %s", (key,))


# 调用位置：pytest 执行该测试用例时调用。
def test_curator_deduplicates_within_each_user(platform):
    """验证记忆整理在各用户范围内分别去重。"""
    _, _, memory, _, _, accounts = platform
    _, (alice, _), (bob, _) = accounts
    key = "theme_" + uuid4().hex[:10]

    # 以下集中定义 Curator：数据结构或组件相关的数据和操作。
    class Curator:
        # 调用位置：src/skysun/memory/memory_curator.py 的 MemoryCurationService.process、tests/test_memory_curator.py 的 test_langchain_curator_uses_typed_json_output_without_network、tests/test_memory_extraction.py 的 test_valid_structured_output_is_accepted_without_retry 等处调用。
        def decide(self, context):
            """根据对话和已有记忆决定新增、更新或删除内容。"""
            return MemoryPlan(decisions=[MemoryDecision(action="upsert", key=key,
                              content="Prefers dark mode", importance="high")])

    service = MemoryCurationService(Curator(), memory)
    with _scope(alice):
        service.process("dark mode", (), str(uuid4()))
        first = memory.get(key)
        service.process("dark mode", (), str(uuid4()))
        assert memory.get(key).id == first.id
    with _scope(bob):
        service.process("dark mode", (), str(uuid4()))
        assert memory.get(key).id != first.id


# 调用位置：pytest 执行该测试用例时调用。
def test_private_knowledge_isolation_and_global_visibility(platform):
    """验证私有知识按用户隔离，全局知识仍可见。"""
    _, _, _, _, knowledge, accounts = platform
    _, (alice, _), (bob, _) = accounts
    vector = (1.0,) + (0.0,) * 1023
    path = "/test/" + uuid4().hex + "/private.md"
    with _scope(alice):
        knowledge.save_document(path, "private.md", "hash-a",
                                (KnowledgeChunkInput("Alice-only cobalt marker", vector),))
        assert knowledge.get_by_path(path).owner_id == alice.user_id
        assert any("Alice-only" in item.content for item in knowledge.search(vector))
    with _scope(bob):
        assert knowledge.get_by_path(path) is None
        assert all("Alice-only" not in item.content for item in knowledge.search(vector))
        assert not knowledge.delete_document(path)
        knowledge.save_document(path, "private.md", "hash-b",
                                (KnowledgeChunkInput("Bob-only amber marker", vector),))
        assert all("Alice-only" not in item.content for item in knowledge.search(vector))
    global_path = "/test/" + uuid4().hex + "/global.md"
    knowledge.save_document(global_path, "global.md", "hash-global",
                            (KnowledgeChunkInput("Shared silver marker", vector),))
    try:
        with _scope(alice):
            assert any("Shared silver" in item.content for item in knowledge.search(vector))
        with _scope(bob):
            assert any("Shared silver" in item.content for item in knowledge.search(vector))
    finally:
        knowledge.delete_document(global_path)


# 调用位置：pytest 执行该测试用例时调用。
def test_task_ownership_events_and_resume(platform):
    """验证任务事件遵守所有权规则，且中断后能够恢复。"""
    _, _, _, tasks, _, accounts = platform
    _, (alice, _), (bob, _) = accounts
    with _scope(alice):
        task = tasks.create("Count then explain", None, ())
        tasks.transition(task.task_id, TaskStatus.PLANNING)
        plan = TaskPlan.model_validate({"goal": task.goal, "steps": [
            {"step_id": "count", "description": "Count", "capabilities": ["count_words"]},
            {"step_id": "explain", "description": "Explain", "capabilities": ["reason"],
             "depends_on": ["count"]},
        ]})
        tasks.save_plan(task.task_id, plan)
        tasks.transition(task.task_id, TaskStatus.RUNNING)
        tasks.start_step(task.task_id, "count")
        tasks.finish_step(StepRecord(task.task_id, "count", StepState.SUCCESS, "3",
                                     ("count_words",), "success"))
        assert all(event.user_id == alice.user_id for event in tasks.events(task.task_id))
    with _scope(bob):
        assert tasks.list_unfinished() == ()
        with pytest.raises(KeyError):
            tasks.get(task.task_id)
        with pytest.raises(KeyError):
            tasks.events(task.task_id)
        with pytest.raises(KeyError):
            tasks.transition(task.task_id, TaskStatus.FAILED)

    from skysun.core.langchain_agent import AgentExecution

    # 以下集中定义 Agent：Agent 执行层相关的数据和操作。
    class Agent:
        # 调用位置：tests/test_multi_user.py 的 platform、tests/test_multi_user.py 的 test_old_admin_name_is_migrated_without_changing_identity_or_token、tests/test_multi_user.py 的 test_creation_api_requires_token_and_rejects_empty_or_duplicate_credentials 等处调用。
        def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
            """执行 Agent 或工作流并返回结构化结果。"""
            return AgentExecution("Explained.", (), (), None, ())

    # 以下集中定义 Planner：计划生成层相关的数据和操作。
    class Planner:
        # 调用位置：Planner 的调用方通过该方法使用。
        def plan(self, goal, capabilities):
            """根据用户目标生成可执行的步骤计划。"""
            raise AssertionError("Resume must use persisted plan")

    workflow = WorkflowAgent(Agent(), Planner(), ("count_words", "reason"), tasks)
    workflow.recover_unfinished()
    with _scope(alice):
        saved = tasks.get(task.task_id)
        assert saved.status is TaskStatus.COMPLETED
        assert saved.steps[0].output == "3"
        assert saved.steps[1].output == "Explained."


# 调用位置：pytest 执行该测试用例时调用。
def test_api_bearer_roles_and_cross_user_rejection(platform, new_username):
    """验证接口检查 Bearer 身份和角色，并拒绝跨用户访问。"""
    users, conversations, _, tasks, _, accounts = platform
    (admin, admin_token), (alice, alice_token), (bob, bob_token) = accounts

    # 以下集中定义 Agent：Agent 执行层相关的数据和操作。
    class Agent:
        # 调用位置：Agent 的调用方通过该方法使用。
        def reply(self, message, history=(), conversation_id=None):
            """生成并返回本次对话的最终回复。"""
            return "Hello"

    app = build_api(ConversationService(Agent(), conversations), users=users, tasks=tasks)
    client = TestClient(app)
    ah = {"Authorization": f"Bearer {alice_token}"}
    bh = {"Authorization": f"Bearer {bob_token}"}
    admin_h = {"Authorization": f"Bearer {admin_token}"}
    assert client.post("/chat", json={"message": "hello"}).status_code == 401
    assert client.post("/chat", json={"message": "hello"},
                       headers={"Authorization": "Bearer invalid"}).status_code == 401
    response = client.post("/chat", json={"message": "hello"}, headers=ah)
    assert response.status_code == 200
    conversation_id = response.json()["conversation_id"]
    assert client.post("/chat", json={"message": "again", "conversation_id": conversation_id},
                       headers=bh).status_code == 404
    assert client.get("/me", headers=ah).json()["user_id"] == str(alice.user_id)
    assert client.get("/admin/summary", headers=ah).status_code == 403
    assert client.get("/admin/summary", headers=admin_h).status_code == 200
    with _scope(alice):
        task = tasks.create("Private task", None, ())
    assert client.get(f"/tasks/{task.task_id}", headers=bh).status_code == 404
    assert client.get(f"/tasks/{task.task_id}/events", headers=bh).status_code == 404
    assert client.get(f"/tasks/{task.task_id}", headers=admin_h).status_code == 404
    assert client.get(f"/tasks/{task.task_id}", headers=ah).json()["user_id"] == str(alice.user_id)
    payload = {"username": new_username(), "token": secrets.token_urlsafe(32)}
    assert client.post("/admin/users", json=payload, headers=ah).status_code == 403
    created = client.post("/admin/users", json=payload,
                          headers=admin_h)
    assert created.status_code == 201
    assert created.json()["token"] == payload["token"]
    new_id = UUID(created.json()["user_id"])
    assert users.authenticate(created.json()["token"]).user_id == new_id
    with psycopg.connect(users._database_url) as conn:
        conn.execute("DELETE FROM skysun_users WHERE user_id = %s", (new_id,))


# 调用位置：pytest 执行该测试用例时调用。
def test_web_read_endpoints_keep_user_data_isolated(platform):
    """验证网页读取接口保持用户数据隔离。"""
    users, conversations, memory, tasks, knowledge, accounts = platform
    _, (alice, alice_token), (bob, bob_token) = accounts
    path = "/test/" + uuid4().hex + "/alice.md"
    with _scope(alice):
        conversation_id = conversations.create()
        conversations.append_exchange(conversation_id, "private question", "private answer")
        memory.remember("web_preference", "Prefers green", "preference")
        task = tasks.create("Private web task", UUID(conversation_id), ())
        knowledge.save_document(path, "alice.md", "hash-web",
                                (KnowledgeChunkInput("Alice document", (1.0,) + (0.0,) * 1023),))

    # 以下集中定义 Agent：Agent 执行层相关的数据和操作。
    class Agent:
        # 调用位置：Agent 的调用方通过该方法使用。
        def reply(self, message, history=(), conversation_id=None):
            """生成并返回本次对话的最终回复。"""
            return "Hello"

    client = TestClient(build_api(ConversationService(Agent(), conversations), users=users,
                                  tasks=tasks, memory=memory, knowledge=knowledge))
    ah = {"Authorization": f"Bearer {alice_token}"}
    bh = {"Authorization": f"Bearer {bob_token}"}
    assert client.get("/api/me", headers=ah).status_code == 200
    assert client.get("/api/tasks").status_code == 401
    assert any(row["task_id"] == str(task.task_id) for row in client.get("/api/tasks", headers=ah).json())
    assert all(row["task_id"] != str(task.task_id) for row in client.get("/api/tasks", headers=bh).json())
    assert any(row["key"] == "web_preference" for row in client.get("/api/memory", headers=ah).json())
    assert all(row["key"] != "web_preference" for row in client.get("/api/memory", headers=bh).json())
    assert any(row["source_path"] == path for row in client.get("/api/knowledge/documents", headers=ah).json())
    assert all(row["source_path"] != path for row in client.get("/api/knowledge/documents", headers=bh).json())
    assert client.get(f"/api/conversations/{conversation_id}/messages", headers=ah).json() == [
        {"role": "user", "content": "private question"},
        {"role": "assistant", "content": "private answer"},
    ]
    assert client.get(f"/api/conversations/{conversation_id}/messages", headers=bh).status_code == 404
    assert any(row["conversation_id"] == conversation_id
               for row in client.get("/api/conversations", headers=ah).json())
    assert all(row["conversation_id"] != conversation_id
               for row in client.get("/api/conversations", headers=bh).json())


# 调用位置：pytest 执行该测试用例时调用。
def test_web_upload_and_task_activity_respect_ownership(platform):
    """验证网页上传和任务活动都遵守数据所有权。"""
    users, conversations, _, tasks, knowledge, accounts = platform
    _, (alice, alice_token), (bob, bob_token) = accounts
    observability = PostgresObservabilityStore(users._database_url)
    observability.initialize_schema()

    # 以下集中定义 Embeddings：数据结构或组件相关的数据和操作。
    class Embeddings:
        dimension = 1024

        # 调用位置：src/skysun/rag/knowledge_ingestion.py 的 KnowledgeIngestionService.ingest_text、tests/test_knowledge.py 的 test_qwen_adapter_uses_compatible_api_and_validates_vectors 调用。
        def embed_passages(self, texts):
            """把多个知识片段转换为向量。"""
            return ((1.0,) + (0.0,) * 1023,) * len(texts)

    # 以下集中定义 Agent：Agent 执行层相关的数据和操作。
    class Agent:
        # 调用位置：Agent 的调用方通过该方法使用。
        def reply(self, message, history=(), conversation_id=None):
            """生成并返回本次对话的最终回复。"""
            return "Hello"

    client = TestClient(build_api(ConversationService(Agent(), conversations), users=users,
                                  tasks=tasks, knowledge=knowledge,
                                  ingestion=KnowledgeIngestionService(knowledge, Embeddings()),
                                  observability=observability))
    ah = {"Authorization": f"Bearer {alice_token}"}
    bh = {"Authorization": f"Bearer {bob_token}"}
    assert client.post("/api/knowledge/documents", headers=ah,
                       files={"file": ("bad.pdf", b"pdf")}).status_code == 422
    assert client.post("/api/knowledge/documents", headers=ah,
                       files={"file": ("bad.md", b"\xff")}).status_code == 422
    assert client.post("/api/knowledge/documents", headers=ah,
                       files={"file": ("large.md", b"x" * (2 * 1024 * 1024 + 1))}).status_code == 413
    uploaded = client.post("/api/knowledge/documents", headers=ah,
                           files={"file": ("notes.md", b"Alice upload marker", "text/markdown")})
    assert uploaded.status_code == 201
    assert uploaded.json()["chunk_count"] == 1
    repeated = client.post("/api/knowledge/documents", headers=ah,
                           files={"file": ("notes.md", b"Alice upload marker", "text/markdown")})
    assert repeated.status_code == 201
    assert repeated.json()["status"] == "unchanged"
    assert repeated.json()["document_id"] == uploaded.json()["document_id"]
    assert any(row["source_path"] == uploaded.json()["source_path"]
               for row in client.get("/api/knowledge/documents", headers=ah).json())
    assert all(row["source_path"] != uploaded.json()["source_path"]
               for row in client.get("/api/knowledge/documents", headers=bh).json())
    with _scope(alice):
        task = tasks.create("Activity test", None, ())
        trace_id = observability.start_trace(task.task_id, "knowledge_retrieval")
        observability.finish_trace(trace_id, "SUCCESS")
    activity = client.get(f"/api/tasks/{task.task_id}/activity", headers=ah)
    assert activity.status_code == 200
    assert activity.json()[0]["stage"] == "knowledge_retrieval"
    assert "metadata" not in activity.json()[0]
    assert client.get(f"/api/tasks/{task.task_id}/activity", headers=bh).status_code == 404
