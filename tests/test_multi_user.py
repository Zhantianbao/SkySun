"""Multi-user ownership and authorization against the dedicated test database."""

import os
import secrets
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from skysun.api import build_api
from skysun.conversation import ConversationService
from skysun.knowledge import KnowledgeChunkInput
from skysun.knowledge_ingestion import KnowledgeIngestionService
from skysun.memory_curator import MemoryCurationService, MemoryDecision, MemoryPlan
from skysun.planner import TaskPlan
from skysun.postgres_conversation import PostgresConversationStore
from skysun.postgres_knowledge import PostgresKnowledgeStore
from skysun.postgres_memory import PostgresMemoryStore
from skysun.postgres_observability import PostgresObservabilityStore
from skysun.postgres_task import PostgresTaskStore
from skysun.postgres_users import PostgresUserStore
from skysun.task_runtime import TaskStatus
from skysun.user_identity import Actor, AuthorizationError, LEGACY_USER_ID, Role, actor_scope
from skysun.workflow import StepRecord, StepState
from skysun.workflow_agent import WorkflowAgent


@pytest.fixture
def platform():
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
    suffix = uuid4().hex[:10]
    admin, admin_token = users.create_user("admin_" + suffix, Role.ADMIN)
    alice, alice_token = users.create_user("alice_" + suffix)
    bob, bob_token = users.create_user("bob_" + suffix)
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


def _scope(user):
    return actor_scope(Actor(user.user_id, user.role))


def test_admin_bootstrap_stores_only_token_hash(platform):
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


def test_identity_authentication_and_conversation_isolation(platform):
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


def test_legacy_rows_keep_their_dedicated_owner(platform):
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


def test_personal_memory_same_key_isolated_and_global_memory_separate(platform):
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


def test_curator_deduplicates_within_each_user(platform):
    _, _, memory, _, _, accounts = platform
    _, (alice, _), (bob, _) = accounts
    key = "theme_" + uuid4().hex[:10]

    class Curator:
        def decide(self, context):
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


def test_private_knowledge_isolation_and_global_visibility(platform):
    _, _, _, _, knowledge, accounts = platform
    _, (alice, _), (bob, _) = accounts
    vector = (1.0,) + (0.0,) * 383
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


def test_task_ownership_events_and_resume(platform):
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

    from skysun.langchain_agent import AgentExecution

    class Agent:
        def execute(self, message, history=(), conversation_id=None, knowledge_question=None):
            return AgentExecution("Explained.", (), (), None, ())

    class Planner:
        def plan(self, goal, capabilities):
            raise AssertionError("Resume must use persisted plan")

    workflow = WorkflowAgent(Agent(), Planner(), ("count_words", "reason"), tasks)
    workflow.recover_unfinished()
    with _scope(alice):
        saved = tasks.get(task.task_id)
        assert saved.status is TaskStatus.COMPLETED
        assert saved.steps[0].output == "3"
        assert saved.steps[1].output == "Explained."


def test_api_bearer_roles_and_cross_user_rejection(platform):
    users, conversations, _, tasks, _, accounts = platform
    (admin, admin_token), (alice, alice_token), (bob, bob_token) = accounts

    class Agent:
        def reply(self, message, history=(), conversation_id=None):
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
    assert client.post("/admin/users", json={"username": "other"}, headers=ah).status_code == 403
    created = client.post("/admin/users", json={"username": "test_" + uuid4().hex[:10]},
                          headers=admin_h)
    assert created.status_code == 201
    new_id = UUID(created.json()["user_id"])
    assert users.authenticate(created.json()["token"]).user_id == new_id
    with psycopg.connect(users._database_url) as conn:
        conn.execute("DELETE FROM skysun_users WHERE user_id = %s", (new_id,))


def test_web_read_endpoints_keep_user_data_isolated(platform):
    users, conversations, memory, tasks, knowledge, accounts = platform
    _, (alice, alice_token), (bob, bob_token) = accounts
    path = "/test/" + uuid4().hex + "/alice.md"
    with _scope(alice):
        conversation_id = conversations.create()
        conversations.append_exchange(conversation_id, "private question", "private answer")
        memory.remember("web_preference", "Prefers green", "preference")
        task = tasks.create("Private web task", UUID(conversation_id), ())
        knowledge.save_document(path, "alice.md", "hash-web",
                                (KnowledgeChunkInput("Alice document", (1.0,) + (0.0,) * 383),))

    class Agent:
        def reply(self, message, history=(), conversation_id=None):
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


def test_web_upload_and_task_activity_respect_ownership(platform):
    users, conversations, _, tasks, knowledge, accounts = platform
    _, (alice, alice_token), (bob, bob_token) = accounts
    observability = PostgresObservabilityStore(users._database_url)
    observability.initialize_schema()

    class Embeddings:
        dimension = 384

        def embed_passages(self, texts):
            return ((1.0,) + (0.0,) * 383,) * len(texts)

    class Agent:
        def reply(self, message, history=(), conversation_id=None):
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
