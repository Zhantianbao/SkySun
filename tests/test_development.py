"""Authenticated screenshot and note flow through real PostgreSQL and WebSocket."""

import base64
import os
from uuid import uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient

from skysun.core.api import build_api
from skysun.memory.conversation import ConversationService, InMemoryConversationStore
from skysun.device.development import PostgresDevelopmentStore
from skysun.device.devices import PostgresDeviceStore
from skysun.user.postgres_users import PostgresUserStore

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/JXcAAAAASUVORK5CYII=")


class StubAgent:
    def reply(self, message, history=(), conversation_id=None):
        return "ok"


@pytest.fixture
def platform(create_account):
    url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("Development tests require a *_test database")
    users = PostgresUserStore(url)
    users.initialize_schema()
    devices = PostgresDeviceStore(url)
    devices.initialize_schema()
    development = PostgresDevelopmentStore(url)
    development.initialize_schema()
    alice, alice_token = create_account(users)
    bob, bob_token = create_account(users)
    client = TestClient(build_api(ConversationService(StubAgent(), InMemoryConversationStore()),
                                  users=users, devices=devices, development=development))
    response = client.post("/api/devices", json={"name": "Laptop", "hostname": "host",
                                                 "operating_system": "Linux"}, headers=auth(alice_token))
    try:
        yield client, devices, development, alice_token, bob_token, response.json()
    finally:
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM skysun_users WHERE user_id = ANY(%s)",
                         ([alice.user_id, bob.user_id],))


def auth(token):
    return {"Authorization": "Bearer " + token}


def test_explicit_screenshot_request_upload_and_owner_isolation(platform):
    client, _, development, alice_token, bob_token, device = platform
    device_id, token = device["device_id"], device["device_token"]
    request_path = f"/api/devices/{device_id}/screenshots"
    assert client.post(request_path, headers=auth(alice_token)).status_code == 409
    with client.websocket_connect(f"/api/ws/devices/{device_id}", headers=auth(token)) as ws:
        ws.receive_json()
        ws.send_json({"type": "heartbeat", "cpu_percent": 1, "memory_percent": 2,
                      "disk_percent": 3, "project_name": "SkySun", "git_branch": "main",
                      "git_modified_count": 2, "vscode_running": True, "codex_active": False})
        assert ws.receive_json()["type"] == "heartbeat_ack"
        device_state = client.get("/api/devices", headers=auth(alice_token)).json()[0]
        assert device_state["project_name"] == "SkySun"
        assert device_state["git_branch"] == "main"
        assert client.post(request_path, headers=auth(bob_token)).status_code == 404
        response = client.post(request_path, headers=auth(alice_token))
        assert response.status_code == 201
        request_id = response.json()["request_id"]
        ws.send_json({"type": "heartbeat", "cpu_percent": 1, "memory_percent": 2,
                      "disk_percent": 3})
        assert ws.receive_json()["type"] == "heartbeat_ack"
        assert ws.receive_json() == {"type": "screenshot_request", "request_id": request_id}
        upload_path = f"{request_path}/{request_id}/upload"
        assert client.post(upload_path, content=PNG, headers=auth(bob_token)).status_code == 401
        assert client.post(upload_path, content=b"not-png", headers=auth(token)).status_code == 422
        assert client.post(upload_path, content=PNG, headers=auth(token)).status_code == 200
        assert client.post(upload_path, content=PNG, headers=auth(token)).status_code == 404
        assert client.get(f"/api/screenshots/{request_id}/image", headers=auth(bob_token)).status_code == 404
        image = client.get(f"/api/screenshots/{request_id}/image", headers=auth(alice_token))
        assert image.status_code == 200 and image.content == PNG
        assert image.headers["cache-control"] == "no-store"
        assert client.get("/api/screenshots", headers=auth(alice_token)).json()[0]["status"] == "COMPLETED"
        assert client.get("/api/screenshots", headers=auth(bob_token)).json() == []
        events = client.get(f"/api/devices/{device_id}/events", headers=auth(alice_token)).json()
        assert {event["event_type"] for event in events} >= {"screenshot_requested", "screenshot_uploaded"}
        with psycopg.connect(development._database_url) as conn:
            conn.execute("""
                UPDATE device_screenshots SET requested_at = now() - interval '25 hours'
                WHERE request_id = %s
            """, (request_id,))
        development.cleanup()
        assert client.get(f"/api/screenshots/{request_id}/image", headers=auth(alice_token)).status_code == 404


def test_note_delivery_is_scoped_and_audited(platform):
    client, _, _, alice_token, bob_token, device = platform
    device_id, token = device["device_id"], device["device_token"]
    path = f"/api/devices/{device_id}/notes"
    assert client.post(path, json={"content": "   "}, headers=auth(alice_token)).status_code == 422
    assert client.post(path, json={"content": "Private reminder"}, headers=auth(bob_token)).status_code == 404
    response = client.post(path, json={"content": "Private reminder"}, headers=auth(alice_token))
    note_id = response.json()["note_id"]
    assert client.get(path, headers=auth(bob_token)).status_code == 404
    with client.websocket_connect(f"/api/ws/devices/{device_id}", headers=auth(token)) as ws:
        ws.receive_json()
        ws.send_json({"type": "heartbeat", "cpu_percent": 1, "memory_percent": 2, "disk_percent": 3})
        assert ws.receive_json()["type"] == "heartbeat_ack"
        assert ws.receive_json() == {"type": "note", "note_id": note_id, "content": "Private reminder"}
        assert client.post(f"{path}/{note_id}/delivered", headers=auth(bob_token)).status_code == 401
        assert client.post(f"{path}/{note_id}/delivered", headers=auth(token)).status_code == 200
        ws.send_json({"type": "heartbeat", "cpu_percent": 1, "memory_percent": 2, "disk_percent": 3})
        assert ws.receive_json()["type"] == "heartbeat_ack"
    assert client.get(path, headers=auth(alice_token)).json()[0]["status"] == "DELIVERED"
    events = client.get(f"/api/devices/{device_id}/events", headers=auth(alice_token)).json()
    assert {event["event_type"] for event in events} >= {"note_created", "note_delivered"}


def test_expired_screenshot_cannot_be_uploaded(platform):
    client, _, development, alice_token, _, device = platform
    device_id, token = device["device_id"], device["device_token"]
    with client.websocket_connect(f"/api/ws/devices/{device_id}", headers=auth(token)) as ws:
        ws.receive_json()
        ws.send_json({"type": "heartbeat", "cpu_percent": 1, "memory_percent": 2,
                      "disk_percent": 3})
        ws.receive_json()
        request_id = client.post(f"/api/devices/{device_id}/screenshots",
                                 headers=auth(alice_token)).json()["request_id"]
        with psycopg.connect(development._database_url) as conn:
            conn.execute("""
                UPDATE device_screenshots SET expires_at = now() - interval '1 second'
                WHERE request_id = %s
            """, (request_id,))
        assert client.post(f"/api/devices/{device_id}/screenshots/{request_id}/upload",
                           content=PNG, headers=auth(token)).status_code == 404
        assert client.get("/api/screenshots", headers=auth(alice_token)).json()[0]["status"] == "EXPIRED"
