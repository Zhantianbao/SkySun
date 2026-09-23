"""Real database and WebSocket checks for the desktop companion boundary."""

import os
from uuid import UUID, uuid4

import psycopg
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from skysun.api import build_api
from skysun.conversation import ConversationService, InMemoryConversationStore
from skysun.devices import PostgresDeviceStore
from skysun.postgres_users import PostgresUserStore


class StubAgent:
    def reply(self, message, history=(), conversation_id=None):
        return "ok"


@pytest.fixture
def platform():
    url = os.environ.get("SKYSUN_TEST_DATABASE_URL", "postgresql:///skysun_test")
    with psycopg.connect(url) as conn:
        if not conn.info.dbname.endswith("_test"):
            raise RuntimeError("Device tests require a *_test database")
    users = PostgresUserStore(url)
    users.initialize_schema()
    devices = PostgresDeviceStore(url)
    devices.initialize_schema()
    suffix = uuid4().hex[:10]
    alice, alice_token = users.create_user("alice_" + suffix)
    bob, bob_token = users.create_user("bob_" + suffix)
    client = TestClient(build_api(ConversationService(StubAgent(), InMemoryConversationStore()),
                                  users=users, devices=devices))
    try:
        yield client, devices, alice, alice_token, bob, bob_token
    finally:
        with psycopg.connect(url) as conn:
            conn.execute("DELETE FROM skysun_users WHERE user_id = ANY(%s)",
                         ([alice.user_id, bob.user_id],))


def _auth(token):
    return {"Authorization": "Bearer " + token}


def test_device_registration_and_owner_isolation(platform):
    client, store, alice, alice_token, _, bob_token = platform
    payload = {"name": "My Computer", "hostname": "laptop", "operating_system": "Linux"}
    assert client.post("/api/devices", json=payload).status_code == 401
    response = client.post("/api/devices", json=payload, headers=_auth(alice_token))
    assert response.status_code == 201
    device_id = UUID(response.json()["device_id"])
    device_token = response.json()["device_token"]
    assert store.authenticate(device_id, device_token) == alice.user_id
    assert store.authenticate(device_id, "wrong") is None
    with psycopg.connect(store._database_url) as conn:
        saved = conn.execute("SELECT token_hash FROM devices WHERE device_id = %s", (device_id,)).fetchone()[0]
    assert saved != device_token and len(saved) == 64
    assert len(client.get("/api/devices", headers=_auth(alice_token)).json()) == 1
    assert client.get("/api/devices", headers=_auth(bob_token)).json() == []
    assert client.delete(f"/api/devices/{device_id}", headers=_auth(bob_token)).status_code == 404
    assert client.delete(f"/api/devices/{device_id}", headers=_auth(alice_token)).status_code == 204
    assert store.authenticate(device_id, device_token) is None


def test_websocket_heartbeat_authentication_and_offline_state(platform):
    client, _, _, alice_token, _, bob_token = platform
    response = client.post("/api/devices", json={"name": "Laptop", "hostname": "host",
                                                 "operating_system": "Linux"}, headers=_auth(alice_token))
    device_id, token = response.json()["device_id"], response.json()["device_token"]
    path = f"/api/ws/devices/{device_id}"
    with pytest.raises(WebSocketDisconnect) as rejected:
        with client.websocket_connect(path, headers=_auth(bob_token)):
            pass
    assert rejected.value.code == 4401
    with client.websocket_connect(path, headers=_auth(token)) as ws:
        assert ws.receive_json()["type"] == "connected"
        ws.send_json({"type": "heartbeat", "cpu_percent": 35,
                      "memory_percent": 60, "disk_percent": 42})
        assert ws.receive_json() == {"type": "heartbeat_ack"}
        status = client.get("/api/devices", headers=_auth(alice_token)).json()[0]
        assert status["online"] is True
        assert status["cpu_percent"] == 35
        assert status["last_heartbeat_at"] is not None
    assert client.get("/api/devices", headers=_auth(alice_token)).json()[0]["online"] is False


def test_invalid_websocket_message_closes_safely(platform):
    client, _, _, alice_token, _, _ = platform
    response = client.post("/api/devices", json={"name": "Laptop", "hostname": "host",
                                                 "operating_system": "Linux"}, headers=_auth(alice_token))
    path = f"/api/ws/devices/{response.json()['device_id']}"
    with client.websocket_connect(path, headers=_auth(response.json()["device_token"])) as ws:
        ws.receive_json()
        ws.send_json({"type": "heartbeat", "cpu_percent": 200,
                      "memory_percent": 30, "disk_percent": 40})
        with pytest.raises(WebSocketDisconnect) as error:
            ws.receive_json()
        assert error.value.code == 1008
