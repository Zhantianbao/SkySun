"""Foreground companion registration and heartbeat behavior without a real desktop."""

import json
import os
from pathlib import Path

import pytest

from skysun_companion import __main__ as companion


def test_server_requires_tls_except_loopback():
    assert companion.server_url("https://sky.example/") == "https://sky.example"
    assert companion.server_url("http://localhost:8080/") == "http://localhost:8080"
    with pytest.raises(ValueError):
        companion.server_url("http://sky.example")


def test_registration_saves_device_credential_privately(tmp_path, monkeypatch):
    class Reply:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def read(self, *_):
            return b'{"device_id":"device-id","device_token":"private-device-token"}'
    requests = []
    def fake_urlopen(request, timeout):
        requests.append(request)
        return Reply()
    monkeypatch.setattr(companion.getpass, "getpass", lambda _: "private-user-token")
    monkeypatch.setattr(companion, "urlopen", fake_urlopen)
    path = tmp_path / "skysun" / "companion.json"
    assert companion.register("https://sky.example", "My Computer", path) == "device-id"
    assert requests[0].get_header("Authorization") == "Bearer private-user-token"
    assert json.loads(path.read_text())["device_token"] == "private-device-token"
    assert os.stat(path).st_mode & 0o777 == 0o600


def test_metrics_and_reconnecting_heartbeat(tmp_path, monkeypatch):
    path = tmp_path / "companion.json"
    path.write_text(json.dumps({"server": "https://sky.example", "device_id": "id",
                                "device_token": "secret"}))
    monkeypatch.setattr(companion, "collect_metrics", lambda: {
        "type": "heartbeat", "cpu_percent": 35, "memory_percent": 60, "disk_percent": 42})
    class Stop:
        active = False
        def is_set(self):
            return self.active
        def wait(self, _):
            return self.active
    stop = Stop()
    attempts = []
    class Channel:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def recv(self, timeout):
            return json.dumps({"type": "connected" if not hasattr(self, "sent") else "heartbeat_ack"})
        def send(self, value):
            self.sent = json.loads(value)
            stop.active = True
    def fake_connect(url, **kwargs):
        attempts.append((url, kwargs))
        if len(attempts) == 1:
            raise OSError("offline")
        return Channel()
    monkeypatch.setattr(companion, "connect", fake_connect)
    companion.run(path, stop=stop, interval=0)
    assert len(attempts) == 2
    assert attempts[-1][0] == "wss://sky.example/api/ws/devices/id"
    assert attempts[-1][1]["additional_headers"]["Authorization"] == "Bearer secret"
