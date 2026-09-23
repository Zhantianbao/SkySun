"""Foreground companion registration and heartbeat behavior without a real desktop."""

import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timedelta, timezone
from uuid import uuid4

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


def test_user_selected_workspace_and_confirmed_codex_status(tmp_path, monkeypatch):
    config_path = tmp_path / "companion.json"
    config_path.write_text('{"server":"https://sky.example","device_id":"id","device_token":"secret"}')
    workspace = tmp_path / "project"
    subprocess.run(["git", "init", "-q", "-b", "main", str(workspace)], check=True)
    (workspace / "draft.txt").write_text("work")
    monkeypatch.setattr(companion.psutil, "process_iter", lambda _: [])
    companion.set_context(config_path, str(workspace), codex_active=True)
    config = json.loads(config_path.read_text())
    state = companion.collect_development(config)
    assert state["project_name"] == "project"
    assert state["git_branch"] == "main"
    assert state["git_modified_count"] == 1
    assert state["last_activity_at"] is not None
    assert state["codex_active"] is True

    marker = tmp_path / "vscode-status.json"
    marker.write_text(json.dumps({"workspace_path": str(workspace),
                                  "updated_at": datetime.now(timezone.utc).isoformat(),
                                  "activity_at": datetime.now(timezone.utc).isoformat()}))
    assert companion.collect_development(config, marker)["vscode_workspace"] == "project"
    marker.write_text(json.dumps({"workspace_path": str(tmp_path / "other"),
                                  "updated_at": datetime.now(timezone.utc).isoformat()}))
    assert companion.collect_development(config, marker)["vscode_workspace"] is None
    marker.write_text(json.dumps({"workspace_path": str(workspace),
                                  "updated_at": (datetime.now(timezone.utc) - timedelta(minutes=3)).isoformat()}))
    assert companion.collect_development(config, marker)["vscode_workspace"] is None


def test_only_allowlisted_actions_are_handled_and_notes_are_idempotent(tmp_path, monkeypatch):
    config = {"server": "https://sky.example", "device_id": "device",
              "device_token": "secret"}
    calls = []
    monkeypatch.setattr(companion, "_device_request", lambda *args: calls.append(args))
    inbox = tmp_path / "inbox"
    note_id = str(uuid4())
    action = {"type": "note", "note_id": note_id, "content": "Review this"}
    companion.handle_action(config, action, inbox)
    companion.handle_action(config, action, inbox)
    assert json.loads((inbox / (note_id + ".json")).read_text())["content"] == "Review this"
    assert len(list(inbox.iterdir())) == 1
    assert len(calls) == 2  # An acknowledgement can be retried safely.
    companion.handle_action(config, {"type": "shell", "command": "rm -rf /"}, inbox)
    assert len(calls) == 2


def test_screenshot_is_captured_only_for_explicit_request(tmp_path, monkeypatch):
    import mss
    import mss.tools
    class Screen:
        monitors = [object()]
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def grab(self, _):
            return type("Frame", (), {"rgb": b"rgb", "size": (1, 1)})()
    monkeypatch.setattr(mss, "mss", lambda: Screen())
    monkeypatch.setattr(mss.tools, "to_png", lambda rgb, size: b"png")
    calls = []
    monkeypatch.setattr(companion, "_device_request", lambda *args: calls.append(args))
    config = {"server": "https://sky.example", "device_id": "device",
              "device_token": "secret"}
    companion.handle_action(config, {"type": "screenshot_request", "request_id": str(uuid4())})
    assert len(calls) == 1
    assert calls[0][2] == b"png"
    assert calls[0][3] == "image/png"
