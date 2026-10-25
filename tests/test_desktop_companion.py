"""Foreground companion registration and heartbeat behavior without a real desktop."""
# 本文件负责测试用例中的 test、desktop、companion 相关流程。

import json
import os
from pathlib import Path
import subprocess
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import pytest

from skysun_companion import __main__ as companion


# 调用位置：pytest 执行该测试用例时调用。
def test_server_requires_tls_except_loopback():
    """验证远程服务地址必须使用 TLS，本机地址可例外。"""
    assert companion.server_url("https://sky.example/") == "https://sky.example"
    assert companion.server_url("http://localhost:8080/") == "http://localhost:8080"
    with pytest.raises(ValueError):
        companion.server_url("http://sky.example")


# 调用位置：pytest 执行该测试用例时调用。
def test_registration_saves_device_credential_privately(tmp_path, monkeypatch):
    """验证设备注册凭据只保存到本机私有配置文件。"""
    # 以下集中定义 Reply：数据结构或组件相关的数据和操作。
    class Reply:
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def __enter__(self):
            """为测试准备 __enter__ 所需的替身或输入。"""
            return self
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def __exit__(self, *_):
            """为测试准备 __exit__ 所需的替身或输入。"""
            pass
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def read(self, *_):
            """为测试提供 read 所需的输入或模拟行为。"""
            return b'{"device_id":"device-id","device_token":"private-device-token"}'
    requests = []
    # 调用位置：当前测试或 fixture test_registration_saves_device_credential_privately 直接调用，或通过 monkeypatch 作为替身调用。
    def fake_urlopen(request, timeout):
        """为测试提供 fake_urlopen 所需的输入或模拟行为。"""
        requests.append(request)
        return Reply()
    monkeypatch.setattr(companion.getpass, "getpass", lambda _: "private-user-token")
    monkeypatch.setattr(companion, "urlopen", fake_urlopen)
    path = tmp_path / "skysun" / "companion.json"
    assert companion.register("https://sky.example", "My Computer", path) == "device-id"
    assert requests[0].get_header("Authorization") == "Bearer private-user-token"
    assert json.loads(path.read_text())["device_token"] == "private-device-token"
    assert os.stat(path).st_mode & 0o777 == 0o600


# 调用位置：pytest 执行该测试用例时调用。
def test_metrics_and_reconnecting_heartbeat(tmp_path, monkeypatch):
    """验证指标与重连心跳。"""
    path = tmp_path / "companion.json"
    path.write_text(json.dumps({"server": "https://sky.example", "device_id": "id",
                                "device_token": "secret"}))
    monkeypatch.setattr(companion, "collect_metrics", lambda: {
        "type": "heartbeat", "cpu_percent": 35, "memory_percent": 60, "disk_percent": 42})
    # 以下集中定义 Stop：数据结构或组件相关的数据和操作。
    class Stop:
        active = False
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def is_set(self):
            """为测试提供 is_set 所需的输入或模拟行为。"""
            return self.active
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def wait(self, _):
            """为测试提供 wait 所需的输入或模拟行为。"""
            return self.active
    stop = Stop()
    attempts = []
    # 以下集中定义 Channel：数据结构或组件相关的数据和操作。
    class Channel:
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def __enter__(self):
            """为测试准备 __enter__ 所需的替身或输入。"""
            return self
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def __exit__(self, *_):
            """为测试准备 __exit__ 所需的替身或输入。"""
            pass
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def recv(self, timeout):
            """为测试提供 recv 所需的输入或模拟行为。"""
            return json.dumps({"type": "connected" if not hasattr(self, "sent") else "heartbeat_ack"})
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def send(self, value):
            """为测试提供 send 所需的输入或模拟行为。"""
            self.sent = json.loads(value)
            stop.active = True
    # 调用位置：当前测试或 fixture test_metrics_and_reconnecting_heartbeat 直接调用，或通过 monkeypatch 作为替身调用。
    def fake_connect(url, **kwargs):
        """为测试提供 fake_connect 所需的输入或模拟行为。"""
        attempts.append((url, kwargs))
        if len(attempts) == 1:
            raise OSError("offline")
        return Channel()
    monkeypatch.setattr(companion, "connect", fake_connect)
    companion.run(path, stop=stop, interval=0)
    assert len(attempts) == 2
    assert attempts[-1][0] == "wss://sky.example/api/ws/devices/id"
    assert attempts[-1][1]["additional_headers"]["Authorization"] == "Bearer secret"


# 调用位置：pytest 执行该测试用例时调用。
def test_user_selected_workspace_and_confirmed_codex_status(tmp_path, monkeypatch):
    """验证只上报用户选定的工作区和可确认的 Codex 状态。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_only_allowlisted_actions_are_handled_and_notes_are_idempotent(tmp_path, monkeypatch):
    """验证桌面伴随程序只处理允许的动作，便笺处理具有幂等性。"""
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


# 调用位置：pytest 执行该测试用例时调用。
def test_screenshot_is_captured_only_for_explicit_request(tmp_path, monkeypatch):
    """验证只有收到明确截图请求时才采集屏幕。"""
    import mss
    import mss.tools
    # 以下集中定义 Screen：数据结构或组件相关的数据和操作。
    class Screen:
        monitors = [object()]
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def __enter__(self):
            """为测试准备 __enter__ 所需的替身或输入。"""
            return self
        # 调用位置：测试替身模拟相应的上下文、I/O 或同步接口时调用。
        def __exit__(self, *_):
            """为测试准备 __exit__ 所需的替身或输入。"""
            pass
        # 调用位置：desktop-agent/skysun_companion/__main__.py 的 handle_action 调用。
        def grab(self, _):
            """为测试提供 grab 所需的输入或模拟行为。"""
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
