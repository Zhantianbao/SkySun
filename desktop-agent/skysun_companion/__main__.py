"""Foreground desktop companion and one-time registration command."""
# 本文件负责桌面伴随程序的注册、状态采集、心跳连接与设备动作处理。

import argparse
from datetime import datetime, timedelta, timezone
import getpass
import json
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from uuid import UUID

import psutil
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

CONFIG_PATH = Path.home() / ".config" / "skysun" / "companion.json"
INBOX_PATH = Path.home() / ".config" / "skysun" / "inbox"
VSCODE_STATUS_PATH = Path.home() / ".config" / "skysun" / "vscode-status.json"


# 调用位置：desktop-agent/skysun_companion/__main__.py 的 register、desktop-agent/skysun_companion/__main__.py 的 set_context 调用。
def _save_config(path: Path, config: dict) -> None:
    """将设备配置写入本机文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump(config, stream)
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


# 调用位置：desktop-agent/skysun_companion/__main__.py 的 register、desktop-agent/skysun_companion/__main__.py 的 run、tests/test_desktop_companion.py 的 test_server_requires_tls_except_loopback 等处调用。
def server_url(value: str) -> str:
    """校验并规范化服务端地址。"""
    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.hostname:
        return value.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        return value.rstrip("/")
    raise ValueError("Use HTTPS, or HTTP only for localhost development")


# 调用位置：本文件的 main 在 register 子命令中调用；设备注册测试也直接调用。
def register(server: str, name: str, config_path: Path = CONFIG_PATH) -> str:
    """登记设备并生成设备凭据。"""
    server = server_url(server)
    user_token = getpass.getpass("SkySun user access token: ")
    payload = json.dumps({"name": name, "hostname": socket.gethostname(),
                          "operating_system": platform.system()}).encode()
    request = Request(server + "/api/devices", data=payload, method="POST",
                      headers={"Authorization": "Bearer " + user_token,
                               "Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    _save_config(config_path, {"server": server, "device_id": result["device_id"],
                               "device_token": result["device_token"]})
    return result["device_id"]


# 调用位置：desktop-agent/skysun_companion/__main__.py 的 collect_heartbeat 调用。
def collect_metrics() -> dict:
    """采集设备运行指标。"""
    disk = Path.home().anchor or "/"
    return {"type": "heartbeat", "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage(disk).percent}


# 调用位置：desktop-agent/skysun_companion/__main__.py 的 main、tests/test_desktop_companion.py 的 test_user_selected_workspace_and_confirmed_codex_status 调用。
def set_context(config_path: Path, workspace: str, codex_active: bool = False) -> None:
    """记录用户明确选定的工作区，后续状态采集只读取该工作区。"""
    path = Path(workspace).expanduser().resolve(strict=True)
    if not path.is_dir():
        raise ValueError("Workspace must be a directory")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["workspace"] = str(path)
    config["activity_at"] = datetime.now(timezone.utc).isoformat()
    config["codex_confirmed_until"] = (datetime.now(timezone.utc) + timedelta(hours=2)).isoformat() if codex_active else None
    _save_config(config_path, config)


# 调用位置：desktop-agent/skysun_companion/__main__.py 的 collect_heartbeat、tests/test_desktop_companion.py 的 test_user_selected_workspace_and_confirmed_codex_status 等处调用。
def collect_development(config: dict, vscode_status_path: Path = VSCODE_STATUS_PATH) -> dict:
    """读取设备的开发环境状态。"""
    workspace = config.get("workspace")
    state = {"project_name": None, "git_branch": None, "git_modified_count": None,
             "last_activity_at": config.get("activity_at"), "vscode_running": False,
             "codex_active": False, "vscode_workspace": None}
    try:
        state["vscode_running"] = any(
            (process.info["name"] or "").lower() in {"code", "code.exe", "codium", "codium.exe"}
            for process in psutil.process_iter(["name"])
        )
    except (psutil.Error, OSError):
        pass
    until = config.get("codex_confirmed_until")
    if until:
        try:
            state["codex_active"] = datetime.fromisoformat(until) > datetime.now(timezone.utc)
        except (ValueError, TypeError):
            pass
    if not workspace:
        return state
    path = Path(workspace)
    if not path.is_dir():
        return state
    state["project_name"] = path.name[:100]
    try:
        marker = json.loads(vscode_status_path.read_text(encoding="utf-8"))
        updated = datetime.fromisoformat(marker["updated_at"])
        matching = (marker.get("workspace_path") and
                    Path(marker["workspace_path"]).resolve() == path.resolve())
        age = datetime.now(timezone.utc) - updated
        if matching and timedelta(0) <= age < timedelta(seconds=90):
            state["vscode_running"] = True
            state["vscode_workspace"] = path.name[:100]
            activity = marker.get("activity_at")
            if isinstance(activity, str) and (not state["last_activity_at"] or
                                              activity > state["last_activity_at"]):
                state["last_activity_at"] = activity
    except (OSError, KeyError, ValueError, TypeError):
        pass
    # 调用位置：desktop-agent/skysun_companion/__main__.py 的 collect_development 调用。
    def git(*args: str) -> str | None:
        """读取工作区的 Git 状态。"""
        try:
            result = subprocess.run(["git", "--no-optional-locks", "-C", str(path),
                                     "-c", "core.fsmonitor=false", *args], capture_output=True,
                                    text=True, timeout=5, check=True)
            return result.stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None
    state["git_branch"] = (git("branch", "--show-current") or None)
    status = git("status", "--porcelain", "--untracked-files=normal")
    if status is not None:
        state["git_modified_count"] = min(len(status.splitlines()), 100000) if status else 0
        recent = None
        for line in status.splitlines()[:100]:
            try:
                relative = line[3:].split(" -> ")[-1]
                candidate = (path / relative).resolve()
                if candidate.is_relative_to(path) and candidate.exists():
                    modified = candidate.stat().st_mtime
                    recent = max(recent or modified, modified)
            except OSError:
                continue
        if recent:
            observed = datetime.fromtimestamp(recent, timezone.utc).isoformat()
            if not state["last_activity_at"] or observed > state["last_activity_at"]:
                state["last_activity_at"] = observed
    return state


# 调用位置：desktop-agent/skysun_companion/__main__.py 的 run 调用。
def collect_heartbeat(config: dict) -> dict:
    """组合心跳消息所需的设备信息。"""
    return {**collect_metrics(), **collect_development(config)}


# 调用位置：desktop-agent/skysun_companion/__main__.py 的 handle_action 等处调用。
def _device_request(config: dict, path: str, data: bytes = b"", content_type: str = "application/json") -> None:
    """携带设备凭据向服务端发送请求。"""
    request = Request(config["server"] + path, data=data, method="POST",
                      headers={"Authorization": "Bearer " + config["device_token"],
                               "Content-Type": content_type})
    with urlopen(request, timeout=30):
        pass


# 调用位置：desktop-agent/skysun_companion/__main__.py 的 run、tests/test_desktop_companion.py 的 test_only_allowlisted_actions_are_handled_and_notes_are_idempotent、tests/test_desktop_companion.py 的 test_screenshot_is_captured_only_for_explicit_request 等处调用。
def handle_action(config: dict, action: dict, inbox_path: Path = INBOX_PATH) -> None:
    """处理服务端允许的截图或便笺动作。"""
    # 服务端只下发截图请求或便笺；按动作类型分别校验标识并回报处理结果。
    device_path = "/api/devices/" + config["device_id"]
    if action.get("type") == "screenshot_request":
        request_id = action.get("request_id")
        if not isinstance(request_id, str):
            return
        try:
            request_id = str(UUID(request_id))
        except ValueError:
            return
        endpoint = device_path + "/screenshots/" + request_id
        try:
            import mss
            import mss.tools
            with mss.mss() as screen:
                frame = screen.grab(screen.monitors[0])
                image = mss.tools.to_png(frame.rgb, frame.size)
        except Exception:
            _device_request(config, endpoint + "/fail")
            return
        if len(image) > 5 * 1024 * 1024:
            _device_request(config, endpoint + "/fail")
            return
        _device_request(config, endpoint + "/upload", image, "image/png")
    elif action.get("type") == "note":
        note_id, content = action.get("note_id"), action.get("content")
        if not isinstance(note_id, str) or not isinstance(content, str) or len(content) > 500:
            return
        try:
            note_id = str(UUID(note_id))
        except ValueError:
            return
        inbox_path.parent.mkdir(parents=True, exist_ok=True)
        inbox_path.mkdir(parents=True, exist_ok=True)
        note_path = inbox_path / (note_id + ".json")
        try:
            fd = os.open(note_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            pass
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as stream:
                json.dump({"note_id": note_id, "content": content,
                           "received_at": datetime.now(timezone.utc).isoformat()}, stream)
            os.chmod(note_path, 0o600)
        _device_request(config, device_path + "/notes/" + note_id + "/delivered")
        print("New SkySun note received in " + str(inbox_path), flush=True)


# 调用位置：本文件的 main 在 run 子命令中调用；桌面伴随程序测试也直接调用。
def run(config_path: Path = CONFIG_PATH, stop=None, interval: float = 15) -> None:
    """维持设备连接、发送心跳并处理动作。"""
    config = json.loads(config_path.read_text(encoding="utf-8"))
    server = server_url(config["server"])
    ws_url = ("wss://" if server.startswith("https://") else "ws://") + server.split("://", 1)[1]
    ws_url += "/api/ws/devices/" + config["device_id"]
    delay = 1
    # 外层循环负责断线重连；内层连接负责定期心跳和接收待处理动作。
    while stop is None or not stop.is_set():
        try:
            with connect(ws_url, additional_headers={
                "Authorization": "Bearer " + config["device_token"]
            }, open_timeout=10, max_size=65536) as channel:
                if json.loads(channel.recv(timeout=10)).get("type") != "connected":
                    raise RuntimeError("Unexpected companion handshake")
                delay = 1
                print("SkySun connected", flush=True)
                while stop is None or not stop.is_set():
                    try:
                        latest = json.loads(config_path.read_text(encoding="utf-8"))
                        for key in ("workspace", "activity_at", "codex_confirmed_until"):
                            config[key] = latest.get(key)
                    except (OSError, ValueError):
                        pass
                    channel.send(json.dumps(collect_heartbeat(config)))
                    while True:
                        message = json.loads(channel.recv(timeout=10))
                        if message.get("type") == "heartbeat_ack":
                            break
                        handle_action(config, message)
                    deadline = time.monotonic() + interval
                    while (stop is None or not stop.is_set()) and time.monotonic() < deadline:
                        try:
                            action = json.loads(channel.recv(timeout=max(0.01, deadline - time.monotonic())))
                        except TimeoutError:
                            break
                        handle_action(config, action)
        except (OSError, TimeoutError, RuntimeError, ValueError, ConnectionClosed) as exc:
            print(f"SkySun disconnected: {type(exc).__name__}; retrying", file=sys.stderr,
                  flush=True)
            if stop is None:
                time.sleep(delay)
            elif stop.wait(delay):
                break
            delay = min(delay * 2, 60)


# 调用位置：命令行入口调用。
def main() -> None:
    """解析启动参数并进入程序主流程。"""
    parser = argparse.ArgumentParser(description="SkySun foreground desktop companion")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("register", help="Pair this computer with your SkySun account")
    setup.add_argument("--server", required=True, help="HTTPS URL of your SkySun deployment")
    setup.add_argument("--name", default=socket.gethostname())
    sub.add_parser("run", help="Connect and report status until stopped")
    context = sub.add_parser("context", help="Select the local workspace to report")
    context.add_argument("--workspace", required=True)
    context.add_argument("--codex-active", action="store_true",
                         help="Confirm Codex work is active for the next two hours")
    args = parser.parse_args()
    if args.command == "register":
        print("Registered device " + register(args.server, args.name))
    elif args.command == "context":
        set_context(CONFIG_PATH, args.workspace, args.codex_active)
        print("Workspace context updated")
    else:
        try:
            run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
