"""Foreground desktop companion and one-time registration command."""

import argparse
import getpass
import json
import os
import platform
import socket
import sys
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import psutil
from websockets.exceptions import ConnectionClosed
from websockets.sync.client import connect

CONFIG_PATH = Path.home() / ".config" / "skysun" / "companion.json"


def server_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme == "https" and parsed.hostname:
        return value.rstrip("/")
    if parsed.scheme == "http" and parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        return value.rstrip("/")
    raise ValueError("Use HTTPS, or HTTP only for localhost development")


def register(server: str, name: str, config_path: Path = CONFIG_PATH) -> str:
    server = server_url(server)
    user_token = getpass.getpass("SkySun user access token: ")
    payload = json.dumps({"name": name, "hostname": socket.gethostname(),
                          "operating_system": platform.system()}).encode()
    request = Request(server + "/api/devices", data=payload, method="POST",
                      headers={"Authorization": "Bearer " + user_token,
                               "Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        result = json.load(response)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(config_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        json.dump({"server": server, "device_id": result["device_id"],
                   "device_token": result["device_token"]}, stream)
    os.chmod(config_path, 0o600)
    return result["device_id"]


def collect_metrics() -> dict:
    disk = Path.home().anchor or "/"
    return {"type": "heartbeat", "cpu_percent": psutil.cpu_percent(interval=None),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage(disk).percent}


def run(config_path: Path = CONFIG_PATH, stop=None, interval: float = 15) -> None:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    server = server_url(config["server"])
    ws_url = ("wss://" if server.startswith("https://") else "ws://") + server.split("://", 1)[1]
    ws_url += "/api/ws/devices/" + config["device_id"]
    delay = 1
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
                    channel.send(json.dumps(collect_metrics()))
                    if json.loads(channel.recv(timeout=10)).get("type") != "heartbeat_ack":
                        raise RuntimeError("Heartbeat not acknowledged")
                    if stop is None:
                        time.sleep(interval)
                    elif stop.wait(interval):
                        break
        except (OSError, TimeoutError, RuntimeError, ValueError, ConnectionClosed) as exc:
            print(f"SkySun disconnected: {type(exc).__name__}; retrying", file=sys.stderr,
                  flush=True)
            if stop is None:
                time.sleep(delay)
            elif stop.wait(delay):
                break
            delay = min(delay * 2, 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="SkySun foreground desktop companion")
    sub = parser.add_subparsers(dest="command", required=True)
    setup = sub.add_parser("register", help="Pair this computer with your SkySun account")
    setup.add_argument("--server", required=True, help="HTTPS URL of your SkySun deployment")
    setup.add_argument("--name", default=socket.gethostname())
    sub.add_parser("run", help="Connect and report status until stopped")
    args = parser.parse_args()
    if args.command == "register":
        print("Registered device " + register(args.server, args.name))
    else:
        try:
            run()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
