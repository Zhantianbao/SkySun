"""Owner-scoped desktop device registration and latest heartbeat state."""
# 本文件负责设备通信中的 devices 相关流程。

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

import psycopg

from skysun.user.postgres_users import ensure_user_schema


# 以下集中定义 Device：设备数据相关的数据和操作。
@dataclass(frozen=True)
class Device:
    device_id: UUID
    user_id: UUID
    name: str
    hostname: str
    operating_system: str
    created_at: datetime
    last_heartbeat_at: datetime | None
    online: bool
    cpu_percent: float | None
    memory_percent: float | None
    disk_percent: float | None
    project_name: str | None = None
    git_branch: str | None = None
    git_modified_count: int | None = None
    last_activity_at: datetime | None = None
    vscode_running: bool | None = None
    codex_active: bool | None = None
    vscode_workspace: str | None = None


# 以下集中定义 PostgresDeviceStore：存储层相关的数据和操作。
class PostgresDeviceStore:
    # 调用位置：创建 PostgresDeviceStore 实例时调用。
    def __init__(self, database_url: str) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._database_url = database_url

    # 调用位置：PostgresDeviceStore 的 initialize_schema 与数据库读写方法调用。
    def _connect(self) -> psycopg.Connection:
        """建立数据库连接，并统一连接参数。"""
        return psycopg.connect(self._database_url, connect_timeout=5)

    # 调用位置：core/main.py 的 create_app 初始化存储时调用；相关测试也会直接调用。
    def initialize_schema(self) -> None:
        """创建或更新当前存储所需的数据表和索引。"""
        with self._connect() as conn:
            ensure_user_schema(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS devices (
                    device_id UUID PRIMARY KEY,
                    user_id UUID NOT NULL REFERENCES skysun_users(user_id) ON DELETE CASCADE,
                    name VARCHAR(100) NOT NULL,
                    hostname VARCHAR(255) NOT NULL,
                    operating_system VARCHAR(100) NOT NULL,
                    token_hash CHAR(64) NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS devices_owner_idx ON devices (user_id, created_at DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_status (
                    device_id UUID PRIMARY KEY REFERENCES devices(device_id) ON DELETE CASCADE,
                    connected_session UUID,
                    last_heartbeat_at TIMESTAMPTZ,
                    cpu_percent DOUBLE PRECISION CHECK (cpu_percent BETWEEN 0 AND 100),
                    memory_percent DOUBLE PRECISION CHECK (memory_percent BETWEEN 0 AND 100),
                    disk_percent DOUBLE PRECISION CHECK (disk_percent BETWEEN 0 AND 100)
                )
            """)
            for column, definition in (
                ("project_name", "VARCHAR(100)"),
                ("git_branch", "VARCHAR(255)"),
                ("git_modified_count", "INTEGER"),
                ("last_activity_at", "TIMESTAMPTZ"),
                ("vscode_running", "BOOLEAN"),
                ("codex_active", "BOOLEAN"),
                ("vscode_workspace", "VARCHAR(100)"),
            ):
                conn.execute(f"ALTER TABLE device_status ADD COLUMN IF NOT EXISTS {column} {definition}")

    # 调用位置：通过 PostgresDeviceStore 实例在设备路由及桌面伴随程序中调用。
    def register(self, user_id: UUID, name: str, hostname: str, operating_system: str) -> tuple[UUID, str]:
        """登记设备并生成设备凭据。"""
        device_id = uuid4()
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("""
                INSERT INTO devices (device_id, user_id, name, hostname, operating_system, token_hash)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (device_id, user_id, name, hostname, operating_system, digest))
            conn.execute("INSERT INTO device_status (device_id) VALUES (%s)", (device_id,))
        return device_id, token

    # 调用位置：通过 PostgresDeviceStore 实例在设备路由及桌面伴随程序中调用。
    def authenticate(self, device_id: UUID, token: str) -> UUID | None:
        """核对凭据并返回对应身份。"""
        if not token or len(token) > 512:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            row = conn.execute("""
                SELECT user_id FROM devices WHERE device_id = %s AND token_hash = %s
            """, (device_id, digest)).fetchone()
        return row[0] if row else None

    # 调用位置：通过 PostgresDeviceStore 实例在设备路由及桌面伴随程序中调用。
    def list_for_user(self, user_id: UUID) -> tuple[Device, ...]:
        """列出当前用户的设备。"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT d.device_id, d.user_id, d.name, d.hostname, d.operating_system,
                       d.created_at, s.last_heartbeat_at,
                       (s.connected_session IS NOT NULL AND s.last_heartbeat_at > now() - interval '90 seconds'),
                       s.cpu_percent, s.memory_percent, s.disk_percent,
                       s.project_name, s.git_branch, s.git_modified_count,
                       s.last_activity_at, s.vscode_running, s.codex_active,
                       s.vscode_workspace
                FROM devices d JOIN device_status s USING (device_id)
                WHERE d.user_id = %s ORDER BY d.created_at DESC
            """, (user_id,)).fetchall()
        return tuple(Device(*row) for row in rows)

    # 调用位置：device/device_api.py 的删除设备路由调用。
    def delete(self, user_id: UUID, device_id: UUID) -> bool:
        """删除指定记录并检查所有权。"""
        with self._connect() as conn:
            row = conn.execute("""
                DELETE FROM devices WHERE device_id = %s AND user_id = %s RETURNING device_id
            """, (device_id, user_id)).fetchone()
        return row is not None

    # 调用位置：通过 PostgresDeviceStore 实例在设备路由及桌面伴随程序中调用。
    def connect(self, device_id: UUID, session_id: UUID) -> None:
        """标记设备连接并更新在线状态。"""
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE device_status SET connected_session = %s
                WHERE device_id = %s RETURNING device_id
            """, (session_id, device_id)).fetchone()
            if row is None:
                raise KeyError(device_id)

    # 调用位置：通过 PostgresDeviceStore 实例在设备路由及桌面伴随程序中调用。
    def heartbeat(self, device_id: UUID, session_id: UUID,
                  cpu_percent: float, memory_percent: float, disk_percent: float) -> bool:
        """保存设备心跳和运行指标。"""
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE device_status SET last_heartbeat_at = now(),
                    cpu_percent = %s, memory_percent = %s, disk_percent = %s
                WHERE device_id = %s AND connected_session = %s RETURNING device_id
            """, (cpu_percent, memory_percent, disk_percent, device_id, session_id)).fetchone()
        return row is not None

    # 调用位置：通过 PostgresDeviceStore 实例在设备路由及桌面伴随程序中调用。
    def disconnect(self, device_id: UUID, session_id: UUID) -> None:
        """标记设备断开连接。"""
        with self._connect() as conn:
            conn.execute("""
                UPDATE device_status SET connected_session = NULL
                WHERE device_id = %s AND connected_session = %s
            """, (device_id, session_id))

    # 调用位置：通过 PostgresDeviceStore 实例在设备路由及桌面伴随程序中调用。
    def update_development(self, device_id: UUID, session_id: UUID, state: dict) -> None:
        """更新设备上报的开发环境信息。"""
        with self._connect() as conn:
            conn.execute("""
                UPDATE device_status SET project_name = %s, git_branch = %s,
                    git_modified_count = %s, last_activity_at = %s,
                    vscode_running = %s, codex_active = %s,
                    vscode_workspace = %s
                WHERE device_id = %s AND connected_session = %s
            """, (state.get("project_name"), state.get("git_branch"),
                  state.get("git_modified_count"), state.get("last_activity_at"),
                  state.get("vscode_running"), state.get("codex_active"),
                  state.get("vscode_workspace"),
                  device_id, session_id))
