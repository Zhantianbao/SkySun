"""Owner-scoped desktop device registration and latest heartbeat state."""

import hashlib
import secrets
from dataclasses import dataclass
from datetime import datetime
from uuid import UUID, uuid4

import psycopg

from skysun.postgres_users import ensure_user_schema


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


class PostgresDeviceStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._database_url, connect_timeout=5)

    def initialize_schema(self) -> None:
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

    def register(self, user_id: UUID, name: str, hostname: str, operating_system: str) -> tuple[UUID, str]:
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

    def authenticate(self, device_id: UUID, token: str) -> UUID | None:
        if not token or len(token) > 512:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            row = conn.execute("""
                SELECT user_id FROM devices WHERE device_id = %s AND token_hash = %s
            """, (device_id, digest)).fetchone()
        return row[0] if row else None

    def list_for_user(self, user_id: UUID) -> tuple[Device, ...]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT d.device_id, d.user_id, d.name, d.hostname, d.operating_system,
                       d.created_at, s.last_heartbeat_at,
                       (s.connected_session IS NOT NULL AND s.last_heartbeat_at > now() - interval '90 seconds'),
                       s.cpu_percent, s.memory_percent, s.disk_percent
                FROM devices d JOIN device_status s USING (device_id)
                WHERE d.user_id = %s ORDER BY d.created_at DESC
            """, (user_id,)).fetchall()
        return tuple(Device(*row) for row in rows)

    def delete(self, user_id: UUID, device_id: UUID) -> bool:
        with self._connect() as conn:
            row = conn.execute("""
                DELETE FROM devices WHERE device_id = %s AND user_id = %s RETURNING device_id
            """, (device_id, user_id)).fetchone()
        return row is not None

    def connect(self, device_id: UUID, session_id: UUID) -> None:
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE device_status SET connected_session = %s
                WHERE device_id = %s RETURNING device_id
            """, (session_id, device_id)).fetchone()
            if row is None:
                raise KeyError(device_id)

    def heartbeat(self, device_id: UUID, session_id: UUID,
                  cpu_percent: float, memory_percent: float, disk_percent: float) -> bool:
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE device_status SET last_heartbeat_at = now(),
                    cpu_percent = %s, memory_percent = %s, disk_percent = %s
                WHERE device_id = %s AND connected_session = %s RETURNING device_id
            """, (cpu_percent, memory_percent, disk_percent, device_id, session_id)).fetchone()
        return row is not None

    def disconnect(self, device_id: UUID, session_id: UUID) -> None:
        with self._connect() as conn:
            conn.execute("""
                UPDATE device_status SET connected_session = NULL
                WHERE device_id = %s AND connected_session = %s
            """, (device_id, session_id))
