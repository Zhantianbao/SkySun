"""Bounded, owner-scoped screenshot and reminder requests for desktop companions."""

from uuid import UUID, uuid4

import psycopg

from skysun.user.postgres_users import ensure_user_schema

MAX_SCREENSHOT_BYTES = 5 * 1024 * 1024
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


class PostgresDevelopmentStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self._database_url, connect_timeout=5)

    def initialize_schema(self) -> None:
        with self._connect() as conn:
            ensure_user_schema(conn)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_screenshots (
                    request_id UUID PRIMARY KEY,
                    device_id UUID NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES skysun_users(user_id) ON DELETE CASCADE,
                    status TEXT NOT NULL CHECK (status IN ('PENDING', 'COMPLETED', 'FAILED', 'EXPIRED')),
                    requested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    expires_at TIMESTAMPTZ NOT NULL DEFAULT (now() + interval '2 minutes'),
                    uploaded_at TIMESTAMPTZ,
                    image_data BYTEA CHECK (octet_length(image_data) <= 5242880)
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS device_screenshots_owner_idx
                ON device_screenshots (user_id, requested_at DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_notes (
                    note_id UUID PRIMARY KEY,
                    device_id UUID NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES skysun_users(user_id) ON DELETE CASCADE,
                    content VARCHAR(500) NOT NULL,
                    status TEXT NOT NULL CHECK (status IN ('PENDING', 'DELIVERED')),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
                    delivered_at TIMESTAMPTZ
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS device_notes_owner_idx
                ON device_notes (user_id, created_at DESC)
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS device_events (
                    event_id BIGSERIAL PRIMARY KEY,
                    device_id UUID NOT NULL REFERENCES devices(device_id) ON DELETE CASCADE,
                    user_id UUID NOT NULL REFERENCES skysun_users(user_id) ON DELETE CASCADE,
                    event_type VARCHAR(40) NOT NULL,
                    reference_id UUID,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                )
            """)

    @staticmethod
    def _owned(conn: psycopg.Connection, user_id: UUID, device_id: UUID) -> bool:
        return conn.execute("""
            SELECT 1 FROM devices WHERE device_id = %s AND user_id = %s
        """, (device_id, user_id)).fetchone() is not None

    def request_screenshot(self, user_id: UUID, device_id: UUID) -> UUID:
        request_id = uuid4()
        with self._connect() as conn:
            if not self._owned(conn, user_id, device_id):
                raise KeyError(device_id)
            online = conn.execute("""
                SELECT connected_session IS NOT NULL
                       AND last_heartbeat_at > now() - interval '90 seconds'
                FROM device_status WHERE device_id = %s FOR UPDATE
            """, (device_id,)).fetchone()[0]
            if not online:
                raise ValueError("Device is offline")
            pending = conn.execute("""
                SELECT 1 FROM device_screenshots
                WHERE device_id = %s AND status = 'PENDING' AND expires_at > now()
                LIMIT 1
            """, (device_id,)).fetchone()
            if pending:
                raise ValueError("A screenshot request is already pending")
            conn.execute("""
                INSERT INTO device_screenshots (request_id, device_id, user_id, status)
                VALUES (%s, %s, %s, 'PENDING')
            """, (request_id, device_id, user_id))
            self._event(conn, device_id, user_id, "screenshot_requested", request_id)
        return request_id

    def list_screenshots(self, user_id: UUID, limit: int = 30) -> list[dict]:
        self.cleanup()
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT request_id, device_id, status, requested_at, uploaded_at
                FROM device_screenshots WHERE user_id = %s
                ORDER BY requested_at DESC LIMIT %s
            """, (user_id, limit)).fetchall()
        return [{"request_id": str(r[0]), "device_id": str(r[1]), "status": r[2],
                 "requested_at": r[3].isoformat(),
                 "uploaded_at": r[4].isoformat() if r[4] else None} for r in rows]

    def image(self, user_id: UUID, request_id: UUID) -> bytes:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT image_data FROM device_screenshots
                WHERE request_id = %s AND user_id = %s AND status = 'COMPLETED'
                  AND requested_at > now() - interval '24 hours'
            """, (request_id, user_id)).fetchone()
        if row is None or row[0] is None:
            raise KeyError(request_id)
        return bytes(row[0])

    def upload_screenshot(self, device_id: UUID, request_id: UUID, image: bytes) -> None:
        if len(image) > MAX_SCREENSHOT_BYTES or not image.startswith(PNG_SIGNATURE):
            raise ValueError("Expected PNG screenshot up to 5 MiB")
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE device_screenshots SET status = 'COMPLETED', uploaded_at = now(), image_data = %s
                WHERE request_id = %s AND device_id = %s AND status = 'PENDING'
                  AND expires_at > now() RETURNING user_id
            """, (image, request_id, device_id)).fetchone()
            if row is None:
                raise KeyError(request_id)
            self._event(conn, device_id, row[0], "screenshot_uploaded", request_id)

    def fail_screenshot(self, device_id: UUID, request_id: UUID) -> None:
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE device_screenshots SET status = 'FAILED'
                WHERE request_id = %s AND device_id = %s AND status = 'PENDING'
                RETURNING user_id
            """, (request_id, device_id)).fetchone()
            if row is None:
                raise KeyError(request_id)
            self._event(conn, device_id, row[0], "screenshot_failed", request_id)

    def create_note(self, user_id: UUID, device_id: UUID, content: str) -> UUID:
        note_id = uuid4()
        with self._connect() as conn:
            if not self._owned(conn, user_id, device_id):
                raise KeyError(device_id)
            conn.execute("""
                INSERT INTO device_notes (note_id, device_id, user_id, content, status)
                VALUES (%s, %s, %s, %s, 'PENDING')
            """, (note_id, device_id, user_id, content))
            self._event(conn, device_id, user_id, "note_created", note_id)
        return note_id

    def list_notes(self, user_id: UUID, device_id: UUID, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            if not self._owned(conn, user_id, device_id):
                raise KeyError(device_id)
            rows = conn.execute("""
                SELECT note_id, content, status, created_at, delivered_at
                FROM device_notes WHERE user_id = %s AND device_id = %s
                ORDER BY created_at DESC LIMIT %s
            """, (user_id, device_id, limit)).fetchall()
        return [{"note_id": str(r[0]), "content": r[1], "status": r[2],
                 "created_at": r[3].isoformat(),
                 "delivered_at": r[4].isoformat() if r[4] else None} for r in rows]

    def deliver_note(self, device_id: UUID, note_id: UUID) -> None:
        with self._connect() as conn:
            row = conn.execute("""
                UPDATE device_notes SET status = 'DELIVERED', delivered_at = now()
                WHERE note_id = %s AND device_id = %s AND status = 'PENDING'
                RETURNING user_id
            """, (note_id, device_id)).fetchone()
            if row is None:
                raise KeyError(note_id)
            self._event(conn, device_id, row[0], "note_delivered", note_id)

    def pending_actions(self, device_id: UUID) -> list[dict]:
        self.cleanup()
        with self._connect() as conn:
            captures = conn.execute("""
                SELECT request_id FROM device_screenshots
                WHERE device_id = %s AND status = 'PENDING' AND expires_at > now()
                ORDER BY requested_at LIMIT 5
            """, (device_id,)).fetchall()
            notes = conn.execute("""
                SELECT note_id, content FROM device_notes
                WHERE device_id = %s AND status = 'PENDING'
                ORDER BY created_at LIMIT 10
            """, (device_id,)).fetchall()
        return ([{"type": "screenshot_request", "request_id": str(r[0])} for r in captures]
                + [{"type": "note", "note_id": str(r[0]), "content": r[1]} for r in notes])

    def list_events(self, user_id: UUID, device_id: UUID, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            if not self._owned(conn, user_id, device_id):
                raise KeyError(device_id)
            rows = conn.execute("""
                SELECT event_type, reference_id, created_at FROM device_events
                WHERE user_id = %s AND device_id = %s
                ORDER BY event_id DESC LIMIT %s
            """, (user_id, device_id, limit)).fetchall()
        return [{"event_type": r[0], "reference_id": str(r[1]) if r[1] else None,
                 "created_at": r[2].isoformat()} for r in rows]

    def cleanup(self) -> None:
        with self._connect() as conn:
            conn.execute("""
                UPDATE device_screenshots SET status = 'EXPIRED'
                WHERE status = 'PENDING' AND expires_at <= now()
            """)
            conn.execute("""
                DELETE FROM device_screenshots
                WHERE requested_at < now() - interval '24 hours'
            """)
            conn.execute("""
                DELETE FROM device_notes
                WHERE created_at < now() - interval '30 days'
            """)

    @staticmethod
    def _event(conn: psycopg.Connection, device_id: UUID, user_id: UUID,
               event_type: str, reference_id: UUID) -> None:
        conn.execute("""
            INSERT INTO device_events (device_id, user_id, event_type, reference_id)
            VALUES (%s, %s, %s, %s)
        """, (device_id, user_id, event_type, reference_id))
