"""Local bearer-token identities; only token hashes are stored."""

import hashlib
import secrets
from uuid import UUID, uuid4

import psycopg

from skysun.user_identity import LEGACY_USER_ID, Role, User, normalize_username


def ensure_user_schema(conn: psycopg.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS skysun_users (
            user_id UUID PRIMARY KEY,
            username TEXT NOT NULL UNIQUE,
            role TEXT NOT NULL CHECK (role IN ('USER', 'ADMIN')),
            token_hash TEXT UNIQUE,
            created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)
    conn.execute("""
        INSERT INTO skysun_users (user_id, username, role)
        VALUES (%s, 'legacy_admin', 'ADMIN') ON CONFLICT (user_id) DO NOTHING
    """, (LEGACY_USER_ID,))


class PostgresUserStore:
    def __init__(self, database_url: str) -> None:
        self._database_url = database_url

    def _connect(self) -> psycopg.Connection:
        try:
            return psycopg.connect(self._database_url, connect_timeout=5)
        except psycopg.Error:
            raise RuntimeError("Cannot connect to SkySun user PostgreSQL") from None

    def initialize_schema(self) -> None:
        with self._connect() as conn:
            ensure_user_schema(conn)

    def bootstrap_admin(self, token: str) -> None:
        """Provision the legacy administrator from a local deployment secret."""
        if len(token) < 32:
            raise ValueError("Administrator token must have at least 32 characters")
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("""
                UPDATE skysun_users SET token_hash = %s, updated_at = now()
                WHERE user_id = %s
            """, (digest, LEGACY_USER_ID))

    def create_user(self, username: str, role: Role = Role.USER) -> tuple[User, str]:
        username = normalize_username(username)
        role = Role(role)
        token = secrets.token_urlsafe(32)
        digest = hashlib.sha256(token.encode()).hexdigest()
        user_id = uuid4()
        with self._connect() as conn:
            row = conn.execute("""
                INSERT INTO skysun_users (user_id, username, role, token_hash)
                VALUES (%s, %s, %s, %s)
                RETURNING user_id, username, role, created_at, updated_at
            """, (user_id, username, role.value, digest)).fetchone()
        return User(row[0], row[1], Role(row[2]), row[3], row[4]), token

    def authenticate(self, token: str) -> User | None:
        if not token or len(token) > 512:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            row = conn.execute("""
                SELECT user_id, username, role, created_at, updated_at
                FROM skysun_users WHERE token_hash = %s
            """, (digest,)).fetchone()
        return User(row[0], row[1], Role(row[2]), row[3], row[4]) if row else None

    def get(self, user_id: UUID) -> User | None:
        with self._connect() as conn:
            row = conn.execute("""
                SELECT user_id, username, role, created_at, updated_at
                FROM skysun_users WHERE user_id = %s
            """, (user_id,)).fetchone()
        return User(row[0], row[1], Role(row[2]), row[3], row[4]) if row else None

    def list_users(self) -> tuple[User, ...]:
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT user_id, username, role, created_at, updated_at
                FROM skysun_users ORDER BY username
            """).fetchall()
        return tuple(User(row[0], row[1], Role(row[2]), row[3], row[4]) for row in rows)
