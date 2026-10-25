"""Local bearer-token identities; only token hashes are stored."""
# 本文件负责用户身份中的 postgres、users 相关流程。

import hashlib
from uuid import UUID, uuid4

import psycopg

from skysun.user.user_identity import ADMIN_USER_ID, Role, User, normalize_token, normalize_username


# 调用位置：src/skysun/device/development.py 的 PostgresDevelopmentStore.initialize_schema、src/skysun/device/devices.py 的 PostgresDeviceStore.initialize_schema、src/skysun/evaluation/postgres_observability.py 的 PostgresObservabilityStore.initialize_schema 等处调用。
def ensure_user_schema(conn: psycopg.Connection) -> None:
    """补齐用户表结构并兼容已有数据。"""
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
        VALUES (%s, '詹天宝', 'ADMIN')
        ON CONFLICT (user_id) DO UPDATE
        SET username = EXCLUDED.username, updated_at = now()
        WHERE skysun_users.username = 'legacy_admin'
    """, (ADMIN_USER_ID,))


# 以下集中定义 PostgresUserStore：存储层相关的数据和操作。
class PostgresUserStore:
    # 调用位置：创建 PostgresUserStore 实例时调用。
    def __init__(self, database_url: str) -> None:
        """保存依赖和初始状态，供该对象后续方法使用。"""
        self._database_url = database_url

    # 调用位置：PostgresUserStore 的 initialize_schema 与数据库读写方法调用。
    def _connect(self) -> psycopg.Connection:
        """建立数据库连接，并统一连接参数。"""
        try:
            return psycopg.connect(self._database_url, connect_timeout=5)
        except psycopg.Error:
            raise RuntimeError("Cannot connect to SkySun user PostgreSQL") from None

    # 调用位置：core/main.py 的 create_app 初始化存储时调用；相关测试也会直接调用。
    def initialize_schema(self) -> None:
        """创建或更新当前存储所需的数据表和索引。"""
        with self._connect() as conn:
            ensure_user_schema(conn)

    # 调用位置：通过 PostgresUserStore 实例在登录接口及各用户数据存储中调用。
    def bootstrap_admin(self, token: str) -> None:
        """同步内置管理员的姓名和令牌，同时保留固定身份及历史数据所有权。"""
        token = normalize_token(token)
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            conn.execute("""
                UPDATE skysun_users
                SET username = '詹天宝', role = 'ADMIN', token_hash = %s, updated_at = now()
                WHERE user_id = %s
            """, (digest, ADMIN_USER_ID))

    # 用于创建用户，返回 User 对象和 token。
    # 调用位置：通过 PostgresUserStore 实例在登录接口及各用户数据存储中调用。
    def create_user(
        self,
        username: str,
        role: Role = Role.USER,
        *,
        token: str,
    ) -> tuple[User, str]:
        """创建新用户并保存令牌摘要。"""
        username = normalize_username(username)
        role = Role(role)
        token = normalize_token(token)
        digest = hashlib.sha256(token.encode()).hexdigest()
        user_id = uuid4()
        with self._connect() as conn:
            row = conn.execute("""
                INSERT INTO skysun_users (user_id, username, role, token_hash)
                VALUES (%s, %s, %s, %s)
                RETURNING user_id, username, role, created_at, updated_at
            """, (user_id, username, role.value, digest)).fetchone()
        return User(row[0], row[1], Role(row[2]), row[3], row[4]), token

    # 调用位置：通过 PostgresUserStore 实例在登录接口及各用户数据存储中调用。
    def authenticate(self, token: str) -> User | None:
        """核对凭据并返回对应身份。"""
        if not token:
            return None
        digest = hashlib.sha256(token.encode()).hexdigest()
        with self._connect() as conn:
            row = conn.execute("""
                SELECT user_id, username, role, created_at, updated_at
                FROM skysun_users WHERE token_hash = %s
            """, (digest,)).fetchone()
        return User(row[0], row[1], Role(row[2]), row[3], row[4]) if row else None

    # 调用位置：core/api.py 的 /me 路由通过用户存储调用。
    def get(self, user_id: UUID) -> User | None:
        """按用户标识读取账户资料。"""
        with self._connect() as conn:
            row = conn.execute("""
                SELECT user_id, username, role, created_at, updated_at
                FROM skysun_users WHERE user_id = %s
            """, (user_id,)).fetchone()
        return User(row[0], row[1], Role(row[2]), row[3], row[4]) if row else None

    # 调用位置：core/api.py 的管理员用户接口调用。
    def list_users(self) -> tuple[User, ...]:
        """列出系统用户。"""
        with self._connect() as conn:
            rows = conn.execute("""
                SELECT user_id, username, role, created_at, updated_at
                FROM skysun_users ORDER BY username
            """).fetchall()
        return tuple(User(row[0], row[1], Role(row[2]), row[3], row[4]) for row in rows)
