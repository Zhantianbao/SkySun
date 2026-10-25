import secrets

import pytest

from skysun.user.user_identity import Role


@pytest.fixture
def new_username():
    """Generate names that follow the application's Chinese-only account rule."""
    def generate():
        return "".join(chr(0x4E00 + secrets.randbelow(0x9FFF - 0x4E00 + 1)) for _ in range(4))

    return generate


@pytest.fixture
def create_account(new_username):
    def create(users, role=Role.USER):
        return users.create_user(new_username(), role, token=secrets.token_urlsafe(32))

    return create
