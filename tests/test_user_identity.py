import pytest

from skysun.user.postgres_users import PostgresUserStore
from skysun.user.user_identity import ADMIN_USER_ID, LEGACY_USER_ID, Role, normalize_token, normalize_username


@pytest.mark.parametrize("name", ["张三", "欧阳明月"])
def test_chinese_names_and_surrounding_spaces(name):
    assert normalize_username(f" {name} ") == name


@pytest.mark.parametrize("name", ["alice", "张", "张三123", "张 三", "", "一二三四五"])
def test_invalid_names_are_rejected(name):
    with pytest.raises(ValueError):
        normalize_username(name)


def test_historical_owner_keeps_the_administrator_id():
    assert LEGACY_USER_ID == ADMIN_USER_ID
    assert str(LEGACY_USER_ID) == "00000000-0000-0000-0000-000000000001"


@pytest.mark.parametrize("token", ["1", "邀请码！", "a" * 513])
def test_tokens_have_no_length_or_character_set_restriction(token):
    assert normalize_token(f" {token} ") == token


@pytest.mark.parametrize("token", ["", " ", "\t\n"])
def test_empty_tokens_are_rejected_before_database_access(token, monkeypatch):
    users = PostgresUserStore("unused")

    def unexpected_connection():
        pytest.fail("Invalid tokens must be rejected before connecting")

    monkeypatch.setattr(users, "_connect", unexpected_connection)
    with pytest.raises(ValueError, match="邀请码不能为空"):
        users.bootstrap_admin(token)
    for role in Role:
        with pytest.raises(ValueError, match="邀请码不能为空"):
            users.create_user("张三", role, token=token)
