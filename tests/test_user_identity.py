# 本文件负责测试用例中的 test、user、identity 相关流程。
import pytest

from skysun.user.postgres_users import PostgresUserStore
from skysun.user.user_identity import ADMIN_USER_ID, LEGACY_USER_ID, Role, normalize_token, normalize_username


# 调用位置：pytest 执行该测试用例时调用。
@pytest.mark.parametrize("name", ["张三", "欧阳明月"])
def test_chinese_names_and_surrounding_spaces(name):
    """验证中文姓名会去除首尾空格并通过校验。"""
    assert normalize_username(f" {name} ") == name


# 调用位置：pytest 执行该测试用例时调用。
@pytest.mark.parametrize("name", ["alice", "张", "张三123", "张 三", "", "一二三四五"])
def test_invalid_names_are_rejected(name):
    """验证无效姓名被拒绝。"""
    with pytest.raises(ValueError):
        normalize_username(name)


# 调用位置：pytest 执行该测试用例时调用。
def test_historical_owner_keeps_the_administrator_id():
    """验证历史数据所有者仍保留固定管理员标识。"""
    assert LEGACY_USER_ID == ADMIN_USER_ID
    assert str(LEGACY_USER_ID) == "00000000-0000-0000-0000-000000000001"


# 调用位置：pytest 执行该测试用例时调用。
@pytest.mark.parametrize("token", ["1", "邀请码！", "a" * 513])
def test_tokens_have_no_length_or_character_set_restriction(token):
    """验证令牌不限制长度或字符种类。"""
    assert normalize_token(f" {token} ") == token


# 调用位置：pytest 执行该测试用例时调用。
@pytest.mark.parametrize("token", ["", " ", "\t\n"])
def test_empty_tokens_are_rejected_before_database_access(token, monkeypatch):
    """验证空令牌在访问数据库前就被拒绝。"""
    users = PostgresUserStore("unused")

    # 调用位置：当前测试或 fixture test_empty_tokens_are_rejected_before_database_access 直接调用，或通过 monkeypatch 作为替身调用。
    def unexpected_connection():
        """为测试提供 unexpected_connection 所需的输入或模拟行为。"""
        pytest.fail("Invalid tokens must be rejected before connecting")

    monkeypatch.setattr(users, "_connect", unexpected_connection)
    with pytest.raises(ValueError, match="邀请码不能为空"):
        users.bootstrap_admin(token)
    for role in Role:
        with pytest.raises(ValueError, match="邀请码不能为空"):
            users.create_user("张三", role, token=token)
