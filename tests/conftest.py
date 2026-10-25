# 本文件负责测试用例中的 conftest 相关流程。
import secrets

import pytest

from skysun.user.user_identity import Role


# 调用位置：pytest 在依赖该 fixture 的测试中调用。
@pytest.fixture
def new_username():
    """为测试生成符合中文姓名规则的随机用户名。"""
    # 调用位置：测试夹具 new_username 返回此函数，创建测试账户时调用。
    def generate():
        """生成符合姓名规则的随机汉字。"""
        return "".join(chr(0x4E00 + secrets.randbelow(0x9FFF - 0x4E00 + 1)) for _ in range(4))

    return generate


# 调用位置：pytest 在依赖该 fixture 的测试中调用。
@pytest.fixture
def create_account(new_username):
    """创建测试账户并返回账户和令牌。"""
    # 调用位置：会话服务或测试夹具通过此接口创建数据。
    def create(users, role=Role.USER):
        """为测试创建带角色和令牌的用户账户。"""
        return users.create_user(new_username(), role, token=secrets.token_urlsafe(32))

    return create
