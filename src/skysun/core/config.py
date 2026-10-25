"""Environment-backed SkySun configuration."""
# 本文件负责核心服务中的 config 相关流程。

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# 以下集中定义 Settings：数据结构或组件相关的数据和操作。
class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        str_strip_whitespace=True,
    )

    deepseek_api_key: SecretStr
    deepseek_base_url: str = "https://api.deepseek.com"
    skysun_model: str = Field(default="deepseek-flash", min_length=1)
    dashscope_api_key: SecretStr | None = None
    dashscope_base_url: str | None = None
    database_url: SecretStr
    skysun_admin_token: SecretStr | None = None

    # 调用位置：通过 Settings 实例在会话服务和应用入口中调用。
    @field_validator("deepseek_api_key", "database_url", "dashscope_api_key", "skysun_admin_token")
    @classmethod
    def require_nonempty_secret(cls, value: SecretStr | None) -> SecretStr | None:
        """检查密钥配置不为空。"""
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("Required configuration must not be empty")
        return value

    # 调用位置：通过 Settings 实例在会话服务和应用入口中调用。
    @field_validator("dashscope_base_url")
    @classmethod
    def require_embedding_url(cls, value: str | None) -> str | None:
        """检查嵌入模型的服务地址。"""
        if value is not None and (not value or "{WorkspaceId}" in value):
            raise ValueError("DASHSCOPE_BASE_URL must contain your actual workspace ID")
        return value


# 调用位置：src/skysun/core/main.py 的 create_app、src/skysun/rag/ingest.py 的 main 调用。
def load_settings() -> Settings:
    """从环境变量读取配置，并由配置模型检查缺失或无效的值。"""
    return Settings()
