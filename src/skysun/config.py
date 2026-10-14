"""Environment-backed SkySun configuration."""

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


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

    @field_validator("deepseek_api_key", "database_url", "dashscope_api_key")
    @classmethod
    def require_nonempty_secret(cls, value: SecretStr | None) -> SecretStr | None:
        if value is not None and not value.get_secret_value().strip():
            raise ValueError("Required configuration must not be empty")
        return value

    @field_validator("dashscope_base_url")
    @classmethod
    def require_embedding_url(cls, value: str | None) -> str | None:
        if value is not None and (not value or "{WorkspaceId}" in value):
            raise ValueError("DASHSCOPE_BASE_URL must contain your actual workspace ID")
        return value


def load_settings() -> Settings:
    """Load settings; Pydantic reports missing or invalid required values."""
    return Settings()
