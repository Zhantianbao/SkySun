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

    @field_validator("deepseek_api_key")
    @classmethod
    def require_api_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("DEEPSEEK_API_KEY must not be empty")
        return value


def load_settings() -> Settings:
    """Load settings; Pydantic reports missing or invalid required values."""
    return Settings()
