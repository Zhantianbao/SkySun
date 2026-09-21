"""DeepSeek model construction, isolated from SkySun's API and tools."""

from langchain_deepseek import ChatDeepSeek

from skysun.config import Settings


def create_deepseek(settings: Settings) -> ChatDeepSeek:
    """Build the configured DeepSeek chat model without making a network call."""
    return ChatDeepSeek(
        model=settings.skysun_model,
        api_key=settings.deepseek_api_key.get_secret_value(),
        api_base=settings.deepseek_base_url,
        timeout=30,
        max_retries=2,
    )
