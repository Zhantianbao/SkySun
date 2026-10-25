"""DeepSeek model construction, isolated from SkySun's API and tools."""
# 本文件负责核心服务中的 deepseek 相关流程。

from langchain_deepseek import ChatDeepSeek

from skysun.core.config import Settings


# 调用位置：src/skysun/core/main.py 的 create_app 调用。
def create_deepseek(settings: Settings) -> ChatDeepSeek:
    """根据配置创建对话模型对象，此过程不发起网络请求。"""
    return ChatDeepSeek(
        model=settings.skysun_model,
        api_key=settings.deepseek_api_key.get_secret_value(),
        api_base=settings.deepseek_base_url,
        timeout=30,
        max_retries=2,
    )
