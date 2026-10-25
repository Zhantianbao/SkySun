"""Shared failure types independent of Agent and storage adapters."""
# 本文件负责工具调用中的 errors 相关流程。


# 以下集中定义 RetryableToolError：数据结构或组件相关的数据和操作。
class RetryableToolError(RuntimeError):
    """An explicitly transient operation failure."""
