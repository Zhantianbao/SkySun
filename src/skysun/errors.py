"""Shared failure types independent of Agent and storage adapters."""


class RetryableToolError(RuntimeError):
    """An explicitly transient operation failure."""
