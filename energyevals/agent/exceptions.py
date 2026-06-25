from typing import Any


class AgentError(Exception):
    """Base exception for all agent-related errors."""

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.context = context or {}


class ProviderError(AgentError):
    """Exception raised when an LLM provider encounters an error.

    This includes API errors, authentication failures, rate limits, etc.
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context)
        self.provider = provider
        self.model = model


class ContextWindowExceededError(ProviderError):
    """Raised when a request exceeds the model's context window.

    Kept distinct from a generic :class:`ProviderError` so callers can react
    specifically (e.g. prune history and retry) instead of failing the run.
    Covers both surfaces of the same overflow: the pre-flight ``400`` rejection
    ("maximum context length is N tokens") and the upstream ``502`` returned as
    an error body ("input exceeds the context window").
    """

    def __init__(
        self,
        message: str,
        provider: str | None = None,
        model: str | None = None,
        requested_tokens: int | None = None,
        max_context_tokens: int | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, provider=provider, model=model, context=context)
        self.requested_tokens = requested_tokens
        self.max_context_tokens = max_context_tokens


class ToolExecutionError(AgentError):
    """Exception raised when a tool fails to execute properly.

    This includes validation errors, runtime errors, and external API failures.
    """

    def __init__(
        self,
        message: str,
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context)
        self.tool_name = tool_name
        self.arguments = arguments


class ConfigurationError(AgentError):
    """Exception raised when configuration is invalid or missing.

    This includes missing API keys, invalid model names, malformed config files, etc.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        super().__init__(message, context)
        self.config_key = config_key
