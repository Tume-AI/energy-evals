from typing import Any


class EnergyEvalsError(Exception):
    """Base exception for all EnergyEvals errors.

    All custom exceptions in EnergyEvals should inherit from this class.
    """

    def __init__(self, message: str, context: dict[str, Any] | None = None):
        """Initialize error with message and optional context.

        Args:
            message: Error message
            context: Additional context about the error
        """
        self.message = message
        self.context = context or {}
        super().__init__(message)

    def __str__(self) -> str:
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            return f"{self.message} ({context_str})"
        return self.message


class ToolError(EnergyEvalsError):
    """Base class for tool-related errors.

    Raised when a tool fails to execute properly.
    """

    def __init__(
        self,
        message: str,
        tool_name: str,
        recoverable: bool = True,
        context: dict[str, Any] | None = None,
    ):
        """Initialize tool error.

        Args:
            message: Error message
            tool_name: Name of the tool that failed
            recoverable: Whether the error is recoverable
            context: Additional context
        """
        self.tool_name = tool_name
        self.recoverable = recoverable
        context = context or {}
        context["tool_name"] = tool_name
        context["recoverable"] = recoverable
        super().__init__(message, context)


class APIError(ToolError):
    """Error calling an external API.

    Raised when an API call fails (network error, HTTP error, timeout, etc).
    """

    def __init__(
        self,
        message: str,
        tool_name: str,
        status_code: int | None = None,
        response_body: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize API error.

        Args:
            message: Error message
            tool_name: Name of the tool that made the API call
            status_code: HTTP status code (if applicable)
            response_body: Response body (if available)
            context: Additional context
        """
        self.status_code = status_code
        self.response_body = response_body
        context = context or {}
        if status_code:
            context["status_code"] = status_code
        super().__init__(message, tool_name, recoverable=True, context=context)


class ProviderError(EnergyEvalsError):
    """Error from an LLM provider.

    Raised when an LLM provider fails to generate a response.
    """

    def __init__(
        self,
        message: str,
        provider: str,
        model: str,
        recoverable: bool = True,
        context: dict[str, Any] | None = None,
    ):
        """Initialize provider error.

        Args:
            message: Error message
            provider: Provider name (e.g., "openai", "anthropic")
            model: Model name
            recoverable: Whether the error is recoverable (e.g., rate limit vs auth error)
            context: Additional context
        """
        self.provider = provider
        self.model = model
        self.recoverable = recoverable
        context = context or {}
        context["provider"] = provider
        context["model"] = model
        context["recoverable"] = recoverable
        super().__init__(message, context)


class ConfigurationError(EnergyEvalsError):
    """Configuration error.

    Raised when configuration is invalid or missing required values.
    """

    def __init__(
        self,
        message: str,
        config_key: str | None = None,
        context: dict[str, Any] | None = None,
    ):
        """Initialize configuration error.

        Args:
            message: Error message
            config_key: Configuration key that is invalid/missing
            context: Additional context
        """
        self.config_key = config_key
        context = context or {}
        if config_key:
            context["config_key"] = config_key
        super().__init__(message, context)
