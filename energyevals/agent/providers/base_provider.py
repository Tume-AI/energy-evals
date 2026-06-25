from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any

from energyevals.agent.constants import MAX_TOKENS
from energyevals.agent.schema import (
    ImageContent,
    Message,
    ProviderResponse,
    TextContent,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    "BaseProvider",
    "ImageContent",
    "Message",
    "ProviderResponse",
    "TextContent",
    "ToolCall",
    "ToolDefinition",
]


class BaseProvider(ABC):
    """Abstract base class for LLM providers.

    All provider implementations must inherit from this class and implement
    the required abstract methods.
    """

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        **kwargs: Any,
    ):
        """Initialize the provider.

        Args:
            model: The model identifier to use.
            api_key: API key for authentication. If None, will try to load from environment.
            base_url: Optional base URL for the API.
            **kwargs: Additional provider-specific configuration.
        """
        self.model = model
        self.api_key = api_key
        self.base_url = base_url
        self.config = kwargs

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of this provider."""
        pass

    @abstractmethod
    async def complete(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = MAX_TOKENS,
        system_prompt: str | None = None,
        stop_sequences: list[str] | None = None,
        **kwargs: Any,
    ) -> ProviderResponse:
        """Generate a completion from the model.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tools available to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            system_prompt: Optional system prompt override (provider-specific handling).
            stop_sequences: Optional list of stop sequences.
            **kwargs: Additional parameters passed through to OpenRouter's
                OpenAI-compatible API (e.g. ``reasoning_effort`` for reasoning
                models, ``tool_choice`` to control tool selection).

        Returns:
            ProviderResponse containing the model's response.
        """
        pass

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        tools: list[ToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = MAX_TOKENS,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Stream a completion from the model.

        Args:
            messages: List of messages in the conversation.
            tools: Optional list of tools available to the model.
            temperature: Sampling temperature (0.0 to 1.0).
            max_tokens: Maximum tokens to generate.
            **kwargs: Additional provider-specific parameters.

        Yields:
            String chunks as they are generated.
        """
        yield ""
        raise NotImplementedError

    @abstractmethod
    def format_tools(self, tools: list[ToolDefinition]) -> list[dict[str, Any]]:
        """Format tools for this provider's API format.

        Args:
            tools: List of tool definitions in standard format.

        Returns:
            Tools formatted for the provider's API.
        """
        pass

    @abstractmethod
    def format_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Format messages for this provider's API format.

        Args:
            messages: List of messages in standard format.

        Returns:
            Messages formatted for the provider's API.
        """

    @staticmethod
    def _extract_content_parts(content: list) -> list["TextContent | ImageContent"]:
        """Normalize a content_parts list into typed TextContent/ImageContent objects.

        Handles both already-typed objects and raw dicts so providers share the
        parsing logic and only need to implement provider-specific formatting.

        Args:
            content: List that may contain TextContent, ImageContent, or raw dicts.

        Returns:
            List of TextContent and ImageContent objects.
        """
        parts: list[TextContent | ImageContent] = []
        for part in content:
            if isinstance(part, (TextContent, ImageContent)):
                parts.append(part)
            elif isinstance(part, dict):
                if part.get("type") == "text":
                    parts.append(TextContent(text=part.get("text", "")))
                elif part.get("type") == "image":
                    parts.append(ImageContent(
                        image_base64=part.get("image_base64", ""),
                        media_type=part.get("media_type", "image/jpeg"),
                        image_url=part.get("image_url"),
                    ))
        return parts

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model={self.model!r})"

    def __str__(self) -> str:
        return self.__repr__()

    def __getstate__(self) -> dict:
        """Exclude api_key from pickle serialization."""
        state = self.__dict__.copy()
        state.pop("api_key", None)
        return state
