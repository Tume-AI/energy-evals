from typing import Any

from energyevals.agent.schema import (
    Message,
    ProviderResponse,
    ToolCall,
    ToolDefinition,
)
from energyevals.core.types import ProviderName

from energyevals.agent.providers.base_provider import BaseProvider
from energyevals.agent.providers.openrouter_provider import OpenRouterProvider

__all__ = [
    # Base classes
    "BaseProvider",
    "Message",
    "ProviderName",
    "ProviderResponse",
    "ToolCall",
    "ToolDefinition",
    # Providers
    "OpenRouterProvider",
]


def get_provider(
    provider_name: str,
    model: str | None = None,
    **kwargs: Any,
) -> BaseProvider:
    """Factory function to get a provider by name.

    OpenRouter is the only supported provider; it routes to models from many
    vendors (OpenAI, Anthropic, Google, DeepSeek, and more) through a single
    OpenAI-compatible API.

    Args:
        provider_name: Name of the provider ("openrouter").
        model: Optional model identifier. If not provided, uses provider default.
        **kwargs: Additional provider configuration.

    Returns:
        Configured provider instance.

    Raises:
        ValueError: If provider_name is not recognized.
    """
    providers: dict[ProviderName, type[BaseProvider]] = {
        ProviderName.OPENROUTER: OpenRouterProvider,
    }

    if provider_name not in providers:
        raise ValueError(
            f"Unknown provider: {provider_name}. "
            f"Available providers: {list(providers.keys())}"
        )

    provider_class = providers[provider_name]  # type: ignore[index]

    if model:
        return provider_class(model=model, **kwargs)
    return provider_class(**kwargs)
