from dataclasses import dataclass
from typing import Any


@dataclass
class ModelSpec:
    """Specification for a single model to evaluate.

    Attributes:
        provider: The provider name (openrouter).
        model: The model identifier (an OpenRouter model slug).
        effort: OpenRouter reasoning.effort; None uses the provider default.
        provider_routing: OpenRouter provider-routing object, forwarded
            verbatim as the request ``provider`` field (e.g.
            ``{"only": ["wandb"]}`` to pin the upstream inference provider).
            None leaves routing to OpenRouter's default price/uptime load
            balancing -- which may land on an upstream that does not support
            prompt caching.
    """

    provider: str
    model: str
    effort: str | None = None 
    provider_routing: dict[str, Any] | None = None

    @property
    def display_name(self) -> str:
        """Return display name like 'openai/gpt-4o-mini'."""
        return f"{self.provider}/{self.model}"

    @property
    def params_summary(self) -> str:
        """Return a bracketed summary of non-default model params, e.g. '[effort=medium]'."""
        parts = []
        if self.effort is not None:
            parts.append(f"effort={self.effort}")
        if self.provider_routing:
            pinned = self.provider_routing.get("only") or self.provider_routing.get("order")
            parts.append(f"provider={'/'.join(pinned)}" if pinned else "provider_routing")
        return f" [{', '.join(parts)}]" if parts else ""

    @property
    def safe_filename(self) -> str:
        """Return filesystem-safe name for output files."""
        return f"{self.provider}_{self.model.replace('/', '_').replace('.', '-')}"
