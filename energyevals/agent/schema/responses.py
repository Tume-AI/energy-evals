from dataclasses import dataclass
from typing import Any

from energyevals.agent.schema.tools import ToolCall


@dataclass
class ProviderResponse:
    """Response from an LLM provider.

    Attributes:
        content: The text content of the response.
        tool_calls: List of tool calls requested by the model.
        input_tokens: Number of input tokens used.
        cached_tokens: Number of cached input tokens used (if reported by provider).
        output_tokens: Number of output tokens generated.
        reasoning_tokens: Number of reasoning tokens used (for reasoning models).
        reasoning_content: The model's reasoning trace text, when the provider
            returns one (e.g. OpenRouter ``reasoning``). None when unavailable.
        latency_ms: Response latency in milliseconds.
        model: The model that generated the response.
        finish_reason: Reason the model stopped generating.
        raw_response: The raw response object from the provider.
    """

    content: str
    tool_calls: list[ToolCall] | None = None
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    reasoning_content: str | None = None
    latency_ms: float = 0.0
    model: str = ""
    finish_reason: str | None = None
    raw_response: Any | None = None
