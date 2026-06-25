from dataclasses import dataclass
from typing import Any


@dataclass
class TextContent:
    """Text content in a message."""

    type: str = "text"
    text: str = ""


@dataclass
class ImageContent:
    """Image content in a message (base64 encoded or URL)."""

    type: str = "image"
    image_base64: str = ""
    image_url: str | None = None
    media_type: str = "image/jpeg"


ContentPart = TextContent | ImageContent


@dataclass
class Message:
    """Represents a message in a conversation.

    Attributes:
        role: The role of the message sender ("system", "user", "assistant", "tool").
        content: The text content of the message (for simple text messages).
        content_parts: List of content parts for multi-modal messages.
        tool_calls: List of tool calls made by the assistant.
        tool_call_id: ID of the tool call this message is responding to.
        name: Name of the tool for tool messages.
    """

    role: str
    content: str = ""
    content_parts: list[ContentPart] | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    name: str | None = None
    # When True, the provider should emit a prompt-cache breakpoint at this
    # message. On OpenRouter this becomes a `cache_control: ephemeral` marker
    # on the content block; providers without prompt caching ignore it.
    cache: bool = False

    @property
    def has_images(self) -> bool:
        """Check if message contains images."""
        if not self.content_parts:
            return False
        return any(
            (isinstance(p, ImageContent) or (isinstance(p, dict) and p.get("type") == "image"))
            for p in self.content_parts
        )

    @property
    def text_content(self) -> str:
        """Get concatenated text content."""
        if self.content:
            return self.content
        if not self.content_parts:
            return ""
        texts = []
        for part in self.content_parts:
            if isinstance(part, TextContent):
                texts.append(part.text)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        return "\n".join(texts)
