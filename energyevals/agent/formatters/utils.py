from typing import Any

from energyevals.agent.schema import ImageContent, Message, TextContent


def separate_system_message(
    messages: list[Message],
) -> tuple[str, list[Message]]:
    """Separate system message from conversation messages.

    Many providers (Anthropic, Google) require system messages to be
    passed separately from conversation messages.

    Args:
        messages: List of messages including potential system message

    Returns:
        Tuple of (system_message_text, remaining_conversation_messages)
    """
    system_msg = ""
    conversation = []

    for msg in messages:
        if msg.role == "system":
            if system_msg:
                system_msg = f"{system_msg}\n\n{msg.content}"
            else:
                system_msg = msg.content
        else:
            conversation.append(msg)

    return system_msg, conversation


def format_multimodal_content(
    content_parts: list[TextContent | ImageContent | dict[str, Any]],
) -> list[dict[str, Any]]:
    """Format multimodal content (text + images) for API requests.

    Handles conversion of TextContent and ImageContent objects into
    the format expected by provider APIs (primarily for Anthropic).

    Args:
        content_parts: List of content parts (text or image)

    Returns:
        List of formatted content blocks for API
    """
    formatted = []

    for part in content_parts:
        if isinstance(part, TextContent):
            formatted.append({
                "type": "text",
                "text": part.text,
            })
        elif isinstance(part, ImageContent):
            image_block: dict[str, Any] = {
                "type": "image",
            }

            if part.image_url:
                image_block["source"] = {
                    "type": "url",
                    "url": part.image_url,
                }
            elif part.image_base64:
                media_type = part.media_type or "image/jpeg"
                image_block["source"] = {
                    "type": "base64",
                    "media_type": media_type,
                    "data": part.image_base64,
                }

            formatted.append(image_block)
        elif isinstance(part, dict):
            formatted.append(part)

    return formatted
