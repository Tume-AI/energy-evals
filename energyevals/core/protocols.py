from collections.abc import Awaitable
from typing import Any, Protocol

from energyevals.agent.schema import Message


class MessageFormatter(Protocol):
    """Protocol for message formatters used by providers.

    Message formatters handle provider-specific formatting of messages,
    such as separating system messages or formatting multimodal content.
    """

    def separate_system(
        self, messages: list[Message]
    ) -> tuple[str | None, list[Message]]:
        """Separate system message from conversation messages.

        Args:
            messages: List of messages including potential system message

        Returns:
            Tuple of (system_message_text, remaining_conversation_messages)
        """
        ...

    def format_content(self, content: str | list) -> Any:
        """Format message content for provider API.

        Args:
            content: Message content (string or list of content parts)

        Returns:
            Formatted content for provider
        """
        ...

    def format_tool_call(self, tool_call: dict[str, Any]) -> Any:
        """Format a tool call for provider API.

        Args:
            tool_call: Tool call dictionary

        Returns:
            Formatted tool call for provider
        """
        ...


class ToolExecutor(Protocol):
    """Protocol for tool executor functions.

    Tool executors handle the execution of tool calls, mapping tool names
    to their implementations and handling argument passing.
    """

    def __call__(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> str | Awaitable[str]:
        """Execute a tool call.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool result as string (or awaitable string for async executors)
        """
        ...
