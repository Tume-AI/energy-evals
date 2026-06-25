import json
import os
from datetime import datetime
from typing import Any

from loguru import logger


def require_api_key(env_var: str, tool_name: str) -> str:
    """Get an API key from environment variable with validation.

    Args:
        env_var: Name of the environment variable containing the API key
        tool_name: Name of the tool requiring the key (for error messages)

    Returns:
        The API key value

    Raises:
        ValueError: If the API key is not set
    """
    api_key = os.getenv(env_var)
    if not api_key:
        logger.warning(f"{env_var} not set. {tool_name} will not function.")
        raise ValueError(f"{env_var} is required for {tool_name}")
    return api_key


def create_error_response(
    error: str,
    source: str,
    context: dict | None = None,
) -> str:
    """Create a standardized error response in JSON format.

    Args:
        error: Error message
        source: Tool/service name that generated the error
        context: Additional context (e.g., request parameters)

    Returns:
        JSON-formatted error response
    """

    response = {
        "error": error,
        "source": source,
        "context": context or {},
        "timestamp": datetime.now().isoformat(),
    }
    return json.dumps(response, indent=2)


def format_json_response(data: Any, indent: int = 2) -> str:
    """Format data as JSON string with consistent formatting.

    Args:
        data: Data to format
        indent: Number of spaces for indentation

    Returns:
        JSON-formatted string
    """
    return json.dumps(data, indent=indent, default=str)
