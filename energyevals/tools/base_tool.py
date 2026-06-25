import inspect
from abc import ABC
from collections.abc import Callable
from typing import Any

from loguru import logger

from energyevals.agent.providers import ToolDefinition
from energyevals.agent.schema.tools import ToolResult
from energyevals.core.errors import APIError, ToolError

from energyevals.tools.schema_builder import build_parameters_schema, get_method_description


def tool_method(
    name: str | None = None,
    *,
    parameters: dict[str, Any] | None = None,
) -> Callable[..., Any]:
    """Mark a method as an exposed tool.

    The method's docstring (summary paragraph before ``Args:`` / ``Returns:``)
    is used as the LLM-facing tool description so there is a single source of
    truth.  When *parameters* is omitted the JSON Schema is auto-generated from
    the method's type hints and docstring ``Args:``/``Parameters:`` section.

    Args:
        name: Tool name exposed to the LLM. Defaults to the method name.
        parameters: Explicit JSON Schema for the tool's accepted parameters.
            When ``None`` (the default) the schema is built automatically from
            the decorated method's signature and docstring.
    """

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        func._tool_metadata = {  # type: ignore[attr-defined]
            "name": name or func.__name__,
            "parameters": parameters,
        }
        return func

    return decorator


class BaseTool(ABC):
    """Abstract base class for standard tools.

    Subclasses decorate callable methods with :func:`tool_method` to expose
    them as LLM tools.  ``get_tools()`` and method registration are handled
    automatically; subclasses only need ``__init__`` and the decorated methods.
    """

    def __init__(self, name: str, description: str):
        """Initialize the tool.

        Args:
            name: Unique name for the tool.
            description: Description of what the tool does.
        """
        self.name = name
        self.description = description
        self._methods: dict[str, Callable[..., Any]] = {}
        self._tool_definitions: list[ToolDefinition] = []
        self._auto_register_tool_methods()

    def _auto_register_tool_methods(self) -> None:
        """Scan for ``@tool_method``-decorated methods and register them."""
        for attr_name in dir(self):
            if attr_name.startswith("_"):
                continue
            attr = getattr(self, attr_name, None)
            if not (callable(attr) and hasattr(attr, "_tool_metadata")):
                continue
            metadata: dict[str, Any] = attr._tool_metadata
            tool_name: str = metadata["name"]
            self._methods[tool_name] = attr
            params = metadata["parameters"]
            if params is None:
                params = build_parameters_schema(attr)
            self._tool_definitions.append(
                ToolDefinition(
                    name=tool_name,
                    description=get_method_description(attr),
                    parameters=params,
                )
            )
            logger.debug(f"Auto-registered tool method: {tool_name}")

    def get_tools(self) -> list[ToolDefinition]:
        """Return tool definitions built at registration time."""
        return self._tool_definitions

    async def execute(self, method_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool method.

        Args:
            method_name: Name of the method to execute.
            **kwargs: Arguments for the method.

        Returns:
            ToolResult with the execution outcome.
        """
        if method_name not in self._methods:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown method: {method_name}",
            )

        method = self._methods[method_name]

        try:
            result = method(**kwargs)

            if inspect.iscoroutine(result):
                result = await result

            return ToolResult(
                success=True,
                data=result,
                metadata={"method": method_name},
            )

        except ToolError as e:
            logger.warning(f"Tool error in {method_name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"method": method_name, "tool_name": e.tool_name, "recoverable": e.recoverable},
            )

        except APIError as e:
            logger.warning(f"API error in {method_name}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=str(e),
                metadata={
                    "method": method_name,
                    "tool_name": e.tool_name,
                    "status_code": e.status_code,
                    "recoverable": e.recoverable,
                },
            )

        except Exception as e:
            logger.error(f"Unexpected error in {method_name}: {type(e).__name__}: {e}")
            return ToolResult(
                success=False,
                data=None,
                error=f"Unexpected error: {type(e).__name__}: {str(e)}",
                metadata={"method": method_name, "error_type": type(e).__name__},
            )
