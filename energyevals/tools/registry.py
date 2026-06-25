from collections.abc import Callable
from importlib.metadata import entry_points
from typing import Any

from loguru import logger

from energyevals.agent.providers import ToolDefinition
from energyevals.agent.schema.tools import ToolResult


class ToolRegistry:
    """Registry for managing multiple tools."""

    def __init__(self) -> None:
        self._tools: dict[str, Any] = {}
        self._method_to_tool: dict[str, str] = {}

    def register(self, tool: Any) -> None:
        self._tools[tool.name] = tool

        for tool_def in tool.get_tools():
            self._method_to_tool[tool_def.name] = tool.name

        logger.debug(f"Registered tool: {tool.name}")

    @classmethod
    def discover_tools(cls) -> "ToolRegistry":
        """Auto-discover and register tools from entry points.

        Discovers tools registered via setuptools entry points under the
        'energyevals.tools' group.

        Returns:
            ToolRegistry with all discovered tools registered
        """
        registry = cls()

        try:
            discovered: Any = entry_points(group="energyevals.tools")  # type: ignore[call-arg]
        except TypeError:
            all_eps: Any = entry_points()
            discovered = all_eps.get("energyevals.tools", []) if isinstance(all_eps, dict) else []

        for ep in discovered:
            try:
                tool_cls = ep.load()
                tool = tool_cls()
                registry.register(tool)
                logger.info(f"Discovered and registered tool: {ep.name}")
            except Exception as e:
                logger.warning(f"Failed to load tool '{ep.name}': {e}")

        return registry

    def get_tool_groups(self) -> dict[str, set[str]]:
        """Return a mapping of parent tool name -> set of method names."""
        groups: dict[str, set[str]] = {}
        for method_name, parent_name in self._method_to_tool.items():
            groups.setdefault(parent_name, set()).add(method_name)
        return groups

    def get_all_tools(self) -> list[ToolDefinition]:
        all_tools = []
        for tool in self._tools.values():
            all_tools.extend(tool.get_tools())
        return all_tools

    async def execute(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Execute a tool by name.

        Args:
            tool_name: Name of the tool method to execute.
            **kwargs: Arguments for the tool.

        Returns:
            ToolResult with the execution outcome.
        """
        if tool_name not in self._method_to_tool:
            return ToolResult(
                success=False,
                data=None,
                error=f"Unknown tool: {tool_name}",
            )

        parent_tool_name = self._method_to_tool[tool_name]
        tool = self._tools[parent_tool_name]

        return await tool.execute(tool_name, **kwargs)

    def get_executor(self) -> Callable[..., Any]:
        """Get a tool executor function for use with ReActAgent."""

        async def executor(tool_name: str, arguments: dict[str, Any]) -> str:
            result = await self.execute(tool_name, **arguments)
            return result.to_json()

        return executor

    async def aclose(self) -> None:
        """Release resources held by registered tools.

        Tools that hold external resources (e.g. database connection pools) may
        expose an async ``aclose()``; this calls each one so the caller can tear
        them down deterministically instead of relying on process exit.
        """
        for tool in self._tools.values():
            aclose = getattr(tool, "aclose", None)
            if callable(aclose):
                try:
                    await aclose()
                except Exception as e:
                    logger.warning(f"Error closing tool '{getattr(tool, 'name', tool)}': {e}")
