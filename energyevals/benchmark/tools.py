import logging
from typing import Any

from energyevals.agent.exceptions import ToolExecutionError
from energyevals.agent.schema import ToolDefinition, ToolExecutor
from energyevals.tools.registry import ToolRegistry

from energyevals.benchmark.config import ToolsConfig
from energyevals.benchmark.constants import TOOL_DESCRIPTION_PREVIEW_LENGTH
from energyevals.benchmark.display import print_header

logger = logging.getLogger(__name__)


def _expand_names(names: list[str], groups: dict[str, set[str]]) -> set[str]:
    """Expand a mix of group names and individual tool names into a flat set."""
    expanded: set[str] = set()
    for name in names:
        if name in groups:
            expanded |= groups[name]
        else:
            expanded.add(name)
    return expanded


def filter_tools(
    all_tools: list[ToolDefinition],
    config: ToolsConfig,
    registry: ToolRegistry | None = None,
) -> list[ToolDefinition]:
    """Filter tools based on include/exclude configuration.

    Both parent tool names (e.g. ``"search"``, ``"openweather"``) and individual
    tool method names (e.g. ``"search_web"``) are accepted in include/exclude
    lists.  Parent tool names are resolved dynamically from the *registry*.

    Args:
        all_tools: List of all available tools
        config: Tools configuration with include/exclude lists
        registry: Optional tool registry used to resolve group names.
            When ``None``, names are matched against individual method names only.

    Returns:
        Filtered list of tools
    """
    if not config.enabled:
        return []

    groups = registry.get_tool_groups() if registry else {}

    if config.include:
        included_names = _expand_names(config.include, groups)
        tools = [t for t in all_tools if t.name in included_names]
        logger.info(f"Including only specified tools: {config.include} -> {included_names}")
    else:
        if config.exclude:
            excluded_names = _expand_names(config.exclude, groups)
            tools = [t for t in all_tools if t.name not in excluded_names]
            logger.info(f"Excluding tools: {config.exclude} -> {excluded_names}")
        else:
            tools = all_tools

    return tools


def list_tools(registry: ToolRegistry) -> int:
    """List all available tools.

    Args:
        registry: Tool registry to list tools from.

    Returns:
        Exit code (0 for success)
    """
    print_header("Available Tools")

    print("\n  Standard Tools:")
    for tool in registry.get_all_tools():
        desc = (
            tool.description[:TOOL_DESCRIPTION_PREVIEW_LENGTH] + "..."
            if len(tool.description) > TOOL_DESCRIPTION_PREVIEW_LENGTH
            else tool.description
        )
        print(f"    - {tool.name}: {desc}")

    return 0


def build_tool_executor(registry: ToolRegistry) -> ToolExecutor:
    """Build a tool executor for the given tool registry."""
    tool_names = {tool.name for tool in registry.get_all_tools()}

    async def executor(tool_name: str, arguments: dict[str, Any]) -> str:
        if tool_name in tool_names:
            result = await registry.execute(tool_name, **arguments)
            return result.to_json()
        raise ToolExecutionError(f"Unknown tool: {tool_name}", tool_name=tool_name)

    return executor
