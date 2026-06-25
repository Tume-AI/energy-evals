from energyevals.tools.base_tool import BaseTool, ToolResult, tool_method
from energyevals.tools.registry import ToolRegistry
from energyevals.tools.battery_tool import BatteryOptimizationTool
from energyevals.tools.dockets import (
    DCDocketTool,
    FERCDocketTool,
    MarylandDocketTool,
    NewYorkDocketTool,
    NorthCarolinaDocketTool,
    SouthCarolinaDocketTool,
    TexasDocketTool,
    VirginiaDocketTool,
)
from energyevals.tools.gridstatus_tool import GridStatusAPITool
from energyevals.tools.openweather_tool import OpenWeatherTool
from energyevals.tools.renewables_tool import RenewablesTool
from energyevals.tools.search_tool import SearchTool
from energyevals.tools.system_tool import SystemTool
from energyevals.tools.tariffs_tool import TariffsTool

# DatabaseTool ships as a separate optional package (energyevals_db).
# When installed it auto-registers; otherwise the framework runs without it.
try:
    from energyevals_db.tool import DatabaseTool
except ImportError:
    DatabaseTool = None

__all__ = [
    "BaseTool",
    "ToolRegistry",
    "ToolResult",
    "tool_method",
    "GridStatusAPITool",
    "SearchTool",
    "TariffsTool",
    "RenewablesTool",
    "BatteryOptimizationTool",
    "OpenWeatherTool",
    "SystemTool",
    "DCDocketTool",
    "FERCDocketTool",
    "MarylandDocketTool",
    "NewYorkDocketTool",
    "NorthCarolinaDocketTool",
    "SouthCarolinaDocketTool",
    "TexasDocketTool",
    "VirginiaDocketTool",
]


def create_default_registry() -> ToolRegistry:
    """Create a tool registry with all default tools registered.

    Returns:
        ToolRegistry with standard energy analytics tools.
    """
    registry = ToolRegistry()

    registry.register(SearchTool())
    registry.register(GridStatusAPITool())
    registry.register(TariffsTool())
    registry.register(RenewablesTool())
    registry.register(BatteryOptimizationTool())
    registry.register(OpenWeatherTool())
    registry.register(SystemTool())

    if DatabaseTool is not None:
        registry.register(DatabaseTool())

    registry.register(FERCDocketTool())
    registry.register(MarylandDocketTool())
    registry.register(TexasDocketTool())
    registry.register(NewYorkDocketTool())
    registry.register(NorthCarolinaDocketTool())
    registry.register(SouthCarolinaDocketTool())
    registry.register(VirginiaDocketTool())
    registry.register(DCDocketTool())

    return registry
