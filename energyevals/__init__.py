from typing import Any  # noqa: E402

from energyevals.agent import (
    AgentRun,
    BaseProvider,
    Message,
    OpenRouterProvider,
    ReActAgent,
    ToolDefinition,
    get_provider,
)
from energyevals.core import (
    APIError,
    ConfigurationError,
    EnergyEvalsError,
    PathLike,
    ProviderError,
    ToolError,
    ensure_path,
)
from energyevals.tools import (
    BatteryOptimizationTool,
    DCDocketTool,
    FERCDocketTool,
    MarylandDocketTool,
    NewYorkDocketTool,
    NorthCarolinaDocketTool,
    RenewablesTool,
    SearchTool,
    SouthCarolinaDocketTool,
    TariffsTool,
    TexasDocketTool,
    ToolRegistry,
    VirginiaDocketTool,
    create_default_registry,
)


def __getattr__(name: str) -> Any:
    """Lazy import for optional modules."""
    if name == "observability":
        import energyevals.observability
        return energyevals.observability
    if name == "utils":
        import energyevals.utils
        return energyevals.utils
    if name == "benchmark":
        import energyevals.benchmark
        return energyevals.benchmark
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "EnergyEvalsError",
    "ToolError",
    "APIError",
    "ProviderError",
    "ConfigurationError",
    "PathLike",
    "ensure_path",
    "ReActAgent",
    "AgentRun",
    "BaseProvider",
    "OpenRouterProvider",
    "get_provider",
    "Message",
    "ToolDefinition",
    "ToolRegistry",
    "create_default_registry",
    "SearchTool",
    "TariffsTool",
    "RenewablesTool",
    "BatteryOptimizationTool",
    "DCDocketTool",
    "FERCDocketTool",
    "MarylandDocketTool",
    "NewYorkDocketTool",
    "NorthCarolinaDocketTool",
    "SouthCarolinaDocketTool",
    "TexasDocketTool",
    "VirginiaDocketTool",
    "observability",
    "utils",
    "benchmark",
]
