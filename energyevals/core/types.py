from enum import StrEnum
from pathlib import Path

PathLike = str | Path


class ProviderName(StrEnum):
    """Known LLM provider identifiers.

    Using StrEnum so values compare equal to plain strings — existing code that
    checks ``model_spec.provider == "openrouter"`` continues to work without
    changes.

    OpenRouter is the only supported provider; it fronts models from many
    vendors through a single OpenAI-compatible API.
    """

    OPENROUTER = "openrouter"


def ensure_path(p: PathLike) -> Path:
    """Convert a path-like object to a Path.

    Args:
        p: String or Path object

    Returns:
        Path object
    """
    return Path(p) if isinstance(p, str) else p
