import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Self

logger = logging.getLogger(__name__)

import yaml

from energyevals.agent.constants import (
    CSV_THRESHOLD,
    HISTORY_WINDOW,
    MAX_TOOL_RESULT_CHARS,
    PROVIDER_MAX_RETRIES,
    PROVIDER_RETRY_BASE_DELAY,
    TOOL_OUTPUT_LOG_DIR,
    TOOL_OUTPUT_LOG_MAX_CHARS,
    TOOL_OUTPUT_LOG_MODE,
    TOOL_OUTPUT_REDACT_SECRETS,
    TOOL_TIMEOUT,
)
from energyevals.agent.schema import ModelSpec
from energyevals.benchmark.constants import DEFAULT_MAX_ITERATIONS
from energyevals.core.errors import ConfigurationError
from energyevals.core.types import ProviderName, ensure_path

VALID_SEED_MODES = {"fixed", "rotate", "random_per_trial"}
VALID_TOOL_OUTPUT_LOG_MODES = {"off", "errors_only", "preview", "full"}

PROVIDER_ENV_VARS: dict[str, str] = {
    ProviderName.OPENROUTER.value: "OPENROUTER_API_KEY",
}


def validate_api_keys(config: "BenchmarkConfig") -> None:
    """Raise ConfigurationError if any required API key env vars are absent.

    Checks only the providers actually used in config.models so that
    unrelated keys are not required.
    """
    missing = []
    for model_spec in config.models:
        env_var = PROVIDER_ENV_VARS.get(model_spec.provider)
        if env_var and not os.getenv(env_var):
            missing.append(f"  - {model_spec.provider}: ${env_var}")
    if missing:
        raise ConfigurationError(
            "Missing required API key(s):\n" +
            "\n".join(missing) +
            "\nSet these environment variables (e.g. in .env) before running."
        )


@dataclass
class ToolsConfig:
    """Tool selection configuration."""

    enabled: bool = True
    include: list[str] = field(default_factory=list)
    exclude: list[str] = field(default_factory=list)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""

    models: list[ModelSpec]
    questions_file: Path
    questions: list[int] | None
    observability_enabled: bool
    observability_output_dir: Path
    observability_run_name: str | None
    max_iterations: int
    results_dir: Path
    save_answers: bool
    num_trials: int = 1
    shuffle: bool = False
    seed: int | None = None
    seed_mode: str = "rotate"
    seeds: list[int] | None = None
    csv_threshold: int = CSV_THRESHOLD
    tool_timeout: float = TOOL_TIMEOUT
    max_retries: int = PROVIDER_MAX_RETRIES
    retry_base_delay: float = PROVIDER_RETRY_BASE_DELAY
    max_tool_result_chars: int = MAX_TOOL_RESULT_CHARS
    history_window: int | None = HISTORY_WINDOW
    tool_output_log_mode: str = TOOL_OUTPUT_LOG_MODE
    tool_output_log_max_chars: int = TOOL_OUTPUT_LOG_MAX_CHARS
    tool_output_log_dir: Path = Path(TOOL_OUTPUT_LOG_DIR)
    tool_output_redact_secrets: bool = TOOL_OUTPUT_REDACT_SECRETS
    tools_config: ToolsConfig = field(default_factory=ToolsConfig)
    config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self.questions_file = ensure_path(self.questions_file)
        self.observability_output_dir = ensure_path(self.observability_output_dir)
        self.results_dir = ensure_path(self.results_dir)
        self.tool_output_log_dir = ensure_path(self.tool_output_log_dir)

        errors = self.validate()
        if errors:
            raise ConfigurationError(
                "Invalid benchmark configuration:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    def validate(self) -> list[str]:
        """Validate configuration and return list of errors.

        Returns:
            List of error messages (empty if valid)
        """
        errors = []

        if not self.models:
            errors.append("At least one model must be specified")

        provider_names = {p.value for p in ProviderName}
        for model in self.models:
            if not model.provider:
                errors.append(f"Model provider is required: {model}")
            if not model.model:
                errors.append(f"Model name is required: {model}")
            if model.provider not in provider_names:
                errors.append(
                    f"Unknown provider '{model.provider}'. "
                    f"Available: {', '.join(sorted(provider_names))}"
                )

        if not self.questions_file.exists():
            errors.append(f"Questions file not found: {self.questions_file}")

        if self.max_iterations < 1:
            errors.append(f"max_iterations must be at least 1, got {self.max_iterations}")

        if self.csv_threshold < 1:
            errors.append(f"csv_threshold must be at least 1, got {self.csv_threshold}")

        if self.tool_timeout <= 0:
            errors.append(f"tool_timeout must be positive, got {self.tool_timeout}")

        if self.max_retries < 0:
            errors.append(f"max_retries must be non-negative, got {self.max_retries}")

        if self.retry_base_delay <= 0:
            errors.append(f"retry_base_delay must be positive, got {self.retry_base_delay}")

        if self.num_trials < 1:
            errors.append(f"num_trials must be at least 1, got {self.num_trials}")

        if self.history_window is not None and not isinstance(self.history_window, int):
            errors.append(
                f"history_window must be an integer or null, got {type(self.history_window).__name__}"
            )

        if self.tool_output_log_mode not in VALID_TOOL_OUTPUT_LOG_MODES:
            errors.append(
                "tool_output_log_mode must be one of "
                f"{sorted(VALID_TOOL_OUTPUT_LOG_MODES)}, got {self.tool_output_log_mode!r}"
            )

        if self.tool_output_log_max_chars < 0:
            errors.append(
                "tool_output_log_max_chars must be non-negative, "
                f"got {self.tool_output_log_max_chars}"
            )

        if not self.tool_output_log_dir:
            errors.append("tool_output_log_dir must be a valid path")

        if self.seed is not None and not isinstance(self.seed, int):
            errors.append(f"seed must be an integer, got {type(self.seed).__name__}")

        if self.seed_mode not in VALID_SEED_MODES:
            errors.append(
                f"seed_mode must be one of {sorted(VALID_SEED_MODES)}, got {self.seed_mode!r}"
            )

        if self.seeds is not None:
            if not isinstance(self.seeds, list):
                errors.append(f"seeds must be a list of integers, got {type(self.seeds).__name__}")
            elif not all(isinstance(s, int) for s in self.seeds):
                errors.append("seeds must contain only integers")
            elif len(self.seeds) != self.num_trials:
                errors.append(
                    f"seeds length ({len(self.seeds)}) must match num_trials ({self.num_trials})"
                )
            if not self.shuffle:
                errors.append("seeds requires shuffle=true")

        if self.questions is not None:
            if not isinstance(self.questions, list):
                errors.append(f"questions must be a list, got {type(self.questions).__name__}")
            elif not all(isinstance(q, int) for q in self.questions):
                errors.append("All question IDs must be integers")
            elif not all(q > 0 for q in self.questions):
                errors.append("All question IDs must be positive")

        return errors

    @classmethod
    def from_dict(cls, data: dict, base_path: Path) -> Self:
        """Create config from dictionary."""
        obs = data.get("observability", {})
        agent = data.get("agent", {})
        output = data.get("output", {})
        tools = data.get("tools", {})

        if "models" not in data:
            raise ConfigurationError(
                "Configuration must include a 'models' list. "
                "Example:\n  models:\n    - provider: openrouter\n      model: openai/gpt-5-mini"
            )

        models = [
            ModelSpec(
                provider=m["provider"],
                model=m["model"],
                effort=m.get("effort"),
                provider_routing=m.get("provider_routing"),
            )
            for m in data["models"]
        ]

        questions = data.get("questions")
        if questions:
            questions = cls.parse_questions(questions)

        if "questions_file" not in data:
            raise ConfigurationError("Configuration must include 'questions_file'")
        questions_file = base_path / str(data["questions_file"])

        tools_config = ToolsConfig(
            enabled=tools.get("enabled", True),
            include=tools.get("include", []),
            exclude=tools.get("exclude", []),
        )

        return cls(
            models=models,
            questions_file=questions_file,
            questions=questions,
            observability_enabled=obs.get("enabled", True),
            observability_output_dir=Path(obs.get("output_dir", "./benchmark_traces")),
            observability_run_name=obs.get("run_name"),
            max_iterations=agent.get("max_iterations", DEFAULT_MAX_ITERATIONS),
            results_dir=Path(output.get("results_dir", "./benchmark_results")),
            save_answers=output.get("save_answers", True),
            num_trials=agent.get("num_trials", 1),
            shuffle=agent.get("shuffle", False),
            seed=agent.get("seed"),
            seed_mode=agent.get("seed_mode", "rotate"),
            seeds=agent.get("seeds"),
            csv_threshold=agent.get("csv_threshold", CSV_THRESHOLD),
            tool_timeout=agent.get("tool_timeout", TOOL_TIMEOUT),
            max_retries=agent.get("max_retries", PROVIDER_MAX_RETRIES),
            retry_base_delay=agent.get("retry_base_delay", PROVIDER_RETRY_BASE_DELAY),
            max_tool_result_chars=agent.get("max_tool_result_chars", MAX_TOOL_RESULT_CHARS),
            history_window=agent.get("history_window", HISTORY_WINDOW),
            tool_output_log_mode=agent.get("tool_output_log_mode", TOOL_OUTPUT_LOG_MODE),
            tool_output_log_max_chars=agent.get(
                "tool_output_log_max_chars", TOOL_OUTPUT_LOG_MAX_CHARS
            ),
            tool_output_log_dir=Path(agent.get("tool_output_log_dir", TOOL_OUTPUT_LOG_DIR)),
            tool_output_redact_secrets=agent.get(
                "tool_output_redact_secrets", TOOL_OUTPUT_REDACT_SECRETS
            ),
            tools_config=tools_config,
        )

    @staticmethod
    def parse_questions(questions: str | list[int] | None) -> list[int] | None:
        """Parse question specification into list of IDs."""
        if questions is None:
            return None
        if isinstance(questions, list):
            return questions
        if isinstance(questions, str):
            result: list[int] = []
            for part in str(questions).split(","):
                part = part.strip()
                if "-" in part:
                    start, end = part.split("-")
                    result.extend(range(int(start), int(end) + 1))
                else:
                    result.append(int(part))
            return result
        raise ValueError(
            f"questions must be null, a list of integers, or a comma/range string, "
            f"got {type(questions).__name__}"
        )


def load_config(config_path: Path | None, base_path: Path) -> BenchmarkConfig:
    """Load configuration from a YAML file.

    Args:
        config_path: Path to YAML config file.
        base_path: Base directory for resolving relative paths.

    Returns:
        Parsed BenchmarkConfig.

    Raises:
        ConfigurationError: If config_path is None or does not exist.
    """
    if config_path is None:
        raise ConfigurationError("A benchmark config file path is required")

    if not config_path.exists():
        raise ConfigurationError(f"Config file not found: {config_path}")
    with open(config_path) as f:
        data = yaml.safe_load(f)
    logger.info(f"Loaded config from: {config_path}")
    config = BenchmarkConfig.from_dict(data, base_path)
    config.config_path = config_path
    return config
