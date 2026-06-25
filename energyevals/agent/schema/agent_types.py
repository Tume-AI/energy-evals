import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal

from energyevals.agent.constants import (
    CSV_THRESHOLD,
    MAX_ITERATIONS,
    PROVIDER_MAX_RETRIES,
    PROVIDER_RETRY_BASE_DELAY,
    TOOL_OUTPUT_LOG_DIR,
    TOOL_OUTPUT_LOG_MAX_CHARS,
    TOOL_OUTPUT_LOG_MODE,
    TOOL_OUTPUT_REDACT_SECRETS,
    TOOL_TIMEOUT,
)


class StepType(Enum):
    """Types of steps in the agent's execution."""

    THOUGHT = "thought"
    ACTION = "action"
    OBSERVATION = "observation"
    ANSWER = "answer"
    ERROR = "error"


ToolOutputLogMode = Literal["off", "errors_only", "preview", "full"]


@dataclass
class AgentStep:
    """Represents a single step in the agent's execution.

    Attributes:
        step_type: The type of step (thought, action, observation, answer, error).
        content: The content or description of this step.
        iteration: 0-based index of the ReAct iteration this step belongs to.
        tool_name: Name of the tool called (for action/observation steps).
        tool_input: Input arguments passed to the tool.
        tool_output: Output returned from the tool.
        tokens_used: Number of tokens used in this step.
        latency_ms: Time taken for this step in milliseconds.
        timestamp: Unix timestamp when this step occurred.
        reasoning: The model's reasoning trace text for this step, when the
            provider returns one (e.g. OpenRouter). None when unavailable.
    """

    step_type: StepType
    content: str
    iteration: int = 0
    tool_name: str | None = None
    tool_input: dict[str, Any] | None = None
    tool_output: str | None = None
    tokens_used: int = 0
    latency_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    reasoning: str | None = None


@dataclass
class AgentRun:
    """Represents a complete agent execution run.

    Attributes:
        query: The original user query.
        steps: List of all steps taken during execution.
        final_answer: The final answer produced by the agent.
        total_input_tokens: Total input tokens used across all steps.
        total_cached_tokens: Total cached input tokens used across all steps.
        total_output_tokens: Total output tokens used across all steps.
        total_reasoning_tokens: Total reasoning tokens used across all steps.
        total_latency_ms: Total latency across all steps.
        tool_calls_count: Number of tool calls made.
        iterations: Number of iterations completed.
        success: Whether the run completed successfully.
        error: Error message if the run failed.
        start_time: Unix timestamp when the run started.
        end_time: Unix timestamp when the run ended.
    """

    query: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None
    total_input_tokens: int = 0
    total_cached_tokens: int = 0
    total_output_tokens: int = 0
    total_reasoning_tokens: int = 0
    total_latency_ms: float = 0.0
    tool_calls_count: int = 0
    iterations: int = 0
    context_prunes: int = 0
    success: bool = False
    error: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None

    @property
    def total_tokens(self) -> int:
        """Total tokens used in this run."""
        return self.total_input_tokens + self.total_output_tokens

    @property
    def duration_seconds(self) -> float:
        """Total duration of the run in seconds."""
        if self.end_time:
            return self.end_time - self.start_time
        return time.time() - self.start_time


@dataclass
class AgentConfig:
    """Configuration for a ReAct agent.

    Attributes:
        max_iterations: Maximum number of iterations before stopping.
        csv_threshold: Row count threshold for saving results to CSV.
        csv_output_dir: Directory to save CSV files.
        system_prompt: Custom system prompt (None uses default).
        tool_output_log_mode: Tool output logging mode: off, errors_only, preview, or full.
        tool_output_log_max_chars: Max preview chars for console tool output logging.
        tool_output_log_dir: Directory used for full mode output files.
        tool_output_redact_secrets: Whether likely secrets are redacted in output logs.
    """

    max_iterations: int = MAX_ITERATIONS
    csv_threshold: int = CSV_THRESHOLD
    csv_output_dir: str = "./agent_outputs"
    system_prompt: str | None = None
    tool_timeout: float = TOOL_TIMEOUT
    max_retries: int = PROVIDER_MAX_RETRIES
    retry_base_delay: float = PROVIDER_RETRY_BASE_DELAY
    tool_output_log_mode: ToolOutputLogMode = TOOL_OUTPUT_LOG_MODE
    tool_output_log_max_chars: int = TOOL_OUTPUT_LOG_MAX_CHARS
    tool_output_log_dir: str = TOOL_OUTPUT_LOG_DIR
    tool_output_redact_secrets: bool = TOOL_OUTPUT_REDACT_SECRETS
