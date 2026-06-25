from abc import ABC, abstractmethod
from typing import Any, TypedDict

from energyevals.agent.schema import AgentRun


class TraceMetadata(TypedDict, total=False):
    """Typed metadata attached to an agent-run trace.

    All fields are optional (``total=False``).  Additional arbitrary keys
    are permitted at runtime since TypedDict is structural.
    """

    provider: str
    model: str
    question_id: str | int
    trial: int
    category: str
    difficulty: str
    model_params: dict[str, Any]
    tools: dict[str, Any]


class BaseObserver(ABC):
    """Abstract base class for observability implementations.

    All observers must implement this interface to ensure consistent
    behavior across supported trace sinks.
    """

    @abstractmethod
    def trace_agent_run(
        self,
        run: AgentRun,
        metadata: "TraceMetadata | None" = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """Trace a complete agent run.

        Args:
            run: The AgentRun to trace (includes all steps, metrics, errors).
            metadata: Typed metadata to attach to the trace (see TraceMetadata).
            tags: Tags for filtering/categorizing traces.
            user_id: User identifier for the trace.
            session_id: Session identifier for grouping related traces.

        Returns:
            Trace ID if successful, None otherwise.
        """
        pass

    def trace_llm_call(
        self,
        trace_id: str,
        model: str,
        messages: list[dict],
        response: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        latency_ms: float = 0.0,
        tool_calls: list[dict] | None = None,
    ) -> None:
        """Trace an individual LLM API call (optional method).

        Some observers support tracing individual LLM calls separately from
        full agent runs. Others may not need this.

        Args:
            trace_id: Parent trace ID to attach this call to.
            model: Model name/identifier.
            messages: Full message history sent to the LLM.
            response: Full LLM response content.
            input_tokens: Number of input tokens used.
            output_tokens: Number of output tokens used.
            latency_ms: Call latency in milliseconds.
            tool_calls: Any tool calls returned by the LLM.
        """
        pass

    def trace_tool_execution(
        self,
        trace_id: str,
        tool_name: str,
        arguments: dict[str, Any],
        result: str,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> None:
        """Trace a tool execution (optional method).

        Some observers support tracing individual tool calls separately.

        Args:
            trace_id: Parent trace ID to attach this execution to.
            tool_name: Name of the tool executed.
            arguments: Full tool input arguments.
            result: Full tool result (not truncated).
            latency_ms: Execution latency in milliseconds.
            error: Error message if the tool call failed.
        """
        pass

    @abstractmethod
    def flush(self) -> None:
        """Flush any pending traces to the backend."""
        pass

    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown the observer and release resources."""
        pass

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """Check if the observer is enabled and functional."""
        pass
