import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict
from uuid import uuid4

try:
    import fcntl
    _HAS_FCNTL = True
except ImportError:
    _HAS_FCNTL = False

from loguru import logger

from energyevals.agent.schema import AgentRun, AgentStep, StepType

from energyevals.observability.constants import ERROR_PREVIEW_LENGTH, RAW_ERROR_LENGTH


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

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]")


def _sanitize_path_component(value: str) -> str:
    """Sanitize a string for safe use as a path component.

    Replaces any character that is not alphanumeric, dot, underscore, or dash
    with an underscore and truncates to 128 characters.
    """
    return _SAFE_NAME_RE.sub("_", value)[:128]


class JSONFileObserver:
    """Observer that writes agent runs to local JSON files.

    Captures complete trace data including:
    - Full query and response
    - All execution steps (action, observation, answer, error, thought)
    - Complete tool inputs and outputs (not truncated)
    - Failed tool calls with error details
    - Token usage and latency metrics
    - Timestamps for each step

    Output Formats:
    - Individual files: One JSON file per trace (easier to inspect)
    - JSONL: All traces in one file, one JSON object per line (easier to process)
    """

    def __init__(
        self,
        output_dir: str = "./observability_logs",
        run_name: str | None = None,
        single_file: bool = False,
        filename: str = "agent_traces.jsonl",
        pretty_print: bool = True,
    ):
        """Initialize the JSON observer.

        Args:
            output_dir: Base directory to store trace files.
            run_name: Optional subdirectory name for organizing runs (e.g., "no_tools").
                     When provided, traces are saved to: {output_dir}/{run_name}/{model}/
            single_file: If True, append all traces to one JSONL file.
                        If False, create one JSON file per trace.
            filename: Filename for single_file mode (should end in .jsonl).
            pretty_print: If True, format JSON with indentation (individual files only).
        """
        self.base_output_dir = Path(output_dir)
        self.run_name = run_name

        if run_name:
            self.output_dir = self.base_output_dir / run_name
        else:
            self.output_dir = self.base_output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.single_file = single_file
        self.filename = filename
        self.pretty_print = pretty_print
        self._enabled = True
        self._trial: int | None = None

        logger.info(f"JSONFileObserver initialized. Output dir: {self.output_dir}")

    @property
    def is_enabled(self) -> bool:
        """Check if the observer is enabled."""
        return self._enabled

    def set_trial(self, trial_num: int | None) -> None:
        """Set the current trial number for trace output path nesting.

        When set to an int, traces are saved under a ``trial_N/`` subdirectory
        inside the model directory.  When set to ``None`` (default / single-trial),
        traces are written directly into the model directory for backward
        compatibility.
        """
        self._trial = trial_num

    def trace_agent_run(
        self,
        run: AgentRun,
        metadata: "TraceMetadata | None" = None,
        tags: list[str] | None = None,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> str | None:
        """Write complete agent run to JSON file.

        Captures ALL data from the run including:
        - Every step (no filtering)
        - Full tool outputs (no truncation)
        - Error details for failed steps
        - Complete metrics

        Args:
            run: The AgentRun to trace.
            metadata: Additional metadata to attach.
            tags: Tags for categorizing the trace.
            user_id: User identifier.
            session_id: Session identifier.

        Returns:
            Trace ID (timestamp-based unique identifier).
        """
        if not self._enabled:
            return None

        try:
            trace_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}"

            trace_data = self._build_trace_data(
                trace_id=trace_id,
                run=run,
                metadata=metadata,
                tags=tags,
                user_id=user_id,
                session_id=session_id,
            )

            self._write_trace(trace_id, trace_data, metadata=metadata)

            logger.debug(f"Traced agent run to JSON: {trace_id}")
            return trace_id

        except Exception as e:
            logger.error(f"Failed to write trace to JSON: {e}")
            return None

    def _build_trace_data(
        self,
        trace_id: str,
        run: AgentRun,
        metadata: "TraceMetadata | None",
        tags: list[str] | None,
        user_id: str | None,
        session_id: str | None,
    ) -> dict[str, Any]:
        """Build complete trace data structure."""

        step_summary = self._analyze_steps(run.steps)

        return {
            "trace_id": trace_id,
            "timestamp": datetime.now().isoformat(),
            "start_time": datetime.fromtimestamp(run.start_time).isoformat() if run.start_time else None,
            "end_time": datetime.fromtimestamp(run.end_time).isoformat() if run.end_time else None,

            "query": run.query,
            "final_answer": run.final_answer,

            "success": run.success,
            "error": run.error,

            "metrics": {
                "iterations": run.iterations,
                "tool_calls_count": run.tool_calls_count,
                "total_input_tokens": run.total_input_tokens,
                "total_output_tokens": run.total_output_tokens,
                "total_cached_tokens": run.total_cached_tokens,
                "total_reasoning_tokens": run.total_reasoning_tokens,
                "total_tokens": run.total_tokens,
                "total_latency_ms": run.total_latency_ms,
                "duration_seconds": run.duration_seconds,
            },
            "step_summary": step_summary,
            "steps": [
                self._serialize_step(step, index)
                for index, step in enumerate(run.steps)
            ],
            "metadata": metadata or {},
            "tags": tags or [],
            "user_id": user_id,
            "session_id": session_id,
        }

    def _analyze_steps(self, steps: list[AgentStep]) -> dict[str, Any]:
        """Analyze steps and provide summary statistics."""
        summary: dict[str, Any] = {
            "total_steps": len(steps),
            "step_types": {},
            "tool_calls": [],
            "failed_tool_calls": [],
            "errors": [],
        }

        for step in steps:
            step_type = step.step_type.value
            summary["step_types"][step_type] = summary["step_types"].get(step_type, 0) + 1

            if step.step_type == StepType.ACTION and step.tool_name:
                summary["tool_calls"].append(step.tool_name)

            if step.step_type == StepType.OBSERVATION and step.tool_output:
                if self._is_tool_error(step.tool_output):
                    summary["failed_tool_calls"].append({
                        "tool": step.tool_name,
                        "error_preview": step.tool_output[:ERROR_PREVIEW_LENGTH] if len(step.tool_output) > ERROR_PREVIEW_LENGTH else step.tool_output,
                    })

            if step.step_type == StepType.ERROR:
                summary["errors"].append(step.content)

        return summary

    def _is_tool_error(self, output: str) -> bool:
        """Check if tool output indicates an error.

        Uses explicit key checks only to avoid false positives from
        incidental mentions of 'error' in tool output values.
        """
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                if data.get("error"):
                    return True
                if data.get("success") is False:
                    return True
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    def _serialize_step(self, step: AgentStep, index: int) -> dict[str, Any]:
        """Serialize a single step with complete data."""
        step_data = {
            "index": index,
            "iteration": step.iteration,
            "step_type": step.step_type.value,
            "timestamp": datetime.fromtimestamp(step.timestamp).isoformat() if step.timestamp else None,
            "timestamp_unix": step.timestamp,
            "content": step.content,
            "reasoning": step.reasoning,
            "tool_name": step.tool_name,
            "tool_input": step.tool_input,
            "tool_output": step.tool_output,
            "tool_output_length": len(step.tool_output) if step.tool_output else 0,
            "tokens_used": step.tokens_used,
            "latency_ms": step.latency_ms,
        }

        if step.step_type == StepType.OBSERVATION and step.tool_output:
            step_data["is_error"] = self._is_tool_error(step.tool_output)
            if step_data["is_error"]:
                step_data["error_details"] = self._extract_error_details(step.tool_output)

        return step_data

    def _extract_error_details(self, output: str) -> dict[str, Any] | None:
        """Extract error details from tool output."""
        try:
            data = json.loads(output)
            if isinstance(data, dict):
                return {
                    "error_message": data.get("error"),
                    "error_type": data.get("error_type"),
                    "success": data.get("success"),
                }
        except (json.JSONDecodeError, TypeError):
            return {"raw_error": output[:RAW_ERROR_LENGTH]}
        return None

    def _write_trace(
        self,
        trace_id: str,
        trace_data: dict[str, Any],
        metadata: "TraceMetadata | None" = None,
    ) -> None:
        """Write trace to file.

        If metadata contains provider, model, and question_id, traces are organized
        into subdirectories by model with question numbers in filenames:
            {output_dir}/{run_name}/{provider}_{model}/trace_q{question_id}_{timestamp}.json

        Otherwise falls back to flat structure:
            {output_dir}/trace_{trace_id}.json
        """
        if metadata and all(k in metadata for k in ("provider", "model", "question_id")):
            provider = _sanitize_path_component(str(metadata["provider"]))
            model = _sanitize_path_component(str(metadata["model"]))
            question_id = _sanitize_path_component(str(metadata["question_id"]))
            model_dir = f"{provider}_{model}"
            trace_output_dir = self.output_dir / model_dir
            if self._trial is not None:
                trace_output_dir = trace_output_dir / f"trial_{self._trial}"
            filename = f"trace_q{question_id}_{trace_id}.json"
        else:
            trace_output_dir = self.output_dir
            filename = f"trace_{trace_id}.json"
        trace_output_dir.mkdir(parents=True, exist_ok=True)

        if self.single_file:
            filepath = trace_output_dir / self.filename
            with open(filepath, "a", encoding="utf-8") as f:
                if _HAS_FCNTL:
                    fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    f.write(json.dumps(trace_data, default=str, ensure_ascii=False) + "\n")
                finally:
                    if _HAS_FCNTL:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)
            filepath.chmod(0o600)
            logger.debug(f"Appended trace to {filepath}")
        else:
            filepath = trace_output_dir / filename
            with open(filepath, "w", encoding="utf-8") as f:
                if self.pretty_print:
                    json.dump(trace_data, f, indent=2, default=str, ensure_ascii=False)
                else:
                    json.dump(trace_data, f, default=str, ensure_ascii=False)
            filepath.chmod(0o600)
            logger.debug(f"Wrote trace to {filepath}")

    def flush(self) -> None:
        pass

    def shutdown(self) -> None:
        self._enabled = False
        logger.info("JSONFileObserver shutdown")

    def _find_trace_file(self, trace_id: str) -> Path | None:
        """Find a trace file by ID, searching flat directory and subdirectories.

        Handles both flat traces (trace_{id}.json) and organized traces
        ({provider}_{model}/trace_q{question_id}_{id}.json).

        Args:
            trace_id: The trace ID to search for.

        Returns:
            Path to the trace file, or None if not found.
        """
        flat_path = self.output_dir / f"trace_{trace_id}.json"
        if flat_path.exists():
            return flat_path

        for match in self.output_dir.rglob(f"*{trace_id}*.json"):
            return match

        return None

    def get_trace_file(self, trace_id: str) -> Path | None:
        """Get the file path for a specific trace.

        Args:
            trace_id: The trace ID to look up.

        Returns:
            Path to the trace file, or None if not found.
        """
        if self.single_file:
            return self.output_dir / self.filename
        return self._find_trace_file(trace_id)

    def list_traces(self) -> list[str]:
        """List all trace IDs in the output directory.

        Searches both the flat directory and any model subdirectories.

        Returns:
            List of trace IDs.
        """
        if self.single_file:
            traces = []
            for jsonl_file in self.output_dir.rglob(self.filename):
                with open(jsonl_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            data = json.loads(line)
                            traces.append(data.get("trace_id", "unknown"))
                        except json.JSONDecodeError:
                            continue
            return traces
        else:
            traces = []
            for trace_path in self.output_dir.rglob("trace_*.json"):
                try:
                    with open(trace_path, encoding="utf-8") as fp:
                        data = json.load(fp)
                        traces.append(data.get("trace_id", trace_path.stem.replace("trace_", "")))
                except (json.JSONDecodeError, OSError):
                    traces.append(trace_path.stem.replace("trace_", ""))
            return traces

    def load_trace(self, trace_id: str) -> dict[str, Any] | None:
        """Load a specific trace by ID.

        Searches both the flat directory and any model subdirectories.

        Args:
            trace_id: The trace ID to load.

        Returns:
            Trace data dict, or None if not found.
        """
        if self.single_file:
            for jsonl_file in self.output_dir.rglob(self.filename):
                with open(jsonl_file, encoding="utf-8") as f:
                    for line in f:
                        try:
                            data: dict[str, Any] = json.loads(line)
                            if data.get("trace_id") == trace_id:
                                return data
                        except json.JSONDecodeError:
                            continue
            return None
        else:
            filepath = self._find_trace_file(trace_id)
            if filepath is None:
                return None
            with open(filepath, encoding="utf-8") as fp:
                loaded: dict[str, Any] = json.load(fp)
                return loaded
