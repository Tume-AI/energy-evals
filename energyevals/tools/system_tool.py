import json
import os
import re
import subprocess
from pathlib import Path

from loguru import logger

from energyevals.utils import generate_timestamp

import energyevals.tools.sandbox as _sandbox
from energyevals.tools.base_tool import BaseTool, tool_method
from energyevals.tools.constants import (
    GREP_EXCLUDE_DIRS,
    GREP_MAX_LINE_CHARS,
    GREP_MAX_TOTAL_CHARS,
    SYSTEM_COMMAND_TIMEOUT,
    SYSTEM_MAX_RESULTS,
)

# Cap inline stdout/stderr to avoid overflowing the context window on large output.
SANDBOX_OUTPUT_MAX_CHARS = int(os.getenv("SANDBOX_OUTPUT_MAX_CHARS", "50000"))


def _offload_large_streams(result: dict) -> dict:
    """Offload oversized ``stdout``/``stderr`` in a sandbox result to a /work file."""
    half = SANDBOX_OUTPUT_MAX_CHARS // 2
    for stream in ("stdout", "stderr"):
        text = result.get(stream)
        if not isinstance(text, str) or len(text) <= SANDBOX_OUTPUT_MAX_CHARS:
            continue
        ref = None
        try:
            _sandbox.SANDBOX_WORK_DIR.mkdir(parents=True, exist_ok=True)
            path = _sandbox.SANDBOX_WORK_DIR / f"{stream}_{generate_timestamp()}.txt"
            path.write_text(text)
            ref = _sandbox.sandbox_path(path)
        except OSError as exc:  # pragma: no cover - best-effort offload
            logger.warning(f"could not offload large {stream}: {exc}")
        omitted = len(text) - 2 * half
        note = (
            f"\n\n...[{omitted:,} chars omitted; full {stream} saved to {ref} -- read it "
            "with run_python_code if needed. Tip: print summaries/aggregates, not raw "
            "data.]...\n\n"
            if ref
            else f"\n\n...[{omitted:,} chars omitted; print summaries, not raw data]...\n\n"
        )
        result[stream] = text[:half] + note + text[-half:]
        if ref:
            result[f"{stream}_file"] = ref
    return result


# run_shell_command / run_python_code execute ONLY in the disposable Docker
# sandbox. There is no in-process fallback: if the sandbox is unavailable,
# the tools fail closed.
_SANDBOX_UNAVAILABLE_MSG = (
    "Execution sandbox (Docker) is unavailable. Start the Docker daemon and build the "
    "image: docker build -t energyevals-sandbox -f sandbox/Dockerfile sandbox/."
)
_SANDBOX_PY_TIMEOUT_S = int(os.getenv("SANDBOX_PYTHON_TIMEOUT_S", "300"))


class SystemTool(BaseTool):
    """Local filesystem search plus sandboxed shell/Python execution."""

    def __init__(self) -> None:
        super().__init__(
            name="system",
            description="Local filesystem search and sandboxed command/code execution",
        )

    @tool_method()
    def list_files(
        self,
        path: str = ".",
        recursive: bool = False,
        max_results: int = SYSTEM_MAX_RESULTS,
    ) -> str:
        """List files and directories under a given path, returning their full paths.

        Args:
            path: Path to list files from (defaults to current directory).
            recursive: Whether to list files recursively through subdirectories.
            max_results: Maximum number of results to return.

        Returns:
            JSON string with path, count, and a list of file/directory paths.
        """
        try:
            base = Path(_sandbox.host_path(path)).expanduser()
            if not base.exists():
                return json.dumps({"error": f"Path not found: {path}"})

            results: list[str] = []
            if recursive:
                for entry in base.rglob("*"):
                    results.append(str(entry))
                    if len(results) >= max_results:
                        break
            else:
                for entry in base.iterdir():
                    results.append(str(entry))
                    if len(results) >= max_results:
                        break

            return json.dumps({"path": str(base), "count": len(results), "results": results}, indent=2)
        except Exception as exc:
            logger.error(f"list_files failed: {exc}")
            return json.dumps({"error": str(exc)})

    @tool_method()
    def grep_files(
        self,
        pattern: str,
        path: str = ".",
        glob: str | None = None,
        case_insensitive: bool = False,
        max_results: int = SYSTEM_MAX_RESULTS,
    ) -> str:
        """Search files for a pattern using ripgrep if available, falling back to Python regex.
        Returns matching lines in file:line:content format.

        Args:
            pattern: Regex pattern to search for.
            path: Path to search (defaults to current directory).
            glob: Optional glob filter (e.g., '*.py').
            case_insensitive: Whether to search case-insensitively.
            max_results: Maximum number of results to return.

        Returns:
            JSON string with count and a list of matching lines (each line and the
            total output are capped; oversized matches are truncated).
        """
        base = Path(_sandbox.host_path(path)).expanduser()
        if not base.exists():
            return json.dumps({"error": f"Path not found: {path}"})

        rg_cmd = ["rg", "--no-messages", "--line-number"]
        if case_insensitive:
            rg_cmd.append("-i")
        if glob:
            rg_cmd.extend(["-g", glob])
        # Never search cache/vendored directories; `**/` matches at any depth.
        for excluded in GREP_EXCLUDE_DIRS:
            rg_cmd.extend(["-g", f"!**/{excluded}/**"])
        rg_cmd.extend([pattern, str(base)])

        try:
            completed = subprocess.run(
                rg_cmd,
                check=False,
                capture_output=True,
                text=True,
            )
            if completed.returncode in {0, 1}:
                lines = completed.stdout.strip().splitlines()
                if max_results:
                    lines = lines[:max_results]
                lines = [self._truncate_grep_line(ln) for ln in lines]
                return self._finalize_grep_results(lines)
        except FileNotFoundError:
            logger.debug("rg not available, falling back to Python search.")
        except Exception as exc:
            logger.error(f"rg search failed: {exc}")

        results: list[str] = []
        flags = re.IGNORECASE if case_insensitive else 0
        regex = re.compile(pattern, flags)

        paths = base.rglob("*") if base.is_dir() else [base]
        for file_path in paths:
            if not file_path.is_file():
                continue
            if any(part in GREP_EXCLUDE_DIRS for part in file_path.parts):
                continue
            if glob and not file_path.match(glob):
                continue
            try:
                for idx, line in enumerate(file_path.read_text(errors="ignore").splitlines(), start=1):
                    if regex.search(line):
                        results.append(self._truncate_grep_line(f"{file_path}:{idx}:{line}"))
                        if len(results) >= max_results:
                            break
            except Exception:
                continue
            if len(results) >= max_results:
                break

        return self._finalize_grep_results(results)

    @staticmethod
    def _truncate_grep_line(line: str) -> str:
        """Cap a single grep match line so one huge line can't flood the output."""
        if len(line) <= GREP_MAX_LINE_CHARS:
            return line
        dropped = len(line) - GREP_MAX_LINE_CHARS
        return line[:GREP_MAX_LINE_CHARS] + f"…[+{dropped} chars truncated]"

    @staticmethod
    def _finalize_grep_results(results: list[str]) -> str:
        """Apply the total-output ceiling and serialize the grep response.

        Even after per-line truncation, many medium lines can add up; drop
        trailing matches past ``GREP_MAX_TOTAL_CHARS`` and flag that we did.
        """
        total_matches = len(results)
        kept: list[str] = []
        size = 2  # opening/closing brackets
        for line in results:
            size += len(line) + 12  # ~JSON overhead per element (quotes, comma, indent)
            if size > GREP_MAX_TOTAL_CHARS and kept:
                break
            kept.append(line)

        payload: dict[str, object] = {"count": len(kept)}
        if len(kept) < total_matches:
            payload["total_matches"] = total_matches
            payload["truncated"] = True
            payload["note"] = (
                f"Output capped at {GREP_MAX_TOTAL_CHARS} chars; "
                f"{total_matches - len(kept)} of {total_matches} matches omitted. "
                "Narrow the pattern/glob or open a specific file."
            )
        payload["results"] = kept
        return json.dumps(payload, indent=2)

    @tool_method()
    def run_python_code(self, code: str) -> str:
        """Execute Python code in a disposable sandbox and return stdout/stderr.

        Runs inside an isolated container with the data-science stack available
        (pandas, numpy, scipy, scikit-learn, statsmodels, geopandas, plotly,
        pymupdf, requests, ...). The container has NO host secrets (no .env /
        ~/.ssh) and no internal-network access. Filesystem layout:

        - ``/work`` -- writable scratch, PERSISTS across calls (also where other
          tools drop CSVs); write files you need later here.
        - ``/data`` -- read-only input datasets.
        - ``/tmp``  -- per-call scratch, wiped after each call.

        Args:
            code: Python code to execute.

        Returns:
            JSON string with status, stdout, and stderr.
        """
        if not _sandbox.sandbox_available():
            return json.dumps(
                {"status": "error", "error": _SANDBOX_UNAVAILABLE_MSG}, indent=2
            )
        return json.dumps(
            _offload_large_streams(_sandbox.run_python(code, _SANDBOX_PY_TIMEOUT_S)),
            indent=2,
        )

    @tool_method()
    def run_shell_command(
        self,
        command: str,
        timeout: int = SYSTEM_COMMAND_TIMEOUT,
    ) -> str:
        """Run a shell command in a disposable sandbox and return stdout/stderr.

        Runs inside an isolated container with a FULL shell -- pipes, heredocs,
        redirects, ``&&``, ``$(...)``, and tools like ``curl`` and ``pdftotext``
        all work. The container has NO host secrets (no .env / ~/.ssh) and no
        access to the internal network. Filesystem layout:

        - ``/work`` -- writable scratch, PERSISTS across calls (also where other
          tools drop CSVs). Put anything you need in a later call here.
        - ``/data`` -- read-only input datasets.
        - ``/tmp``  -- per-call scratch, wiped after each call (each call is a
          fresh container).

        Args:
            command: Shell command to run (full shell syntax supported).
            timeout: Timeout in seconds.

        Returns:
            JSON string with status, returncode, stdout, and stderr.
        """
        if not command.strip():
            return json.dumps({"status": "error", "error": "Empty command"}, indent=2)
        if not _sandbox.sandbox_available():
            return json.dumps(
                {"status": "error", "error": _SANDBOX_UNAVAILABLE_MSG}, indent=2
            )
        safe_timeout = max(1, min(timeout, SYSTEM_COMMAND_TIMEOUT))
        return json.dumps(
            _offload_large_streams(_sandbox.run_shell(command, safe_timeout)),
            indent=2,
        )
