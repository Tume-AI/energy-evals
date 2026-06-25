import csv
import json
import os
import re
from pathlib import Path
from typing import Any

from loguru import logger

from energyevals.agent.constants import CSV_THRESHOLD, PREVIEW_ROWS
from energyevals.utils import generate_timestamp

_FORMULA_PREFIXES = ("=", "+", "-", "@")
_SAFE_FILENAME_RE = re.compile(r"[^a-zA-Z0-9_-]")


def _sanitize_csv_value(value: Any) -> Any:
    """Prepend a single quote to strings that start with formula-injection prefixes.

    Prevents spreadsheet applications from interpreting cell values as formulas.
    """
    if isinstance(value, str) and value.startswith(_FORMULA_PREFIXES):
        return "'" + value
    return value


class CSVProcessor:
    """Processor for handling large CSV results from tools.

    When tool results contain large tabular data (exceeding the row threshold),
    this processor saves the data to CSV and returns a reference instead of
    the full data.
    """

    def __init__(
        self,
        threshold: int = CSV_THRESHOLD,
        output_dir: str | Path = "./agent_outputs",
    ):
        """Initialize CSV processor.

        Args:
            threshold: Row count threshold for saving to CSV
            output_dir: Directory to save CSV files
        """
        self.threshold = threshold
        self.output_dir = Path(output_dir)
        os.makedirs(self.output_dir, exist_ok=True)

    def process(
        self,
        tool_name: str,
        result: str,
    ) -> tuple[str, str | None]:
        """Process tool result, saving to CSV if it exceeds threshold.

        Args:
            tool_name: Name of the tool that produced the result
            result: Tool result as JSON string

        Returns:
            Tuple of (context_result, csv_path):
                - context_result: Result to pass to LLM (may be reference to CSV)
                - csv_path: Path to saved CSV file (None if not saved)
        """
        try:
            data = json.loads(result)
        except json.JSONDecodeError:
            return result, None

        # Tool already saved its own CSV — nothing to do
        if isinstance(data, dict) and (data.get("saved_csv") or
                (isinstance(data.get("data"), dict) and data["data"].get("saved_csv"))):
            return result, None

        # Unwrap the ToolResult envelope: {"success": ..., "data": <payload>}.
        # The framework serializes ``data`` as a JSON *string* (the tool's own
        # json.dumps output), so parse it back before looking for ``rows``;
        # otherwise large query results never get offloaded to CSV.
        inner: dict = data if isinstance(data, dict) else {}
        if "success" in inner:
            payload = inner.get("data")
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    payload = None
            if isinstance(payload, dict):
                inner = payload

        rows = inner.get("rows")
        columns = inner.get("columns")

        if not rows or not isinstance(rows, list):
            return result, None

        row_count = len(rows)

        if row_count <= self.threshold:
            return result, None

        safe_tool_name = _SAFE_FILENAME_RE.sub("_", tool_name)[:64]
        timestamp = generate_timestamp()
        csv_filename = f"{safe_tool_name}_{timestamp}.csv"
        csv_path = self.output_dir / csv_filename

        try:
            with open(csv_path, "w", newline="") as f:
                if rows and isinstance(rows[0], dict):
                    fieldnames = [_sanitize_csv_value(k) for k in rows[0].keys()]
                    sanitized_rows = [
                        {_sanitize_csv_value(k): _sanitize_csv_value(v) for k, v in row.items()}
                        for row in rows
                    ]
                    dict_writer = csv.DictWriter(f, fieldnames=fieldnames)
                    dict_writer.writeheader()
                    dict_writer.writerows(sanitized_rows)
                elif columns:
                    list_writer = csv.writer(f)
                    list_writer.writerow([_sanitize_csv_value(c) for c in columns])
                    list_writer.writerows(
                        [[_sanitize_csv_value(cell) for cell in row] for row in rows]
                    )
                else:
                    return result, None

            preview_rows = rows[:PREVIEW_ROWS]
            context_data = {
                "status": "success",
                "row_count": row_count,
                "csv_file": str(csv_path),
                "message": f"Query returned {row_count} rows. Results saved to {csv_path}. Use Python to read and analyze the CSV file.",
                "columns": columns or (list(rows[0].keys()) if rows else []),
                "preview": preview_rows,
            }

            for key in ["database", "query", "table"]:
                if key in inner:
                    context_data[key] = inner[key]

            return json.dumps(context_data, indent=2, default=str), str(csv_path)

        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return result, None
