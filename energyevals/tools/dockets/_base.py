import json
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup, Tag
from loguru import logger
import pandas as pd

from energyevals.tools.base_tool import BaseTool
from energyevals.tools.sandbox import SANDBOX_WORK_DIR, sandbox_path


DOCKET_PREVIEW_ROWS = 25
DOCKET_INLINE_MAX_CHARS = 40000


class DocketBaseTool(BaseTool):
    """Base class for jurisdiction-specific docket tools.

    Provides shared helpers (e.g. CSV export) used by every docket scraper.
    """

    @staticmethod
    def _collect_hidden_fields(soup: BeautifulSoup) -> dict[str, str]:
        """Extract all hidden input fields from a parsed HTML page.

        Used by ASP.NET WebForms scrapers to capture ViewState, EventValidation,
        and other hidden fields needed to construct a valid POST payload.
        """
        data: dict[str, str] = {}
        for inp in soup.select("input[type=hidden]"):
            name = inp.get("name")
            if name:
                data[str(name)] = str(inp.get("value", ""))
        return data

    @staticmethod
    def _result_json(payload: dict[str, Any]) -> str:
        """JSON-encode a docket result, offloading an oversized result list.

        Finds the largest list-of-rows field; if it's large (by count or serialized
        size) it's replaced inline with a short preview plus a pointer to the saved
        CSV, so a broad search can't blow the context window. The full set is already
        on disk (``payload['saved_csv']``) for run_python_code to read.
        """
        list_key = None
        for k, v in payload.items():
            if isinstance(v, list) and (
                list_key is None or len(v) > len(payload[list_key])
            ):
                list_key = k
        if list_key is not None:
            rows = payload[list_key]
            oversized = len(rows) > DOCKET_PREVIEW_ROWS or (
                len(json.dumps(rows, default=str)) > DOCKET_INLINE_MAX_CHARS
            )
            if oversized:
                payload = dict(payload)
                payload[list_key] = rows[:DOCKET_PREVIEW_ROWS]
                payload["truncated"] = True
                csv = payload.get("saved_csv")
                payload["message"] = (
                    f"{len(rows)} results; showing the first {DOCKET_PREVIEW_ROWS}. "
                    + (
                        f"Full results saved to {csv} -- read it with run_python_code "
                        f"(pandas.read_csv('{csv}'))."
                        if csv
                        else "Narrow the date range or add a keyword filter."
                    )
                )
        return json.dumps(payload, indent=2)

    @staticmethod
    def _save_csv(rows: list[dict[str, Any]], save_csv_path: str | None) -> str | None:
        if not save_csv_path:
            return None
        try:
            SANDBOX_WORK_DIR.mkdir(parents=True, exist_ok=True)
            out_path = SANDBOX_WORK_DIR / Path(save_csv_path).name
            pd.DataFrame(rows).to_csv(out_path, index=False)
            return sandbox_path(out_path)
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return None
