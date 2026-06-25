import json
import os
import threading
import time
from typing import Any, Literal

import pandas as pd
import requests
from loguru import logger

from energyevals.utils import generate_timestamp, http_error_detail
from energyevals.tools.base_tool import BaseTool, tool_method
from energyevals.tools.constants import GRID_STATUS_PAGE_SIZE
from energyevals.tools.sandbox import SANDBOX_WORK_DIR, sandbox_path

_GRIDSTATUS_BASE_URL = "https://api.gridstatus.io/v1"
_MIN_REQUEST_INTERVAL_S: float = 1.05  # Stay just under the ~1 req/sec account-level cap.
_REQUEST_TIMEOUT_S: int = 60
_MAX_RETRIES: int = 3
_RETRY_BASE_DELAY_S: float = 2.0


class GridStatusAPITool(BaseTool):

    # Process-wide throttle: rate limit is per account so all instances must share one
    # lock, held across the inter-request sleep so concurrent callers queue correctly.
    _rate_lock = threading.Lock()
    _last_request_monotonic = 0.0

    def __init__(self, api_key: str | None = None):
        super().__init__(
            name="gridstatus_api_tool",
            description="Access electricity market data from Grid Status API",
        )
        self.api_key = api_key or os.getenv("GRIDSTATUS_API_KEY")
        if not self.api_key:
            logger.warning("GRIDSTATUS_API_KEY not set. Tool will not function.")

    def _throttle(self) -> None:
        """Block until _MIN_REQUEST_INTERVAL_S has elapsed since the last request."""
        with self._rate_lock:
            elapsed = time.monotonic() - GridStatusAPITool._last_request_monotonic
            wait = _MIN_REQUEST_INTERVAL_S - elapsed
            if wait > 0:
                time.sleep(wait)
            GridStatusAPITool._last_request_monotonic = time.monotonic()

    def _make_request(
        self, endpoint: str, params: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        if not self.api_key:
            return {"error": "GRIDSTATUS_API_KEY not configured"}

        url = f"{_GRIDSTATUS_BASE_URL}/{endpoint}"
        headers = {"x-api-key": self.api_key}

        for attempt in range(_MAX_RETRIES + 1):
            self._throttle()
            try:
                response = requests.get(
                    url,
                    headers=headers,
                    params=params,
                    timeout=_REQUEST_TIMEOUT_S,
                )
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                if attempt < _MAX_RETRIES:
                    delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        f"GridStatus network error (attempt {attempt + 1}/{_MAX_RETRIES + 1}); "
                        f"retrying in {delay:.1f}s: {e}"
                    )
                    time.sleep(delay)
                    continue
                logger.error(f"GridStatus request failed after retries: {e}")
                return {"error": str(e)}
            except requests.exceptions.RequestException as e:
                logger.error(f"GridStatus request failed: {e}")
                return {"error": str(e)}

            if response.status_code == 429 or response.status_code >= 500:
                if attempt < _MAX_RETRIES:
                    retry_after = (
                        response.headers.get("Retry-After")
                        if response.status_code == 429
                        else None
                    )
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        delay = 0.0
                    if delay <= 0:
                        delay = _RETRY_BASE_DELAY_S * (2 ** attempt)
                    logger.warning(
                        f"GridStatus {response.status_code} (attempt {attempt + 1}/{_MAX_RETRIES + 1}); "
                        f"retrying in {delay:.1f}s."
                    )
                    time.sleep(delay)
                    continue

            try:
                response.raise_for_status()
                return response.json()
            except requests.exceptions.RequestException as e:
                detail = http_error_detail(e)
                logger.error(f"GridStatus request failed: {e}" + (f" | {detail}" if detail else ""))
                err: dict[str, Any] = {"error": str(e)}
                if detail:
                    err["detail"] = detail
                return err

        return {"error": "GridStatus request failed after retries"}

    @tool_method()
    def list_gridstatus_datasets(self) -> str:
        """Return a JSON list of available Grid Status datasets with each dataset's id and name.
        Use this first to discover valid dataset IDs, then call inspect_gridstatus_dataset(id)
        for a specific dataset's full description, schema, and query options."""
        result = self._make_request("datasets")

        if "error" in result:
            return json.dumps(result)

        # Only id + name: the catalog has ~500 datasets and per-dataset descriptions
        # account for ~80% of the payload, enough to overflow the context window.
        datasets = [
            {"id": ds["id"], "name": ds["name"]}
            for ds in result.get("data", [])
        ]
        return json.dumps(datasets, separators=(",", ":"))

    @tool_method()
    def inspect_gridstatus_dataset(self, dataset_id: str) -> str:
        """Return full metadata for a specific Grid Status dataset to understand its schema and query options before running dataset queries.

        Args:
            dataset_id: The id of the gridstatus dataset to inspect.

        Returns:
            A JSON string of the full metadata associated with the dataset.
        """
        result = self._make_request(f"datasets/{dataset_id}")
        return json.dumps(result, indent=2, default=str)

    @tool_method()
    def query_gridstatus_dataset(
        self,
        dataset_id: str,
        filter_column: str | None = None,
        filter_value: str | None = None,
        limit: int | None = None,
        columns: list[str] | None = None,
        start_time: str | None = None,
        end_time: str | None = None,
        time_val: str | None = None,
        publish_time: str | None = None,
        publish_time_start: str | None = None,
        publish_time_end: str | None = None,
        resample_frequency: str | None = None,
        resample_by: str | None = None,
        resample_function: Literal["mean", "sum", "min", "max", "count", "stddev", "variance"] | None = "mean",
    ) -> str:
        """Query data from a Grid Status dataset with optional filtering, column selection, and resampling.
        On success, saves data to CSV and returns preview rows, filepath, and row count; on error, returns
        the full JSON error. Notes: filter_operator and time_comparison are fixed to '=' and cannot be
        changed. time_val cannot be used together with start_time or end_time. For time-related filters,
        NEVER use filter_column and filter_value; use start_time/end_time or time_val instead. If an input
        is not used, set it to None or an empty string. Also important to note that each query can only return
        up to 50,000 rows at a time so it is important to split the queries when data pulls over longer periods
        are required. The resample inputs are also very powerful and you should use them often when calculating
        data aggregations

        Args:
            dataset_id: The ID of the dataset to query.
            filter_column: Column name to filter results by. Do NOT use for time-related filtering.
            filter_value: Value to filter results by. Do NOT use for time-related filtering.
            limit: Maximum number of rows to return.
            columns: Columns to return.
            start_time: ISO 8601 start time (for time_index_column). Cannot be used with time_val.
            end_time: ISO 8601 end time (for time_index_column). Cannot be used with time_val.
            time_val: 'latest' or ISO 8601 timestamp. Cannot be used with start_time or end_time.
            publish_time: Advanced filtering for forecast datasets.
            publish_time_start: Start of publish_time filter.
            publish_time_end: End of publish_time filter.
            resample_frequency: e.g. '1 minute', '5 minutes', '1 hour', etc.
            resample_by: Columns to group by before resampling.
            resample_function: One of 'mean', 'sum', 'min', 'max', 'count', 'stddev', 'variance'. Default is 'mean'.

        Returns:
            JSON string. On success contains 'preview' (first rows as list of dicts), 'filepath'
            (absolute path to saved CSV), and 'row_count'. On error contains the full API error response.
        """
        params: dict[str, Any] = {
            "order": "asc",
            "return_format": "json",
            "filter_operator": "=",
            "time_comparison": "=",
            "timezone": "market",
            "page": 1,
            "page_size": GRID_STATUS_PAGE_SIZE,
        }

        if filter_column is not None:
            params["filter_column"] = filter_column
        if filter_value is not None:
            params["filter_value"] = filter_value
        if limit is not None:
            params["limit"] = limit
        if columns is not None:
            params["columns"] = ",".join(columns)
        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time
        if time_val is not None:
            params["time"] = time_val
        if publish_time is not None:
            params["publish_time"] = publish_time
        if publish_time_start is not None:
            params["publish_time_start"] = publish_time_start
        if publish_time_end is not None:
            params["publish_time_end"] = publish_time_end
        if resample_frequency is not None:
            params["resample_frequency"] = resample_frequency
        if resample_by is not None:
            params["resample_by"] = resample_by
        if resample_function is not None:
            params["resample_function"] = resample_function

        params = {k: v for k, v in params.items() if v not in ("", None)}

        result = self._make_request(f"datasets/{dataset_id}/query", params)

        if "error" in result:
            return json.dumps(result, indent=2)

        data = result.get("data", [])

        if not data:
            return json.dumps({"message": "No data found in response", "response": result}, indent=2, default=str)

        try:
            df = pd.DataFrame(data)
            timestamp = generate_timestamp()
            SANDBOX_WORK_DIR.mkdir(parents=True, exist_ok=True)
            filepath = SANDBOX_WORK_DIR / f"{dataset_id}_{timestamp}.csv"
            df.to_csv(filepath, index=False)
            return json.dumps(
                {
                    "preview": df.head().to_dict("records"),
                    "filepath": sandbox_path(filepath),
                    "row_count": len(df),
                },
                indent=2,
                default=str,
            )
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")
            return json.dumps({"error": str(e), "data": data}, indent=2, default=str)
