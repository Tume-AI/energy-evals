import json
from datetime import datetime, timedelta
from urllib.parse import urlencode

from loguru import logger

from energyevals.utils import generate_timestamp, request_with_retry

from energyevals.tools.base_tool import tool_method
from energyevals.tools.dockets._base import DocketBaseTool

_SCC_BASE = "https://www.scc.virginia.gov/docketsearchapi/breeze/dailyfilings/getalldailyfilings"
_SCC_DOCS_BASE = "https://www.scc.virginia.gov/docketsearch/DOCS/"


class VirginiaDocketTool(DocketBaseTool):
    """Search Virginia SCC daily filings by date range."""

    def __init__(self) -> None:
        super().__init__(
            name="virginia_dockets",
            description="Search Virginia SCC daily filings by date range",
        )

    @tool_method(name="search_virginia_dockets")
    def search_virginia(
        self,
        start_date: str,
        end_date: str,
        docname_contains: str | None = None,
        case_contains: str | None = None,
        timeout: int = 30,
    ) -> str:
        """Search Virginia SCC daily filings by date range.

        Args:
            start_date: Start date in YYYY-MM-DD format (inclusive).
            end_date: End date in YYYY-MM-DD format (inclusive).
            docname_contains: Keyword to search within document names.
            case_contains: Keyword to search within case numbers.
            timeout: Timeout in seconds. Defaults to 30.

        Returns:
            JSON string with the search results.
        """
        try:
            timestamp = generate_timestamp()
            save_csv_path = f"virginia_dockets_{timestamp}.csv"

            start = datetime.fromisoformat(start_date).date()
            end = datetime.fromisoformat(end_date).date() + timedelta(days=1)

            start_utc = f"{start.isoformat()}T05:00:00.000Z"
            end_utc = f"{end.isoformat()}T05:00:00.000Z"

            filt = (
                f"(DateFiled ge datetime'{start_utc}') and "
                f"(DateFiled lt datetime'{end_utc}')"
            )
            params = {
                "$filter": filt,
                "$orderby": "Month,Day",
                "$select": "CaseNumber,DocName,Month,Day,Year,DocID,FileName",
            }
            url = f"{_SCC_BASE}?{urlencode(params)}"

            resp = request_with_retry("get", url, timeout=timeout)
            resp.raise_for_status()
            data = resp.json() or []

            results = []
            for row in data:
                if docname_contains and docname_contains.lower() not in (
                    row.get("DocName") or ""
                ).lower():
                    continue
                if case_contains and case_contains.lower() not in (
                    row.get("CaseNumber") or ""
                ).lower():
                    continue
                y, m, d = row.get("Year"), row.get("Month"), row.get("Day")
                filed_date = None
                try:
                    filed_date = datetime(int(y), int(m), int(d)).date().isoformat()
                except Exception:
                    filed_date = None
                filename = row.get("FileName")
                doc_url = f"{_SCC_DOCS_BASE}{filename}" if filename else None
                results.append(
                    {
                        "case_number": row.get("CaseNumber"),
                        "doc_name": row.get("DocName"),
                        "year": y,
                        "month": m,
                        "day": d,
                        "filed_date": filed_date,
                        "doc_id": row.get("DocID"),
                        "document_url": doc_url,
                    }
                )

            saved_csv = self._save_csv(results, save_csv_path)
            return self._result_json(
                {
                    "results": results,
                    "num_results": len(results),
                    "saved_csv": saved_csv,
                }
            )
        except Exception as e:
            logger.error(f"Virginia SCC search failed: {e}")
            return json.dumps({"error": str(e), "source": "Virginia SCC"})
