import json
import os
import re
from datetime import UTC, datetime
from html import unescape
from typing import Literal
from urllib.parse import urlencode, urljoin

import requests
from bs4 import BeautifulSoup, Tag
from loguru import logger

from energyevals.utils import generate_timestamp, request_with_retry

from energyevals.tools.base_tool import tool_method
from energyevals.tools.dockets._base import DocketBaseTool

_DPS_BASE = "https://documents.dps.ny.gov/public"
_MS_DATE_RE = re.compile(r"/Date\((\d+)\)/")


class NewYorkDocketTool(DocketBaseTool):
    """Search New York DPS cases or documents between dates."""

    def __init__(self) -> None:
        super().__init__(
            name="new_york_dockets",
            description="Search New York DPS cases or documents",
        )

    @staticmethod
    def _ms_date_to_str(s: str | None) -> str | None:
        """Convert a Microsoft /Date(ms)/ string to YYYY-MM-DD."""
        if not s or not isinstance(s, str):
            return None
        m = _MS_DATE_RE.fullmatch(s)
        if not m:
            return None
        dt = datetime.fromtimestamp(int(m.group(1)) / 1000.0, tz=UTC)
        return dt.strftime("%Y-%m-%d")

    @staticmethod
    def _extract_anchor(html_snippet: str | None) -> tuple[str | None, str | None]:
        """Return (text, absolute_url) from an HTML anchor snippet."""
        if not html_snippet:
            return None, None
        try:
            soup = BeautifulSoup(unescape(html_snippet), "html.parser")
            a = soup.find("a")
            if not a or not isinstance(a, Tag):
                return soup.get_text(strip=True) or None, None
            text = a.get_text(strip=True) or None
            href = str(a.get("href", ""))
            abs_url = urljoin(_DPS_BASE + "/", href) if href else None
            return text, abs_url
        except Exception:
            return unescape(str(html_snippet)), None

    @staticmethod
    def _build_search_url(
        mode: str,
        case_number: str | None,
        keyword: str | None,
        start_date: str,
        end_date: str,
    ) -> str:
        params = {
            "MC": "1" if mode.lower().startswith("case") else "0",
            "IA": "",
            "MT": "",
            "MST": "",
            "CN": case_number or "",
            "MCT": keyword or "",
            "SDF": start_date,
            "SDT": end_date,
            "C": "",
            "M": "",
            "CO": "",
        }
        return f"{_DPS_BASE}/Common/SearchResults.aspx?{urlencode(params)}"

    @tool_method(name="search_new_york_dockets")
    def search_new_york(
        self,
        start_date: str,
        end_date: str,
        keyword: str | None = None,
        case_number: str | None = None,
        mode: Literal["cases", "documents"] = "cases",
        timeout: int = 30,
    ) -> str:
        """Search New York DPS cases or documents between dates.

        Requires the NY_DPS_TOKEN environment variable (obtain from the NY DPS
        public portal).

        Args:
            start_date: Start date (MM/DD/YYYY)
            end_date: End date (MM/DD/YYYY)
            keyword: Keyword to search for
            case_number: Case number to filter by
            mode: Search mode - "cases" or "documents"
            timeout: Timeout in seconds

        Returns:
            JSON string with the search results.
        """
        try:
            token = os.environ.get("NY_DPS_TOKEN")
            if not token:
                return json.dumps({
                    "error": "NY_DPS_TOKEN environment variable is not set. "
                    "Please set it to use the New York DPS search.",
                    "source": "New York PSC",
                })

            timestamp = generate_timestamp()
            save_csv_path = f"new_york_dps_{mode}_{timestamp}.csv"

            session = requests.Session()
            session.headers.update({"User-Agent": "Mozilla/5.0"})

            search_url = self._build_search_url(mode, case_number, keyword, start_date, end_date)
            resp = request_with_retry("get", search_url, session=session, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            qstring = soup.find("input", id="GridPlaceHolder_hdnQueryString")
            is_matter = soup.find("input", id="GridPlaceHolder_hdnbIsMatter")
            if not qstring or not is_matter or not isinstance(qstring, Tag) or not isinstance(is_matter, Tag):
                raise RuntimeError("Could not locate required hidden fields on results page.")

            query_string = unescape(str(qstring.get("value", "")))
            is_cases_mode = str(is_matter.get("value", "")).lower() == "true"
            data_url = (
                f"{_DPS_BASE}/CaseMaster/MatterExternal/{token}?{query_string}"
                if is_cases_mode
                else f"{_DPS_BASE}/CaseMaster/DocumentExternal/{token}?{query_string}"
            )

            data_resp = request_with_retry("get", data_url, session=session, timeout=timeout)
            data_resp.raise_for_status()

            records = []
            try:
                data = json.loads(data_resp.text.strip())
                if isinstance(data, list):
                    for item in data:
                        record = {
                            "MatterID": item.get("MatterID"),
                            "MatterType": item.get("MatterType"),
                            "MatterSubType": item.get("MatterSubType"),
                            "MatterTitle": item.get("MatterTitle"),
                            "Company": item.get("MatterCompanies"),
                            "SubmitDate": item.get("strSubmitDate"),
                            "TotalRecords": item.get("TotalRecords"),
                        }
                        start_date_str = self._ms_date_to_str(item.get("StartDate"))
                        if start_date_str:
                            record["StartDate"] = start_date_str
                        txt, url = self._extract_anchor(item.get("CaseOrMatterNumber"))
                        record["CaseOrMatterNumber"] = txt
                        record["CaseOrMatterNumber_url"] = url
                        records.append(record)
            except json.JSONDecodeError:
                logger.warning(f"NY DPS returned non-JSON response from {data_url}")

            saved_csv = self._save_csv(records, save_csv_path)
            return self._result_json(
                {
                    "search_url": search_url,
                    "data_url": data_url,
                    "mode": "cases" if is_cases_mode else "documents",
                    "records": records,
                    "saved_csv": saved_csv,
                }
            )
        except Exception as e:
            logger.error(f"New York PSC search failed: {e}")
            return json.dumps({"error": str(e), "source": "New York PSC"})
