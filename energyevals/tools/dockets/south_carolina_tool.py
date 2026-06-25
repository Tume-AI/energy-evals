import json
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from loguru import logger

from energyevals.utils import generate_timestamp, request_with_retry

from energyevals.tools.base_tool import tool_method
from energyevals.tools.dockets._base import DocketBaseTool


class SouthCarolinaDocketTool(DocketBaseTool):
    """Search South Carolina PSC dockets by date range."""

    def __init__(self) -> None:
        super().__init__(
            name="south_carolina_dockets",
            description="Search South Carolina PSC dockets by date range",
        )

    @tool_method(name="search_south_carolina_dockets")
    def search_south_carolina(
        self,
        start_date: str,
        end_date: str,
        organization: str | None = None,
        individual: str | None = None,
        summary: str | None = None,
        number_year: str | None = None,
        number_sequence: str | None = None,
        number_type: str | None = None,
        timeout: int = 30,
    ) -> str:
        """Search South Carolina PSC dockets by date range.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            organization: Organization name to filter by
            individual: Individual name to filter by
            summary: Keyword(s) to search within filing summaries
            number_year: Docket number year component
            number_sequence: Docket number sequence component
            number_type: Docket number type code
            timeout: Timeout in seconds

        Returns:
            JSON string with the search results.
        """
        try:
            base_url = "https://dms.psc.sc.gov"
            search_path = "/Web/Dockets/Search"

            params = {
                "IndividualName": individual or "",
                "OrganizationName": organization or "",
                "Summary": summary or "",
                "StartDate": start_date or "",
                "EndDate": end_date or "",
            }
            if number_year is not None:
                params["NumberYear"] = number_year
            if number_sequence is not None:
                params["NumberSequence"] = number_sequence
            if number_type is not None:
                params["NumberType"] = number_type

            resp = request_with_retry("get", urljoin(base_url, search_path), params=params, timeout=timeout)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.select_one("table.datatable-standard-savestate")
            items: list[dict[str, Any]] = []
            timestamp = generate_timestamp()
            save_csv_path = f"south_carolina_dockets_{timestamp}.csv"

            if table and isinstance(table, Tag):
                tbody_result = table.find("tbody")
                tbody = tbody_result if isinstance(tbody_result, Tag) else table
                for tr in tbody.find_all("tr"):
                    tds = tr.find_all("td")
                    if len(tds) < 2:
                        continue
                    link_a = tds[0].find("a", class_="detailNumber")
                    docket_number_value = (
                        link_a.get_text(strip=True) if link_a else tds[0].get_text(strip=True)
                    )
                    docket_link = (
                        urljoin(base_url, str(link_a["href"]))
                        if link_a and link_a.has_attr("href")
                        else None
                    )
                    summary_el = tds[1].find("span")
                    strong = summary_el.find("strong") if summary_el else None
                    summary_text = (
                        strong.get_text(" ", strip=True)
                        if strong
                        else (
                            summary_el.get_text(" ", strip=True)
                            if summary_el
                            else tds[1].get_text(" ", strip=True)
                        )
                    )
                    parties_div = tds[1].find("div", class_="parties")
                    parties = []
                    if parties_div:
                        for a in parties_div.find_all("a"):
                            txt = a.get_text(" ", strip=True)
                            if txt:
                                parties.append(txt)
                    items.append(
                        {
                            "docket_number": docket_number_value,
                            "summary": summary_text,
                            "docket_link": docket_link,
                            "parties": parties,
                        }
                    )

            saved_csv = self._save_csv(items, save_csv_path)
            return self._result_json(
                {
                    "items": items,
                    "source_url": resp.url,
                    "saved_csv": saved_csv,
                }
            )
        except Exception as e:
            logger.error(f"South Carolina PSC search failed: {e}")
            return json.dumps({"error": str(e), "source": "South Carolina PSC"})
