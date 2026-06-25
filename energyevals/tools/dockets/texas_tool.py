import json
from urllib.parse import urlencode

from bs4 import BeautifulSoup
from loguru import logger

from energyevals.utils import generate_timestamp, get_system_ca_bundle, request_with_retry

from energyevals.tools.base_tool import tool_method
from energyevals.tools.dockets._base import DocketBaseTool


class TexasDocketTool(DocketBaseTool):
    """Search Texas Public Utility Commission (PUCT) filings or dockets."""

    def __init__(self) -> None:
        super().__init__(
            name="texas_dockets",
            description="Search Texas PUCT filings or dockets",
        )

    @tool_method(name="search_texas_dockets")
    def search_texas(
        self,
        date_from: str,
        date_to: str,
        utility_name: str | None = None,
        control_number: str | None = None,
        description: str | None = None,
        utility_type: str = "A",
        document_type: str = "ALL",
        item_match: str = "Equal",
        sort_order: str = "Descending",
        timeout: int = 30,
    ) -> str:
        """Search Texas Public Utility Commission (PUCT) filings or dockets.

        Parameters:
            date_from: Start of filing date range in "MM/DD/YYYY" format.
            date_to: End of filing date range in "MM/DD/YYYY" format.
            utility_name: Name of the utility to search (for docket search).
            control_number: Docket control number (for filings search).
            description: Text to match in docket descriptions.
            utility_type: Utility type. Default is "A" (ALL).
            document_type: Filter by document type. Default is "ALL".
            item_match: Match logic for string filters.
            sort_order: Sort by date filed.
            timeout: Timeout in seconds. Defaults to 30.

        Returns:
            JSON string with the search results.
        """
        try:
            timestamp = generate_timestamp()
            save_csv_path = f"texas_puc_filings_{timestamp}.csv"
            if not (date_from and date_to):
                raise ValueError("date_from and date_to must be provided in MM/DD/YYYY format")

            is_filing_search = bool(control_number)
            base_url = (
                "https://interchange.puc.texas.gov/search/filings/"
                if is_filing_search
                else "https://interchange.puc.texas.gov/search/dockets/"
            )

            query_params = {
                "UtilityType": utility_type,
                "ItemMatch": item_match,
                "DocumentType": document_type,
                "SortOrder": sort_order,
                "DateFiledFrom": f"{date_from} 00:00:00",
                "DateFiledTo": f"{date_to} 00:00:00",
            }

            if control_number:
                query_params["ControlNumber"] = control_number
            if utility_name:
                query_params["UtilityName"] = utility_name
            if description:
                query_params["Description"] = description
            full_url = f"{base_url}?{urlencode(query_params)}"
            # Use the system CA bundle instead of certifi's — the Texas PUC
            # cert chains to a root CA that certifi has dropped.
            response = request_with_retry(
                "get",
                full_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=timeout,
                verify=get_system_ca_bundle(),
            )
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.select("table tr")[1:]
            results = []
            for row in rows:
                cols = row.find_all("td")
                if len(cols) == 4:
                    link_tag = cols[0].find("a")
                    if not link_tag:
                        continue
                    control_number_text = link_tag.text.strip()
                    control_link = "https://interchange.puc.texas.gov" + str(link_tag.get("href", ""))
                    filings = cols[1].text.strip()
                    utility = cols[2].text.strip()
                    summary = cols[3].text.strip()
                    results.append(
                        {
                            "control_number": control_number_text,
                            "description": summary,
                            "utility": utility,
                            "filings": filings,
                            "link": control_link,
                        }
                    )

            saved_csv = self._save_csv(results, save_csv_path)
            return self._result_json(
                {
                    "source": "Texas PUC",
                    "search_url": full_url,
                    "num_results": len(results),
                    "results": results,
                    "saved_csv": saved_csv,
                }
            )
        except Exception as e:
            logger.error(f"Texas PUC search failed: {e}")
            return json.dumps({"error": str(e), "source": "Texas PUC"})
