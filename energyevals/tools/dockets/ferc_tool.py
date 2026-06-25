import json
from typing import Literal

from loguru import logger

from energyevals.utils import generate_timestamp, request_with_retry

from energyevals.tools.base_tool import tool_method
from energyevals.tools.dockets._base import DocketBaseTool


class FERCDocketTool(DocketBaseTool):
    """Search the FERC eLibrary for filings and dockets."""

    def __init__(self) -> None:
        super().__init__(
            name="ferc_dockets",
            description="Search the FERC eLibrary for filings and dockets",
        )

    @tool_method(name="search_ferc_dockets")
    def search_ferc(
        self,
        start_date: str,
        end_date: str,
        keyword: str | None = "",
        affiliation_type: Literal["agent", "author", "recipient"] | None = None,
        affiliation: str | None = None,
        last_name: str | None = None,
        first_initial: str | None = None,
        middle_initial: str | None = None,
        docket_number: str | None = None,
        sub_docket_numbers: list[str] | None = None,
        search_full_text: bool = True,
        search_description: bool = True,
        results_per_page: int = 50,
        page: int = 0,
    ) -> str:
        """Search the FERC eLibrary using the AdvancedSearch API for filings and dockets.
        Returns source, keyword, date range, number of results, detailed results, and a saved CSV path.

        Parameters:
            start_date (str): Start date of the search window in "YYYY-MM-DD" format.
            end_date (str): End date of the search window in "YYYY-MM-DD" format.
            keyword (str,optional): Keyword or phrase to search within documents. Default is ""
            affiliation_type (str,optional): Keyword showing the role of the docket filer. Can only be one of ["agent","author","recipient"]
                                    and can be used to provide specific filters such as the affiliated organization
            affiliation (str,optional): Name of the organization affiliated with the docket
            last_name (str,optional): Last name of the filing entity or personnel
            first_initial(str,optional): First initial of the filing entity or personnel
            middle_initial(str,optional): Middle initial of the filing entity or personnel
            docket_number (str, optional): Docket number to filter the search (e.g., "ER25-1234").
            sub_docket_numbers (List[str], optional): List of sub-docket numbers to include in the filter.
            search_full_text (bool): Whether to search within the full text of documents (default: True).
            search_description (bool): Whether to search within document descriptions (default: True).
            results_per_page (int): Number of results to return per page.
            page (int): Zero-based index for pagination (e.g., page=0 returns the first set of results).

        Returns:
            JSON string with source, keyword, date_range, num_results, results
            (title, filed_date, docket_numbers, category, libraries, accession_number,
            pdf_files, affiliations), and saved_csv path.
        """
        try:

            url = "https://elibrary.ferc.gov/eLibraryWebAPI/api/Search/AdvancedSearch"

            affiliation_inputs = {
                "afType": affiliation_type,
                "affiliation": affiliation,
                "lastName": last_name,
                "firstInitial": first_initial,
                "middleInitial": middle_initial,
            }

            payload = {
                "searchText": keyword,
                "searchFullText": search_full_text,
                "searchDescription": search_description,
                "dateSearches": [
                    {
                        "dateType": "filed_date",
                        "startDate": start_date,
                        "endDate": end_date,
                    }
                ],
                "availability": None,
                "affiliations": [affiliation_inputs],
                "categories": [],
                "libraries": [],
                "accessionNumber": None,
                "eFiling": False,
                "docketSearches": [
                    {
                        "docketNumber": docket_number or "",
                        "subDocketNumbers": sub_docket_numbers or [],
                    }
                ],
                "resultsPerPage": results_per_page,
                "curPage": page,
                "classTypes": [],
                "sortBy": "",
                "groupBy": "NONE",
                "idolResultID": "",
                "allDates": False,
            }

            headers = {"Content-Type": "application/json", "Accept": "application/json"}
            response = request_with_retry("post", url, json=payload, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()

            timestamp = generate_timestamp()
            save_csv_path = f"ferc_search_{timestamp}.csv"

            results = []
            for hit in data.get("searchHits", []):
                result = {
                    "title": hit.get("description", ""),
                    "filed_date": hit.get("filedDate"),
                    "docket_numbers": hit.get("docketNumbers", []),
                    "category": hit.get("category"),
                    "libraries": hit.get("libraries", []),
                    "accession_number": hit.get("accessionNumber"),
                    "pdf_files": [],
                    "affiliations": [
                        f"{a.get('afType')}: {a.get('affiliation')}"
                        for a in hit.get("affiliations", [])
                    ],
                }

                for file in hit.get("transmittals", []):
                    if file.get("fileType") == "PDF":
                        result["pdf_files"].append(
                            {
                                "file_name": file.get("fileName"),
                                "file_desc": file.get("fileDesc"),
                                "file_size": file.get("fileSize"),
                                "download_url": (
                                    "https://elibrary.ferc.gov/eLibrary/filedownload?fileid="
                                    f"{file.get('fileId')}"
                                ),
                            }
                        )
                results.append(result)

            saved_csv = self._save_csv(results, save_csv_path)
            return self._result_json(
                {
                    "source": "FERC",
                    "keyword": keyword,
                    "date_range": f"{start_date} to {end_date}",
                    "num_results": len(results),
                    "results": results,
                    "saved_csv": saved_csv,
                }
            )
        except Exception as e:
            logger.error(f"FERC search failed: {e}")
            return json.dumps({"error": str(e), "source": "FERC"})
