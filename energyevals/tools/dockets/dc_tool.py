import json
import re

from loguru import logger

from energyevals.utils import generate_timestamp, request_with_retry

from energyevals.tools.base_tool import tool_method
from energyevals.tools.dockets._base import DocketBaseTool


class DCDocketTool(DocketBaseTool):
    """Search DC PSC filings by date range."""

    def __init__(self) -> None:
        super().__init__(
            name="dc_dockets",
            description="Search DC PSC filings by date range",
        )

    @tool_method(name="search_dc_dockets")
    def search_dc(
        self,
        start_date: str,
        end_date: str,
        keywords: str = "",
        company_individual: str = "",
        case_type_id: str = "",
        docket_number: str = "",
        filing_type_id: str = "",
        sub_filing_type_id: str = "",
        industry_type: str = "",
        records_to_show: int = 50,
        records_to_skip: int = 0,
        timeout: int = 30,
    ) -> str:
        """Search DC PSC filings by date range from the eDocket system.

        Args:
            start_date: Start date in MM/DD/YYYY format.
            end_date: End date in MM/DD/YYYY format.
            keywords: Keyword to search for in filings.
            company_individual: Company or individual to filter by.
            case_type_id: Case type to filter by.
            docket_number: Docket/order number to filter by.
            filing_type_id: Filing type identifier to filter by.
            sub_filing_type_id: Sub-filing type identifier to filter by.
            industry_type: Industry type identifier to filter by.
            records_to_show: Number of records to show. Defaults to 50.
            records_to_skip: Number of records to skip. Defaults to 0.
            timeout: Timeout in seconds. Defaults to 30.

        Returns:
            JSON string with the search results.
        """
        try:
            url = "https://edocket.dcpsc.org/apis/api/Filing/GetFilings"
            params = {
                "isAdmin": "false",
                "orderByColumn": "receivedDate",
                "sortBy": "desc",
                "recordsToSkip": str(records_to_skip),
                "recordsToShow": str(records_to_show),
                "keywords": keywords,
                "isExactMatch": "false",
                "searchThruPDF": "false",
                "companyIndividual": company_individual,
                "caseTypeId": case_type_id,
                "caseNumber": "",
                "itemNumber": "",
                "orderNumber": docket_number,
                "filingTypeId": filing_type_id,
                "filingTypeOther": "",
                "subFilingTypeId": sub_filing_type_id,
                "subFilingTypeOther": "",
                "startDate": start_date,
                "endDate": end_date,
                "industryType": industry_type,
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            response = request_with_retry("get", url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            data = response.json()
            timestamp = generate_timestamp()
            save_csv_path = f"dc_psc_filings_{timestamp}.csv"

            filings_list = []
            for filing in data.get("resultsSet", []):
                download_url = (
                    f"https://edocket.dcpsc.org/public_filesearch/filing/{filing['attachmentId']}"
                    if filing.get("attachmentId") and not filing.get("isConfidential")
                    else None
                )
                description = re.sub(r"<[^>]+>", "", filing.get("description", ""))

                filings_list.append(
                    {
                        "filing_id": filing.get("filingId"),
                        "docket_number": filing.get("docketNumber"),
                        "company_or_individual": filing.get("companyOrIndividual"),
                        "filing_type": filing.get("filingType"),
                        "received_date": filing.get("receivedDate"),
                        "description": description,
                        "attachment_file_name": filing.get("attachmentFileName"),
                        "download_url": download_url,
                        "is_confidential": filing.get("isConfidential"),
                    }
                )

            saved_csv = self._save_csv(filings_list, save_csv_path)
            return self._result_json(
                {
                    "results": filings_list,
                    "num_results": len(filings_list),
                    "saved_csv": saved_csv,
                }
            )
        except Exception as e:
            logger.error(f"DC PSC search failed: {e}")
            return json.dumps({"error": str(e), "source": "DC PSC"})
