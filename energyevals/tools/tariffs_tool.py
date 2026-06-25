import json
import os
import re
from typing import Any, Literal

from loguru import logger

from energyevals.utils import request_with_retry

from energyevals.tools.base_tool import BaseTool, tool_method

_OPENEI_URL = "https://api.openei.org/utility_rates"

# 12×24 schedule matrices that bloat responses without adding analytical value.
# The rate structures (energyratestructure / demandratestructure) are kept because
# they contain the actual prices; the schedules only map time slots to tier indices.
_SCHEDULE_FIELDS = {
    "energyweekdayschedule",
    "energyweekendschedule",
    "demandweekdayschedule",
    "demandweekendschedule",
}

# Analytically essential fields returned by default. Everything else (free-text
# prose like ``description``/``energycomments`` and provenance such as ``source``,
# ``uri``, ``revisions``) is dropped to cut tokens. Free-text/provenance accounts
# for ~60% of a typical response; pass ``verbose=True`` to get the full records.
_ESSENTIAL_FIELDS = {
    "label",
    "utility",
    "eiaid",
    "name",
    "sector",
    "servicetype",
    "startdate",
    "enddate",
    "fixedchargefirstmeter",
    "fixedchargeunits",
    "energyratestructure",
    "demandratestructure",
    "flatdemandstructure",
    "demandunits",
    "demandrateunit",
    "flatdemandunit",
    "is_default",
    "approved",
}


class TariffsTool(BaseTool):

    def __init__(self, api_key: str | None = None):
        super().__init__(
            name="tariffs",
            description="Look up utility electricity tariffs and rate structures",
        )
        self.api_key = api_key or os.getenv("OPEN_EI_API_KEY")
        if not self.api_key:
            logger.warning("OPEN_EI_API_KEY not set. Tool will not function.")

    @tool_method()
    def get_utility_tariffs(
        self,
        sector: Literal["Residential", "Commercial", "Industrial", "Lighting"],
        address: str = "",
        state: str = "",
        eia_id: int | None = None,
        active_only: bool = True,
        approved: Literal["true", "false"] = "true",
        include_schedules: bool = False,
        verbose: bool = True,
        max_results: int | None = None,
    ) -> str:
        """Look up utility electricity tariff records from the OpenEI IURDB for a given location and customer sector.

        At least one of `address`, `state`, or `eia_id` MUST be provided. Omitting all three
        causes the API to return every tariff in the country, which will time out.

        Args:
            sector: Customer type. One of "Residential", "Commercial", "Industrial", or "Lighting".
            address: Full street address including city, state, and ZIP code
                     (e.g., "123 Main St, Richmond, VA 23219"). Preferred over `state` alone
                     because it resolves the specific utility serving that location.
                     Leave empty if using `state` or `eia_id` instead.
            state: Two-letter US state abbreviation (e.g., "VA") to retrieve all tariffs for
                   utilities in that state. Use when you don't have a specific address.
                   Ignored if `address` is provided.
            eia_id: EIA utility ID to retrieve tariffs for a specific utility
                    (e.g., 13781 for Northern States Power Company - Wisconsin).
                    Can be combined with `state` or `address`.
            active_only: If True (default), returns only currently active tariffs (no end date).
                         Set to False to include retired tariffs.
            approved: Filter records by OpenEI approval status. "true" (default) returns only
                      approved rates; "false" returns only unapproved ones. Passed through to
                      the OpenEI API's `approved` query parameter.
            include_schedules: Only relevant with verbose=True. If False (default), omits the 12×24
                               time-of-use schedule matrices (energyweekdayschedule, energyweekendschedule,
                               demandweekdayschedule, demandweekendschedule). Set to True only when you
                               need to know which rate tier applies at a specific hour.
            verbose: If True (default), returns the complete raw records (all fields). Set False to
                     return only the analytically essential fields per record (utility, name, rate
                     structures, charges, dates, etc.), dropping free-text prose (description,
                     energycomments) and provenance (source, uri, revisions) to save tokens.
            max_results: Optional cap on the number of tariff records returned. A single address can
                         return a dozen near-variant schedules; set this (e.g. 5) to limit output.
                         None (default) returns all matching records.

        Returns:
            JSON string with a list of tariff records, or an error message.
        """
        if not self.api_key:
            return json.dumps({"error": "OPEN_EI_API_KEY not configured"})

        address = (address or "").strip()
        state = (state or "").strip()

        if not address and not state and eia_id is None:
            return json.dumps({
                "error": (
                    "At least one of 'address', 'state', or 'eia_id' must be provided. "
                    "Querying without any location filter returns every tariff in the country "
                    "and will time out. Example: state='VA' for all Virginia tariffs."
                )
            })

        params: dict[str, Any] = {
            "version": 7,
            "format": "json",
            "api_key": self.api_key,
            "sector": sector,
            "detail": "full",
            "approved": approved,
        }
        if address:
            params["address"] = address
        elif state:
            params["state"] = state
        if eia_id is not None:
            params["eia"] = eia_id

        try:
            response = request_with_retry("get", _OPENEI_URL, params=params, timeout=60)
            response.raise_for_status()

            cleaned_text = re.sub(r'[\x00-\x1F\x7F]', '', response.text)
            data = json.loads(cleaned_text)
            items = data.get("items") or []

            if not items:
                return json.dumps({"error": "Tariffs not found for this location at this time"})

            if active_only:
                items = [item for item in items if not item.get("enddate")]

            if verbose:
                if not include_schedules:
                    items = [{k: v for k, v in item.items() if k not in _SCHEDULE_FIELDS} for item in items]
            else:
                items = [{k: v for k, v in item.items() if k in _ESSENTIAL_FIELDS} for item in items]

            if max_results is not None and max_results >= 0:
                items = items[:max_results]

            return json.dumps(items, separators=(",", ":"))

        except Exception as e:
            logger.error(f"Tariff lookup failed: {e}")
            return json.dumps({"error": str(e), "address": address, "state": state})
