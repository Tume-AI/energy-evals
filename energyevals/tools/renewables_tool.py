import json
import os
import threading
import time
from io import StringIO
from typing import Any, Literal

import pandas as pd
import requests
from loguru import logger

from energyevals.tools.base_tool import BaseTool, tool_method
from energyevals.utils import generate_timestamp, http_error_detail

from energyevals.tools.constants import (
    CSV_PREVIEW_ROWS,
    DATA_PREVIEW_SIZE,
    HTTP_TIMEOUT_LONG,
    RENEWABLES_MAX_RETRIES,
    RENEWABLES_MIN_REQUEST_INTERVAL_S,
    RENEWABLES_RETRY_BASE_DELAY_S,
)
from energyevals.tools.sandbox import SANDBOX_WORK_DIR, sandbox_path

_RENEWABLES_BASE_URL = "https://www.renewables.ninja/api"


class RenewablesTool(BaseTool):

    # Process-wide throttle: Renewables.ninja limits to 1 request/second (burst)
    # and 50/hour (sustained), per account. Share one limiter across instances.
    _rate_lock = threading.Lock()
    _last_request_monotonic = 0.0

    def __init__(
        self,
        api_key: str | None = None,
        min_request_interval: float = RENEWABLES_MIN_REQUEST_INTERVAL_S,
        max_retries: int = RENEWABLES_MAX_RETRIES,
        retry_base_delay: float = RENEWABLES_RETRY_BASE_DELAY_S,
    ):
        super().__init__(
            name="renewables",
            description="Get solar and wind generation profiles",
        )
        self.api_key = api_key or os.getenv("RENEWABLES_NINJA_API_KEY")
        self.min_request_interval = min_request_interval
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay

        if not self.api_key:
            logger.warning("RENEWABLES_NINJA_API_KEY not set. Tool will not function.")

    def _throttle(self) -> None:
        """Block until at least ``min_request_interval`` has passed since the last
        request, process-wide, to honor the 1/second burst limit."""
        with self._rate_lock:
            elapsed = time.monotonic() - RenewablesTool._last_request_monotonic
            wait = self.min_request_interval - elapsed
            if wait > 0:
                time.sleep(wait)
            RenewablesTool._last_request_monotonic = time.monotonic()

    @staticmethod
    def _extract_numeric_values(data: object) -> list[float]:
        """Collect numeric values from nested data structures."""
        values: list[float] = []

        def _add(value: object) -> None:
            if isinstance(value, bool):
                return
            if isinstance(value, (int, float)):
                values.append(float(value))
                return
            if isinstance(value, dict):
                for nested in value.values():
                    _add(nested)
                return
            if isinstance(value, (list, tuple)):
                for nested in value:
                    _add(nested)

        _add(data)
        return values

    def _make_request(self, endpoint: str, params: dict[str, Any]) -> Any:
        """Make a rate-limited request to the Renewables.ninja API, retrying on 429.

        Honors the 1/second burst limit via a client-side throttle and backs off
        on 429 using the server's ``Retry-After`` header (which also covers the
        50/hour sustained limit). 4xx other than 429 return immediately.
        """
        if not self.api_key:
            return {"error": "RENEWABLES_NINJA_API_KEY not configured"}

        headers = {
            "x-api-key": self.api_key,
            "Authorization": f"Token {self.api_key}",
        }
        url = f"{_RENEWABLES_BASE_URL}/{endpoint}"

        for attempt in range(self.max_retries + 1):
            self._throttle()
            try:
                response = requests.get(
                    url, headers=headers, params=params, timeout=HTTP_TIMEOUT_LONG
                )
            except requests.exceptions.RequestException as e:
                logger.error(f"Renewables.ninja API request failed: {e}")
                return {"error": str(e)}

            if response.status_code == 429:
                if attempt < self.max_retries:
                    retry_after = response.headers.get("Retry-After")
                    try:
                        delay = float(retry_after) if retry_after else 0.0
                    except ValueError:
                        delay = 0.0
                    if delay <= 0:
                        delay = self.retry_base_delay * (2**attempt)
                    logger.warning(
                        f"Renewables.ninja 429 (attempt {attempt + 1}/"
                        f"{self.max_retries + 1}); backing off {delay:.1f}s before retry."
                    )
                    time.sleep(delay)
                    continue
                logger.error("Renewables.ninja 429: rate limit exceeded after retries.")
                return {"error": "Renewables.ninja rate limit (429) exceeded after retries"}

            try:
                response.raise_for_status()
                if params.get("format") == "csv":
                    return response.text
                return response.json()
            except requests.exceptions.RequestException as e:
                detail = http_error_detail(e)
                logger.error(
                    f"Renewables.ninja API request failed: {e}"
                    + (f" | {detail}" if detail else "")
                )
                return {"error": str(e), "detail": detail} if detail else {"error": str(e)}

        return {"error": "Renewables.ninja request failed after retries"}

    def _parse_csv_response(
        self, result: Any, lat: float, lon: float, save_csv_path: Any
    ) -> str:
        """Parse a Renewables.ninja CSV response, save it, and return a preview."""
        data = result if isinstance(result, str) else result.get("data", "")
        # renewables.ninja prefixes the CSV with a multi-line "#" metadata
        # preamble (the 3rd line is a JSON blob full of commas); without
        # comment="#" pandas reads the banner as the header and dies with
        # "Expected 2 fields in line 3, saw 12".
        df = pd.read_csv(StringIO(data), comment="#")
        df.to_csv(save_csv_path, index=False)
        return json.dumps(
            {
                "location": {"lat": lat, "lon": lon},
                "rows": len(df),
                "data_preview": df.head(CSV_PREVIEW_ROWS).to_dict(orient="records"),
                "saved_csv": sandbox_path(save_csv_path),
            },
            indent=2,
            default=str,
        )

    def _build_profile_response(
        self,
        result: Any,
        lat: float,
        lon: float,
        capacity: float,
        save_csv_path: Any,
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Build the JSON profile summary from a Renewables.ninja JSON result."""
        data = result.get("data", {})
        values = self._extract_numeric_values(data)
        summary: dict[str, Any] = {
            "location": {"lat": lat, "lon": lon},
            "capacity_kw": capacity,
            **(extra or {}),
            # The API returns electricity already scaled to `capacity` (kW), so
            # normalize by capacity for the 0-1 factors, and the kWh total is just
            # the sum of the hourly kW values (no extra * capacity).
            "annual_capacity_factor": (sum(values) / len(values) / capacity) if values and capacity else 0,
            "peak_output": (max(values) / capacity) if values and capacity else 0,
            "annual_generation_kwh": sum(values) if values else 0,
            "num_hours": len(values),
            "sample_data": dict(list(data.items())[:DATA_PREVIEW_SIZE]) if isinstance(data, dict) else values[:DATA_PREVIEW_SIZE],
        }
        if isinstance(data, dict):
            try:
                df = pd.DataFrame(list(data.items()), columns=["timestamp", "value"])
                df.to_csv(save_csv_path, index=False)
                summary["saved_csv"] = sandbox_path(save_csv_path)
            except Exception as e:
                summary["csv_error"] = str(e)
        return json.dumps(summary, indent=2, default=str)

    @tool_method()
    def get_solar_profile(
        self,
        lat: float,
        lon: float,
        date_from: str,
        date_to: str,
        capacity: float = 1.0,
        tilt: int = 35,
        azimuth: int = 180,
        system_loss: float = 0.1,
        tracking: int = 0,
        format: Literal["json", "csv"] = "json",
    ) -> str:
        """Get an hourly solar PV generation profile for any location via Renewables.ninja, returning
        normalized capacity factors (0-1) for each hour, plus summary metrics and sample data.
        Useful for solar project analysis and energy modeling.

        Each request must span AT MOST 1 year: the API rejects ranges where date_to is more than
        one year after date_from (HTTP 400 "date_to exceeds allowed max_daterange of P1Y"). To cover
        multiple years, make one call per year and combine the results.

        Args:
            lat: Latitude of the location in degrees.
            lon: Longitude of the location in degrees.
            date_from (str): Starting date for observations in "yyyy-mm-dd" format.
            date_to (str): End date in "yyyy-mm-dd" format. Must be no more than 1 year after date_from.
            capacity: System capacity in kW (default: 1.0).
            tilt: How far the panel is inclined from the horizontal, in degrees. A tilt of 0 degrees is a panel facing
                  directly upwards, 90 degrees is a panel installed vertically, facing sideways.
            azimuth: Compass direction the panel is facing (clockwise). An azimuth angle of 180 degrees means poleward facing,
                     so for latitudes >=0 is interpreted as southwards facing, else northwards facing.
            system_loss (float): The combined losses from all system components (e.g. inverter, tracking system) as a
                           fraction between 0 and 1 (default: 0.1 = 10%).
            tracking (int): One of three options - [{'value': 0, 'label': 'None'}, {'value': 1, 'label': '1-axis (azimuth)'},
                        {'value': 2, 'label': '2-axis (tilt & azimuth)'}]
            format: Response format (json or csv).

        Returns:
            JSON string with the following fields:
            "location": dict with lat and lon,
            "capacity_kw": system capacity in kW,
            "annual_capacity_factor": mean capacity factor over the period (0-1),
            "peak_output": maximum hourly capacity factor observed,
            "annual_generation_kwh": estimated total generation in kWh,
            "num_hours": number of hourly data points,
            "sample_data": first 10 data points as a preview,
            "saved_csv": file path of saved CSV with full time-series results.
        """
        params = {
            "lat": lat,
            "lon": lon,
            "date_from": date_from,
            "date_to": date_to,
            "dataset": "merra2",
            "capacity": capacity,
            "system_loss": system_loss,
            "tracking": tracking,
            "tilt": tilt,
            "azim": azimuth,
            "format": format,
        }

        result = self._make_request("data/pv", params)
        if isinstance(result, dict) and "error" in result:
            return json.dumps(result)

        SANDBOX_WORK_DIR.mkdir(parents=True, exist_ok=True)
        save_csv_path = SANDBOX_WORK_DIR / f"solar_profile_{generate_timestamp()}.csv"

        if format.lower() == "csv":
            try:
                return self._parse_csv_response(result, lat, lon, save_csv_path)
            except Exception as e:
                return json.dumps({"error": str(e), "raw": result}, indent=2)

        return self._build_profile_response(result, lat, lon, capacity, save_csv_path)

    @tool_method()
    def get_wind_profile(
        self,
        lat: float,
        lon: float,
        date_from: str,
        date_to: str,
        capacity: float = 1.0,
        height: float = 80.0,
        turbine: str = "Vestas V90 2000",
        format: Literal["json", "csv"] = "json",
    ) -> str:
        """Get an hourly wind generation profile for any location via Renewables.ninja, returning
        normalized capacity factors (0-1) for each hour, plus summary metrics and sample data.
        Useful for wind project analysis and energy modeling.

        Each request must span AT MOST 1 year: the API rejects ranges where date_to is more than
        one year after date_from (HTTP 400 "date_to exceeds allowed max_daterange of P1Y"). To cover
        multiple years, make one call per year and combine the results.

        Args:
            lat: Latitude of the location in degrees.
            lon: Longitude of the location in degrees.
            date_from (str): Starting date for observations in "yyyy-mm-dd" format.
            date_to (str): End date in "yyyy-mm-dd" format. Must be no more than 1 year after date_from.
            capacity: Turbine capacity in kW. Default is 1.
            height: The height of the turbine tower, that is, how far the blades are centred above
                        the ground. Hub heights are limited to between 10 and 300 m. Default is 80.
            turbine: Wind turbine model name. Default is "Vestas V90 2000". Use a valid model from
                  the following list: ['Acciona AW77 1500',
                 'Alstom Eco 110',
                 'Alstom Eco 74',
                 'Alstom Eco 80',
                 'Bonus B23 150',
                 'Bonus B33 300',
                 'Bonus B37 450',
                 'Bonus B41 500',
                 'Bonus B44 600',
                 'Bonus B54 1000',
                 'Bonus B62 1300',
                 'Bonus B82 2300',
                 'Dewind D4 41 500',
                 'Dewind D6 62 1000',
                 'Enercon E101 3000',
                 'Enercon E101 3500',
                 'Enercon E103 2350',
                 'Enercon E112 4500',
                 'Enercon E115 2500',
                 'Enercon E115 3000',
                 'Enercon E115 3200',
                 'Enercon E126 3500',
                 'Enercon E126 4000',
                 'Enercon E126 4200',
                 'Enercon E126 6500',
                 'Enercon E126 7000',
                 'Enercon E126 7500',
                 'Enercon E138 3500',
                 'Enercon E138 4260',
                 'Enercon E141 4200',
                 'Enercon E160 4600',
                 'Enercon E160 5560',
                 'Enercon E175 6000',
                 'Enercon E40 500',
                 'Enercon E40 600',
                 'Enercon E44 900',
                 'Enercon E48 800',
                 'Enercon E53 800',
                 'Enercon E66 1500',
                 'Enercon E66 1800',
                 'Enercon E66 2000',
                 'Enercon E70 2000',
                 'Enercon E70 2300',
                 'Enercon E82 1800',
                 'Enercon E82 2000',
                 'Enercon E82 2300',
                 'Enercon E82 3000',
                 'Enercon E92 2300',
                 'Enercon E92 2350',
                 'EWT DirectWind 52 900',
                 'Gamesa G114 2000',
                 'Gamesa G114 2100',
                 'Gamesa G114 2500',
                 'Gamesa G114 2625',
                 'Gamesa G128 4500',
                 'Gamesa G128 5000',
                 'Gamesa G47 660',
                 'Gamesa G52 850',
                 'Gamesa G58 850',
                 'Gamesa G80 2000',
                 'Gamesa G87 2000',
                 'Gamesa G90 2000',
                 'Gamesa G97 2000',
                 'GE 1.5s',
                 'GE 1.5se',
                 'GE 1.5sl',
                 'GE 1.5sle',
                 'GE 1.5xle',
                 'GE 1.6',
                 'GE 1.6-82.5',
                 'GE 1.7',
                 'GE 1.85-82.5',
                 'GE 1.85-87',
                 'GE 2.5-100',
                 'GE 2.5-103',
                 'GE 2.5-120',
                 'GE 2.5-88',
                 'GE 2.5xl',
                 'GE 2.75-103',
                 'GE 2.75-120',
                 'GE 2.85-103',
                 'GE 3.2-103',
                 'GE 3.2-130',
                 'GE 3.4-130',
                 'GE 3.4-137',
                 'GE 3.6sl',
                 'GE 3.8-130',
                 'GE 5.3-158',
                 'GE 5.5-158',
                 'GE 6.0-164',
                 'GE 900S',
                 'GE Haliade 6-150',
                 'GE Haliade-X 12-220',
                 'Goldwind GW109 2500',
                 'Goldwind GW121 2500',
                 'Goldwind GW140 3400',
                 'Goldwind GW140 3000',
                 'Goldwind GW154 6700',
                 'Goldwind GW82 1500',
                 'Hitachi HTW5.2-127',
                 'Hitachi HTW5.2-136',
                 'MHI Vestas V117 4200',
                 'MHI Vestas V164 10000',
                 'MHI Vestas V164 8400',
                 'MHI Vestas V164 9500',
                 'MHI Vestas V174 9500',
                 'Mingyang SCD 3000 100',
                 'Mingyang SCD 3000 108',
                 'Mingyang SCD 3000 92',
                 'NEG Micon M1500 500',
                 'NEG Micon M1500 750',
                 'NEG Micon NM48 750',
                 'NEG Micon NM52 900',
                 'NEG Micon NM60 1000',
                 'NEG Micon NM64c 1500',
                 'NEG Micon NM80 2750',
                 'Nordex N100 2500',
                 'Nordex N100 3300',
                 'Nordex N117 2400',
                 'Nordex N117 3000',
                 'Nordex N117 3600',
                 'Nordex N131 3000',
                 'Nordex N131 3300',
                 'Nordex N131 3600',
                 'Nordex N131 3900',
                 'Nordex N149 4500',
                 'Nordex N27 150',
                 'Nordex N29 250',
                 'Nordex N43 600',
                 'Nordex N50 800',
                 'Nordex N60 1300',
                 'Nordex N80 2500',
                 'Nordex N90 2300',
                 'Nordex N90 2500',
                 'Nordtank NTK500',
                 'Nordtank NTK600',
                 'PowerWind 56 900',
                 'REpower 3.4M',
                 'REpower 5M',
                 'REpower 6M',
                 'REpower MD70 1500',
                 'REPower MD77 1500',
                 'REpower MM70 2000',
                 'REpower MM82 2000',
                 'REpower MM92 2000',
                 'Senvion 3.2M114',
                 'Senvion 6.3M152',
                 'Senvion MM82 2050',
                 'Senvion MM92 2050',
                 'Shanghai Electric W2000 105',
                 'Shanghai Electric W2000 111',
                 'Shanghai Electric W2000 116',
                 'Shanghai Electric W2000 87',
                 'Shanghai Electric W2000 93',
                 'Shanghai Electric W2000 99',
                 'Shanghai Electric W3600 116',
                 'Shanghai Electric W3600 122',
                 'Siemens Gamesa SG 10.0-193',
                 'Siemens Gamesa SG 4.5-145',
                 'Siemens Gamesa SG 5.0-132',
                 'Siemens Gamesa SG 5.0-145',
                 'Siemens Gamesa SG 6.0-154',
                 'Siemens Gamesa SG 6.2-170',
                 'Siemens Gamesa SG 6.6-155',
                 'Siemens Gamesa SG 6.6-170',
                 'Siemens Gamesa SG 7.0-154',
                 'Siemens Gamesa SG 8.0-167',
                 'Siemens Gamesa SG 8.5-167',
                 'Siemens SWT-1.3-62',
                 'Siemens SWT-2.3-101',
                 'Siemens SWT-2.3-108',
                 'Siemens SWT-2.3-82',
                 'Siemens SWT-2.3-93',
                 'Siemens SWT-2.5-108',
                 'Siemens SWT-2.625-120',
                 'Siemens SWT-3.0-101',
                 'Siemens SWT-3.15-142',
                 'Siemens SWT-3.2-101',
                 'Siemens SWT-3.2-108',
                 'Siemens SWT-3.2-113',
                 'Siemens SWT-3.3-130',
                 'Siemens SWT-3.6-107',
                 'Siemens SWT-3.6-120',
                 'Siemens SWT-3.6-130',
                 'Siemens SWT-4.0-120',
                 'Siemens SWT-4.0-130',
                 'Siemens SWT-4.1-142',
                 'Siemens SWT-4.3-120',
                 'Siemens SWT-4.3-130',
                 'Siemens SWT-6.0-154',
                 'Siemens SWT-7.0-154',
                 'Siemens SWT-8.0-154',
                 'Suzlon S88 2100',
                 'Suzlon S97 2100',
                 'Tacke TW600 43',
                 'Vestas V100 1800',
                 'Vestas V100 2000',
                 'Vestas V100 2600',
                 'Vestas V105 3300',
                 'Vestas V105 3450',
                 'Vestas V110 2000',
                 'Vestas V112 3000',
                 'Vestas V112 3300',
                 'Vestas V112 3450',
                 'Vestas V117 3300',
                 'Vestas V117 3450',
                 'Vestas V117 3600',
                 'Vestas V117 4000',
                 'Vestas V120 2200',
                 'Vestas V126 3000',
                 'Vestas V126 3300',
                 'Vestas V126 3450',
                 'Vestas V136 3450',
                 'Vestas V136 4000',
                 'Vestas V150 4000',
                 'Vestas V150 4200',
                 'Vestas V150 4500',
                 'Vestas V150 5600',
                 'Vestas V150 6000',
                 'Vestas V162 5600',
                 'Vestas V162 6000',
                 'Vestas V162 6200',
                 'Vestas V162 7200',
                 'Vestas V164 7000',
                 'Vestas V164 8000',
                 'Vestas V164 9500',
                 'Vestas V172 7200',
                 'Vestas V27 225',
                 'Vestas V29 225',
                 'Vestas V39 500',
                 'Vestas V42 600',
                 'Vestas V44 600',
                 'Vestas V47 660',
                 'Vestas V52 850',
                 'Vestas V66 1650',
                 'Vestas V66 1750',
                 'Vestas V66 2000',
                 'Vestas V80 1800',
                 'Vestas V80 2000',
                 'Vestas V82 1650',
                 'Vestas V90 1800',
                 'Vestas V90 2000',
                 'Vestas V90 3000',
                 'Wind World W3700',
                 'Wind World W4200',
                 'Windflow 33 500',
                 'Windmaster WM28 300',
                 'Windmaster WM43 750',
                 'XANT M21 100',
                 'Dewind D6 1000',
                 'Windflow 500',
                 'GE 2.75 103',
                 'GE 3.4 130',
                 'GE 3.2 103',
                 'GE 3.2 130',
                 'GE 3.8 130',
                 'Siemens SWT 1.3 62',
                 'Siemens SWT 2.3 82',
                 'Siemens SWT 2.3 93',
                 'Siemens SWT 2.3 101',
                 'Siemens SWT 3.0 101',
                 'Siemens SWT 3.6 107',
                 'Siemens SWT 3.6 120',
                 'Siemens SWT 3.6 130',
                 'Siemens SWT 4.0 130',
                 'Siemens SWT 4.3 130',
                 'Siemens SWT 4.1 142',
                 'Siemens Gamesa SG 4.5 145',
                 'GE 5.5 158',
                 'GE 5.3 158']
            format: Response format (json or csv).

        Returns:
            JSON string with the following fields:
            "location": dict with lat and lon,
            "capacity_kw": turbine capacity in kW,
            "hub_height_m": hub height in meters,
            "annual_capacity_factor": mean capacity factor over the period (0-1),
            "peak_output": maximum hourly capacity factor observed,
            "annual_generation_kwh": estimated total generation in kWh,
            "num_hours": number of hourly data points,
            "sample_data": first 10 data points as a preview,
            "saved_csv": file path of saved CSV with full time-series results.
        """
        params = {
            "lat": lat,
            "lon": lon,
            "date_from": date_from,
            "date_to": date_to,
            "dataset": "merra2",
            "capacity": capacity,
            "height": height,
            "turbine": turbine,
            "format": format,
        }

        result = self._make_request("data/wind", params)
        if isinstance(result, dict) and "error" in result:
            return json.dumps(result)

        SANDBOX_WORK_DIR.mkdir(parents=True, exist_ok=True)
        save_csv_path = SANDBOX_WORK_DIR / f"wind_profile_{generate_timestamp()}.csv"

        if format.lower() == "csv":
            try:
                return self._parse_csv_response(result, lat, lon, save_csv_path)
            except Exception as e:
                return json.dumps({"error": str(e), "raw": result}, indent=2)

        return self._build_profile_response(result, lat, lon, capacity, save_csv_path, extra={"hub_height_m": height})
