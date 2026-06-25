import json
import os
import shutil
from pathlib import Path

from loguru import logger

from energyevals.utils import generate_timestamp
import energyevals.tools.sandbox as _sandbox
from energyevals.tools.base_tool import BaseTool, tool_method
from energyevals.tools.sandbox import SANDBOX_WORK_DIR, SANDBOX_WORK_MOUNT, host_path, sandbox_path

_SCRIPT_SRC = Path(__file__).resolve().parents[2] / "sandbox" / "scripts" / "battery_optimize.py"
_SANDBOX_TIMEOUT_S = 600
_SANDBOX_UNAVAILABLE_MSG = (
    "Execution sandbox (Docker) is unavailable. Start the Docker daemon and build the "
    "image: docker build -t energyevals-sandbox -f sandbox/Dockerfile sandbox/."
)


class BatteryOptimizationTool(BaseTool):

    def __init__(self) -> None:
        super().__init__(
            name="battery_optimization",
            description="Optimize battery storage operations for maximum revenue",
        )

    @tool_method()
    def battery_revenue_optimization(
        self,
        run_description: str,
        csv_path: str,
        energy_price_column: str,
        battery_size_mw: float,
        battery_duration: float,
        battery_degradation_cost: float = 24.0,
        round_trip_efficiency: float = 0.81,
        minimum_state_of_charge: float = 0.1,
        maximum_state_of_charge: float = 0.8,
        timestep_in_hours: float = 1.0,
        days: float = 365.0,
    ) -> str:
        """Produces optimal potential net revenues for battery storage projects using energy arbitrage only
        (perfect price foresight, no price-uncertainty penalty). Reads a CSV of energy prices and returns
        summary metrics, time-series rows, and a saved CSV path.

        Args:
            run_description: One word description of the model run, used for the filename of the output csv file
            csv_path: Path to the CSV file containing the energy prices.
            energy_price_column: Column name of the energy prices.
            battery_size_mw: Battery size in MW. Battery size is the maximum power output of the battery.
            battery_duration: Battery duration in hours. Battery duration is the maximum amount of time the battery can store energy.
            battery_degradation_cost: cost in $/MWh reflecting additional costs to preserve battery life based on number
                                    of cycles and capital cost. Default is $24/MWh based on $110,000/MW and $205,000/MWh
                                    default capex and 6000 cycles
            round_trip_efficiency: float reflecting battery charging and discharging efficiency. Should always be between 0 and 1
            minimum_state_of_charge: float reflecting battery minimum state of charge. Should always be between 0 and 1
            maximum_state_of_charge: float reflecting battery maximum state of charge. Should always be between 0 and 1
            timestep_in_hours: float representing minimum time step
            days: float representing number of days in the input price data

        Returns:
            JSON string with the optimization results. The value fields are as follows.

            "run_description": description of the run,
            "net_revenue": float reflecting net battery revenues after removing charging and degradation costs in $
            "total_revenue": float reflecting total battery discharge revenues in $
            "charging_cost": float reflecting annual charging costs in $
            "degradation_penalty": float reflecting additional O&M for degradation management in $
            "rows": contains time series results including operating profile, state_of_charge, net_revenue, total_revenue,
                    charging_cost an degradation penalty
            "saved_csv": saved file path for the output csv

        """
        if battery_size_mw <= 0:
            return json.dumps({"error": "battery_size_mw must be positive."})
        if battery_duration <= 0:
            return json.dumps({"error": "battery_duration must be positive."})
        if timestep_in_hours <= 0:
            return json.dumps({"error": "timestep_in_hours must be positive."})
        if not (0 < round_trip_efficiency <= 1):
            return json.dumps({"error": "round_trip_efficiency must be in (0, 1]."})
        if not (0 <= minimum_state_of_charge < 1):
            return json.dumps({"error": "minimum_state_of_charge must be in [0, 1)."})
        if not (0 < maximum_state_of_charge <= 1):
            return json.dumps({"error": "maximum_state_of_charge must be in (0, 1]."})
        if minimum_state_of_charge >= maximum_state_of_charge:
            return json.dumps({"error": "minimum_state_of_charge must be less than maximum_state_of_charge."})

        # The model may pass a sandbox path (e.g. /work/prices.csv); map it to the host.
        actual_csv = host_path(csv_path)
        if not os.path.exists(actual_csv):
            return json.dumps({"error": f"CSV file not found: {csv_path}"})

        if not _sandbox.sandbox_available():
            return json.dumps({"status": "error", "error": _SANDBOX_UNAVAILABLE_MSG})

        try:
            timestamp = generate_timestamp()
            SANDBOX_WORK_DIR.mkdir(parents=True, exist_ok=True)

            inputs_file = SANDBOX_WORK_DIR / f"battery_inputs_{timestamp}.json"
            outputs_file = SANDBOX_WORK_DIR / f"battery_outputs_{timestamp}.json"
            output_csv = f"{SANDBOX_WORK_MOUNT}/battery_{run_description}_{timestamp}.csv"

            inputs_file.write_text(json.dumps({
                "csv_path": sandbox_path(actual_csv),
                "energy_price_column": energy_price_column,
                "battery_size_mw": battery_size_mw,
                "battery_duration": battery_duration,
                "battery_degradation_cost": battery_degradation_cost,
                "round_trip_efficiency": round_trip_efficiency,
                "minimum_state_of_charge": minimum_state_of_charge,
                "maximum_state_of_charge": maximum_state_of_charge,
                "timestep_in_hours": timestep_in_hours,
                "days": days,
                "run_description": run_description,
                "output_csv_path": output_csv,
                "output_path": f"{SANDBOX_WORK_MOUNT}/battery_outputs_{timestamp}.json",
            }))

            shutil.copy2(_SCRIPT_SRC, SANDBOX_WORK_DIR / "battery_optimize.py")

            result = _sandbox.run_shell(
                f"python3 /work/battery_optimize.py /work/battery_inputs_{timestamp}.json",
                _SANDBOX_TIMEOUT_S,
            )

            if not outputs_file.exists():
                stderr = (result.get("stderr") or "").strip()
                return json.dumps({"error": "Optimization script produced no output.", "stderr": stderr})

            return outputs_file.read_text()

        except Exception as e:
            logger.error(f"Battery revenue optimization failed: {e}")
            return json.dumps({"error": str(e)})
