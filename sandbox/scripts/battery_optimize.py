"""Battery revenue optimization script — runs inside the Docker sandbox.

Reads inputs from the JSON file path given as argv[1], writes results to a
sibling battery_outputs_*.json file. All paths are sandbox-relative (/work/...).
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyomo.environ import (
    ConcreteModel,
    ConstraintList,
    NonNegativeReals,
    Objective,
    Var,
    maximize,
)
from pyomo.opt import SolverFactory, TerminationCondition

_ROUNDING_PROFILE = 4
_ROUNDING_SUMMARY = 2
_INITIAL_SOC_FRACTION = 0.5
_CSV_MAX_ROWS = 10_000_000
_CSV_MAX_MB = 200


def fail(output_path: Path, msg: str) -> None:
    output_path.write_text(json.dumps({"error": msg}))
    sys.exit(0)


def main() -> None:
    inputs_path = Path(sys.argv[1])
    inputs = json.loads(inputs_path.read_text())
    output_path = Path(inputs["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    csv_path = inputs["csv_path"]
    energy_price_column = inputs["energy_price_column"]
    battery_size_mw = inputs["battery_size_mw"]
    battery_duration = inputs["battery_duration"]
    battery_degradation_cost = inputs["battery_degradation_cost"]
    round_trip_efficiency = inputs["round_trip_efficiency"]
    minimum_soc = inputs["minimum_state_of_charge"]
    maximum_soc = inputs["maximum_state_of_charge"]
    timestep_in_hours = inputs["timestep_in_hours"]
    days = inputs["days"]
    run_description = inputs["run_description"]
    output_csv_path = inputs["output_csv_path"]

    # Validate CSV size before loading.
    size_bytes = Path(csv_path).stat().st_size
    if size_bytes > _CSV_MAX_MB * 1024 * 1024:
        fail(output_path, f"CSV is {size_bytes / 1024 / 1024:.1f} MB, exceeding the {_CSV_MAX_MB} MB limit.")

    df = pd.read_csv(csv_path)

    if len(df) > _CSV_MAX_ROWS:
        fail(output_path, f"CSV has {len(df):,} rows, exceeding the {_CSV_MAX_ROWS:,} row limit.")

    if energy_price_column not in df.columns:
        fail(output_path, f"Column '{energy_price_column}' not found in CSV.")

    energy_price = df[energy_price_column].values
    horizon = int(days * 24 / timestep_in_hours)

    if horizon > len(energy_price):
        fail(output_path, (
            f"Requested horizon ({horizon} intervals = {days} days) "
            f"exceeds available price data ({len(energy_price)} intervals). "
            f"Reduce 'days' or provide a longer CSV."
        ))

    battery_size_mwh = battery_size_mw * battery_duration
    charge_eff = np.sqrt(round_trip_efficiency)
    discharge_eff = np.sqrt(round_trip_efficiency)

    model = ConcreteModel()
    model.storage_soc = Var(range(horizon), domain=NonNegativeReals)
    model.charge_power = Var(range(horizon), domain=NonNegativeReals)
    model.discharge_power = Var(range(horizon), domain=NonNegativeReals)
    model.abs_power = Var(range(horizon), domain=NonNegativeReals)
    model.constraints = ConstraintList()

    for t in range(horizon):
        model.constraints.add(model.storage_soc[t] >= minimum_soc * battery_size_mwh)
        model.constraints.add(model.storage_soc[t] <= maximum_soc * battery_size_mwh)
        model.constraints.add(model.charge_power[t] <= battery_size_mw)
        model.constraints.add(model.discharge_power[t] <= battery_size_mw)
        if t > 0:
            model.constraints.add(
                model.storage_soc[t]
                == model.storage_soc[t - 1]
                + timestep_in_hours * (
                    (charge_eff * model.charge_power[t])
                    - ((1 / discharge_eff) * model.discharge_power[t])
                )
            )
        model.constraints.add(model.charge_power[t] - model.discharge_power[t] <= model.abs_power[t])
        model.constraints.add(model.discharge_power[t] - model.charge_power[t] <= model.abs_power[t])

    model.constraints.add(model.storage_soc[0] == _INITIAL_SOC_FRACTION * battery_size_mwh)
    model.constraints.add(model.storage_soc[0] == model.storage_soc[horizon - 1])

    def total_cost(m: ConcreteModel):
        energy_rev = sum(
            timestep_in_hours * energy_price[t] * (m.discharge_power[t] - m.charge_power[t])
            for t in range(horizon)
        )
        degradation_penalty = sum(
            timestep_in_hours * battery_degradation_cost * m.abs_power[t]
            for t in range(horizon)
        )
        return energy_rev - degradation_penalty

    model.cost = Objective(rule=total_cost, sense=maximize)

    solver = SolverFactory("ipopt")
    if not solver.available():
        fail(output_path, "IPOPT solver not available in sandbox.")

    results = solver.solve(model)
    termination = results.solver.termination_condition

    if termination not in {
        TerminationCondition.optimal,
        TerminationCondition.feasible,
        TerminationCondition.locallyOptimal,
    }:
        fail(output_path, json.dumps({
            "error": "Optimization did not converge to a feasible solution.",
            "termination_condition": str(termination),
            "solver_status": str(results.solver.status),
        }))

    if any(model.discharge_power[t].value is None for t in range(horizon)):
        fail(output_path, json.dumps({
            "error": "Optimization produced incomplete results.",
            "termination_condition": str(termination),
        }))

    total_revenue_series = [
        timestep_in_hours * energy_price[t] * model.discharge_power[t].value
        for t in range(horizon)
    ]
    total_revenue = float(np.sum(total_revenue_series))

    degradation_penalty_series = [
        -1 * timestep_in_hours * battery_degradation_cost * model.abs_power[t].value
        for t in range(horizon)
    ]
    degradation_penalty = float(np.sum(degradation_penalty_series))

    operation_profile = [
        model.discharge_power[t].value - model.charge_power[t].value
        for t in range(horizon)
    ]
    charging_only = np.minimum(0, np.array(operation_profile))
    charging_cost_series = timestep_in_hours * (charging_only * energy_price[: len(charging_only)])
    charging_cost = float(-1 * np.sum(charging_only * energy_price[: len(charging_only)]) * timestep_in_hours)

    # degradation and charging cost series are pre-negated, so addition is correct here.
    net_revenue_series = (
        np.array(total_revenue_series) + charging_cost_series + np.array(degradation_penalty_series)
    )
    net_revenue = float(np.sum(net_revenue_series))

    state_of_charge = [model.storage_soc[t].value / battery_size_mwh for t in range(horizon)]

    output_rows = [
        {
            "operation_profile": round(operation_profile[i], _ROUNDING_PROFILE),
            "state_of_charge": round(state_of_charge[i], _ROUNDING_PROFILE),
            "net_revenue": round(float(net_revenue_series[i]), _ROUNDING_PROFILE),
            "total_revenue": round(float(total_revenue_series[i]), _ROUNDING_PROFILE),
            "charging_cost": round(float(charging_cost_series[i]), _ROUNDING_PROFILE),
            "degradation_penalty": round(float(degradation_penalty_series[i]), _ROUNDING_PROFILE),
        }
        for i in range(horizon)
    ]

    pd.DataFrame(output_rows).to_csv(output_csv_path, index=False)

    output_path.write_text(json.dumps(
        {
            "run_description": run_description,
            "net_revenue": round(net_revenue, _ROUNDING_SUMMARY),
            "total_revenue": round(total_revenue, _ROUNDING_SUMMARY),
            "charging_cost": round(charging_cost, _ROUNDING_SUMMARY),
            "degradation_penalty": round(degradation_penalty, _ROUNDING_SUMMARY),
            "row_count": len(output_rows),
            "saved_csv": output_csv_path,
        },
        indent=2,
    ))


if __name__ == "__main__":
    main()
