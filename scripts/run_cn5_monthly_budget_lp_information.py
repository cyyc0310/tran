"""Explore monthly-to-hourly CIF identified sets under public-budget constraints."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, lil_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as fuel_model  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_MONTHLY_BUDGET_LP_INFORMATION_PROTOCOL_20261102.md"
CAPACITY_PATH = ROOT / "data/eurostat-capacity-energy-charts-r1/capacity.csv"
GROUPS = ("thermal", "hydro", "nuclear", "wind", "solar", "other_renewable")
THERMAL_EF_LOWER = 400.0
THERMAL_EF_UPPER = 1000.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def monthly_groups(fine: np.ndarray) -> np.ndarray:
    """Aggregate the hourly proxy into coarse thermal and nonthermal groups."""
    return np.column_stack((
        fine[:, :4].sum(axis=1),  # coal, gas, oil, other thermal
        fine[:, 5],               # hydro
        fine[:, 4],               # nuclear
        fine[:, 6],               # wind
        fine[:, 7],               # solar
        fine[:, 8],               # extra oracle group, absent from CN5 inputs
    ))


def capacity_vector(row: pd.Series) -> np.ndarray:
    return np.array((row.thermal_mw, row.hydro_mw, row.nuclear_other_mw,
                     row.wind_mw, row.solar_pv_mw, np.inf), dtype=float)


def build_transport(equal_hour_totals: np.ndarray,
                    group_totals: np.ndarray) -> csr_matrix:
    """Build monthly group-margin and hourly-total constraints."""
    hours = len(equal_hour_totals)
    group_count = len(group_totals)
    matrix = lil_matrix((group_count + hours, group_count * hours), dtype=float)
    for group in range(group_count):
        matrix[group, group * hours:(group + 1) * hours] = 1.0
    for hour in range(hours):
        for group in range(group_count):
            matrix[group_count + hour, group * hours + hour] = 1.0
    return csr_matrix(matrix)


def solve_hour(matrix: csr_matrix, rhs: np.ndarray, bounds: list[tuple[float, float | None]],
               hours: int, hour: int, total_mw: float) -> tuple[float, float]:
    """Return identified CIF bounds at one hour using endpoint thermal EFs."""
    objective_min = np.zeros(6 * hours, dtype=float)
    objective_max = np.zeros(6 * hours, dtype=float)
    objective_min[hour] = THERMAL_EF_LOWER
    objective_max[hour] = -THERMAL_EF_UPPER
    low = linprog(objective_min, A_eq=matrix, b_eq=rhs, bounds=bounds, method="highs")
    high = linprog(objective_max, A_eq=matrix, b_eq=rhs, bounds=bounds, method="highs")
    if not low.success or not high.success:
        raise RuntimeError(f"transport LP failed: min={low.message}; max={high.message}")
    return float(low.fun / total_mw), float(-high.fun / total_mw)


def oracle_bounds(base: list[tuple[float, float | None]], groups: np.ndarray,
                  hour: int, fixed_groups: tuple[int, ...]) -> list[tuple[float, float | None]]:
    """Fix selected target-hour group outputs to their observed oracle values."""
    bounds = list(base)
    hours = groups.shape[0]
    for group in fixed_groups:
        value = float(groups[hour, group])
        bounds[group * hours + hour] = (value, value)
    return bounds


def source_paths() -> tuple[list[Path], list[Path]]:
    inputs = [CAPACITY_PATH, PROTOCOL]
    for country in nine.COUNTRIES:
        folder, manifest, _ = nine.six_group.data_locations(country)
        inputs.extend((manifest, folder / f"{country}_2022_2024.json"))
        if country in nine.six_group.OLD_COUNTRIES:
            inputs.append(folder / "audit.json")
    code = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
            Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
            Path(fuel_model.__file__).resolve(), Path(model.__file__).resolve(),
            ROOT / "src/transcif/research/cross_country.py"]
    return sorted(set(inputs)), sorted(set(code))


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    input_paths, code_paths = source_paths()
    startup = {
        "input_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in input_paths},
        "code_sha256": {str(path.relative_to(ROOT)): sha256(path) for path in code_paths},
    }
    (output / "startup_hashes.json").write_text(json.dumps(startup, indent=2) + "\n")

    capacity_data = pd.read_csv(CAPACITY_PATH)
    rows: list[dict] = []
    capacity_support_rows: list[dict] = []
    missing_capacity: list[str] = []
    for country in nine.COUNTRIES:
        capacity_row = capacity_data.loc[
            capacity_data.country.eq(country.upper()) & capacity_data.stat_year.eq(2022)]
        has_capacity = len(capacity_row) == 1
        if not has_capacity:
            missing_capacity.append(country.upper())
        raw, features, _ = nine.prepare_country(country)
        part = features.loc[features.year.eq(2024)].copy()
        fine = fuel_model.fuel_for(raw, part)
        time = raw.iloc[part.raw_row.to_numpy()].clock.reset_index(drop=True)
        periods = time.dt.to_period("M")
        hourly_total = fine.sum(axis=1)
        hourly_truth = np.divide(fine @ model.FACTORS, hourly_total,
                                 out=np.zeros(len(hourly_total)), where=hourly_total > 0)
        groups = monthly_groups(fine)
        caps = capacity_vector(capacity_row.iloc[0]) if has_capacity else None
        if has_capacity:
            for group, name in enumerate(GROUPS[:-1]):
                maximum = float(groups[:, group].max())
                capacity = float(caps[group])
                capacity_support_rows.append({
                    "country": country.upper(), "group": name,
                    "capacity_stat_year": 2022, "capacity_mw": capacity,
                    "maximum_observed_hourly_generation_mw": maximum,
                    "hours_over_capacity": int(np.sum(groups[:, group] > capacity + 1e-6)),
                    "max_generation_to_capacity_ratio": (
                        maximum / capacity if capacity > 0 else (0.0 if maximum == 0 else np.inf)),
                    "hard_upper_bound_contains_all_observed_hours": bool(
                        np.all(groups[:, group] <= capacity + 1e-6)),
                })

        for period in periods.unique():
            index = np.flatnonzero(periods.eq(period).to_numpy())
            month_groups = groups[index]
            group_totals = month_groups.sum(axis=0)
            hour_totals = hourly_total[index]
            if not np.isclose(group_totals.sum(), hour_totals.sum(), atol=1e-5):
                raise ValueError(f"monthly energy closure failed: {country} {period}")
            if (hour_totals <= 0).any():
                raise ValueError(f"nonpositive hourly generation: {country} {period}")

            matrix = build_transport(hour_totals, group_totals)
            rhs = np.r_[group_totals, hour_totals]
            mix_only_bounds: list[tuple[float, float | None]] = []
            capacity_bounds: list[tuple[float, float | None]] = []
            for group, total in enumerate(group_totals):
                mix_only_bounds.extend([(0.0, float(total))] * len(index))
                if has_capacity:
                    upper = min(float(total), float(caps[group]))
                    capacity_bounds.extend([(0.0, upper)] * len(index))

            # Select low, median-nearest, and high gross-generation hours.
            sampled = np.unique((int(np.argmin(hour_totals)),
                                 int(np.argmin(np.abs(hour_totals - np.median(hour_totals)))),
                                 int(np.argmax(hour_totals))))
            for hour in sampled:
                records = []
                arms = [("M+T_oracle", mix_only_bounds, ())]
                arms.append(("M+T+H_oracle", oracle_bounds(
                    mix_only_bounds, month_groups, int(hour), (1,)), (1,)))
                arms.append(("M+T+HN_oracle", oracle_bounds(
                    mix_only_bounds, month_groups, int(hour), (1, 2)), (1, 2)))
                arms.append(("M+T+WS_oracle", oracle_bounds(
                    mix_only_bounds, month_groups, int(hour), (3, 4)), (3, 4)))
                arms.append(("M+T+HNW_oracle", oracle_bounds(
                    mix_only_bounds, month_groups, int(hour), (1, 2, 3, 4)),
                    (1, 2, 3, 4)))
                arms.append(("M+T+NTH_oracle", oracle_bounds(
                    mix_only_bounds, month_groups, int(hour), (1, 2, 3, 4, 5)),
                    (1, 2, 3, 4, 5)))
                if has_capacity:
                    arms.append(("M+C+T_oracle", capacity_bounds, ()))
                for budget, bounds, fixed_groups in arms:
                    lower, upper = solve_hour(matrix, rhs, bounds, len(index), int(hour),
                                              float(hour_totals[hour]))
                    records.append({"budget": budget, "lower": lower, "upper": upper,
                                    "width": upper - lower,
                                    "fixed_oracle_groups": [GROUPS[g] for g in fixed_groups],
                                    "sampled_label_contained": bool(
                                        lower - 1e-7 <= hourly_truth[index[hour]] <= upper + 1e-7)})
                for record in records:
                    rows.append({"country": country.upper(), "month": str(period),
                        "hour_index": int(hour), "hourly_total_generation_mw": float(hour_totals[hour]),
                        "actual_proxy_cif_g_per_kwh": float(hourly_truth[index[hour]]),
                        **record})

    result = pd.DataFrame(rows)
    result.to_csv(output / "hourly_bounds.csv", index=False)
    capacity_support = pd.DataFrame(capacity_support_rows)
    capacity_support.to_csv(output / "capacity_support.csv", index=False)
    by_country = (result.groupby(["budget", "country"], as_index=False)
        .agg(sampled_hours=("width", "size"), mean_width=("width", "mean"),
             median_width=("width", "median"),
             sampled_label_containment=("sampled_label_contained", "mean")))
    by_country.to_csv(output / "country_metrics.csv", index=False)
    macro = (by_country.groupby("budget", as_index=False)
        .agg(country_equal_mean_width=("mean_width", "mean"),
             country_equal_sampled_containment=("sampled_label_containment", "mean")))
    summary = {
        "protocol": str(PROTOCOL.relative_to(ROOT)),
        "countries_with_capacity": sorted(capacity_data.loc[
            capacity_data.stat_year.eq(2022), "country"].astype(str).str.upper().unique().tolist()),
        "countries_evaluated": sorted(result.country.unique().tolist()),
        "countries_without_capacity": missing_capacity,
        "capacity_support_all_observed_hours_within_2022_bound": bool(
            capacity_support.hard_upper_bound_contains_all_observed_hours.all()),
        "capacity_support_violations": int(capacity_support.hours_over_capacity.sum()),
        "sampled_hours_per_country": int(result.loc[result.budget.eq("M+T_oracle")].shape[0]
                                          // result.country.nunique()),
        "months": 12,
        "sampling": "monthly low, median-nearest, and high actual gross-generation hours; deduplicated",
        "endpoint_thermal_factors_g_per_kwh": [THERMAL_EF_LOWER, THERMAL_EF_UPPER],
        "macro_metrics": macro.to_dict(orient="records"),
        "limitations": [
            "development-only Energy-Charts fixed-EF generation proxy",
            "target-month generation and hourly total generation are oracle inputs",
            "other_renewable monthly total is an additional oracle group",
            "six-hour-category transportation LP omits dispatch chronology, network, and storage",
            "Eurostat installed capacity is coarse and only paired for five regions",
            "selected hours are descriptive, not a random sample for inferential claims",
        ],
    }
    summary["runtime"] = {"numpy": np.__version__, "pandas": pd.__version__}
    summary["output_sha256"] = {
        name: sha256(output / name) for name in (
            "hourly_bounds.csv", "country_metrics.csv", "capacity_support.csv")
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output.resolve()), indent=2))


if __name__ == "__main__":
    main()
