"""V271: compare monthly/daily/hourly tail sets under asymmetric shared EF ranges."""
from __future__ import annotations

import hashlib
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v225_partial_snapshot_sharp_risk_bounds as partial  # noqa: E402
import v226_frozen_four_field_confirmation as confirm  # noqa: E402
import v237_nested_daily_hourly_risk as bounds  # noqa: E402

PROTOCOL = ROOT / "docs/V271_ASYMMETRIC_EF_TEMPORAL_VALUE_PROTOCOL.md"
EF_INPUT = ROOT / "results/v269-asymmetric-empirical-ef-support-r2/cems_asymmetric_support_intervals.csv"
OUT = ROOT / "results/v271-asymmetric-ef-temporal-value-r2"
YEAR = 2025
TAU = 500.0
THRESHOLD_MARGIN = 1e-4
TIME_LIMIT_SECONDS = 20.0
REGIONS = {"AZPS": (4, 7, 10), "CPLE": (4, 7, 10), "ERCO": (4, 7, 10),
           "PACE": (4, 7, 10), "SOCO": (4, 7, 10), "PSCO": (4,)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def factor_bounds(quarter: int, intervals: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    q = intervals[intervals.quarter == quarter].set_index("fuel_group")
    low = np.array([q.loc["coal", "support_low_g_kwh"],
                    q.loc["oil", "support_low_g_kwh"],
                    q.loc["gas", "support_low_g_kwh"], 0.0])
    high = np.array([q.loc["coal", "support_high_g_kwh"],
                     q.loc["oil", "support_high_g_kwh"],
                     q.loc["gas", "support_high_g_kwh"], 0.0])
    return low, high


def solve_regime(hourly: np.ndarray, daily: np.ndarray, margins: np.ndarray,
                 ef_low: np.ndarray, ef_high: np.ndarray, hourly_known: bool) -> dict:
    lower_solutions = bounds.solve_bounds(hourly, daily, margins, ef_low, TAU, hourly_known)
    upper_solutions = bounds.solve_bounds(hourly, daily, margins, ef_high, TAU, hourly_known)
    low_endpoint = lower_solutions[0]
    high_endpoint = upper_solutions[1]
    return {"min_bound": low_endpoint["bound"], "min_optimal": low_endpoint["optimal"],
            "min_status": low_endpoint["status"], "min_gap": low_endpoint["mip_gap"],
            "max_bound": high_endpoint["bound"], "max_optimal": high_endpoint["optimal"],
            "max_status": high_endpoint["status"], "max_gap": high_endpoint["mip_gap"],
            "all_solved": lower_solutions + upper_solutions}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    intervals = pd.read_csv(EF_INPUT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)
    bounds.TIME_LIMIT_SECONDS = TIME_LIMIT_SECONDS
    endpoint_rows = []
    case_rows = []
    for region, months in REGIONS.items():
        for month in months:
            key = (region, YEAR, month)
            if key not in series:
                raise RuntimeError(f"missing frozen EIA case {key}")
            entry = series[key]
            fuel_totals = np.asarray(entry["fuel_totals"], float)
            fuel = np.asarray(entry["fuel_profile"], float) * fuel_totals[None, :]
            hourly = fuel.sum(axis=1)
            daily = hourly.reshape(-1, 24).sum(axis=1)
            fuel_labels = np.asarray(partial.EF, float)
            margins = np.array([fuel_totals[fuel_labels == ef].sum()
                                for ef in (1000.0, 800.0, 400.0, 0.0)], dtype=float)
            quarter = (month - 1) // 3 + 1
            ef_low, ef_high = factor_bounds(quarter, intervals)
            nominal_ef = np.array([1000., 800., 400., 0.])
            actual_proxy_tail = int(np.sum(np.asarray(entry["cif"], float) >= TAU + THRESHOLD_MARGIN))
            if not np.isclose(hourly.sum(), margins.sum(), rtol=1e-10, atol=1e-6):
                raise RuntimeError(f"grouped margin conservation failed for {key}")
            monthly_min = bounds.monthly_only_bounds(margins, ef_low, TAU + THRESHOLD_MARGIN, len(hourly))[0]
            monthly_max = bounds.monthly_only_bounds(margins, ef_high, TAU + THRESHOLD_MARGIN, len(hourly))[1]
            for regime, hourly_known in (("daily_total_only", False), ("exact_hourly_total", True)):
                solved = solve_regime(hourly, daily, margins, ef_low, ef_high, hourly_known)
                low_min, low_max, high_min, high_max = solved["all_solved"]
                if low_min["bound"] > high_min["bound"] or low_max["bound"] > high_max["bound"]:
                    raise RuntimeError(f"factor monotonicity failed for {key}, {regime}")
                for scenario, result in (("all_factors_low", solved["all_solved"][0]),
                                         ("all_factors_low_max_diagnostic", solved["all_solved"][1]),
                                         ("all_factors_high_min_diagnostic", solved["all_solved"][2]),
                                         ("all_factors_high", solved["all_solved"][3])):
                    endpoint_rows.append({"region": region, "year": YEAR, "month": month,
                                          "quarter": quarter, "regime": regime, "scenario": scenario,
                                          "endpoint": result["endpoint"], "bound": result["bound"],
                                          "optimal": result["optimal"], "status": result["status"],
                                          "mip_gap": result["mip_gap"]})
                width = solved["max_bound"] - solved["min_bound"]
                case_rows.append({"region": region, "year": YEAR, "month": month, "quarter": quarter,
                                  "regime": regime, "min_bound": solved["min_bound"],
                                  "max_bound": solved["max_bound"], "width": width,
                                  "both_endpoints_optimal": solved["min_optimal"] and solved["max_optimal"],
                                  "monthly_min": monthly_min, "monthly_max": monthly_max,
                                  "monthly_width": monthly_max - monthly_min,
                                  "proxy_tail_count": actual_proxy_tail,
                                  "ef_low": ef_low.tolist(), "ef_high": ef_high.tolist(),
                                  "all_corners_monotone_diagnostic": True})
                print(f"solved {key} {regime}: bounds=({solved['min_bound']},{solved['max_bound']}), "
                      f"optimal={solved['min_optimal'] and solved['max_optimal']}", flush=True)

    endpoints = pd.DataFrame(endpoint_rows)
    cases = pd.DataFrame(case_rows)
    cases["proxy_witness_in_set"] = ((cases.min_bound <= cases.proxy_tail_count)
                                     & (cases.max_bound >= cases.proxy_tail_count))
    cases["nested_in_monthly"] = ((cases.min_bound >= cases.monthly_min)
                                  & (cases.max_bound <= cases.monthly_max))
    pivot = cases.pivot(index=["region", "year", "month", "quarter"], columns="regime",
                        values=["width", "both_endpoints_optimal", "min_bound", "max_bound"])
    paired_keys = pivot.index[pivot[("both_endpoints_optimal", "daily_total_only")]
                              & pivot[("both_endpoints_optimal", "exact_hourly_total")]]
    paired = cases.set_index(["region", "year", "month", "quarter"]).loc[paired_keys].reset_index()
    paired_wide = paired.pivot(index=["region", "year", "month", "quarter"], columns="regime",
                               values=["width", "min_bound", "max_bound"])
    paired_result = paired_wide.reset_index()
    paired_result.columns = ["_".join(str(x) for x in col if str(x)) if isinstance(col, tuple) else col
                             for col in paired_result.columns]
    paired_result["hourly_width_reduction"] = (paired_result["width_daily_total_only"]
                                                - paired_result["width_exact_hourly_total"])
    complete = cases[cases.both_endpoints_optimal]
    if not complete.proxy_witness_in_set.all() or not complete.nested_in_monthly.all():
        raise RuntimeError("witness containment or monthly nesting failed")
    for _, row in paired_result.iterrows():
        if row.hourly_width_reduction < 0:
            raise RuntimeError(f"hourly set wider than daily set: {row.to_dict()}")

    OUT.mkdir(parents=True)
    endpoints.to_csv(OUT / "milp_endpoints.csv", index=False)
    cases.to_csv(OUT / "risk_count_bounds.csv", index=False)
    paired_result.to_csv(OUT / "paired_daily_hourly.csv", index=False)
    summary = complete.groupby("regime", as_index=False).agg(
        complete_cases=("region", "size"), mean_monthly_width=("monthly_width", "mean"),
        mean_identified_width=("width", "mean"), mean_witness_containment=("proxy_witness_in_set", "mean"),
        mean_monthly_nesting=("nested_in_monthly", "mean"))
    summary.to_csv(OUT / "summary.csv", index=False)
    metadata = {"status": "asymmetric_factor_set_information_value_not_forecast_validation",
                "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
                "ef_input_sha256": sha256(EF_INPUT), "eia_input_hashes": input_hashes,
                "scipy_version": scipy.__version__, "threshold": TAU + THRESHOLD_MARGIN,
                "candidate_ba_months": int(cases[["region", "month"]].drop_duplicates().shape[0]),
                "case_regimes": int(len(cases)), "endpoint_rows": int(len(endpoints)),
                "optimal_endpoint_rows": int(endpoints.optimal.sum()),
                "complete_paired_months": int(len(paired_result)),
                "complete_cases": int(len(complete)),
                "all_proved_witnesses_inside": bool(complete.proxy_witness_in_set.all()),
                "all_proved_nested_in_monthly": bool(complete.nested_in_monthly.all()),
                "warning": "V269 US CEMS empirical supports transferred to already-open EIA BA-month mixes; fixed-EF EIA proxy labels; no China or predictive coverage validation"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(paired_result.to_string(index=False))
    print(f"complete paired months={metadata['complete_paired_months']}; "
          f"endpoints={metadata['optimal_endpoint_rows']}/{metadata['endpoint_rows']}")


if __name__ == "__main__":
    main()
