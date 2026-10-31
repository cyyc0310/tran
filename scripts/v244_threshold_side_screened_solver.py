"""V240: temporal tail-count bounds under hour-varying EF uncertainty outer sets."""
from __future__ import annotations

import hashlib
import json
import sys
import time as time_module
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v225_partial_snapshot_sharp_risk_bounds as partial  # noqa: E402
import v226_frozen_four_field_confirmation as confirm  # noqa: E402

PROTOCOL = ROOT / "docs/V244_THRESHOLD_SIDE_SCREENED_SOLVER_PROTOCOL.md"
OUT = ROOT / "results/v244-threshold-side-screened-r1"
BASELINE = ROOT / "results/v240-ef-uncertainty-temporal-risk-r1/risk_count_bounds.csv"
BA_MONTHS = {
    "AZPS": (4, 7, 10), "CPLE": (4, 7, 10), "ERCO": (4, 7, 10),
    "PACE": (4, 7, 10), "SOCO": (4, 7, 10), "PSCO": (4,),
}
YEAR = 2025
TAU = 500.0
EF_TOL = 0.10
GROUP_EF = np.array([1000., 800., 400., 0.])
TIME_LIMIT_SECONDS = 30.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def group_margins(margins: np.ndarray) -> np.ndarray:
    return np.array([margins[partial.EF == g].sum() for g in GROUP_EF], dtype=float)


def solve_uncertain(hourly: np.ndarray, daily: np.ndarray, margins: np.ndarray,
                    hourly_known: bool) -> list[dict]:
    """Bound K when each hour/group EF may vary independently within its box."""
    n_hours, n_groups = len(hourly), len(margins)
    n_days = len(daily)
    x_count = n_hours * n_groups
    n_vars = x_count + n_hours
    ef_lo = GROUP_EF * (1. - EF_TOL)
    ef_hi = GROUP_EF * (1. + EF_TOL)
    a_lo = ef_lo - (TAU + 1e-4)
    a_hi = ef_hi - (TAU + 1e-4)
    min_hour = float(margins.sum()) / n_hours * 1e-6
    n_rows = n_groups + n_days + (n_hours if hourly_known else 0) + n_hours + 2 * n_hours
    lower = np.full(n_rows, -np.inf)
    upper = np.full(n_rows, np.inf)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    row = 0

    for k, amount in enumerate(margins):
        for h in range(n_hours):
            rows.append(row); cols.append(h * n_groups + k); vals.append(1.)
        lower[row] = upper[row] = float(amount)
        row += 1

    for day, amount in enumerate(daily):
        for h in range(day * 24, (day + 1) * 24):
            for k in range(n_groups):
                rows.append(row); cols.append(h * n_groups + k); vals.append(1.)
        lower[row] = upper[row] = float(amount)
        row += 1

    if hourly_known:
        for h, amount in enumerate(hourly):
            for k in range(n_groups):
                rows.append(row); cols.append(h * n_groups + k); vals.append(1.)
            lower[row] = upper[row] = float(amount)
            row += 1

    for h in range(n_hours):
        for k in range(n_groups):
            rows.append(row); cols.append(h * n_groups + k); vals.append(1.)
        lower[row] = min_hour
        row += 1

    for h in range(n_hours):
        cap = float(hourly[h]) if hourly_known else float(daily[h // 24])
        e_hi_min, e_lo_max = float(a_hi.min() * cap), float(a_lo.max() * cap)
        z = x_count + h
        for k in range(n_groups):
            rows.append(row); cols.append(h * n_groups + k); vals.append(float(a_hi[k]))
        rows.append(row); cols.append(z); vals.append(e_hi_min)
        lower[row] = e_hi_min
        row += 1
        for k in range(n_groups):
            rows.append(row); cols.append(h * n_groups + k); vals.append(float(a_lo[k]))
        rows.append(row); cols.append(z); vals.append(-e_lo_max)
        upper[row] = 0.
        row += 1

    if row != n_rows:
        raise RuntimeError(f"constraint row mismatch: {row} != {n_rows}")
    matrix = coo_matrix((vals, (rows, cols)), shape=(n_rows, n_vars)).tocsr()
    constraints = LinearConstraint(matrix, lower, upper)
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[x_count:] = 1
    bounds = Bounds(np.zeros(n_vars), np.concatenate((np.full(x_count, np.inf), np.ones(n_hours))))
    answers = []
    for endpoint in ("min", "max"):
        objective = np.zeros(n_vars)
        objective[x_count:] = 1. if endpoint == "min" else -1.
        started = time_module.perf_counter()
        result = milp(objective, integrality=integrality, bounds=bounds, constraints=constraints,
                      options={"time_limit": TIME_LIMIT_SECONDS, "mip_rel_gap": 0.})
        elapsed = time_module.perf_counter() - started
        count = int(np.rint(result.x[x_count:].sum())) if result.x is not None else None
        answers.append({"endpoint": endpoint, "bound": count, "status": int(result.status),
                        "message": str(result.message), "optimal": bool(result.status == 0 and result.x is not None),
                        "mip_gap": float(getattr(result, "mip_gap", np.nan)), "solve_seconds": elapsed})
    return answers


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    run_started = time_module.perf_counter()
    protocol_hash = sha256(PROTOCOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)
    records = []
    for region, months in BA_MONTHS.items():
        for month in months:
            key = (region, YEAR, month)
            if key not in series:
                raise RuntimeError(f"frozen EIA query absent: {key}")
            d = series[key]
            time = pd.DatetimeIndex(d["time"])
            fuel_totals = np.asarray(d["fuel_totals"], float)
            fuel = np.asarray(d["fuel_profile"], float) * fuel_totals[None, :]
            expected_hours = pd.Period(f"{YEAR}-{month:02d}").days_in_month * 24
            if len(time) != expected_hours or not time.is_monotonic_increasing or time.has_duplicates:
                raise RuntimeError(f"incomplete/nonmonotone EIA hours at {key}")
            hourly = fuel.sum(axis=1)
            daily = hourly.reshape(-1, 24).sum(axis=1)
            margins = group_margins(fuel_totals)
            actual = np.asarray(d["cif"], float)
            if np.max(np.abs(fuel @ partial.EF / hourly - actual)) > 1e-7:
                raise RuntimeError(f"fixed EF witness reconstruction mismatch at {key}")
            observed = int(np.sum(actual >= TAU + 1e-4))
            nominal_mean = float(GROUP_EF @ margins / margins.sum())
            mean_low, mean_high = nominal_mean * (1. - EF_TOL), nominal_mean * (1. + EF_TOL)
            threshold_straddled = mean_low < TAU + 1e-4 and mean_high >= TAU + 1e-4
            if threshold_straddled:
                allocation = hourly[:, None] * (margins / margins.sum())[None, :]
                if (np.max(np.abs(allocation.sum(axis=0) - margins)) > 1e-6
                        or np.max(np.abs(allocation.sum(axis=1) - hourly)) > 1e-8):
                    raise RuntimeError(f"threshold-side proportional witness failed at {key}")
            for regime, hourly_known in (("daily_total_only", False), ("hourly_total", True)):
                if threshold_straddled:
                    solved = [{"endpoint": endpoint, "bound": (0 if endpoint == "min" else len(time)),
                               "status": -1, "message": "analytic threshold-side full-range certificate",
                               "optimal": True, "mip_gap": 0.0, "solve_seconds": 0.0,
                               "solver_method": "analytic_threshold_side"}
                              for endpoint in ("min", "max")]
                else:
                    solved = solve_uncertain(hourly, daily, margins, hourly_known)
                    for endpoint in solved:
                        endpoint["solver_method"] = "MILP"
                for endpoint in solved:
                    bound = endpoint["bound"]
                    witness_ok = bound is not None and (
                        bound <= observed if endpoint["endpoint"] == "min" else bound >= observed)
                    records.append({"region": region, "year": YEAR, "month": month, "tau": TAU,
                                    "regime": regime, "endpoint": endpoint["endpoint"],
                                    "bound": bound, "optimal": endpoint["optimal"],
                                    "status": endpoint["status"], "mip_gap": endpoint["mip_gap"],
                                    "message": endpoint["message"], "observed_nominal_fixed_ef_tail_count": observed,
                                    "solver_method": endpoint["solver_method"],
                                    "threshold_straddled": threshold_straddled,
                                    "monthly_mean_low": mean_low, "monthly_mean_high": mean_high,
                                    "solve_seconds": endpoint["solve_seconds"],
                                    "nominal_witness_in_uncertainty_outer_set": witness_ok})
    endpoints = pd.DataFrame(records)
    cases = endpoints.groupby(["region", "month", "tau", "regime"], as_index=False).agg(
        min_bound=("bound", "first"), max_bound=("bound", "last"), both_optimal=("optimal", "all"),
        observed_tail_count=("observed_nominal_fixed_ef_tail_count", "first"),
        witness_included=("nominal_witness_in_uncertainty_outer_set", "all"))
    cases["identified_width"] = cases.max_bound - cases.min_bound
    pivot = cases.pivot(index=["region", "month", "tau"], columns="regime",
                        values=["min_bound", "max_bound", "both_optimal"])
    complete = pivot[("both_optimal", "daily_total_only")] & pivot[("both_optimal", "hourly_total")]
    for idx in pivot.index[complete]:
        d_lo = pivot.loc[idx, ("min_bound", "daily_total_only")]
        d_hi = pivot.loc[idx, ("max_bound", "daily_total_only")]
        h_lo = pivot.loc[idx, ("min_bound", "hourly_total")]
        h_hi = pivot.loc[idx, ("max_bound", "hourly_total")]
        if h_lo < d_lo or h_hi > d_hi:
            raise RuntimeError(f"nested uncertainty-set invariant failed at {idx}")
    if not endpoints.nominal_witness_in_uncertainty_outer_set.all():
        raise RuntimeError("nominal fixed-EF observed witness escaped the EF uncertainty outer set")
    baseline = pd.read_csv(BASELINE)
    comparison = cases.merge(baseline, on=["region", "month", "tau", "regime"],
                             suffixes=("_screened", "_full"), validate="one_to_one")
    exact_match = bool((comparison.min_bound_screened == comparison.min_bound_full).all()
                       and (comparison.max_bound_screened == comparison.max_bound_full).all())
    if not exact_match:
        raise RuntimeError("threshold-side screening disagrees with full V240 MILP endpoints")
    OUT.mkdir(parents=True)
    endpoints.to_csv(OUT / "milp_endpoints.csv", index=False)
    cases.to_csv(OUT / "risk_count_bounds.csv", index=False)
    metadata = {"status": "hour_varying_ef_uncertainty_outer_set_not_probability_forecast",
                "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
                "input_hashes": input_hashes, "scipy_version": scipy.__version__,
                "regions": list(BA_MONTHS), "ba_months": {k: list(v) for k, v in BA_MONTHS.items()},
                "year": YEAR, "threshold": TAU, "ef_tolerance": EF_TOL,
                "ef_group_nominal": GROUP_EF.tolist(), "n_ba_months": int(len(BA_MONTHS["AZPS"]) * 5 + len(BA_MONTHS["PSCO"])),
                "endpoint_count": int(len(endpoints)), "optimal_endpoints": int(endpoints.optimal.sum()),
                "optimal_fraction": float(endpoints.optimal.mean()), "complete_paired_cases": int(complete.sum()),
                "all_nominal_witnesses_in_outer_set": bool(endpoints.nominal_witness_in_uncertainty_outer_set.all()),
                "threshold_straddled_cases": int(endpoints.loc[endpoints.solver_method == "analytic_threshold_side", ["region", "month"]].drop_duplicates().shape[0]),
                "analytic_milp_endpoint_skips": int((endpoints.solver_method == "analytic_threshold_side").sum()),
                "milp_endpoints_called": int((endpoints.solver_method == "MILP").sum()),
                "milp_calls": int((endpoints.solver_method == "MILP").sum()),
                "exact_endpoint_match_to_full_v240": exact_match,
                "mean_milp_solve_seconds": float(endpoints.loc[endpoints.solver_method == "MILP", "solve_seconds"].mean()),
                "total_elapsed_seconds": float(time_module.perf_counter() - run_started),
                "time_limit_seconds_per_endpoint": TIME_LIMIT_SECONDS,
                "caveat": "EF varies independently by hour/group inside ±10% box, a conservative outer set; no direct hourly CO2 label"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(cases.to_string(index=False))
    print(f"optimal={metadata['optimal_endpoints']}/{metadata['endpoint_count']} paired={metadata['complete_paired_cases']} skips={metadata['analytic_milp_endpoint_skips']} exact={metadata['exact_endpoint_match_to_full_v240']}")


if __name__ == "__main__":
    main()
