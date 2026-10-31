"""V243: repeat the 600-threshold MILP with monthly-energy normalized variables."""
from __future__ import annotations

import hashlib
import json
import sys
import time
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
import v237_nested_daily_hourly_risk as nested  # noqa: E402

PROTOCOL = ROOT / "docs/V243_SCALED_MILP_THRESHOLD600_PROTOCOL.md"
OUT = ROOT / "results/v243-scaled-threshold600-r1"
BA_MONTHS = {
    "AZPS": (4, 7, 10), "CPLE": (4, 7, 10), "ERCO": (4, 7, 10),
    "PACE": (4, 7, 10), "SOCO": (4, 7, 10), "PSCO": (4,),
}
YEAR = 2025
TAUS = (600.,)
GROUP_EF = np.array([1000., 800., 400., 0.])


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def group_margins(margins: np.ndarray, ef: np.ndarray) -> np.ndarray:
    return np.array([margins[ef == g].sum() for g in GROUP_EF], dtype=float)


def solve_bounds(hourly_proxy: np.ndarray, daily_totals: np.ndarray, margins: np.ndarray,
                 ef: np.ndarray, tau: float, hourly_known: bool) -> list[dict]:
    """Monthly-energy-scaled V237 MILP; positive scaling preserves its feasible set."""
    month_energy = float(np.sum(margins))
    hourly_proxy = np.asarray(hourly_proxy, float) / month_energy
    daily_totals = np.asarray(daily_totals, float) / month_energy
    margins = np.asarray(margins, float) / month_energy
    n_hours, n_fuels = len(hourly_proxy), len(margins)
    n_days = len(daily_totals)
    x_count = n_hours * n_fuels
    n_vars = x_count + n_hours
    aef = ef - (tau + nested.THRESHOLD_MARGIN)
    min_hour = float(margins.sum()) / n_hours * 1e-6
    total_rows = n_fuels + n_days + (n_hours if hourly_known else 0) + n_hours + 2 * n_hours
    row_lower = np.full(total_rows, -np.inf)
    row_upper = np.full(total_rows, np.inf)
    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    row = 0
    for k, amount in enumerate(margins):
        for h in range(n_hours):
            rows.append(row); cols.append(h * n_fuels + k); vals.append(1.)
        row_lower[row] = row_upper[row] = float(amount)
        row += 1
    for day, amount in enumerate(daily_totals):
        for h in range(day * 24, (day + 1) * 24):
            for k in range(n_fuels):
                rows.append(row); cols.append(h * n_fuels + k); vals.append(1.)
        row_lower[row] = row_upper[row] = float(amount)
        row += 1
    if hourly_known:
        for h, amount in enumerate(hourly_proxy):
            for k in range(n_fuels):
                rows.append(row); cols.append(h * n_fuels + k); vals.append(1.)
            row_lower[row] = row_upper[row] = float(amount)
            row += 1
    for h in range(n_hours):
        for k in range(n_fuels):
            rows.append(row); cols.append(h * n_fuels + k); vals.append(1.)
        row_lower[row] = min_hour
        row += 1
    for h in range(n_hours):
        cap = float(hourly_proxy[h]) if hourly_known else float(daily_totals[h // 24])
        s_min, s_max = float(aef.min() * cap), float(aef.max() * cap)
        z = x_count + h
        for k in range(n_fuels):
            rows.append(row); cols.append(h * n_fuels + k); vals.append(float(aef[k]))
        rows.append(row); cols.append(z); vals.append(s_min)
        row_lower[row] = s_min
        row += 1
        for k in range(n_fuels):
            rows.append(row); cols.append(h * n_fuels + k); vals.append(float(aef[k]))
        rows.append(row); cols.append(z); vals.append(-s_max)
        row_upper[row] = 0.
        row += 1
    if row != total_rows:
        raise RuntimeError(f"constraint row mismatch: {row} != {total_rows}")
    matrix = coo_matrix((vals, (rows, cols)), shape=(total_rows, n_vars)).tocsr()
    constraints = LinearConstraint(matrix, row_lower, row_upper)
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[x_count:] = 1
    bounds = Bounds(np.zeros(n_vars), np.concatenate((np.full(x_count, np.inf), np.ones(n_hours))))
    solved = []
    for endpoint in ("min", "max"):
        objective = np.zeros(n_vars)
        objective[x_count:] = 1. if endpoint == "min" else -1.
        start_time = time.perf_counter()
        result = milp(objective, integrality=integrality, bounds=bounds, constraints=constraints,
                      options={"time_limit": nested.TIME_LIMIT_SECONDS, "mip_rel_gap": 0.})
        elapsed = time.perf_counter() - start_time
        count = int(np.rint(result.x[x_count:].sum())) if result.x is not None else None
        gap = getattr(result, "mip_gap", np.nan)
        solved.append({"endpoint": endpoint, "bound": count, "status": int(result.status),
                       "message": str(result.message), "optimal": bool(result.status == 0 and result.x is not None),
                       "mip_gap": float(gap) if gap is not None else np.nan,
                       "solve_seconds": elapsed})
    return solved


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)
    rows = []
    for region, months in BA_MONTHS.items():
        for month in months:
            key = (region, YEAR, month)
            if key not in series:
                raise RuntimeError(f"frozen EIA query absent: {key}")
            d = series[key]
            time = pd.DatetimeIndex(d["time"])
            fuel_totals = np.asarray(d["fuel_totals"], float)
            fuel_profile = np.asarray(d["fuel_profile"], float)
            fuel = fuel_profile * fuel_totals[None, :]
            n_hours = len(time)
            expected_hours = pd.Period(f"{YEAR}-{month:02d}").days_in_month * 24
            if n_hours != expected_hours or not time.is_monotonic_increasing or time.has_duplicates:
                raise RuntimeError(f"incomplete/nonmonotone EIA hours at {key}: {n_hours} vs {expected_hours}")
            diffs = np.diff(time.asi8)
            if len(diffs) and not np.all(diffs == diffs[0]):
                raise RuntimeError(f"nonuniform timestamp cadence at {key}")
            hourly_total = fuel.sum(axis=1)
            margins = fuel_totals.copy()
            if np.any(~np.isfinite(fuel)) or np.any(fuel < -1e-9):
                raise RuntimeError(f"nonfinite or negative generation at {key}")
            present = margins > 1e-12
            if not np.allclose(fuel_profile[:, present].sum(axis=0), 1., rtol=0, atol=1e-8):
                raise RuntimeError(f"monthly-normalized fuel profiles do not sum to one at {key}")
            if not np.allclose(fuel.sum(axis=0), margins, rtol=1e-10, atol=1e-6):
                raise RuntimeError(f"hourly fuel values do not reproduce monthly margins at {key}")
            if not np.isclose(hourly_total.sum(), margins.sum(), rtol=1e-10, atol=1e-6):
                raise RuntimeError(f"hourly totals do not conserve monthly generation at {key}")
            grouped = group_margins(margins, partial.EF)
            daily = hourly_total.reshape(-1, 24).sum(axis=1)
            actual = np.asarray(d["cif"], float)
            reconstructed = fuel @ partial.EF / hourly_total
            reconstruction_error = float(np.max(np.abs(actual - reconstructed)))
            if reconstruction_error > 1e-7:
                raise RuntimeError(f"EIA CIF witness reconstruction mismatch {key}: {reconstruction_error}")

            for tau in TAUS:
                month_bounds = nested.monthly_only_bounds(grouped, GROUP_EF, tau + nested.THRESHOLD_MARGIN, n_hours)
                for regime, hourly_known in (("daily_total_only", False), ("hourly_total", True)):
                    solved = solve_bounds(hourly_total, daily, grouped, GROUP_EF, tau, hourly_known)
                    observed_count = int(np.sum(actual >= tau + nested.THRESHOLD_MARGIN))
                    for endpoint in solved:
                        rows.append({
                            "region": region, "year": YEAR, "month": month, "tau": tau,
                            "regime": regime, "n_hours": n_hours,
                            "monthly_min": month_bounds[0], "monthly_max": month_bounds[1],
                            "monthly_width": month_bounds[1] - month_bounds[0],
                            "endpoint": endpoint["endpoint"], "bound": endpoint["bound"],
                            "status": endpoint["status"], "message": endpoint["message"],
                            "optimal": endpoint["optimal"], "mip_gap": endpoint["mip_gap"],
                            "solve_seconds": endpoint["solve_seconds"],
                            "observed_fixed_ef_tail_count": observed_count,
                            "witness_contained_in_primal_bounds": (None if not endpoint["optimal"] else
                                (endpoint["bound"] <= observed_count if endpoint["endpoint"] == "min" else endpoint["bound"] >= observed_count)),
                            "max_fixed_ef_cif_reconstruction_error": reconstruction_error,
                            "max_hourly_total_mwh": float(hourly_total.max()),
                        })
    endpoints = pd.DataFrame(rows)
    cases = endpoints.groupby(["region", "month", "tau", "regime"], as_index=False).agg(
        min_bound=("bound", "first"), max_bound=("bound", "last"), both_optimal=("optimal", "all"),
        monthly_min=("monthly_min", "first"), monthly_max=("monthly_max", "first"),
        monthly_width=("monthly_width", "first"), observed_tail_count=("observed_fixed_ef_tail_count", "first"),
        witness_included=("witness_contained_in_primal_bounds", lambda s: bool(s.dropna().all()) if s.notna().any() else None))
    cases["identified_width"] = cases.max_bound - cases.min_bound
    cases["width_reduction"] = cases.monthly_width - cases.identified_width
    pivot = cases.pivot(index=["region", "month", "tau"], columns="regime",
                        values=["min_bound", "max_bound", "both_optimal", "witness_included"])
    daily_proved = pivot[("both_optimal", "daily_total_only")]
    complete = daily_proved & pivot[("both_optimal", "hourly_total")]
    monthly_bounds = cases.drop_duplicates(["region", "month", "tau"]).set_index(["region", "month", "tau"])
    for idx in pivot.index[daily_proved]:
        d_lo, d_hi = pivot.loc[idx, ("min_bound", "daily_total_only")], pivot.loc[idx, ("max_bound", "daily_total_only")]
        m_lo, m_hi = monthly_bounds.loc[idx, "monthly_min"], monthly_bounds.loc[idx, "monthly_max"]
        if d_lo < m_lo or d_hi > m_hi:
            raise RuntimeError(f"monthly/daily nesting invariant failed for {idx}")
    for idx in pivot.index[complete]:
        d_lo, d_hi = pivot.loc[idx, ("min_bound", "daily_total_only")], pivot.loc[idx, ("max_bound", "daily_total_only")]
        h_lo, h_hi = pivot.loc[idx, ("min_bound", "hourly_total")], pivot.loc[idx, ("max_bound", "hourly_total")]
        if h_lo < d_lo or h_hi > d_hi:
            raise RuntimeError(f"nested information invariant failed for {idx}")
    proven_witness_checks = endpoints.loc[endpoints.optimal, "witness_contained_in_primal_bounds"]
    if not proven_witness_checks.all():
        raise RuntimeError("actual fixed-EF fuel schedule escaped a feasible-set primal endpoint")

    OUT.mkdir(parents=True)
    endpoints.to_csv(OUT / "milp_endpoints.csv", index=False)
    cases.to_csv(OUT / "risk_count_bounds.csv", index=False)
    meta = {"status": "EIA_generation_geometry_replication_not_measured_emissions_validation",
            "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
            "input_hashes": input_hashes, "scipy_version": scipy.__version__,
            "regions": list(BA_MONTHS), "ba_months": {k: list(v) for k, v in BA_MONTHS.items()}, "year": YEAR,
            "thresholds": list(TAUS), "group_ef": GROUP_EF.tolist(),
            "n_ba_months": int(cases[["region", "month"]].drop_duplicates().shape[0]),
            "endpoint_count": int(len(endpoints)), "optimal_endpoints": int(endpoints.optimal.sum()),
            "optimal_fraction": float(endpoints.optimal.mean()), "complete_nested_cases": int(complete.sum()),
            "candidate_nested_cases": int(len(complete)),
            "all_optimal_endpoints_contain_actual_witness": bool(proven_witness_checks.all()),
            "max_cif_reconstruction_error": float(endpoints.max_fixed_ef_cif_reconstruction_error.max()),
            "variable_scaling": "all generation amounts divided by each BA-month total; objective, constraints, and feasible set otherwise identical to V242",
            "caveat": "EIA hourly generation is actual activity data but fixed-EF CIF remains a derived proxy; already-opened V226 regions; 2025 Apr/Jul/Oct; threshold 600 numerical scaling comparison"}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(cases.to_string(index=False))
    print(f"optimal={meta['optimal_endpoints']}/{meta['endpoint_count']} nested={meta['complete_nested_cases']}/{meta['candidate_nested_cases']} witness={meta['all_optimal_endpoints_contain_actual_witness']}")


if __name__ == "__main__":
    main()
