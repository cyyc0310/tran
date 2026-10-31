"""V262: robust tail-count identification under noisy hourly-total observations."""
from __future__ import annotations

import hashlib
import json
import sys
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

PROTOCOL = ROOT / "docs/V262_HOURLY_TOTAL_MEASUREMENT_ERROR_PROTOCOL.md"
OUT = ROOT / "results/v262-hourly-total-measurement-error-r1"
BA_MONTHS = {"PACE": (4, 7, 10), "CPLE": (7,), "ERCO": (10,), "AZPS": (4,)}
YEAR = 2025
TAU = 500.0
EPSILONS = (0.05, 0.10)
GROUP_EF = np.array([1000., 800., 400., 0.])
TIME_LIMIT_SECONDS = 20.0


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def group_margins(margins: np.ndarray) -> np.ndarray:
    return np.array([margins[partial.EF == group].sum() for group in GROUP_EF], float)


def solve_uncertain_hourly_bounds(observed: np.ndarray, daily: np.ndarray,
                                  margins: np.ndarray, epsilon: float) -> list[dict]:
    """Exact MILP over fuel allocation and uncertain but bounded hourly totals."""
    n_hours, n_fuels = len(observed), len(margins)
    n_days = len(daily)
    x_count, n_vars = n_hours * n_fuels, n_hours * n_fuels + n_hours
    lo_total = observed * (1. - epsilon)
    hi_total = observed * (1. + epsilon)
    threshold = TAU + nested.THRESHOLD_MARGIN
    excess_ef = GROUP_EF - threshold
    min_hour = float(margins.sum()) / n_hours * 1e-6

    # Rows: fuel margins; daily totals; uncertain hourly totals; positive hourly
    # denominators; two standard big-M implications for each threshold event.
    n_rows = n_fuels + n_days + n_hours + n_hours + 2 * n_hours
    row_lo = np.full(n_rows, -np.inf)
    row_hi = np.full(n_rows, np.inf)
    rr: list[int] = []
    cc: list[int] = []
    vv: list[float] = []
    row = 0

    for fuel_id, amount in enumerate(margins):
        for hour in range(n_hours):
            rr.append(row); cc.append(hour * n_fuels + fuel_id); vv.append(1.)
        row_lo[row] = row_hi[row] = float(amount)
        row += 1

    for day, amount in enumerate(daily):
        for hour in range(day * 24, (day + 1) * 24):
            for fuel_id in range(n_fuels):
                rr.append(row); cc.append(hour * n_fuels + fuel_id); vv.append(1.)
        row_lo[row] = row_hi[row] = float(amount)
        row += 1

    for hour in range(n_hours):
        for fuel_id in range(n_fuels):
            rr.append(row); cc.append(hour * n_fuels + fuel_id); vv.append(1.)
        row_lo[row], row_hi[row] = float(lo_total[hour]), float(hi_total[hour])
        row += 1

    for hour in range(n_hours):
        for fuel_id in range(n_fuels):
            rr.append(row); cc.append(hour * n_fuels + fuel_id); vv.append(1.)
        row_lo[row] = min_hour
        row += 1

    for hour in range(n_hours):
        cap = float(hi_total[hour])
        s_min, s_max = float(excess_ef.min() * cap), float(excess_ef.max() * cap)
        z = x_count + hour
        for fuel_id in range(n_fuels):
            rr.append(row); cc.append(hour * n_fuels + fuel_id); vv.append(float(excess_ef[fuel_id]))
        rr.append(row); cc.append(z); vv.append(s_min)
        row_lo[row] = s_min
        row += 1
        for fuel_id in range(n_fuels):
            rr.append(row); cc.append(hour * n_fuels + fuel_id); vv.append(float(excess_ef[fuel_id]))
        rr.append(row); cc.append(z); vv.append(-s_max)
        row_hi[row] = 0.
        row += 1
    if row != n_rows:
        raise RuntimeError(f"constraint row mismatch {row} != {n_rows}")

    matrix = coo_matrix((vv, (rr, cc)), shape=(n_rows, n_vars)).tocsr()
    constraints = LinearConstraint(matrix, row_lo, row_hi)
    integrality = np.zeros(n_vars, dtype=np.uint8)
    integrality[x_count:] = 1
    variable_bounds = Bounds(np.zeros(n_vars), np.concatenate((np.full(x_count, np.inf), np.ones(n_hours))))
    solved = []
    for endpoint in ("min", "max"):
        objective = np.zeros(n_vars)
        objective[x_count:] = 1. if endpoint == "min" else -1.
        result = milp(objective, integrality=integrality, bounds=variable_bounds,
                      constraints=constraints,
                      options={"time_limit": TIME_LIMIT_SECONDS, "mip_rel_gap": 0.})
        count = int(np.rint(result.x[x_count:].sum())) if result.x is not None else None
        solved.append({"endpoint": endpoint, "bound": count, "status": int(result.status),
                       "message": str(result.message), "optimal": bool(result.status == 0 and result.x is not None),
                       "mip_gap": float(getattr(result, "mip_gap", np.nan))})
    return solved


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    protocol_hash = sha256(PROTOCOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        native, schemas, input_hashes, helpers = confirm.load_native()
        series = confirm.build_series(native, schemas, helpers)

    endpoint_rows = []
    for region, months in BA_MONTHS.items():
        for month in months:
            key = (region, YEAR, month)
            if key not in series:
                raise RuntimeError(f"frozen EIA query absent: {key}")
            entry = series[key]
            time = pd.DatetimeIndex(entry["time"])
            fuel_totals = np.asarray(entry["fuel_totals"], float)
            fuel = np.asarray(entry["fuel_profile"], float) * fuel_totals[None, :]
            hourly_total = fuel.sum(axis=1)
            daily = hourly_total.reshape(-1, 24).sum(axis=1)
            margins = group_margins(fuel_totals)
            actual_cif = np.asarray(entry["cif"], float)
            observed_tail = int(np.sum(actual_cif >= TAU + nested.THRESHOLD_MARGIN))
            month_bounds = nested.monthly_only_bounds(margins, GROUP_EF,
                                                       TAU + nested.THRESHOLD_MARGIN, len(time))
            if not np.isclose(hourly_total.sum(), margins.sum(), rtol=1e-10, atol=1e-6):
                raise RuntimeError(f"monthly total conservation failed for {key}")

            for epsilon in EPSILONS:
                # Alternating signed perturbation is deterministic, bounded by
                # epsilon/2, and deliberately not selected using CIF outcomes.
                sign = np.where(np.arange(len(hourly_total)) % 2 == 0, 1., -1.)
                observed = hourly_total * (1. + sign * epsilon / 2.)
                if np.any(hourly_total < observed * (1. - epsilon) - 1e-9) or np.any(
                        hourly_total > observed * (1. + epsilon) + 1e-9):
                    raise RuntimeError(f"true total outside frozen error box for {key}, epsilon={epsilon}")
                solutions = solve_uncertain_hourly_bounds(observed, daily, margins, epsilon)
                for endpoint in solutions:
                    endpoint_rows.append({
                        "region": region, "year": YEAR, "month": month, "tau": TAU,
                        "epsilon": epsilon, "endpoint": endpoint["endpoint"], "bound": endpoint["bound"],
                        "status": endpoint["status"], "message": endpoint["message"],
                        "optimal": endpoint["optimal"], "mip_gap": endpoint["mip_gap"],
                        "monthly_min": month_bounds[0], "monthly_max": month_bounds[1],
                        "observed_fixed_ef_tail_count": observed_tail,
                        "true_hourly_total_in_measurement_box": True,
                        "daily_totals_exact": True,
                    })
                print(f"solved {key} epsilon={epsilon:.0%} "
                      f"optimal={all(row['optimal'] for row in solutions)}", flush=True)

    endpoints = pd.DataFrame(endpoint_rows)
    cases = endpoints.groupby(["region", "year", "month", "tau", "epsilon"], as_index=False).agg(
        min_bound=("bound", "first"), max_bound=("bound", "last"), both_optimal=("optimal", "all"),
        monthly_min=("monthly_min", "first"), monthly_max=("monthly_max", "first"),
        observed_tail_count=("observed_fixed_ef_tail_count", "first"),
        true_profile_inside=("true_hourly_total_in_measurement_box", "all"))
    cases["monthly_width"] = cases.monthly_max - cases.monthly_min
    cases["identified_width"] = cases.max_bound - cases.min_bound
    cases["width_reduction_vs_month"] = cases.monthly_width - cases.identified_width
    cases["witness_included"] = ((cases.min_bound <= cases.observed_tail_count)
                                 & (cases.max_bound >= cases.observed_tail_count))

    # Exact daily-only reference is deterministic from the already-frozen V239 output.
    v239_path = ROOT / "results/v239-eia-seasonal-nested-risk-r1/risk_count_bounds.csv"
    daily_ref = pd.read_csv(v239_path)
    daily_ref = daily_ref[(daily_ref.regime == "daily_total_only") & (daily_ref.tau == TAU)]
    if "year" not in daily_ref.columns:
        daily_ref["year"] = YEAR
    daily_ref = daily_ref[["region", "year", "month", "min_bound", "max_bound", "both_optimal"]].rename(
        columns={"min_bound": "daily_min", "max_bound": "daily_max", "both_optimal": "daily_both_optimal"})
    cases = cases.merge(daily_ref, on=["region", "year", "month"], how="left", validate="many_to_one")
    cases["daily_width"] = cases.daily_max - cases.daily_min
    cases["width_reduction_vs_daily"] = cases.daily_width - cases.identified_width
    cases["relative_reduction_vs_daily"] = cases.width_reduction_vs_daily / cases.daily_width.replace(0, np.nan)
    cases["nested_in_daily"] = ((cases.min_bound >= cases.daily_min) & (cases.max_bound <= cases.daily_max))

    proved_cases = cases[cases.both_optimal].copy()
    if not proved_cases.true_profile_inside.all() or not proved_cases.witness_included.all():
        raise RuntimeError("observed true profile or its fixed-EF tail count escaped robust set")
    OUT.mkdir(parents=True)
    endpoints.to_csv(OUT / "milp_endpoints.csv", index=False)
    cases.to_csv(OUT / "robust_risk_count_bounds.csv", index=False)
    summary = cases.groupby("epsilon", as_index=False).agg(
        candidates=("region", "size"), complete=("both_optimal", "sum"))
    proved_summary = proved_cases.groupby("epsilon", as_index=False).agg(
        daily_mean_width=("daily_width", "mean"), robust_mean_width=("identified_width", "mean"),
        daily_to_robust_mean_shrink=("width_reduction_vs_daily", "mean"),
        daily_to_robust_relative_shrink=("relative_reduction_vs_daily", "mean"),
        witness_rate=("witness_included", "mean"), nested_rate=("nested_in_daily", "mean"))
    summary = summary.merge(proved_summary, on="epsilon", how="left", validate="one_to_one")
    summary.to_csv(OUT / "summary.csv", index=False)
    metadata = {
        "status": "noisy_hourly_total_identification_sensitivity_not_forecast_validation",
        "protocol_sha256": protocol_hash, "script_sha256": sha256(Path(__file__)),
        "input_hashes": input_hashes, "v239_reference_sha256": sha256(v239_path),
        "scipy_version": scipy.__version__, "regions": list(BA_MONTHS),
        "ba_months": {key: list(value) for key, value in BA_MONTHS.items()},
        "year": YEAR, "threshold": TAU, "epsilons": list(EPSILONS),
        "endpoint_count": int(len(endpoints)), "optimal_endpoint_count": int(endpoints.optimal.sum()),
        "optimal_fraction": float(endpoints.optimal.mean()),
        "all_proved_witnesses_inside": bool(proved_cases.witness_included.all()),
        "all_proved_robust_sets_nested_in_daily": bool(proved_cases.nested_in_daily.all()),
        "warning": "EIA activity totals are real, but CIF is fixed-EF proxy; already-opened V239 cohort; not Chinese measurement validation"
    }
    (OUT / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    if sha256(PROTOCOL) != protocol_hash:
        raise RuntimeError("protocol changed during run")
    print(summary.to_string(index=False))
    print(f"optimal={metadata['optimal_endpoint_count']}/{metadata['endpoint_count']} "
          f"witness={metadata['all_proved_witnesses_inside']} "
          f"nested={metadata['all_proved_robust_sets_nested_in_daily']}")


if __name__ == "__main__":
    main()
