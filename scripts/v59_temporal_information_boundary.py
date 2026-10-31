"""Oracle information-budget bounds for monthly equal-hour CIF means.

This is a posthoc identifiability sensitivity analysis, not a forecast.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as data  # noqa: E402
import v58_monthly_marginal_identifiability as v58  # noqa: E402

TARGETS = data.TARGET_REGIONS
N_FUELS = len(data.FUELS)
THERMAL_INDEX = np.array([data.FUELS.index(fuel) for fuel in data.THERMAL], dtype=int)
SAMPLE_HOURS = {"1_per_day_12": (12,), "2_per_day_06_18": (6, 18),
                "4_per_day_00_06_12_18": (0, 6, 12, 18)}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def lp_bounds(hourly_totals: np.ndarray, ef: np.ndarray, *, monthly_margins: np.ndarray | None = None,
              daily_margins: np.ndarray | None = None, day_index: np.ndarray | None = None,
              thermal_samples: dict[int, float] | None = None) -> tuple[float, float, int, int]:
    """Sharp LP bounds given aggregate fuel totals and selected hourly summaries."""
    n_hours = len(hourly_totals)
    n_vars = n_hours * len(ef)
    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    rhs: list[float] = []
    row = 0
    for hour in range(n_hours):
        for fuel in range(len(ef)):
            rows.append(row); cols.append(hour * len(ef) + fuel); values.append(1.0)
        rhs.append(float(hourly_totals[hour])); row += 1
    if monthly_margins is not None:
        for fuel, margin in enumerate(monthly_margins):
            for hour in range(n_hours):
                rows.append(row); cols.append(hour * len(ef) + fuel); values.append(1.0)
            rhs.append(float(margin)); row += 1
    if daily_margins is not None:
        assert day_index is not None
        for day in range(len(daily_margins)):
            hours = np.flatnonzero(day_index == day)
            for fuel in range(len(ef)):
                for hour in hours:
                    rows.append(row); cols.append(int(hour) * len(ef) + fuel); values.append(1.0)
                rhs.append(float(daily_margins[day, fuel])); row += 1
    for hour, thermal_mwh in (thermal_samples or {}).items():
        for fuel in THERMAL_INDEX:
            rows.append(row); cols.append(hour * len(ef) + int(fuel)); values.append(1.0)
        rhs.append(float(thermal_mwh)); row += 1
    a_eq = coo_matrix((values, (rows, cols)), shape=(row, n_vars)).tocsr()
    b_eq = np.asarray(rhs)
    cost = np.tile(ef, n_hours) / np.repeat(hourly_totals, len(ef)) / n_hours
    low = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    high = linprog(-cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not low.success or not high.success:
        raise RuntimeError(f"LP failed: low={low.message}; high={high.message}")
    return float(low.fun), float(-high.fun), int(low.status), int(high.status)


def weighted_daily_margin_bounds(day_values: np.ndarray, day_index: np.ndarray) -> tuple[float, float]:
    lows, highs, weights = [], [], []
    for day in range(int(day_index.max()) + 1):
        selected = day_values[day_index == day]
        if not len(selected):
            continue
        margins = selected.sum(axis=0)
        lo, hi, _ = v58.margin_only_bounds(margins, len(selected))
        lows.append(lo * len(selected)); highs.append(hi * len(selected)); weights.append(len(selected))
    return float(sum(lows) / sum(weights)), float(sum(highs) / sum(weights))


def main() -> None:
    protocol = ROOT / "docs/V59_TEMPORAL_INFORMATION_BOUNDARY_PROTOCOL.md"
    protocol_hash = sha256(protocol)
    hourly, _, _, input_hashes = data.prepare_regions()
    rows: list[dict] = []
    fuels = [f"generation_{fuel}" for fuel in data.FUELS]
    for region in TARGETS:
        frame = hourly[region].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[frame.time.dt.year.eq(2026) & frame.time.dt.month.le(6) & frame.valid].copy()
        for month_no in range(1, 7):
            month = frame.loc[frame.time.dt.month.eq(month_no)].sort_values("time").copy()
            if len(month) < 480:
                continue
            values = month[fuels].to_numpy(float)
            valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
            month = month.loc[valid].copy()
            values = values[valid]
            totals = values.sum(axis=1)
            keep = totals > 0
            month, values, totals = month.loc[keep].copy(), values[keep], totals[keep]
            timestamps = pd.to_datetime(month.time, utc=True)
            day_keys = timestamps.dt.strftime("%Y-%m-%d").to_numpy()
            unique_days, day_index = np.unique(day_keys, return_inverse=True)
            monthly_margins = values.sum(axis=0)
            daily_margins = np.vstack([values[day_index == day].sum(axis=0)
                                       for day in range(len(unique_days))])
            actual = float(np.mean(values @ data.EF / totals))
            month_lo, month_hi, _ = v58.margin_only_bounds(monthly_margins, len(totals))
            day_lo, day_hi = weighted_daily_margin_bounds(values, day_index)
            scenarios: list[tuple[str, float, float, int, int]] = [
                ("monthly_margins_only", month_lo, month_hi, 0, 0),
                ("daily_margins_only", day_lo, day_hi, 0, 0),
            ]
            lo, hi, ls, hs = lp_bounds(totals, data.EF, monthly_margins=monthly_margins)
            scenarios.append(("monthly_margins_plus_hourly_total", lo, hi, ls, hs))
            lo, hi, ls, hs = lp_bounds(totals, data.EF, daily_margins=daily_margins,
                                       day_index=day_index)
            scenarios.append(("daily_margins_plus_hourly_total", lo, hi, ls, hs))
            time_hours = timestamps.dt.hour.to_numpy()
            for name, hours in SAMPLE_HOURS.items():
                sample_idx = np.flatnonzero(np.isin(time_hours, hours))
                thermal = values[:, THERMAL_INDEX].sum(axis=1)
                sample_map = {int(i): float(thermal[i]) for i in sample_idx}
                lo, hi, ls, hs = lp_bounds(totals, data.EF, monthly_margins=monthly_margins,
                                           thermal_samples=sample_map)
                scenarios.append((f"monthly_margins_hourly_total_{name}", lo, hi, ls, hs))
            for name, lo, hi, ls, hs in scenarios:
                tol = 1e-6
                if not lo - tol <= actual <= hi + tol:
                    raise ValueError(f"bound misses observed mean for {region}/{month_no}/{name}: {lo} {actual} {hi}")
                rows.append({"region": region, "month": f"2026-{month_no:02d}",
                             "valid_hours": len(totals), "days": len(unique_days), "scenario": name,
                             "actual_equal_hour_mean": actual, "lower": lo, "upper": hi,
                             "width": hi - lo, "width_over_ef_range": (hi - lo) / v58.EF_RANGE,
                             "low_status": ls, "high_status": hs})

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No eligible region-months")
    output = ROOT / "results/v59-temporal-information-boundary-r1"
    output.mkdir(parents=True, exist_ok=False)
    detail.to_csv(output / "region_month_scenarios.csv", index=False)
    region_summary = (detail.groupby(["scenario", "region"]).agg(months=("month", "count"),
                     mean_width=("width", "mean"), mean_width_over_ef_range=("width_over_ef_range", "mean"),
                     mean_lower=("lower", "mean"), mean_upper=("upper", "mean")).reset_index())
    region_summary.to_csv(output / "region_summary.csv", index=False)
    overall = (detail.groupby("scenario").agg(region_months=("month", "count"), mean_width=("width", "mean"),
              mean_width_over_ef_range=("width_over_ef_range", "mean"), worst_width=("width", "max"),
              all_actual_means_inside=("actual_equal_hour_mean", "count"))
              .reset_index())
    overall.to_csv(output / "scenario_summary.csv", index=False)
    script = Path(__file__).resolve()
    payload = {"status": "POSTHOC_ORACLE_IDENTIFIABILITY_NOT_PREDICTION",
               "regions": list(TARGETS), "period": "2026 H1", "region_months": int(detail.month.nunique() * len(TARGETS)),
               "scenario_count": int(detail.scenario.nunique()), "all_bounds_contain_actual": True,
               "all_lp_status_optimal": bool(detail.loc[detail.low_status.ne(0) | detail.high_status.ne(0)].empty),
               "script_sha256": sha256(script), "protocol_sha256": protocol_hash,
               "raw_input_sha256": input_hashes,
               "limitations": ["All target-month margins and hourly totals in these oracle sensitivities are posthoc truth.",
                               "The sampled thermal summaries are target-local measurements and are not claimed public in China.",
                               "Bounds concern monthly equal-hour mean under fixed-EF proxy, not hourly CIF tail coverage."]}
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Frozen protocol changed during run")
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(overall.to_string(index=False))
    print(region_summary.to_string(index=False))


if __name__ == "__main__":
    main()
