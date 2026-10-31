"""Sharp feasibility bounds for hourly CIF means under monthly fuel margins."""

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

TARGETS = data.TARGET_REGIONS
N_FUELS = len(data.FUELS)
EF_RANGE = float(np.max(data.EF) - np.min(data.EF))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def margin_only_bounds(margins: np.ndarray, hours: int) -> tuple[float, float, float]:
    active = margins > max(1e-10, float(margins.sum()) * 1e-12)
    ef = data.EF[active]
    mass = margins[active]
    overall = float(ef @ mass / mass.sum())
    lo, hi = float(ef.min()), float(ef.max())
    if np.isclose(lo, hi):
        return lo, hi, overall
    below = ef < hi
    mean_below = float(ef[below] @ mass[below] / mass[below].sum())
    lower = ((hours - 1) * lo + overall) / hours
    upper = ((hours - 1) * hi + mean_below) / hours
    return lower, upper, overall


def hourly_total_lp(margins: np.ndarray, hourly_totals: np.ndarray) -> tuple[float, float, float, int, int]:
    n_hours = len(hourly_totals)
    n_vars = n_hours * N_FUELS
    rows, cols, values = [], [], []
    for hour in range(n_hours):
        base = hour * N_FUELS
        for fuel in range(N_FUELS):
            variable = base + fuel
            rows.append(hour)
            cols.append(variable)
            values.append(1.0)
            rows.append(n_hours + fuel)
            cols.append(variable)
            values.append(1.0)
    a_eq = coo_matrix((values, (rows, cols)), shape=(n_hours + N_FUELS, n_vars)).tocsr()
    b_eq = np.concatenate([hourly_totals, margins])
    cost = np.tile(data.EF, n_hours) / np.repeat(hourly_totals, N_FUELS) / n_hours
    low = linprog(cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    high = linprog(-cost, A_eq=a_eq, b_eq=b_eq, bounds=(0, None), method="highs")
    if not low.success or not high.success:
        raise RuntimeError(f"LP failed: low={low.message}; high={high.message}")
    return float(low.fun), float(-high.fun), float(np.mean(hourly_totals)), int(low.status), int(high.status)


def main() -> None:
    protocol = ROOT / "docs/V58_MONTHLY_MARGINAL_IDENTIFIABILITY_PROTOCOL.md"
    protocol_hash = sha256(protocol)
    hourly, _, _, input_hashes = data.prepare_regions()
    rows = []
    for region in TARGETS:
        frame = hourly[region].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[frame.time.dt.year.eq(2026) & frame.time.dt.month.le(6) & frame.valid]
        fuels = [f"generation_{fuel}" for fuel in data.FUELS]
        for month_no in range(1, 7):
            month = frame.loc[frame.time.dt.month.eq(month_no)]
            if len(month) < 480:
                continue
            values = month[fuels].to_numpy(float)
            valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
            values = values[valid]
            totals = values.sum(axis=1)
            keep = totals > 0
            values, totals = values[keep], totals[keep]
            margins = values.sum(axis=0)
            lower_margin, upper_margin, energy_mean = margin_only_bounds(margins, len(totals))
            lower_hour, upper_hour, _, low_status, high_status = hourly_total_lp(margins, totals)
            hourly_intensity = values @ data.EF / totals
            actual_mean = float(hourly_intensity.mean())
            tolerance = 1e-6
            if not (lower_margin - tolerance <= actual_mean <= upper_margin + tolerance):
                raise ValueError(f"Analytical margin-only bound misses actual {region}/{month_no}")
            if not (lower_hour - tolerance <= actual_mean <= upper_hour + tolerance):
                raise ValueError(f"Hourly-total LP bound misses actual {region}/{month_no}")
            rows.append({"region": region, "month": f"2026-{month_no:02d}", "valid_hours": len(totals),
                         "active_fuels": int(np.count_nonzero(margins > 1e-9)),
                         "energy_weighted_cif_exact": energy_mean, "actual_equal_hour_mean": actual_mean,
                         "margin_only_lower": lower_margin, "margin_only_upper": upper_margin,
                         "margin_only_width": upper_margin - lower_margin,
                         "margin_width_over_ef_range": (upper_margin - lower_margin) / EF_RANGE,
                         "hourly_total_lp_lower": lower_hour, "hourly_total_lp_upper": upper_hour,
                         "hourly_total_lp_width": upper_hour - lower_hour,
                         "hourly_lp_width_over_ef_range": (upper_hour - lower_hour) / EF_RANGE,
                         "hourly_lp_low_status": low_status, "hourly_lp_high_status": high_status})

    detail = pd.DataFrame(rows)
    if detail.empty:
        raise ValueError("No eligible target region-months")
    output = ROOT / "results/v58-monthly-marginal-identifiability-r2"
    output.mkdir(parents=True, exist_ok=False)
    detail.to_csv(output / "region_month_bounds.csv", index=False)
    summary = (detail.groupby("region").agg(months=("month", "count"),
                 exact_energy_weighted_cif=("energy_weighted_cif_exact", "mean"),
                 actual_equal_hour_mean=("actual_equal_hour_mean", "mean"),
                 margin_only_width=("margin_only_width", "mean"),
                 margin_width_over_ef_range=("margin_width_over_ef_range", "mean"),
                 hourly_total_lp_width=("hourly_total_lp_width", "mean"),
                 hourly_lp_width_over_ef_range=("hourly_lp_width_over_ef_range", "mean"))
                 .reset_index())
    summary.to_csv(output / "region_summary.csv", index=False)
    overall = detail.agg({"margin_only_width": "mean", "margin_width_over_ef_range": "mean",
                          "hourly_total_lp_width": "mean", "hourly_lp_width_over_ef_range": "mean",
                          "energy_weighted_cif_exact": "mean", "actual_equal_hour_mean": "mean"}).to_dict()
    script = Path(__file__).resolve()
    payload = {"status": "POSTHOC_IDENTIFIABILITY_ANALYSIS_NOT_PREDICTION",
               "regions": list(TARGETS), "period": "2026 H1", "region_months": len(detail),
               "ef_range": EF_RANGE, "lp_successful_for_every_row": bool(
                   detail.hourly_lp_low_status.eq(0).all() and detail.hourly_lp_high_status.eq(0).all()),
               "actual_means_inside_both_bounds": True, "script_sha256": sha256(script),
               "protocol_sha256": protocol_hash, "raw_input_sha256": input_hashes,
               "overall_mean_metrics": overall,
               "limitations": ["Hourly-total LP uses target hourly total as an oracle, not a deployed feature.",
                              "Fixed-factor EIA proxy is not full-grid measured CIF.",
                              "Margin-only bounds assume no per-hour capacity or ramp constraints."]}
    if sha256(protocol) != protocol_hash:
        raise RuntimeError("Frozen identifiability protocol changed during run")
    (output / "summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload["overall_mean_metrics"], indent=2))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
