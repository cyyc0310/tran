"""V281 exploratory audit: compare EIA target BA factors with transferred CEMS support.

This is post-hoc diagnostics on already opened PACE/LGEE workbooks, not a
confirmatory experiment. Factors are workbook-reported lbs CO2/kWh converted
to g/kWh and compared with V269's quarter-specific CEMS intervals.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/v280-eia-official-label-pace-lgee"
SUPPORT = ROOT / "results/v269-asymmetric-empirical-ef-support-r2/cems_asymmetric_support_intervals.csv"
OUT = ROOT / "results/v281-eia-target-ef-support-audit"
TARGETS = ("PACE", "LGEE")
FACTORS = {
    "coal": ("CO2 Factor: COL", "NG: COL"),
    "gas": ("CO2 Factor: NG", "NG: NG"),
    "oil": ("CO2 Factor: OIL", "NG: OIL"),
}
LB_KWH_TO_G_KWH = 453.59237


def target_rows(ba: str) -> list[dict]:
    path = DATA / f"{ba}.xlsx"
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]
    rows = ws.iter_rows(values_only=True)
    headers = next(rows)
    ix = {str(value): i for i, value in enumerate(headers) if value is not None}
    required = ["Local date", *(col for pair in FACTORS.values() for col in pair)]
    missing = [col for col in required if col not in ix]
    if missing:
        raise ValueError(f"{ba}: missing factor audit columns {missing}")
    result = []
    for row in rows:
        day = row[ix["Local date"]]
        if not hasattr(day, "year") or day.year != 2025:
            continue
        quarter = (day.month - 1) // 3 + 1
        for fuel, (factor_col, generation_col) in FACTORS.items():
            try:
                factor = float(row[ix[factor_col]]) * LB_KWH_TO_G_KWH
            except (TypeError, ValueError):
                continue
            try:
                generation = float(row[ix[generation_col]] or 0.)
            except (TypeError, ValueError):
                generation = 0.
            result.append({"ba": ba, "quarter": quarter, "fuel": fuel,
                           "factor_g_kwh": factor, "active": generation > 0})
    wb.close()
    return result


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    support = pd.read_csv(SUPPORT).set_index(["quarter", "fuel_group"])
    raw = pd.DataFrame([r for ba in TARGETS for r in target_rows(ba)])
    summaries = []
    for (ba, quarter, fuel), group in raw.groupby(["ba", "quarter", "fuel"], sort=True):
        active = group[group.active]
        support_row = support.loc[(quarter, fuel)]
        low = float(support_row.support_low_g_kwh)
        high = float(support_row.support_high_g_kwh)
        values = active.factor_g_kwh.to_numpy(float)
        summaries.append({
            "ba": ba, "quarter": quarter, "fuel": fuel,
            "support_low_g_kwh": low, "support_high_g_kwh": high,
            "all_hour_factor_median_g_kwh": float(group.factor_g_kwh.median()),
            "active_hours": int(len(active)),
            "active_factor_min_g_kwh": float(np.min(values)) if len(values) else np.nan,
            "active_factor_median_g_kwh": float(np.median(values)) if len(values) else np.nan,
            "active_factor_max_g_kwh": float(np.max(values)) if len(values) else np.nan,
            "active_below_fraction": float(np.mean(values < low)) if len(values) else np.nan,
            "active_inside_fraction": float(np.mean((values >= low) & (values <= high))) if len(values) else np.nan,
            "active_above_fraction": float(np.mean(values > high)) if len(values) else np.nan,
        })
    OUT.mkdir(parents=True)
    frame = pd.DataFrame(summaries)
    frame.to_csv(OUT / "quarterly_target_factor_support.csv", index=False)
    pooled_rows = []
    for (ba, fuel), group in frame.groupby(["ba", "fuel"], sort=True):
        active = group[group.active_hours > 0]
        outside = active.active_below_fraction + active.active_above_fraction
        pooled_rows.append({
            "ba": ba, "fuel": fuel,
            "active_hours": int(group.active_hours.sum()),
            "active_factor_min_g_kwh": float(group.active_factor_min_g_kwh.min()),
            "active_factor_max_g_kwh": float(group.active_factor_max_g_kwh.max()),
            "outside_active_hour_fraction": float(np.average(
                outside, weights=active.active_hours)) if len(active) else np.nan,
        })
    pooled = pd.DataFrame(pooled_rows)
    pooled.to_csv(OUT / "pooled_target_factor_support.csv", index=False)
    print(frame.to_string(index=False))
    print("\nPooled active-hour summary")
    print(pooled.to_string(index=False))


if __name__ == "__main__":
    main()
