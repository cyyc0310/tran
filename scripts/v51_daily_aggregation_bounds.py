"""Sharp daily-mean CIF bounds from monthly fuel inventories and hourly total."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from identifiability_v15 import EF, FUELS, functional_bounds, read_evaluation

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/research/data"
OUT = ROOT / "results/v51-daily-aggregation-bounds"
REGIONS = ("BPAT", "CISO", "ERCO", "FPL", "ISNE", "MISO", "NYIS")
YEARS = (2022, 2023, 2024)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if any(OUT.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty result directory: {OUT}")
    records = []
    inputs = {}
    for region in REGIONS:
        path = DATA / f"{region}_hourly.csv"
        inputs[str(path)] = sha256(path)
        for year in YEARS:
            days, generation = read_evaluation(path, region, year)
            totals = generation.sum(axis=1)
            truth_hourly = generation @ EF / totals
            day_keys = days.tz_localize(None).to_period("M")
            for month in day_keys.unique():
                selected_days = np.flatnonzero(day_keys == month)
                rows = np.concatenate([np.arange(d * 24, (d + 1) * 24) for d in selected_days])
                month_generation = generation[rows]
                month_total = month_generation.sum(axis=0)
                month_rows = month_generation.sum(axis=1)
                for local_day, day_index in enumerate(selected_days):
                    local_hours = np.arange(local_day * 24, (local_day + 1) * 24)
                    coefficients = np.zeros(len(rows))
                    coefficients[local_hours] = 1. / (24. * month_rows[local_hours])
                    lower, upper = functional_bounds(month_rows, month_total, EF, coefficients)
                    global_rows = rows[local_hours]
                    observed = float(truth_hourly[global_rows].mean())
                    if lower > observed + 1e-8 or observed > upper + 1e-8:
                        raise AssertionError("Observed daily mean outside sharp bounds")
                    records.append({
                        "region": region, "year": year, "month": str(month),
                        "day": str(days[day_index].date()), "hours": 24,
                        "truth_daily_mean": observed, "lower": lower, "upper": upper,
                        "width": upper - lower,
                        "within_day_hourly_sd": float(truth_hourly[global_rows].std()),
                        "monthly_weighted_mean": float(month_total @ EF / month_total.sum()),
                    })
    frame = pd.DataFrame(records)
    frame.to_csv(OUT / "daily_bounds.csv", index=False)
    by_source = frame.groupby("region", as_index=False).agg(
        days=("width", "size"), mean_width=("width", "mean"),
        median_width=("width", "median"), p90_width=("width", lambda x: x.quantile(.9)),
        actual_daily_mean_sd=("truth_daily_mean", "std"),
        within_day_hourly_sd=("within_day_hourly_sd", "mean"),
        truth_inside=("truth_daily_mean", lambda x: 1.0),
    )
    by_source.to_csv(OUT / "source_summary.csv", index=False)
    overall = frame.groupby("year", as_index=False).agg(
        days=("width", "size"), mean_width=("width", "mean"),
        median_width=("width", "median"), p90_width=("width", lambda x: x.quantile(.9)),
        actual_daily_mean_sd=("truth_daily_mean", "std"),
        within_day_hourly_sd=("within_day_hourly_sd", "mean"),
    )
    overall.to_csv(OUT / "year_summary.csv", index=False)
    pooled = {
        "days": len(frame), "sources": int(frame.region.nunique()),
        "mean_width": float(frame.width.mean()), "median_width": float(frame.width.median()),
        "p90_width": float(frame.width.quantile(.9)),
        "actual_daily_mean_sd": float(frame.truth_daily_mean.std()),
        "mean_within_day_hourly_sd": float(frame.within_day_hourly_sd.mean()),
        "truth_inside_fraction": 1.0,
    }
    outputs = ("daily_bounds.csv", "source_summary.csv", "year_summary.csv")
    summary = {
        "status": "RETROSPECTIVE_DAILY_MEAN_SHARP_BOUNDS",
        "contract": "Exact target-month fuel totals and exact hourly total generation; optimize each day's arithmetic mean hourly CIF over all monthly fuel allocations.",
        "support": "Seven already inspected sources, H1 2022-2024 complete valid UTC days only.",
        "input_sha256": inputs,
        "outputs_sha256": {name: sha256(OUT / name) for name in outputs},
        "pooled": pooled,
        "interpretation": "Sharp physical bounds under monthly fuel margins and observed hourly total; target-month margins are retrospective oracle data, not forecast-time information. Not a prediction interval or coverage guarantee.",
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(pooled, indent=2))
    print(by_source.to_string(index=False))


if __name__ == "__main__":
    main()
