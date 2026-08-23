#!/usr/bin/env python
"""Extract day-ahead load forecasts from raw EIA-930 (FD-15, E/C-class).

The raw files carry ``Demand Forecast (MW)`` — the balancing authority's
own DAY-AHEAD forecast — plus the actual.  The forecast is deployment-
perfect for day-ahead CIF prediction (it is what utilities publish /
plan on), and load shape is the primary driver of the thermal-residual
component (duck-curve evening ramp, coal/gas dispatch).

Writes ``data_2023/demand/{REGION}_demand_2023_hourly.csv`` with columns
``hour`` (UTC), ``demand_forecast_mw``, ``demand_actual_mw``.

Usage:
    .venv/bin/python scripts/data/extract_eia_demand.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
RAW = DATA / "raw_eia930"
OUT = DATA / "demand"

# EIA balancing-authority code -> TransCIF region name.
BA_MAP = {
    "CISO": "US_CISO", "PJM": "US_PJM", "MISO": "US_MISO",
    "ERCO": "US_ERCO", "ISNE": "US_ISNE", "NYIS": "US_NYIS",
    "FPL": "US_FPL", "BPAT": "US_BPAT",
}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    frames = []
    for f in sorted(RAW.glob("gen_*.csv")):
        print(f"[demand] reading {f.name} ...")
        df = pd.read_csv(
            f, usecols=["Balancing Authority", "UTC Time at End of Hour",
                        "Demand Forecast (MW)", "Demand (MW)"],
            parse_dates=["UTC Time at End of Hour"])
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    full = full.rename(columns={
        "UTC Time at End of Hour": "hour",
        "Demand Forecast (MW)": "demand_forecast_mw",
        "Demand (MW)": "demand_actual_mw"})
    for ba, region in BA_MAP.items():
        sub = full[full["Balancing Authority"] == ba]
        g = (sub.groupby("hour")[["demand_forecast_mw", "demand_actual_mw"]]
             .mean().sort_index())
        # Interval-ENDING labels -> shift to interval-beginning to match
        # the main pipeline's convention (verified: main CSVs label the
        # hour the energy belongs to).
        g.index = g.index - pd.Timedelta(hours=1)
        g = g[~g.index.duplicated(keep="first")]
        out = OUT / f"{region}_demand_2023_hourly.csv"
        g.to_csv(out)
        print(f"[demand] {region}: {len(g)} hours -> {out.name}")


if __name__ == "__main__":
    main()
