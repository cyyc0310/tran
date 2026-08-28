#!/usr/bin/env python
"""Extract day-ahead load forecasts + net interchange from raw EIA-930.

The raw files carry ``Demand Forecast (MW)`` (day-ahead), ``Demand (MW)``
and ``Total Interchange (MW)`` (net imports, positive = importing) —
the last is the physical carbon-flow channel for import-heavy BAs
(ISNE's Quebec hydro imports are ~15-20 % of energy and absent from the
rs telemetry definition).

Writes ``data_2023/demand/{REGION}_demand_{year}_hourly.csv``.

Usage:
    .venv/bin/python scripts/data/extract_eia_demand.py
"""

from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
RAW = DATA / "raw_eia930"
OUT = DATA / "demand"

BA_MAP = {
    "CISO": "US_CISO", "PJM": "US_PJM", "MISO": "US_MISO",
    "ERCO": "US_ERCO", "ISNE": "US_ISNE", "NYIS": "US_NYIS",
    "FPL": "US_FPL", "BPAT": "US_BPAT",
}
COLS = ["Balancing Authority", "UTC Time at End of Hour",
        "Demand Forecast (MW)", "Demand (MW)", "Total Interchange (MW)"]
RENAME = {"UTC Time at End of Hour": "hour",
          "Demand Forecast (MW)": "demand_forecast_mw",
          "Demand (MW)": "demand_actual_mw",
          "Total Interchange (MW)": "net_interchange_mw"}


def main():
    OUT.mkdir(exist_ok=True)
    year_files = {}
    for f in sorted(RAW.glob("gen_*.csv")):
        yr = "".join(c for c in f.stem if c.isdigit())
        year_files.setdefault(yr, []).append(f)
    for ba, region in BA_MAP.items():
        for yr, files in sorted(year_files.items()):
            out = OUT / f"{region}_demand_{yr}_hourly.csv"
            rows = []
            for f in files:
                df = pd.read_csv(f, usecols=COLS, parse_dates=[COLS[1]])
                rows.append(df[df["Balancing Authority"] == ba])
            fy = pd.concat(rows, ignore_index=True).rename(columns=RENAME)
            g = (fy.groupby("hour")[list(RENAME.values())[1:]]
                 .mean().sort_index())
            # interval-ENDING -> beginning (matches the main pipeline)
            g.index = g.index - pd.Timedelta(hours=1)
            g = g[~g.index.duplicated(keep="first")]
            g.to_csv(out)
            print(f"[demand] {region} {yr}: {len(g)} hours -> {out.name}")


if __name__ == "__main__":
    main()
