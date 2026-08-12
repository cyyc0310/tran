"""Extract per-fuel generation mix for UK regions from the Carbon Intensity API.

The legacy ``download_uk_regions.py`` collapses the API's 9-fuel generationmix
into a single renew/nonrenew split.  This script re-queries the same API and
keeps the per-fuel percentages so that the multi-fuel config vector (Stage A)
can be built for UK regions too.

Outputs (does NOT overwrite the existing ``UK_*_2023_hourly.csv``):
    data_2023/fuel/UK_<id>_<name>_fuel_2023_hourly.csv — augmented hourly CSV
    data_2023/fuel/fuel_shares_uk.json                 — annual mean fuel-share
                                                          vectors (train-split
                                                          aware)

The API serves 30-min intervals; we aggregate to hourly (mean of 2 half-hours)
to match the legacy schema.

Usage:
    PYTHONPATH=. python scripts/data/extract_uk_fuel_breakdown.py
"""

import json
import os
import time
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_2023"
FUEL_DIR = DATA_DIR / "fuel"
FUEL_DIR.mkdir(exist_ok=True)
TRAIN_FRACTION = 0.8
BASE_URL = "https://api.carbonintensity.org.uk"

# All fuels returned by the UK Carbon Intensity API generationmix.
# Order is fixed so the share vector is consistent across regions.
FUEL_KEYS = ["coal", "gas", "biomass", "nuclear", "hydro", "imports", "other", "solar", "wind"]

# Lifecycle emission factors (gCO2/kWh) — UK-specific where they differ from US.
# imports ≈ the grid-average of interconnected neighbours (FR/IE/NL/BE), ~250.
# biomass treated as ~0 (sustainable harvest assumption, matches DECC convention).
FUEL_EMISSION_FACTORS = {
    "coal": 980.0, "gas": 410.0, "biomass": 0.0, "nuclear": 0.0,
    "hydro": 0.0, "imports": 250.0, "other": 500.0, "solar": 0.0, "wind": 0.0,
}
# UK convention: solar/wind/hydro/biomass/nuclear are "renewable/low-carbon".
RENEWABLE_FUELS = {"solar", "wind", "hydro", "biomass", "nuclear"}

UK_REGIONS = [
    {"id": 1, "name": "North Scotland"}, {"id": 2, "name": "South Scotland"},
    {"id": 3, "name": "North West England"}, {"id": 4, "name": "North East England"},
    {"id": 5, "name": "Yorkshire"}, {"id": 6, "name": "North Wales Merseyside"},
    {"id": 7, "name": "South Wales"}, {"id": 8, "name": "West Midlands"},
    {"id": 9, "name": "East Midlands"}, {"id": 10, "name": "East England"},
    {"id": 11, "name": "South West England"}, {"id": 12, "name": "South England"},
    {"id": 13, "name": "London"}, {"id": 14, "name": "South East England"},
    {"id": 15, "name": "England"}, {"id": 16, "name": "Scotland"},
    {"id": 17, "name": "Wales"}, {"id": 18, "name": "GB"},
]


def fetch_chunk(region_id, start, end):
    s = start.strftime("%Y-%m-%dT%H:%MZ")
    e = end.strftime("%Y-%m-%dT%H:%MZ")
    url = f"{BASE_URL}/regional/intensity/{s}/{e}/regionid/{region_id}"
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        return json.load(resp)["data"]["data"]
    except Exception as ex:
        print(f"    WARN: region {region_id} {s}→{e}: {ex}")
        return []


def parse_intervals(intervals):
    """Keep all 9 fuel percentages + CIF per 30-min interval."""
    rows = []
    for iv in intervals:
        t = datetime.fromisoformat(iv["from"].replace("Z", "+00:00"))
        intensity = iv["intensity"].get("forecast")
        mix = {g["fuel"]: g["perc"] for g in iv.get("generationmix", [])}
        row = {"time": t.replace(tzinfo=None), "cif_gco2_per_kwh": intensity}
        for f in FUEL_KEYS:
            row[f"perc_{f}"] = mix.get(f, 0.0)
        rows.append(row)
    return rows


def download_region(region, year=2023):
    rid, rname = region["id"], region["name"]
    safe = rname.replace(" ", "_")
    out_path = FUEL_DIR / f"UK_{rid:02d}_{safe}_fuel_2023_hourly.csv"
    if out_path.exists():
        print(f"  [skip] {out_path.name} exists")
        return out_path

    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    all_rows, cursor = [], start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=14), end)
        intervals = fetch_chunk(rid, cursor, chunk_end)
        all_rows.extend(parse_intervals(intervals))
        cursor = chunk_end
        time.sleep(0.3)

    if not all_rows:
        print(f"  [EMPTY] region {rid} ({rname})")
        return None

    df = pd.DataFrame(all_rows)
    df["hour"] = df["time"].dt.floor("h")
    perc_cols = [f"perc_{f}" for f in FUEL_KEYS]
    agg = {"cif_gco2_per_kwh": ("cif_gco2_per_kwh", "mean")}
    for c in perc_cols:
        agg[c] = (c, "mean")
    hourly = df.groupby("hour").agg(**agg).reset_index()

    hourly.insert(0, "Region", f"UK_{rid:02d}")
    hourly["cif_real_gco2_per_kwh"] = hourly["cif_gco2_per_kwh"]
    hourly["cif_real_tco2_per_mwh"] = hourly["cif_gco2_per_kwh"] / 1000.0
    hourly.to_csv(out_path, index=False)
    print(f"  [OK] {out_path.name}: {len(hourly)} hours, "
          f"mean CIF={hourly['cif_gco2_per_kwh'].mean():.1f}")
    return out_path


def compute_fuel_shares(df, train_fraction=TRAIN_FRACTION):
    """Annual mean fuel-share vector over the training split."""
    split = int(len(df) * train_fraction)
    perc_cols = [f"perc_{f}" for f in FUEL_KEYS]
    mean_perc = df[perc_cols].iloc[:split].mean(axis=0).values  # (9,)
    total = mean_perc.sum()
    if total <= 0:
        return {f: 0.0 for f in FUEL_KEYS}
    return {f: float(mean_perc[i] / total) for i, f in enumerate(FUEL_KEYS)}


def main():
    print("=" * 70)
    print("UK Carbon Intensity Fuel Breakdown Extraction (Stage A.2)")
    print("=" * 70)

    fuel_shares_all = {}
    for r in UK_REGIONS:
        print(f"\n[{r['id']:02d}] {r['name']}...")
        path = download_region(r)
        if path is None:
            continue
        df = pd.read_csv(path, parse_dates=["hour"])
        shares = compute_fuel_shares(df)
        fuel_shares_all[f"UK_{r['id']:02d}_{r['name'].replace(' ', '_')}"] = shares
        mix = "  ".join(f"{k}:{v*100:4.1f}%" for k, v in shares.items() if v > 0.01)
        print(f"    fuel mix (train split): {mix}")

    out = FUEL_DIR / "fuel_shares_uk.json"
    with open(out, "w") as f:
        json.dump({
            "_emission_factors": FUEL_EMISSION_FACTORS,
            "_renewable_fuels": sorted(RENEWABLE_FUELS),
            "_fuel_order": FUEL_KEYS,
            "regions": fuel_shares_all,
        }, f, indent=2)
    print(f"\n[WRITE] {out}")
    print(f"Done. {len(fuel_shares_all)} UK regions processed.")


if __name__ == "__main__":
    main()
