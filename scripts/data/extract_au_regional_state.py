#!/usr/bin/env python
"""Extract public AEMO/NEMED regional state signals for TransCIF.

This writes causal, timestamped system-state features from the public NEM
dispatch stream.  It deliberately does not use target CIF labels.  The
current release contains sent-out generation and emissions-weighted output;
future demand forecasts can be joined later when an AEMO forecast feed is
available.

Usage (NEMED's pinned Python environment):
    .venv-nemed311/bin/python scripts/data/extract_au_regional_state.py \
        --year 2023 --regions NSW1 VIC1
"""
import argparse
from pathlib import Path

import pandas as pd

import nemed.downloader as dl
import nemed.process as proc

CACHE = ".nemed_cache"
OUT = Path("data_2023/state")
ASOF_DATE = "2023/12/01 00:00"
REGIONS = ["QLD1", "NSW1", "VIC1", "SA1", "TAS1"]

_dds = dl.download_dudetailsummary
_gm = dl.download_genset_map
proc.download_dudetailsummary = lambda cache, asof_date=None: _dds(
    cache, asof_date=ASOF_DATE)
proc.download_genset_map = lambda cache, asof_date=None: _gm(
    cache, asof_date=ASOF_DATE)


def extract_region(region, year):
    frames = []
    for month in range(1, 13):
        ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
        start = f"{year}/{month:02d}/01 00:00"
        end = f"{ny}/{nm:02d}/01 00:00"
        print(f"[{region}] {start} .. {end}", flush=True)
        raw = proc.get_total_emissions_by_DI_DUID(
            start, end, CACHE, filter_regions=[region],
            generation_sent_out=True, return_all=True)
        frames.append(raw[["Time", "Energy_SO", "Total_Emissions"]])
    raw = pd.concat(frames, ignore_index=True)
    raw["hour"] = raw["Time"].dt.floor("h")
    out = raw.groupby("hour").agg(
        generation_sent_out_mw=("Energy_SO", "sum"),
        emissions_tco2=("Total_Emissions", "sum"),
    ).reset_index()
    out.insert(0, "Region", region)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for region in args.regions:
        path = OUT / f"{region}_state_{args.year}_hourly.csv"
        if path.exists():
            print(f"[skip] {path}")
            continue
        df = extract_region(region, args.year)
        df.to_csv(path, index=False)
        print(f"[write] {path}: {len(df)} rows", flush=True)


if __name__ == "__main__":
    main()
