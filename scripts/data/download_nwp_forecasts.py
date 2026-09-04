#!/usr/bin/env python
"""Download multi-model historical NWP forecasts (FD-43, curl-based).

Open-Meteo historical-forecast API serves archived GFS/ICON(/ECMWF-2024+)
model outputs keyed by valid time — the deployment-real day-ahead weather
that WAS available, unlike the ERA5 reanalysis proxy.  We keep two
independent models for 2023 and derive inter-model disagreement
(predictability signal) rather than replacing the proxy:

    data_2023/nwp/{REGION}_nwp_2023_hourly.csv
        hour, gfs_wind100, icon_wind100, gfs_temp, icon_temp

Usage:
    .venv/bin/python scripts/data/download_nwp_forecasts.py [--regions A B]
"""

import argparse
import subprocess
import time
from pathlib import Path

DATA = Path("data_2023")
OUT = DATA / "nwp"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODELS = "gfs_seamless,icon_seamless"


def fetch(lat, lon, year):
    q = (f"latitude={lat}&longitude={lon}"
         f"&start_date={year}-01-01&end_date={year}-12-31"
         f"&hourly=wind_speed_100m,temperature_2m&models={MODELS}"
         f"&wind_speed_unit=ms&timeformat=unixtime")
    r = subprocess.run(["curl", "-s", "--max-time", "120", f"{URL}?{q}"],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise RuntimeError(f"curl failed: {r.stderr[:120]}")
    return r.stdout


def main():
    from transcif.config.region_meta import REGION_META
    from transcif.data.loaders import all_region_configs

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--year", type=int, default=2023)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    import json
    cfgs = all_region_configs()
    names = args.regions or [n for n in cfgs if n in REGION_META]
    for i, name in enumerate(names):
        out = OUT / f"{name}_nwp_{args.year}_hourly.csv"
        if out.exists():
            print(f"[nwp] {name}: exists, skip")
            continue
        lat, lon, _ = REGION_META[name]
        try:
            doc = json.loads(fetch(lat, lon, args.year))
            h = doc["hourly"]
            import pandas as pd
            df = pd.DataFrame({
                "hour": pd.to_datetime(h["time"], unit="s"),
                "gfs_wind100": h["wind_speed_100m_gfs_seamless"],
                "icon_wind100": h["wind_speed_100m_icon_seamless"],
                "gfs_temp": h["temperature_2m_gfs_seamless"],
                "icon_temp": h["temperature_2m_icon_seamless"],
            }).drop_duplicates("hour").sort_values("hour")
            df.to_csv(out, index=False)
            print(f"[nwp] {i + 1}/{len(names)} {name}: {len(df)} hours "
                  f"-> {out.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[nwp] {name}: FAILED {e}")
        time.sleep(1.0)


if __name__ == "__main__":
    main()
