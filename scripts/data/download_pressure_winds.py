#!/usr/bin/env python
"""Download pressure-level wind + gust tracks (roadmap B-class, #3).

Open-Meteo ERA5 archive (keyless): hourly ``wind_gusts_10m`` and
``pressure_msl`` at each region's demand-weighted coordinate for 2023
(pressure-level winds are NOT carried by the archive API — probed and
returned all-null).  Gusts proxy ramp events; MSL pressure anomalies
carry the synoptic frontal signal that drives multi-hour wind regime
shifts.

Writes ``data_2023/weather2/{REGION}_wind2_2023_hourly.csv`` per region.

Usage:
    .venv/bin/python scripts/data/download_pressure_winds.py [--regions A B]
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
OUT = DATA / "weather2"
URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region(name, lat, lon):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": "2023-01-01", "end_date": "2023-12-31",
        "hourly": "wind_gusts_10m,pressure_msl",
        "timezone": "UTC",
    })
    with urllib.request.urlopen(f"{URL}?{q}", timeout=120) as r:
        doc = json.load(r)
    hourly = doc["hourly"]
    rows = zip(hourly["time"], hourly["wind_gusts_10m"],
               hourly["pressure_msl"])
    out = OUT / f"{name}_wind2_2023_hourly.csv"
    with open(out, "w") as f:
        f.write("hour,wind_gusts_10m,pressure_msl\n")
        for t, w950, gust in rows:
            f.write(f"{t},{w950},{gust}\n")
    return out


def main():
    from transcif.config.region_meta import REGION_META
    from transcif.data.loaders import all_region_configs

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=None)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    cfgs = all_region_configs()
    names = args.regions or [n for n in cfgs if n in REGION_META]
    for i, name in enumerate(names):
        out = OUT / f"{name}_wind2_2023_hourly.csv"
        if out.exists():
            print(f"[wind2] {name}: exists, skip")
            continue
        lat, lon, _ = REGION_META[name]
        try:
            p = fetch_region(name, lat, lon)
            print(f"[wind2] {i + 1}/{len(names)} {name}: wrote {p.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[wind2] {name}: FAILED ({e})")
        time.sleep(1.2)


if __name__ == "__main__":
    main()
