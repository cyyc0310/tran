#!/usr/bin/env python
"""Download ERA5 precipitation for extreme-weather attribution analysis.

Open-Meteo ERA5 archive (keyless): hourly ``precipitation`` (rain +
showers, mm) at each region's coordinate for 2023.  Heavy-rain hours are
the direct "大暴雨" channel; combined with gusts + MSL pressure
(weather2) they separate convective rain from frontal/cyclone systems.

Writes ``data_2023/weather3/{REGION}_precip_2023_hourly.csv`` per region.

Usage:
    .venv/bin/python scripts/data/download_precipitation.py [--regions A B]
"""

import argparse
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
OUT = DATA / "weather3"
URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_region(name, lat, lon):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": "2023-01-01", "end_date": "2023-12-31",
        "hourly": "precipitation",
        "timezone": "UTC",
    })
    with urllib.request.urlopen(f"{URL}?{q}", timeout=120) as r:
        doc = json.load(r)
    hourly = doc["hourly"]
    out = OUT / f"{name}_precip_2023_hourly.csv"
    with open(out, "w") as f:
        f.write("hour,precipitation\n")
        for t, p in zip(hourly["time"], hourly["precipitation"]):
            f.write(f"{t},{p}\n")
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
        out = OUT / f"{name}_precip_2023_hourly.csv"
        if out.exists():
            print(f"[precip] {name}: exists, skip")
            continue
        lat, lon, _ = REGION_META[name]
        try:
            p = fetch_region(name, lat, lon)
            print(f"[precip] {i + 1}/{len(names)} {name}: wrote {p.name}")
        except Exception as e:  # noqa: BLE001
            print(f"[precip] {name}: FAILED ({e})")
        time.sleep(1.2)


if __name__ == "__main__":
    main()
