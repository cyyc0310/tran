#!/usr/bin/env python
"""Download day-ahead GFS + ICON forecast archive for FD-47 (P0 rewrite).

Reality check (2026-09-09): ECMWF IFS/AIFS open-data archives start
2024-02 (0.25deg) — NOTHING is available for the 2023 evaluation year.
The only archived day-ahead models covering 2023 on Open-Meteo are
gfs_seamless and icon_seamless.  FD-47 therefore replaces the ERA5
reanalysis proxy in ``fut_weather`` with these two OPERATIONAL forecast
archives (the weather a deployer would actually receive), keeping ERA5
for the 336 h history.

Per region (REGION_META centroid), 2023 full year, hourly:
    wind_speed_100m (m/s), temperature_2m (C), shortwave_radiation (W/m2)
for BOTH models in one request each.

Output: data_2023/nwp/{REGION}_nwp_2023_hourly.csv
    hour, gfs_wind100, gfs_temp, gfs_swr, icon_wind100, icon_temp, icon_swr

Usage:
    env PYTHONPATH=src .venv-nemed/bin/python scripts/data/download_nwp_gfs_icon.py
"""
import argparse
import calendar
import csv
import json
import subprocess
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
OUT = DATA / "nwp"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODELS = "gfs_seamless,icon_seamless"
CHUNKS = [(f"2023-{m:02d}-01",
           f"2023-{m:02d}-{calendar.monthrange(2023, m)[1]}")
          for m in range(1, 13)]


def fetch_chunk(lat, lon, start, end, retries=4):
    q = (f"latitude={lat}&longitude={lon}"
         f"&start_date={start}&end_date={end}"
         f"&hourly=wind_speed_100m,temperature_2m,shortwave_radiation"
         f"&models={MODELS}&wind_speed_unit=ms&timeformat=unixtime")
    for i in range(retries):
        r = subprocess.run(["curl", "-s", "--max-time", "120",
                            f"{URL}?{q}"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip().startswith("{"):
            return r.stdout
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"fetch failed {start}..{end}: {r.stderr[:120]}")


def parse_rows(js):
    import datetime as dt
    doc = json.loads(js)
    h = doc["hourly"]
    keys = h.keys()
    g_w = h.get("wind_speed_100m_gfs_seamless",
                h.get("wind_speed_100m", [None] * len(h["time"])))
    g_t = h.get("temperature_2m_gfs_seamless",
                h.get("temperature_2m", [None] * len(h["time"])))
    g_s = h.get("shortwave_radiation_gfs_seamless",
                h.get("shortwave_radiation", [None] * len(h["time"])))
    i_w = h.get("wind_speed_100m_icon_seamless",
                h.get("wind_speed_100m", [None] * len(h["time"])))
    i_t = h.get("temperature_2m_icon_seamless",
                h.get("temperature_2m", [None] * len(h["time"])))
    i_s = h.get("shortwave_radiation_icon_seamless",
                h.get("shortwave_radiation", [None] * len(h["time"])))
    rows = []
    for k, t in enumerate(h["time"]):
        stamp = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
        rows.append((stamp.strftime("%Y-%m-%d %H:%M:%S"),
                     g_w[k] if g_w and g_w[k] is not None else "",
                     g_t[k] if g_t and g_t[k] is not None else "",
                     g_s[k] if g_s and g_s[k] is not None else "",
                     i_w[k] if i_w and i_w[k] is not None else "",
                     i_t[k] if i_t and i_t[k] is not None else "",
                     i_s[k] if i_s and i_s[k] is not None else ""))
    return rows


def download_region(region, lat, lon, year=2023):
    out = OUT / f"{region}_nwp_{year}_hourly.csv"
    if out.exists():
        print(f"[skip] {out.name}")
        return
    all_rows = []
    for start, end in CHUNKS:
        all_rows += parse_rows(fetch_chunk(lat, lon, start, end))
        time.sleep(1.2)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["hour", "gfs_wind100", "gfs_temp", "gfs_swr",
                    "icon_wind100", "icon_temp", "icon_swr"])
        w.writerows(all_rows)
    nonempty = sum(1 for r in all_rows if r[1] != "")
    print(f"[ok] {region}: {len(all_rows)} rows, gfs non-empty {nonempty}")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from transcif.config.region_meta import REGION_META
    from transcif.data.loaders import all_region_configs

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # every LORO target region (29) — the model needs fut_weather everywhere
    regions = args.regions or sorted(all_region_configs())
    print(f"[nwp-dl] {len(regions)} regions")
    for r in regions:
        meta = REGION_META.get(r)
        if meta is None:
            print(f"[ERR] {r}: no REGION_META")
            continue
        lat, lon, _ = meta
        try:
            download_region(r, lat, lon)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {r}: {e}")


if __name__ == "__main__":
    main()
