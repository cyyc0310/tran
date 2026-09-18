#!/usr/bin/env python
"""Download day-ahead ECMWF IFS 0.25° forecast archive for FD-47 (P0).

Open-Meteo historical-forecast API serves archived operational IFS
(ecmwf_ifs025) forecasts keyed by valid time.  Unlike the ERA5 reanalysis
proxy currently used for ``fut_weather``, these are the forecasts a
deployer would ACTUALLY receive day-ahead — including IFS wind bias and
forecast error.

Per region (centroid of REGION_META): 2023 full year, hourly
    wind_speed_100m (m/s), temperature_2m (C), shortwave_radiation (W/m2)

Output: data_2023/nwp_ifs/{REGION}_ifs_2023_hourly.csv
Usage:
    .venv-nemed/bin/python scripts/data/download_ifs_forecasts.py
    .venv-nemed/bin/python scripts/data/download_ifs_forecasts.py --regions VIC1 SA1
"""
import argparse
import subprocess
import time
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
OUT = DATA / "nwp_ifs"
URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
MODEL = "ecmwf_ifs025"
# API limit: ~8 days per request comfortably; we chunk by month.
CHUNKS = [(f"2023-{m:02d}-01", f"2023-{m:02d}-28") for m in range(1, 13)]
CHUNKS[1] = ("2023-02-01", "2023-02-28")   # 2023 not a leap year


def fetch_chunk(lat, lon, start, end, retries=3):
    q = (f"latitude={lat}&longitude={lon}"
         f"&start_date={start}&end_date={end}"
         f"&hourly=wind_speed_100m,temperature_2m,shortwave_radiation"
         f"&models={MODEL}&wind_speed_unit=ms&timeformat=unixtime")
    for i in range(retries):
        r = subprocess.run(["curl", "-s", "--max-time", "120",
                            f"{URL}?{q}"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip().startswith("{"):
            return r.stdout
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"fetch failed for {start}..{end}: {r.stderr[:120]}")


def parse_to_rows(json_text):
    import json
    doc = json.loads(json_text)
    h = doc["hourly"]
    import datetime as dt
    rows = []
    for t, w, temp, swr in zip(h["time"], h["wind_speed_100m"],
                               h["temperature_2m"],
                               h["shortwave_radiation"]):
        stamp = dt.datetime.fromtimestamp(t, tz=dt.timezone.utc)
        rows.append((stamp.strftime("%Y-%m-%d %H:%M:%S"),
                     w if w is not None else "",
                     temp if temp is not None else "",
                     swr if swr is not None else ""))
    return rows


def download_region(region, lat, lon, year=2023):
    import csv
    out = OUT / f"{region}_ifs_{year}_hourly.csv"
    if out.exists():
        print(f"[skip] {out.name} exists")
        return
    all_rows = []
    for start, end in CHUNKS:
        js = fetch_chunk(lat, lon, start, end)
        all_rows += parse_to_rows(js)
        time.sleep(1.5)
    with open(out, "w", newline="") as f:
        wcsv = csv.writer(f)
        wcsv.writerow(["hour", "wind_speed_100m", "temperature_2m",
                       "shortwave_radiation"])
        wcsv.writerows(all_rows)
    print(f"[ok] {region}: {len(all_rows)} rows -> {out.name}")


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    from transcif.config.region_meta import REGION_META

    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    # regions that actually have centroid weather files (the FD stack
    # consumes these); UK regions without own file share UK_18_GB later.
    have_weather = sorted(
        p.name.split("_weather_")[0] for p in (DATA / "weather").glob(
            "*_weather_2023_hourly.csv"))
    regions = args.regions or have_weather
    print(f"[ifs-dl] {len(regions)} regions")
    for r in regions:
        meta = REGION_META.get(r)
        if meta is None:
            print(f"[skip] {r}: no REGION_META entry")
            continue
        lat, lon, _ = meta
        try:
            download_region(r, lat, lon)
        except Exception as e:  # noqa: BLE001
            print(f"[ERR] {r}: {e}")


if __name__ == "__main__":
    main()
