"""Download hourly ERA5 weather data from Open-Meteo for all 29+ regions.

Stage B.1: weather as causal upstream of renewable generation.  Open-Meteo's
free historical archive API serves ERA5 reanalysis (no API key, non-commercial
use).  We fetch three variables that physically drive solar/wind/hydro output:

    shortwave_radiation  — solar resource (W/m²), drives solar PV
    wind_speed_100m      — wind resource (km/h), drives wind turbines
    temperature_2m       — demand proxy (°C), heating/cooling load

Each region is queried at a single representative-city coordinate (the project
convention per README §"Optional: Temperature Data").  Output is one CSV per
region in data_2023/weather/{REGION}_weather_2023_hourly.csv with columns:
    hour, temperature_c, shortwave_radiation, wind_speed_100m

Usage:
    PYTHONPATH=. python scripts/data/download_weather_openmeteo.py
"""

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_2023"
WEATHER_DIR = DATA_DIR / "weather"
WEATHER_DIR.mkdir(exist_ok=True)

START_DATE = "2023-01-01"
END_DATE = "2023-12-31"

# Representative city per region (lat, lon).  US = BA load centre, AU = NEM
# capital, UK = DNO region's largest city.  Coordinates are intentionally
# coarse — a single point per region is the project's established convention.
REGION_COORDS = {
    # AU NEM
    "QLD1": (-27.47, 153.02, "Brisbane"),
    "NSW1": (-33.87, 151.21, "Sydney"),
    "VIC1": (-37.81, 144.96, "Melbourne"),
    "SA1":  (-34.93, 138.60, "Adelaide"),
    "TAS1": (-42.88, 147.33, "Hobart"),
    # US EIA-930 balancing authorities
    "US_CISO": (38.58, -121.49, "Sacramento CA"),
    "US_PJM":  (39.95, -75.16, "Philadelphia PA"),
    "US_MISO": (39.77, -86.16, "Indianapolis IN"),
    "US_ERCO": (30.27, -97.74, "Austin TX"),
    "US_ISNE": (42.36, -71.06, "Boston MA"),
    "US_NYIS": (42.65, -73.76, "Albany NY"),
    "US_FPL":  (25.76, -80.19, "Miami FL"),
    "US_BPAT": (45.52, -122.67, "Portland OR"),
    # UK DNO regions (IDs match download_uk_regions.py)
    "UK_01_North_Scotland":          (57.48, -4.22, "Inverness"),
    "UK_02_South_Scotland":          (55.86, -4.25, "Glasgow"),
    "UK_03_North_West_England":      (53.41, -2.99, "Manchester"),
    "UK_04_North_East_England":      (54.60, -1.23, "Middlesbrough"),
    "UK_05_Yorkshire":               (53.80, -1.54, "Leeds"),
    "UK_06_North_Wales_Merseyside":  (53.40, -3.00, "Liverpool"),
    "UK_07_South_Wales":             (51.48, -3.18, "Cardiff"),
    "UK_08_West_Midlands":           (52.48, -1.90, "Birmingham"),
    "UK_09_East_Midlands":           (52.63, -1.14, "Leicester"),
    "UK_10_East_England":            (52.63, 1.30, "Norwich"),
    "UK_11_South_West_England":      (51.45, -2.59, "Bristol"),
    "UK_12_South_England":           (50.90, -1.40, "Southampton"),
    "UK_13_London":                  (51.51, -0.13, "London"),
    "UK_14_South_East_England":      (51.13, 0.27, "Tunbridge Wells"),
    "UK_15_England":                 (52.48, -1.90, "Birmingham"),  # aggregate proxy
    "UK_16_Scotland":                (55.95, -3.19, "Edinburgh"),
    "UK_17_Wales":                   (52.41, -3.58, "Aberystwyth"),
    "UK_18_GB":                      (52.91, -1.47, "Nottingham"),  # GB centroid proxy
}

VARIABLES = "shortwave_radiation,wind_speed_100m,temperature_2m"
API_BASE = "https://archive-api.open-meteo.com/v1/archive"


def fetch_one_region(region, lat, lon, city):
    """Fetch one region's full-year hourly weather from Open-Meteo."""
    params = urllib.parse.urlencode({
        "latitude": lat,
        "longitude": lon,
        "start_date": START_DATE,
        "end_date": END_DATE,
        "hourly": VARIABLES,
        "timezone": "UTC",
    })
    url = f"{API_base}?{params}" if False else f"{API_BASE}?{params}"
    out_path = WEATHER_DIR / f"{region}_weather_2023_hourly.csv"
    if out_path.exists():
        print(f"  [skip] {region} exists")
        return out_path

    try:
        resp = urllib.request.urlopen(url, timeout=60)
        data = json.load(resp)
    except Exception as ex:
        print(f"  [FAIL] {region} ({city}): {ex}")
        return None

    hourly = data.get("hourly", {})
    times = hourly.get("time", [])
    if not times:
        print(f"  [FAIL] {region}: no hourly data in response")
        return None

    df = pd.DataFrame({
        "hour": pd.to_datetime(times),
        "temperature_c": hourly.get("temperature_2m", [None] * len(times)),
        "shortwave_radiation": hourly.get("shortwave_radiation", [None] * len(times)),
        "wind_speed_100m": hourly.get("wind_speed_100m", [None] * len(times)),
    })
    # Fill any NaN with forward-fill then backward-fill (ERA5 gaps are rare).
    df[["temperature_c", "shortwave_radiation", "wind_speed_100m"]] = (
        df[["temperature_c", "shortwave_radiation", "wind_speed_100m"]]
        .interpolate(method="linear").ffill().bfill())
    df.to_csv(out_path, index=False)
    print(f"  [OK] {region:32s} ({city:18s}) {len(df)} rows  "
          f"temp={df.temperature_c.mean():.1f}°C  "
          f"rad={df.shortwave_radiation.mean():.0f}W/m²  "
          f"wind={df.wind_speed_100m.mean():.1f}km/h")
    return out_path


def main():
    print("=" * 80)
    print("Stage B.1: Download ERA5 weather (Open-Meteo) for 30 regions")
    print("=" * 80)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Variables:  {VARIABLES}")
    print(f"Output:     {WEATHER_DIR}/")

    ok, fail = 0, 0
    for region, (lat, lon, city) in REGION_COORDS.items():
        result = fetch_one_region(region, lat, lon, city)
        if result is not None:
            ok += 1
        else:
            fail += 1
        time.sleep(1.0)  # polite rate limiting

    print(f"\n[SUMMARY] {ok} ok, {fail} failed, {len(REGION_COORDS)} total")
    if fail:
        print("Re-run to retry failed regions (existing files are skipped).")


if __name__ == "__main__":
    main()
