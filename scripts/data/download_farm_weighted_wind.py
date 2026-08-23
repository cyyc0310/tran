#!/usr/bin/env python
"""Capacity-weighted wind-farm weather for VIC1/SA1 (representativeness fix).

Root cause (FD-14 router): the centroid ERA5 cell badly misrepresents
regions whose wind fleet clusters far from the centroid (VIC1 Melbourne
centroid vs the Western District/Macarthur belt; SA1 Adelaide centroid vs
the Mid-North/Spencer Gulf belt).  This fetches hourly 100 m wind at each
major farm's coordinate and writes a blended weather file:

    wind_speed_100m = capacity-weighted mean over farms
    temperature/shortwave = unchanged centroid values (demand channels)

Farms in operation during 2023, capacities from Wikipedia/windpower.net
(see FARM_TABLE docstrings); coordinates are site centroids, accurate to
~0.1 deg — inside the ERA5 land grid cell scale.

Output: data_2023/weather/{REGION}_farmblend_weather_2023_hourly.csv with
the standard 3-column schema; ``load_raw_weather`` prefers it when present.

Usage:
    .venv/bin/python scripts/data/download_farm_weighted_wind.py
"""

import json
import subprocess
import time
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent.parent / "data_2023"
URL = "https://archive-api.open-meteo.com/v1/archive"

# (name, MW, lat, lon) — 2023 operating fleet, ~2.7 GW total.
VIC1_FARMS = [
    ("Stockyard Hill", 530, -37.55, 143.25),
    ("Macarthur", 420, -38.05, 142.28),
    ("Dundonnell", 336, -37.87, 142.97),
    ("Moorabool N+S", 312, -37.74, 144.17),
    ("Waubra", 192, -37.35, 143.12),
    ("Berrybank", 180, -37.99, 143.58),
    ("Mt Gellibrand", 132, -38.16, 143.90),
    ("Portland/Yambuk", 130, -38.35, 141.90),
    ("Bald Hills", 107, -38.49, 146.28),
    ("Mortlake South", 91, -38.15, 142.75),
    ("Crowlands", 80, -36.90, 143.20),
    ("Yaloak South", 66, -37.87, 144.35),
    ("Oaklands Hill", 63, -37.35, 142.86),
    ("Challicum Hills", 52.5, -37.60, 143.20),
]

# NSW1 (~1.9 GW): three disjoint clusters — New England tablelands,
# Southern Tablelands (Yass/Bungendore), far-west Broken Hill — none near
# the Sydney centroid, which is why the fixed wind channel regressed
# NSW1 (wind share 0.18 at centroid R2 0.08).
NSW1_FARMS = [
    ("Sapphire", 270, -29.60, 151.60),
    ("Bango", 244, -34.75, 148.85),
    ("White Rock", 175, -29.79, 151.50),
    ("Gullen Range", 165, -34.55, 149.65),
    ("Capital", 141, -35.20, 149.40),
    ("Silverton", 200, -31.85, 141.45),
    ("Boco Rock", 113, -36.59, 149.09),
    ("Bodangora", 113, -32.40, 148.60),
    ("Woodlawn 1+2", 161, -35.05, 149.55),
    ("Broken Hill", 107, -31.95, 141.40),
    ("Crudine Ridge", 134, -33.10, 149.40),
    ("Crookwell 2", 46, -34.50, 149.50),
    ("Blayney", 37, -33.50, 149.70),
]

# ~2.35 GW total; the Mid-North/Spencer Gulf belt dominates.
SA1_FARMS = [
    ("Snowtown 1+2", 369, -33.75, 138.22),
    ("Hallett 1-5", 350, -33.40, 138.40),
    ("Hornsdale", 315, -32.70, 138.05),
    ("Lake Bonney 1-3", 278.5, -37.90, 140.40),
    ("Lincoln Gap", 212, -32.54, 137.55),
    ("North Brown Hill", 132.3, -33.30, 138.20),
    ("Waterloo", 131, -33.87, 138.62),
    ("Tailem Bend 1+2", 129, -35.10, 139.30),
    ("Willogoleche", 119, -33.60, 138.40),
    ("Wattle Point", 91, -35.10, 137.70),
    ("Clements Gap", 74, -33.55, 138.10),
    ("Mount Millar", 70, -33.65, 135.65),
    ("Canunda", 46, -37.90, 140.30),
    ("Starfish Hill", 34.5, -35.50, 138.30),
]


def fetch_wind(lat, lon, year=2023):
    q = urllib.parse.urlencode({
        "latitude": lat, "longitude": lon,
        "start_date": f"{year}-01-01", "end_date": f"{year}-12-31",
        "hourly": "wind_speed_100m",
        "timezone": "UTC",
    })
    # urllib TLS to this host is cut by the network path (curl works);
    # shell out and parse.
    out = subprocess.run(
        ["curl", "-s", "-m", "120", f"{URL}?{q}"], capture_output=True,
        check=True).stdout
    doc = json.loads(out)
    hours = pd.DatetimeIndex(doc["hourly"]["time"])
    return hours, np.asarray(doc["hourly"]["wind_speed_100m"], float)


# UK_01 North Scotland (SHEPD, ~2.4 GW in 2023): dominated by the Outer
# Moray Firth OFFSHORE cluster (Moray East 950 + Beatrice 588) which the
# regional centroid cell badly misrepresents (pre-fix wcf~wind_share
# R2 = 0.17).  Coords: Moray East/Beatrice verified (Wikipedia/4C).
UK01_FARMS = [
    ("Moray East (offshore)", 950, 58.17, -2.70),
    ("Beatrice (offshore)", 588, 58.25, -3.00),
    ("Kincardine (floating)", 50, 57.00, -1.90),
    ("Aberdeen EOWDC", 93, 57.10, -1.85),
    ("Hywind (floating)", 30, 57.42, -1.35),
    ("Gordonbush + ext", 119, 58.25, -4.10),
    ("Strathy North", 67, 58.55, -3.85),
    ("Baillie", 48, 58.50, -3.70),
    ("Halsary", 44, 58.40, -3.90),
    ("Millennium/Farr", 45, 58.40, -4.10),
    ("Dorenell", 94, 57.35, -3.15),
    ("Berry Burn", 66, 57.55, -3.40),
    ("Paul's Hill", 64, 57.55, -3.25),
    ("Mid Hill", 75, 57.10, -2.40),
    ("Rothes", 50, 57.55, -3.20),
    ("Tullo", 47, 57.25, -2.90),
    ("Clashindarroch", 47, 57.35, -2.75),
]

# Scotland-wide (UK_16) = North set + Central/South fleet (~6.3 GW).
UK16_EXTRA_FARMS = [
    ("Seagreen (offshore)", 1142, 56.90, -2.20),
    ("Whitelee", 539, 55.70, -3.90),
    ("Clyde + ext", 522, 55.45, -3.75),
    ("Crystal Rig", 400, 55.85, -2.35),
    ("Kilgallioch", 239, 55.10, -4.60),
    ("Black Law + ext", 179, 55.65, -3.70),
    ("Griffin", 156, 56.30, -4.20),
    ("Fallago Rig", 144, 55.70, -2.55),
    ("Harestanes", 136, 55.35, -3.55),
    ("Arecleoch", 120, 55.10, -4.85),
    ("Tangy + ext", 87, 55.40, -5.60),
    ("Whitehill", 67, 55.60, -2.60),
    ("Mark Hill", 56, 55.15, -4.95),
    ("Middle Muir", 47, 55.35, -3.85),
    ("Beinn an Tuirc", 34, 55.50, -5.60),
]
UK16_FARMS = UK01_FARMS + UK16_EXTRA_FARMS

# South Scotland (UK_02): Central-belt + Borders + Tayside/Angus fleet.
UK02_FARMS = UK16_EXTRA_FARMS[:5] + [
    ("Fallago Rig", 144, 55.70, -2.55),
    ("Harestanes", 136, 55.35, -3.55),
    ("Arecleoch", 120, 55.10, -4.85),
    ("Whitehill", 67, 55.60, -2.60),
    ("Mark Hill", 56, 55.15, -4.95),
    ("Middle Muir", 47, 55.35, -3.85),
]


# GB national fleet (~14 GW listed, offshore-weighted): GB synoptic wind
# is coherent nationwide, so regions without their own blend (Midlands,
# Wales, East/South England, GB aggregate) share this national blend —
# the same FD-17 representativeness fix applied where per-region farm
# tables would be indistinguishable from the national fleet.
GB_FARMS = UK16_FARMS + [
    ("Hornsea 2 (offshore)", 1400, 53.90, 1.90),
    ("Hornsea 1 (offshore)", 1200, 53.90, 1.70),
    ("Dogger Bank A (offshore, part-2023)", 300, 54.70, 1.80),
    ("East Anglia ONE (offshore)", 700, 52.30, 2.00),
    ("Triton Knoll (offshore)", 860, 53.40, 0.70),
    ("Race Bank (offshore)", 570, 52.90, 0.40),
    ("Dudgeon (offshore)", 400, 53.30, 1.40),
    ("Sheringham Shoal (offshore)", 320, 53.20, 1.20),
    ("Humber Gateway (offshore)", 220, 53.80, 0.10),
    ("Westermost Rough (offshore)", 240, 53.70, 0.20),
    ("London Array (offshore)", 630, 51.60, 1.50),
    ("Thanet (offshore)", 300, 51.40, 1.70),
    ("Kentish Flats+ (offshore)", 200, 51.40, 1.10),
    ("Rampion (offshore)", 400, 50.70, 0.40),
    ("Walney 1-4 (offshore)", 1000, 54.10, -3.60),
    ("West of Duddon Sands (offshore)", 390, 53.90, -3.30),
    ("Ormonde (offshore)", 240, 53.90, -3.60),
    ("Robin Rigg (offshore)", 180, 54.40, -3.50),
    ("Gwynt y Mor (offshore)", 580, 53.40, -3.60),
    ("Pen y Cymoedd", 230, 51.60, -3.60),
    ("Brechfa", 90, 52.10, -4.10),
    ("Scout Moor", 70, 53.60, -2.30),
]

# QLD1 (north + central Queensland fleet; Brisbane centroid misses all).
QLD1_FARMS = [
    ("Coopers Gap", 458, -26.1, 151.2),
    ("Kaban", 157, -17.3, 145.5),
    ("Mount Emerald", 181, -19.0, 146.4),
    ("Windy Hill", 45, -17.7, 145.6),
    ("Kennedy Energy Park", 43, -18.2, 145.9),
    ("Dulacca", 180, -26.9, 149.7),
    ("Clarke Creek", 191, -23.0, 149.9),
    ("Crow's Nest", 80, -27.4, 151.9),
    ("Bungaban", 180, -26.0, 150.3),
    ("Bowen", 68, -20.0, 148.2),
    ("Ravenswood", 60, -20.7, 147.3),
    ("Morgan's Camp/other NQ", 60, -19.5, 146.5),
]


# UK_10 East England: its own offshore fleet hugs the coast, so the GB
# national blend (Scottish-weighted) degrades it — region-specific table.
UK10_FARMS = [
    ("Dudgeon (offshore)", 400, 53.30, 1.40),
    ("Sheringham Shoal (offshore)", 320, 53.20, 1.20),
    ("Race Bank (offshore)", 570, 52.90, 0.40),
    ("Dudgeon ext (offshore, part)", 100, 53.40, 1.40),
    ("Lynn + Inner Dowsing (offshore)", 194, 53.20, 0.50),
    ("London Array (offshore)", 630, 51.60, 1.50),
    ("Thanet (offshore)", 300, 51.40, 1.70),
    ("Kentish Flats+ (offshore)", 200, 51.40, 1.10),
    ("Gunfleet Sands (offshore)", 140, 51.50, 1.10),
    ("Scroby Sands (offshore)", 60, 52.50, 1.80),
    ("Sheringham ext (offshore, part)", 100, 53.20, 1.10),
    ("Earls Hall / onshore Essex", 40, 51.80, 0.90),
]


# US_ERCO (~28 GW listed of ~33 GW, 2023): three disjoint clusters —
# West Texas/Permian, Panhandle (Amarillo NW), South-Texas coastal —
# none near the Austin centroid (pre-fix wcf~wind_share R2 = 0.21).
ERCO_FARMS = [
    ("Pyron + Western Trail (West TX)", 1000, 32.30, -100.90),
    ("Horse Hollow", 735, 32.20, -100.05),
    ("Roscoe", 781, 32.45, -100.55),
    ("Sweetwater", 585, 32.40, -100.40),
    ("Buffalo Gap", 523, 32.25, -99.85),
    ("Panther Creek", 458, 32.35, -100.05),
    ("Clearwater (West TX)", 468, 32.30, -101.20),
    ("Los Vientos 1-5 (coastal)", 612, 26.40, -97.85),
    ("Penascal (coastal)", 408, 26.50, -97.65),
    ("Palm Creek (coastal)", 266, 26.35, -97.75),
    ("Magic Valley + San Roman (coastal)", 300, 26.20, -97.70),
    ("Midway (Panhandle)", 1000, 35.20, -101.55),
    ("Sagamore (Panhandle)", 519, 35.45, -101.60),
    ("Nordheim (Panhandle)", 422, 28.95, -97.20),
    ("Lincoln + Chaves (NM edge)", 300, 33.40, -103.60),
    ("Jumbo Hill (Panhandle)", 250, 35.30, -101.70),
    ("Sneed (Panhandle)", 250, 33.60, -102.50),
    ("Mesteno/other Panhandle", 600, 35.00, -102.50),
    ("Barton Chapel", 200, 32.15, -99.60),
    ("Longhorn", 200, 32.20, -100.20),
    ("Silver Star", 200, 32.30, -100.00),
    ("Whirlwind/Forest Grove/other West TX", 500, 32.10, -100.30),
    ("Snapshot/others", 400, 32.80, -100.90),
    ("Comanche/other Central TX", 300, 32.00, -99.20),
    ("Ancho/other NM-W TX", 300, 33.80, -103.30),
]

# US_MISO (~10 GW listed): Iowa/Minnesota plains cluster, far from the
# Indiana/Illinois centroid.
MISO_FARMS = [
    ("Alta + Rolling Hills (IA)", 1040, 42.70, -95.30),
    ("Worth + Winnebago (IA top)", 800, 43.40, -93.20),
    ("Victory + Windsor (IA)", 800, 43.20, -92.90),
    ("Intrepid (IA)", 310, 43.30, -94.60),
    ("Buffalo Ridge I-III (MN)", 700, 44.10, -96.40),
    ("Lake Benton (MN)", 400, 44.20, -96.20),
    ("Noble/Chanute-Kansas edge", 400, 37.70, -95.50),
    ("Flat Ridge (KS)", 500, 37.30, -98.40),
    ("Central IL (Railsplitter etc)", 300, 40.20, -89.40),
    ("N.D. basin ( PrairieWinds etc)", 700, 46.50, -101.00),
    ("Meridian Way (OK/ KS)", 200, 36.80, -98.70),
    ("Crofton Bluffs (NE)", 100, 42.70, -97.20),
]

# US_CISO: Solano/Montezuma delta + Tehachapi/San Gorgonio desert passes.
CISO_FARMS = [
    ("Tehachapi cluster", 3000, 35.10, -118.30),
    ("Solano/Montezuma delta", 700, 38.15, -121.85),
    ("San Gorgonio cluster", 600, 33.90, -116.60),
    ("Ocotillo + rest Imperial", 350, 32.75, -116.00),
    ("Shasta/Lassen small", 200, 40.60, -122.00),
]


YEARS = [2023, 2022, 2024]  # multi-year blends for the multi_year protocol


def build_region(region, farms):
    for year in YEARS:
        build_region_year(region, farms, year)


def build_region_year(region, farms, year=2023):
    out_path = DATA / "weather" / f"{region}_farmblend_weather_{year}_hourly.csv"
    if out_path.exists():
        print(f"[farmblend] {region} {year}: exists, skip")
        return out_path
    hours_ref = None
    blend = None
    wsum = sum(f[1] for f in farms)
    for name, mw, lat, lon in farms:
        try:
            hours, ws = fetch_wind(lat, lon, year)
        except Exception as e:  # noqa: BLE001
            print(f"[farmblend] {region}/{name}: FAILED ({e}), skipped")
            continue
        if hours_ref is None:
            hours_ref, blend = hours, np.zeros_like(ws)
        elif not hours.equals(hours_ref):
            print(f"[farmblend] {region}/{name}: hour grid mismatch, skipped")
            continue
        blend += (mw / wsum) * ws
        if year == 2023:
            print(f"[farmblend] {region}/{name}: {mw} MW at ({lat},{lon})")
        time.sleep(1.0)
    if blend is None:
        return None
    # Keep centroid temp/swrad (demand channels), swap the wind column.
    base = pd.read_csv(DATA / "weather" / f"{region}_weather_{year}_hourly.csv",
                       parse_dates=["hour"])
    base = base.set_index("hour").reindex(hours_ref)
    df = pd.DataFrame({
        "hour": hours_ref,
        "temperature_c": base["temperature_c"].values,
        "shortwave_radiation": base["shortwave_radiation"].values,
        "wind_speed_100m": blend,
    })
    df.to_csv(out_path, index=False)
    print(f"[farmblend] {region}: wrote {out_path.name} "
          f"({len(farms)} farms, {wsum:.0f} MW)")
    return out_path


def main():
    tables = [("VIC1", VIC1_FARMS), ("SA1", SA1_FARMS),
              ("NSW1", NSW1_FARMS),
              ("UK_01_North_Scotland", UK01_FARMS),
              ("UK_16_Scotland", UK16_FARMS),
              ("UK_02_South_Scotland", UK02_FARMS),
              ("UK_18_GB", GB_FARMS),
              ("QLD1", QLD1_FARMS),
              ("UK_10_East_England", UK10_FARMS),
              ("US_ERCO", ERCO_FARMS),
              ("US_MISO", MISO_FARMS),
              ("US_CISO", CISO_FARMS)]
    for region, farms in tables:
        build_region(region, farms)


if __name__ == "__main__":
    main()
