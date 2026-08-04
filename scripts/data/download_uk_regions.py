"""Download UK Carbon Intensity regional data (18 zones) for 2023.

Free API, no key required: https://carbonintensity.org.uk/
Output: data_2023/UK_{regionid}_{shortname}_2023_hourly.csv
Schema: Region, hour, renew_out, nonrenew_out, total_emissions, total_energy_so,
        renew_share, cif_real_tco2_per_mwh, cif_real_gco2_per_kwh
"""
import json, os, time, urllib.request
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

OUT_DIR = "data_2023"
os.makedirs(OUT_DIR, exist_ok=True)

# Renewable fuels (zero/low carbon for CIF purposes)
RENEW_FUELS = {"wind", "solar", "hydro", "nuclear", "biomass"}
NONRENEW_FUELS = {"coal", "gas", "other", "imports"}

BASE_URL = "https://api.carbonintensity.org.uk"


def fetch_region_chunk(region_id: int, start: datetime, end: datetime):
    """Fetch up to 14 days of regional data."""
    s = start.strftime("%Y-%m-%dT%H:%MZ")
    e = end.strftime("%Y-%m-%dT%H:%MZ")
    url = f"{BASE_URL}/regional/intensity/{s}/{e}/regionid/{region_id}"
    try:
        resp = urllib.request.urlopen(url, timeout=30)
        data = json.load(resp)
        return data["data"]["data"]
    except Exception as ex:
        print(f"    WARN: {region_id} {s}→{e}: {ex}")
        return []


def parse_intervals(intervals):
    """Parse API intervals into rows."""
    rows = []
    for iv in intervals:
        t = datetime.fromisoformat(iv["from"].replace("Z", "+00:00"))
        intensity = iv["intensity"].get("forecast", None)  # actual not always available
        mix = {g["fuel"]: g["perc"] for g in iv.get("generationmix", [])}
        
        renew_pct = sum(mix.get(f, 0) for f in RENEW_FUELS)
        nonrenew_pct = sum(mix.get(f, 0) for f in NONRENEW_FUELS)
        total_pct = renew_pct + nonrenew_pct
        
        if total_pct > 0 and intensity is not None:
            rs = renew_pct / total_pct
            rows.append({
                "time": t.replace(tzinfo=None),
                "renew_pct": renew_pct,
                "nonrenew_pct": nonrenew_pct,
                "cif_gco2_per_kwh": intensity,
                "renew_share": rs,
            })
    return rows


def get_all_regions():
    """Hardcoded UK DNO regions (IDs 1-18)."""
    return [
        {"id": 1, "name": "North Scotland"},
        {"id": 2, "name": "South Scotland"},
        {"id": 3, "name": "North West England"},
        {"id": 4, "name": "North East England"},
        {"id": 5, "name": "Yorkshire"},
        {"id": 6, "name": "North Wales Merseyside"},
        {"id": 7, "name": "South Wales"},
        {"id": 8, "name": "West Midlands"},
        {"id": 9, "name": "East Midlands"},
        {"id": 10, "name": "East England"},
        {"id": 11, "name": "South West England"},
        {"id": 12, "name": "South England"},
        {"id": 13, "name": "London"},
        {"id": 14, "name": "South East England"},
        {"id": 15, "name": "England"},
        {"id": 16, "name": "Scotland"},
        {"id": 17, "name": "Wales"},
        {"id": 18, "name": "GB"},
    ]


def download_region_year(region_id, region_name, year=2023):
    """Download full year in 14-day chunks, aggregate to hourly."""
    safe_name = region_name.replace(" ", "_")
    out_path = f"{OUT_DIR}/UK_{region_id:02d}_{safe_name}_2023_hourly.csv"
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} exists")
        return out_path
    
    start = datetime(year, 1, 1)
    end = datetime(year + 1, 1, 1)
    chunk_days = 14
    
    all_rows = []
    cursor = start
    while cursor < end:
        chunk_end = min(cursor + timedelta(days=chunk_days), end)
        intervals = fetch_region_chunk(region_id, cursor, chunk_end)
        rows = parse_intervals(intervals)
        all_rows.extend(rows)
        cursor = chunk_end
        time.sleep(0.3)  # polite rate limiting
    
    if not all_rows:
        print(f"  [EMPTY] region {region_id} ({region_name})")
        return None
    
    df = pd.DataFrame(all_rows)
    df["hour"] = df["time"].dt.floor("h")
    
    # Aggregate 30-min → hourly (mean of 2 half-hours)
    hourly = df.groupby("hour").agg(
        renew_pct=("renew_pct", "mean"),
        nonrenew_pct=("nonrenew_pct", "mean"),
        cif_real_gco2_per_kwh=("cif_gco2_per_kwh", "mean"),
        renew_share=("renew_share", "mean"),
    ).reset_index()
    
    # Normalize pct to pseudo-energy (total=100 per hour)
    hourly["renew_out"] = hourly["renew_pct"]
    hourly["nonrenew_out"] = hourly["nonrenew_pct"]
    hourly["total_energy_so"] = hourly["renew_out"] + hourly["nonrenew_out"]
    # Back-calculate total_emissions from CIF × total_energy
    # CIF (gCO2/kWh) = total_emissions / total_energy, so:
    hourly["total_emissions"] = hourly["cif_real_gco2_per_kwh"] * hourly["total_energy_so"] / 1000.0
    hourly["cif_real_tco2_per_mwh"] = hourly["cif_real_gco2_per_kwh"] / 1000.0
    hourly.insert(0, "Region", f"UK_{region_id:02d}")
    
    out_cols = ["Region", "hour", "nonrenew_out", "renew_out", "total_emissions",
                "total_energy_so", "renew_share", "cif_real_tco2_per_mwh", "cif_real_gco2_per_kwh"]
    hourly[out_cols].to_csv(out_path, index=False)
    print(f"  [OK] {out_path}: {len(hourly)} hours, mean CIF={hourly['cif_real_gco2_per_kwh'].mean():.1f}")
    return out_path


if __name__ == "__main__":
    print("Fetching UK region list...")
    regions = get_all_regions()
    print(f"Found {len(regions)} regions: {[r['name'] for r in regions]}")
    
    for r in regions:
        print(f"\n[{r['id']:02d}] {r['name']}...")
        download_region_year(r["id"], r["name"])
    
    print("\n=== Done ===")
