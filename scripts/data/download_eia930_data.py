"""Phase 1.2: Download and preprocess US EIA-930 data for expanding to 25+ regions.

EIA-930 provides hourly generation by source for US balancing authorities.
We download the 2023 CSV files and compute renew_share + CIF for each region.

Target regions: CISO, PJM, MISO, ERCO, ISNE, NYIS, FPL, BPAT
(8 US regions + 4 AU + 17 UK = 29 total)

Usage: python scripts/download_eia930_data.py
"""

import os
import sys
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data_2023"
RAW_DIR = DATA_DIR / "raw_eia930"
RAW_DIR.mkdir(exist_ok=True)

# EIA-930 download URLs (6-month bulk files)
EIA_URLS = {
    "gen_jan_jun_2023": "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jan_Jun.csv",
    "gen_jul_dec_2023": "https://www.eia.gov/electricity/gridmonitor/sixMonthFiles/EIA930_BALANCE_2023_Jul_Dec.csv",
}

# US Balancing Authority regions to extract
US_REGIONS = {
    "CISO": "California ISO",
    "PJM": "PJM Interconnection",
    "MISO": "Midcontinent ISO",
    "ERCO": "ERCOT (Texas)",
    "ISNE": "ISO New England",
    "NYIS": "New York ISO",
    "FPL": "Florida Power & Light",
    "BPAT": "Bonneville Power (Pacific NW)",
}

# Direct emission factors for US sources (gCO₂/kWh) — IPCC/EPA median values
# Source: EPA eGRID 2022 + IPCC AR6
EMISSION_FACTORS = {
    "coal": 980,
    "natural gas": 410,    # combined cycle average
    "petroleum": 650,
    "nuclear": 0,
    "solar": 0,
    "wind": 0,
    "hydro": 0,
    "other renewables": 0,  # biomass/geothermal (small, treat as 0 for simplicity)
    "other": 500,           # mixed/unknown
    "unknown": 400,
}

# EIA-930 source column name mapping to emission factor categories
SOURCE_TO_CATEGORY = {
    "Net Generation (MW) from Coal": "coal",
    "Net Generation (MW) from Natural Gas": "natural gas",
    "Net Generation (MW) from Nuclear": "nuclear",
    "Net Generation (MW) from Petroleum": "petroleum",
    "Net Generation (MW) from Hydropower and Pumped Storage": "hydro",
    "Net Generation (MW) from Solar": "solar",
    "Net Generation (MW) from Wind": "wind",
    "Net Generation (MW) from Other Fuel Sources": "other",
    "Net Generation (MW) from Unknown Fuel Sources": "unknown",
}

# Renewable sources
RENEWABLE_SOURCES = {"solar", "wind", "hydro", "other renewables"}


def download_files():
    """Download EIA-930 CSV files if not already present."""
    for name, url in EIA_URLS.items():
        filepath = RAW_DIR / f"{name}.csv"
        if filepath.exists():
            print(f"  Already exists: {filepath.name}")
            continue
        print(f"  Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, filepath)
            print(f"  ✓ Downloaded: {filepath.name} ({filepath.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            print(f"  Manual download: {url}")
            print(f"  Save to: {filepath}")


def load_and_merge_eia():
    """Load and merge the two 6-month CSV files."""
    dfs = []
    for name in EIA_URLS:
        filepath = RAW_DIR / f"{name}.csv"
        if not filepath.exists():
            print(f"  Missing: {filepath}. Run download first.")
            continue
        print(f"  Loading {filepath.name}...")
        df = pd.read_csv(filepath, low_memory=False)
        dfs.append(df)

    if not dfs:
        return None
    return pd.concat(dfs, ignore_index=True)


def process_region(df_region, region_code):
    """Process one US region's data into our standard format.

    Returns DataFrame with columns matching our AU/UK format:
    hour, nonrenew_out, renew_out, total_emissions, total_energy_so,
    renew_share, cif_real_tco2_per_mwh, cif_real_gco2_per_kwh
    """
    # Parse timestamp
    if "UTC Time at End of Hour" in df_region.columns:
        df_region = df_region.copy()
        df_region["hour"] = pd.to_datetime(df_region["UTC Time at End of Hour"])
    elif "Local Time at End of Hour" in df_region.columns:
        df_region = df_region.copy()
        df_region["hour"] = pd.to_datetime(df_region["Local Time at End of Hour"])
    else:
        print(f"    No time column found for {region_code}")
        return None

    df_region = df_region.sort_values("hour").reset_index(drop=True)

    # Extract generation by source
    total_gen = np.zeros(len(df_region))
    renew_gen = np.zeros(len(df_region))
    total_emissions = np.zeros(len(df_region))

    for col_name, category in SOURCE_TO_CATEGORY.items():
        if col_name in df_region.columns:
            gen = pd.to_numeric(df_region[col_name], errors="coerce").fillna(0).values
            gen = np.maximum(gen, 0)  # No negative generation
            total_gen += gen
            if category in RENEWABLE_SOURCES:
                renew_gen += gen
            total_emissions += gen * EMISSION_FACTORS.get(category, 0)

    # Compute metrics
    mask = total_gen > 0
    renew_share = np.where(mask, renew_gen / total_gen, 0.0)
    cif_gco2_kwh = np.where(mask, total_emissions / total_gen, 0.0)

    result = pd.DataFrame({
        "Region": region_code,
        "hour": df_region["hour"],
        "nonrenew_out": total_gen - renew_gen,
        "renew_out": renew_gen,
        "total_emissions": total_emissions / 1000,  # to tCO₂
        "total_energy_so": total_gen,
        "renew_share": renew_share,
        "cif_real_tco2_per_mwh": cif_gco2_kwh / 1000,  # gCO₂/kWh → tCO₂/MWh
        "cif_real_gco2_per_kwh": cif_gco2_kwh,
    })

    # Drop rows with zero total generation
    result = result[result["total_energy_so"] > 0].reset_index(drop=True)

    # Resample to hourly (some entries may be duplicated or missing)
    result = result.drop_duplicates(subset=["hour"], keep="first")
    result = result.set_index("hour").resample("h").first().reset_index()
    result = result.dropna(subset=["renew_share"])

    return result


def estimate_emission_factors(df):
    """Estimate ef_r and ef_nr from the processed data."""
    rs = df["renew_share"].values
    cif = df["cif_real_gco2_per_kwh"].values
    mask = (rs > 0.05) & (rs < 0.95) & (cif > 0)
    if mask.sum() < 100:
        return 0.0, float(np.median(cif[cif > 0]))

    # ef_r ≈ 0 for most grids (solar/wind/hydro)
    ef_r = 0.0
    # ef_nr = CIF / (1 - rs) when ef_r ≈ 0
    ef_nr_values = cif[mask] / (1 - rs[mask])
    ef_nr = float(np.median(ef_nr_values))
    return ef_r, ef_nr


def main():
    print("=" * 70)
    print("Phase 1.2: EIA-930 US Data Download & Processing")
    print("=" * 70)

    # Step 1: Download
    print("\n[1/3] Downloading EIA-930 data files...")
    download_files()

    # Step 2: Load
    print("\n[2/3] Loading CSV files...")
    df = load_and_merge_eia()
    if df is None:
        print("No data available. Please download manually.")
        print("URLs:")
        for name, url in EIA_URLS.items():
            print(f"  {url}")
        sys.exit(1)

    print(f"  Total rows: {len(df):,}")
    print(f"  Columns: {list(df.columns[:10])}...")

    # Identify the BA column
    ba_col = None
    for candidate in ["Balancing Authority", "BA Code", "Respondent"]:
        if candidate in df.columns:
            ba_col = candidate
            break
    if ba_col is None:
        print(f"  Available columns: {list(df.columns)}")
        print("  Cannot find BA column. Checking data format...")
        print(df.head(2).to_string())
        sys.exit(1)

    print(f"  BA column: '{ba_col}'")
    available_bas = df[ba_col].unique()
    print(f"  Available BAs: {len(available_bas)}")

    # Step 3: Process each target region
    print("\n[3/3] Processing target US regions...")
    results_summary = []

    for region_code, region_name in US_REGIONS.items():
        # Check if this BA exists in data
        if region_code not in available_bas:
            print(f"  ⚠ {region_code} ({region_name}) not found in data, skipping")
            continue

        print(f"\n  Processing {region_code} ({region_name})...")
        df_region = df[df[ba_col] == region_code].copy()
        print(f"    Raw rows: {len(df_region):,}")

        result = process_region(df_region, region_code)
        if result is None or len(result) < 1000:
            print(f"    ⚠ Insufficient data ({len(result) if result is not None else 0} hours)")
            continue

        # Estimate emission factors
        ef_r, ef_nr = estimate_emission_factors(result)

        # Save to CSV
        outfile = DATA_DIR / f"US_{region_code}_2023_hourly.csv"
        result.to_csv(outfile, index=False)

        mean_rs = result["renew_share"].mean()
        mean_cif = result["cif_real_gco2_per_kwh"].mean()
        n_hours = len(result)

        print(f"    ✓ Saved: {outfile.name}")
        print(f"    Hours: {n_hours}, mean_rs: {mean_rs:.3f}, mean_CIF: {mean_cif:.1f}, ef_nr_est: {ef_nr:.1f}")
        results_summary.append({
            "region": region_code, "name": region_name,
            "hours": n_hours, "mean_rs": mean_rs,
            "mean_cif": mean_cif, "ef_nr": ef_nr
        })

    # Summary table
    if results_summary:
        print("\n\n" + "=" * 70)
        print("US REGIONS SUMMARY")
        print("=" * 70)
        print(f"{'Region':<8} {'Name':<30} {'Hours':<7} {'mean_rs':<9} {'mean_CIF':<10} {'ef_nr':<8}")
        print("-" * 70)
        for r in results_summary:
            print(f"{r['region']:<8} {r['name']:<30} {r['hours']:<7} {r['mean_rs']:<9.3f} "
                  f"{r['mean_cif']:<10.1f} {r['ef_nr']:<8.1f}")
        print(f"\nTotal new US regions: {len(results_summary)}")
        print(f"Combined with AU(4) + UK(17): {4 + 17 + len(results_summary)} total regions")
    else:
        print("\nNo US regions processed. Check data availability.")


if __name__ == "__main__":
    main()
