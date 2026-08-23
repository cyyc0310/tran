"""Extract per-fuel generation breakdown from EIA-930 for US regions.

The legacy ``download_eia930_data.py`` collapses the 7 fuel-specific generation
columns into a single ``renew_share`` scalar, discarding the fuel mix that
distinguishes e.g. wind-dominated ERCO from coal-dominated MISO (both land at
similar mean_rs but have very different CIF dynamics).  This script re-reads the
raw EIA-930 bulk CSVs and emits an *augmented* per-region CSV that keeps the
original 9 columns **plus** 7 per-fuel generation columns (MW), using the
cleaner "Adjusted" series.

Outputs (does NOT overwrite the existing ``US_*_2023_hourly.csv``):
    data_2023/fuel/US_<BA>_fuel_2023_hourly.csv  — augmented hourly CSV
    data_2023/fuel/fuel_shares_us.json           — annual mean fuel-share vectors
                                                    (training-split aware) for
                                                    direct injection into
                                                    constants.py

Usage:
    PYTHONPATH=. python scripts/data/extract_fuel_breakdown.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_2023"
RAW_DIR = DATA_DIR / "raw_eia930"
FUEL_DIR = DATA_DIR / "fuel"
FUEL_DIR.mkdir(exist_ok=True)

# Six-month bulk files produced by download_eia930_data.py
YEAR = 2023  # --year override; files are named by the downloader

# Target US balancing authorities (must match constants.py US_REGIONS keys)
US_REGIONS = {
    "CISO", "PJM", "MISO", "ERCO", "ISNE", "NYIS", "FPL", "BPAT",
}

# EIA-930 "Adjusted" columns are the cleanest (EIA fills structural gaps).
# Map the long EIA column name -> our short fuel key.  We keep the 7 fuels that
# matter for carbon intensity; "Other" and "Unknown" are lumped into a residual.
FUEL_ADJUSTED_COLS = {
    "coal":      "Net Generation (MW) from Coal (Adjusted)",
    "gas":       "Net Generation (MW) from Natural Gas (Adjusted)",
    "nuclear":   "Net Generation (MW) from Nuclear (Adjusted)",
    "petroleum": "Net Generation (MW) from All Petroleum Products (Adjusted)",
    "hydro":     "Net Generation (MW) from Hydropower and Pumped Storage (Adjusted)",
    "solar":     "Net Generation (MW) from Solar (Adjusted)",
    "wind":      "Net Generation (MW) from Wind (Adjusted)",
}
FUEL_KEYS = list(FUEL_ADJUSTED_COLS.keys())

# Lifecycle emission factors (gCO2/kWh) — IPCC AR6 / EPA eGRID medians.
# Used to (a) sanity-check the reconstructed CIF against the legacy file and
# (b) provide the physics layer with per-fuel EFs for multi-fuel decomposition.
FUEL_EMISSION_FACTORS = {
    "coal": 980.0, "gas": 410.0, "nuclear": 0.0, "petroleum": 650.0,
    "hydro": 0.0, "solar": 0.0, "wind": 0.0,
}
# Fuels counted as "renewable" — must match download_eia930_data.py:71 semantics
# (solar/wind/hydro renewable; nuclear stays non-renewable in the US convention).
RENEWABLE_FUELS = {"hydro", "solar", "wind"}

TRAIN_FRACTION = 0.8


def load_raw_eia():
    """Concatenate the two 6-month EIA-930 bulk CSVs."""
    eia_files = [f"gen_jan_jun_{YEAR}.csv", f"gen_jul_dec_{YEAR}.csv"]
    dfs = []
    for name in eia_files:
        path = RAW_DIR / name
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run download_eia930_data.py first.")
        print(f"  Loading {name}...")
        dfs.append(pd.read_csv(path, low_memory=False))
    return pd.concat(dfs, ignore_index=True)


def process_region(df_region: pd.DataFrame, region_code: str) -> pd.DataFrame:
    """Extract per-fuel generation + reconstructed CIF for one BA.

    Returns a DataFrame indexed by hour with the 7 fuel columns plus the
    standard renew_share / cif columns for cross-checking against the legacy
    file.
    """
    df = df_region.copy()
    # Prefer UTC timestamp for cross-region consistency.
    if "UTC Time at End of Hour" in df.columns:
        df["hour"] = pd.to_datetime(df["UTC Time at End of Hour"])
    elif "Local Time at End of Hour" in df.columns:
        df["hour"] = pd.to_datetime(df["Local Time at End of Hour"])
    else:
        raise KeyError(f"No time column for {region_code}")
    df = df.sort_values("hour").reset_index(drop=True)

    # Pull each fuel's Adjusted generation.
    fuel_gen = {}
    for key, col in FUEL_ADJUSTED_COLS.items():
        if col not in df.columns:
            raise KeyError(f"{region_code}: missing column {col!r}")
        gen = pd.to_numeric(df[col], errors="coerce").fillna(0.0).values
        fuel_gen[key] = np.maximum(gen, 0.0)

    gen_matrix = np.stack([fuel_gen[k] for k in FUEL_KEYS], axis=1)  # (T, 7)
    total_gen = gen_matrix.sum(axis=1)  # (T,)

    # Reconstruct CIF = Σ share_f × ef_f  (gCO2/kWh)
    ef_vector = np.array([FUEL_EMISSION_FACTORS[k] for k in FUEL_KEYS])  # (7,)
    total_emissions = gen_matrix @ ef_vector  # (T,) gCO2/kWh × MW ≈ g·MW/(kWh·MW)
    mask = total_gen > 0
    safe_total = np.where(mask, total_gen, 1.0)  # avoid div-by-zero where gen==0
    cif_gco2_kwh = np.where(mask, total_emissions / safe_total, 0.0)
    renew_gen = fuel_gen["hydro"] + fuel_gen["solar"] + fuel_gen["wind"]
    renew_share = np.where(mask, renew_gen / safe_total, 0.0)

    out = pd.DataFrame({
        "Region": region_code,
        "hour": df["hour"],
        **{f"gen_{k}": fuel_gen[k] for k in FUEL_KEYS},
        "total_gen": total_gen,
        "renew_out": renew_gen,
        "nonrenew_out": total_gen - renew_gen,
        "total_emissions_tco2": total_emissions / 1000.0,
        "renew_share": renew_share,
        "cif_real_gco2_per_kwh": cif_gco2_kwh,
        "cif_real_tco2_per_mwh": cif_gco2_kwh / 1000.0,
    })
    # Clean: drop zero-generation hours, resample to strict hourly grid.
    out = out[out["total_gen"] > 0].reset_index(drop=True)
    out = out.drop_duplicates(subset=["hour"], keep="first")
    out = out.set_index("hour").resample("h").first().reset_index()
    out = out.dropna(subset=["renew_share"])
    return out


def compute_fuel_shares(df: pd.DataFrame, train_fraction=TRAIN_FRACTION) -> dict:
    """Annual mean fuel-share vector over the training split.

    Returns ``{fuel_key: share}`` summing to ~1.0.  Training-split aware so the
    config vector stays leak-free (consistent with loaders.py post-leak-fix).
    """
    split = int(len(df) * train_fraction)
    gen_cols = [f"gen_{k}" for k in FUEL_KEYS]
    gen_train = df[gen_cols].iloc[:split].values  # (T, 7)
    mean_gen = gen_train.mean(axis=0)  # (7,)
    total = mean_gen.sum()
    if total <= 0:
        return {k: 0.0 for k in FUEL_KEYS}
    return {k: float(mean_gen[i] / total) for i, k in enumerate(FUEL_KEYS)}


def main():
    import argparse
    global YEAR
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    args = ap.parse_args()
    YEAR = args.year
    print("=" * 70)
    print("EIA-930 Fuel Breakdown Extraction (Stage A.1)")
    print("=" * 70)

    df_all = load_raw_eia()
    print(f"  Total rows: {len(df_all):,}")

    ba_col = None
    for cand in ("Balancing Authority", "BA Code", "Respondent"):
        if cand in df_all.columns:
            ba_col = cand
            break
    if ba_col is None:
        raise KeyError(f"Cannot find BA column. Cols: {list(df_all.columns)[:15]}")
    print(f"  BA column: {ba_col!r}")

    fuel_shares_all = {}
    for region in US_REGIONS:
        if region not in df_all[ba_col].unique():
            print(f"  ⚠ {region} not in data, skipping")
            continue
        print(f"\n  Processing {region}...")
        df_region = df_all[df_all[ba_col] == region]
        out = process_region(df_region, region)

        outfile = FUEL_DIR / f"US_{region}_fuel_{YEAR}_hourly.csv"
        out.to_csv(outfile, index=False)
        print(f"    ✓ {outfile.name}: {len(out)} rows")

        shares = compute_fuel_shares(out)
        fuel_shares_all[f"US_{region}"] = shares

        # Sanity check against legacy CIF file.
        legacy_path = DATA_DIR / f"US_{region}_{YEAR}_hourly.csv"
        if legacy_path.exists():
            legacy = pd.read_csv(legacy_path)
            common = min(len(legacy), len(out))
            old = legacy["cif_real_gco2_per_kwh"].iloc[:common].values
            new = out["cif_real_gco2_per_kwh"].iloc[:common].values
            mae = np.mean(np.abs(old - new))
            print(f"    CIF recon vs legacy: MAE={mae:.2f} gCO2/kWh "
                  f"(expect <1.0; legacy used raw cols, we use Adjusted)")

        # Print fuel mix.
        mix = "  ".join(f"{k}:{v*100:4.1f}%" for k, v in shares.items() if v > 0.01)
        print(f"    fuel mix (train split): {mix}")

    # Write fuel shares JSON for constants.py injection (2023 = canonical;
    # extra years write side files so they never clobber the baseline).
    shares_file = FUEL_DIR / (
        "fuel_shares_us.json" if YEAR == 2023 else f"fuel_shares_us_{YEAR}.json")
    with open(shares_file, "w") as f:
        json.dump({
            "_emission_factors": FUEL_EMISSION_FACTORS,
            "_renewable_fuels": sorted(RENEWABLE_FUELS),
            "_fuel_order": FUEL_KEYS,
            "regions": fuel_shares_all,
        }, f, indent=2)
    print(f"\n[WRITE] {shares_file}")
    print(f"\nDone. {len(fuel_shares_all)} US regions processed.")


if __name__ == "__main__":
    main()
