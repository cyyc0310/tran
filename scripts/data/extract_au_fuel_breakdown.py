#!/usr/bin/env python
"""Extract per-fuel hourly generation for AU NEM regions via NEMED (FD-14).

Fuel classification is computed FROM THE DISPATCH FRAME ITSELF (the AEMO
registration table is Cloudflare-blocked):

    * name join  — nemed's bundled duid_mapping + existing_gen_data_summary
                   (station-name match; primary where available)
    * intensity  — Plant_Emissions_Intensity > 0.8 -> coal, (0, 0.8] -> gas
                   (CIF-safe: AU coal >= ~0.85 sent-out, gas <= ~0.7)
    * night test — zero-intensity DUIDs with <0.1% night energy -> solar
    * default    — remaining zero-intensity DUIDs -> wind (renewables are
                   all EF = 0, so intra-renewable confusion is CIF-neutral;
                   AU hydro ~6% national is absorbed into wind)

Outputs per region (2023): data_2023/fuel/{REGION}_fuel_2023_hourly.csv
(US schema: gen_{fuel}, total_gen) + fuel_shares_au.json (annual config).

Run inside the 3.11 NEMED venv:
    .venv-nemed311/bin/python scripts/data/extract_au_fuel_breakdown.py
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

import nemed.downloader as dl
import nemed.process as proc

CACHE = ".nemed_cache"
OUT_DIR = Path("data_2023/fuel")
ASOF_DATE = "2023/12/01 00:00"
REGIONS = ["QLD1", "NSW1", "VIC1", "SA1"]
CANON = ["coal", "gas", "petroleum", "nuclear", "hydro", "solar", "wind",
         "biomass", "imports", "other"]
RENEWABLES = ["hydro", "solar", "wind", "biomass"]
EFS = {"coal": 920.0, "gas": 500.0, "petroleum": 650.0, "nuclear": 0.0,
       "hydro": 0.0, "solar": 0.0, "wind": 0.0, "biomass": 0.0,
       "imports": 0.0, "other": 0.0}

_orig_dds = dl.download_dudetailsummary
_orig_gm = dl.download_genset_map
proc.download_dudetailsummary = lambda cache, asof_date=None: _orig_dds(cache, asof_date=ASOF_DATE)
proc.download_genset_map = lambda cache, asof_date=None: _orig_gm(cache, asof_date=ASOF_DATE)

NEMED_DATA = Path(proc.__file__).parent / "data"


def name_fuel_map():
    """Station-name join: DUID -> canonical fuel (primary source)."""
    mapping = pd.read_csv(NEMED_DATA / "duid_mapping.csv")
    gen = pd.read_csv(NEMED_DATA / "existing_gen_data_summary.csv")
    gen["clean"] = gen["Generator"].str.upper().str.strip()
    mapping["clean"] = mapping["GeneratorsandScheduledLoads_StationName"].str.upper().str.strip()
    joined = mapping.merge(gen[["clean", "Fuel/Technology type"]],
                           on="clean", how="left")

    def parse(s):
        if not isinstance(s, str):
            return None
        s = s.upper()
        if "COAL" in s:
            return "coal"
        if any(k in s for k in ("OCGT", "CCGT", "GAS")):
            return "gas"
        if "LIQUID" in s or "DIESEL" in s or "FUEL OIL" in s:
            return "petroleum"
        if "SOLAR" in s:
            return "solar"
        if "WIND" in s:
            return "wind"
        if "HYDRO" in s or "PUMP" in s:
            return "hydro"
        if "BATTERY" in s or "BESS" in s:
            return "other"
        return None

    joined["fuel"] = joined["Fuel/Technology type"].map(parse)
    out = joined.dropna(subset=["fuel"]).drop_duplicates("DUID")
    return dict(zip(out["DUID"], out["fuel"]))


# Curated unit registry (FD-28): confirmed dispatch-storage / hydro units
# that masquerade as wind in the zero-intensity default bucket.  These
# are station identities from public AEMO knowledge, not fitted rules —
# VIC1's MURRAY (Snowy hydro, 3.1 TWh = 6 % of state generation!) alone
# explains a quarter of its "wind" share; QLD's Wivenhoe PHS and the
# NSW batteries/PHS likewise fire at peaks and wind lulls.
REGISTRY_OVERRIDES = {
    "MURRAY": "hydro",            # Snowy Murray 1&2 (VIC1)
    "GUTHEGA": "hydro",           # Snowy Guthega (NSW1)
    "SHGEN": "hydro",             # Shoalhaven pumped storage (NSW1)
    "W/HOE#1": "hydro", "W/HOE#2": "hydro",   # Wivenhoe PHS (QLD1)
    "WALGRVG1": "other", "QBYNBG1": "other", "RIVNBG2": "other",
    "DPNTBG1": "other", "CAPBES1G": "other", "BHBG1": "other",  # BESS
}


def classify_duids(raw, name_map, wind_ref=None, region="", tz=10.0):
    # FD-25 (attempted, reverted): weather-correlation relabelling of the
    # default wind bucket needs a COMPLETE farm registry to calibrate —
    # the incomplete NSW table over-relabeled real wind (0.176 -> 0.095).
    # Storage misfiling (~0.4-4 % share) stands as a documented residue
    # until the AEMO registration list is obtainable.
    """DUID -> canonical fuel from dispatch behaviour + intensity bands."""
    g = raw.groupby("DUID").agg(
        total=("Energy_SO", "sum"),
        intensity=("Plant_Emissions_Intensity", "first"))
    # NEMED ``Time`` is already local (AEST) — verified: the name-mapped
    # solar fleet peaks at Time hours 08-16.  Night test on raw hours.
    night = (raw["Time"].dt.hour >= 21) | (raw["Time"].dt.hour <= 5)
    night_e = raw["Energy_SO"].where(night, 0.0).groupby(raw["DUID"]).sum()
    night_frac = (night_e / g["total"].clip(lower=1e-9))

    fuel = pd.Series("wind", index=g.index)          # renewable default
    fuel[g["intensity"] > 0.8] = "coal"
    fuel[(g["intensity"] > 0) & (g["intensity"] <= 0.8)] = "gas"
    solar_like = (g["intensity"] <= 0) & (night_frac < 0.02) & (g["total"] > 0)
    fuel[solar_like] = "solar"
    # Dispatch LOADS (pump/battery charging) carry negative/zero energy —
    # anything with (near-)zero net generation is excluded downstream.
    import re as _re
    duid_fuel = {}
    for d in g.index:
        u = d.upper()
        if d in REGISTRY_OVERRIDES:
            duid_fuel[d] = REGISTRY_OVERRIDES[d]
        elif "BESS" in u or _re.search(r"BG\d*$", u):
            duid_fuel[d] = "other"
    return fuel.to_dict() | name_map | duid_fuel


def load_wind_ref(region, year):
    """Hourly fleet wind-capacity-factor reference (UTC-indexed) for the
    weather-correlation classifier: the region's farmblend file when
    present, else its centroid weather; IEC transform of m/s wind."""
    root = Path(__file__).resolve().parent.parent.parent / "data_2023"
    farm = root / "weather" / f"{region}_farmblend_weather_2023_hourly.csv"
    path = farm if farm.exists() else (
        root / "weather" / f"{region}_weather_2023_hourly.csv")
    if not path.exists():
        return None
    w = pd.read_csv(path, parse_dates=["hour"])
    v_in, v_rated = 3.0, 12.0
    v = w["wind_speed_100m"].values / 3.6
    cf = np.clip((v ** 3 - v_in ** 3) / (v_rated ** 3 - v_in ** 3), 0, 1)
    cf = np.where((v >= v_rated) & (v <= 25.0), 1.0, cf)
    return pd.Series(cf.astype(float), index=pd.DatetimeIndex(w["hour"]))


def extract_region(region, year):
    frames = []
    for month in range(1, 13):
        m_start = f"{year}/{month:02d}/01 00:00"
        ny, nm = (year + 1, 1) if month == 12 else (year, month + 1)
        m_end = f"{ny}/{nm:02d}/01 00:00"
        print(f"[{region}] {m_start} .. {m_end}", flush=True)
        raw = proc.get_total_emissions_by_DI_DUID(
            m_start, m_end, CACHE, filter_regions=[region],
            generation_sent_out=True, return_all=True)
        frames.append(raw[["DUID", "Time", "Energy_SO",
                           "Plant_Emissions_Intensity"]])
    raw = pd.concat(frames, ignore_index=True)
    fuel_of = classify_duids(raw, name_fuel_map())
    raw["fuel"] = raw["DUID"].map(fuel_of).fillna("wind")
    raw = raw[raw["Energy_SO"] > 0]
    raw["hour"] = raw["Time"].dt.floor("h")
    piv = (raw.groupby(["hour", "fuel"])["Energy_SO"].sum()
           .unstack("fuel").fillna(0.0))
    for f in CANON:
        if f not in piv.columns:
            piv[f] = 0.0
    piv = piv[CANON]
    piv["total_gen"] = piv.sum(axis=1)
    piv = piv[piv["total_gen"] > 0]
    out = piv.reset_index()
    out.insert(0, "Region", region)
    for f in CANON:
        out[f"gen_{f}"] = out[f]
    return out[["Region", "hour"] + [f"gen_{f}" for f in CANON] + ["total_gen"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    args = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    shares = {}
    for region in args.regions:
        path = OUT_DIR / f"{region}_fuel_{args.year}_hourly.csv"
        if path.exists():
            print(f"[{region}] exists, skip")
            df = pd.read_csv(path)
        else:
            df = extract_region(region, args.year)
            df.to_csv(path, index=False)
            print(f"[{region}] wrote {path}: {len(df)} hours")
        split = int(len(df) * 0.8)
        ann = {f: float(df[f"gen_{f}"].iloc[:split].sum()
                        / df["total_gen"].iloc[:split].sum())
               for f in CANON}
        shares[region] = ann
        print(f"[{region}] annual: " +
              ", ".join(f"{k}={v:.3f}" for k, v in ann.items() if v > 0.005),
              flush=True)

    if set(args.regions) == set(REGIONS):
        doc = {
            "_emission_factors": EFS,
            "_renewable_fuels": RENEWABLES,
            "_fuel_order": CANON,
            "regions": shares,
        }
        with open(OUT_DIR / "fuel_shares_au.json", "w") as f:
            json.dump(doc, f, indent=2)
        print(f"wrote {OUT_DIR / 'fuel_shares_au.json'}")


if __name__ == "__main__":
    main()
