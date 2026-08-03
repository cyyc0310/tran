"""Faithfully regenerate the AEMO/NEMED per-region hourly carbon-intensity dataset that
originally lived under /tmp/nemed_output (now lost). Emissions are computed by NEMED
(nemed 0.3.3) from AEMO MMS DISPATCH_UNIT_SCADA + generator emission factors -- exactly the
original source. We only reconstruct the *aggregation* the original export did:

  1. Pull DUID-level 5-min emissions (`get_total_emissions_by_DI_DUID(return_all=True)`).
  2. Classify each DUID renewable iff its NEMED Plant_Emissions_Intensity <= 0 (wind/solar/
     hydro/battery carry a 0 factor in NEMED's CDEII table; thermal carries >0). This is the
     same binary renew/non-renew split the physics layer (physics/cif.py) assumes.
  3. Aggregate sent-out energy (Energy_SO) to hourly renew_out / nonrenew_out and sum
     Total_Emissions, then derive renew_share and the real CIF.

Environment quirk this script works around: NEMED fetches the generator registration
(DUDETAILSUMMARY / genset_map) as of `now - 90 days`. This machine's clock is set to 2026,
so that date's AEMO archive does not exist yet on AEMO's servers. We pin the registration
snapshot to a date *inside the data year* (ASOF_DATE), which is also more correct for 2023
emissions than a 2026 registration. No emissions math is altered.

Run inside the 3.11 NEMED venv:
  .venv-nemed311/bin/python scripts/generate_nemed_regions.py --validate   # check vs fixture
  .venv-nemed311/bin/python scripts/generate_nemed_regions.py --year 2023  # full generation
"""

import argparse
import os

import pandas as pd

import nemed.downloader as dl
import nemed.process as proc

CACHE = ".nemed_cache"
OUT_DIR = "data_2023"
ASOF_DATE = "2023/12/01 00:00"  # registration snapshot inside the data year
REGIONS = ["QLD1", "NSW1", "VIC1", "SA1", "TAS1"]

# --- pin the generator-registration snapshot (see module docstring) ------------------------
_orig_dds = dl.download_dudetailsummary
_orig_gm = dl.download_genset_map
proc.download_dudetailsummary = lambda cache, asof_date=None: _orig_dds(cache, asof_date=ASOF_DATE)
proc.download_genset_map = lambda cache, asof_date=None: _orig_gm(cache, asof_date=ASOF_DATE)


def fetch_region_hourly(region: str, start: str, end: str) -> pd.DataFrame:
    """Return an hourly frame for one NEM region with the fixture's column schema."""
    raw = proc.get_total_emissions_by_DI_DUID(
        start, end, CACHE, filter_regions=[region],
        generation_sent_out=True, return_all=True,
    )
    r = raw[["DUID", "Time", "Plant_Emissions_Intensity", "Energy_SO", "Total_Emissions"]].copy()
    is_renew = r["Plant_Emissions_Intensity"] <= 0.0
    r["renew_out"] = r["Energy_SO"].where(is_renew, 0.0)
    r["nonrenew_out"] = r["Energy_SO"].where(~is_renew, 0.0)
    # hour label = interval-ending floored to the hour (matches the original SA1 fixture)
    r["hour"] = r["Time"].dt.floor("h")

    g = (
        r.groupby("hour")
        .agg(
            renew_out=("renew_out", "sum"),
            nonrenew_out=("nonrenew_out", "sum"),
            total_emissions=("Total_Emissions", "sum"),
        )
        .reset_index()
    )
    g["total_energy_so"] = g["renew_out"] + g["nonrenew_out"]
    g = g[g["total_energy_so"] > 0].copy()
    g["renew_share"] = g["renew_out"] / g["total_energy_so"]
    g["cif_real_tco2_per_mwh"] = g["total_emissions"] / g["total_energy_so"]
    g["cif_real_gco2_per_kwh"] = g["cif_real_tco2_per_mwh"] * 1000.0
    g.insert(0, "Region", region)
    return g[[
        "Region", "hour", "nonrenew_out", "renew_out", "total_emissions",
        "total_energy_so", "renew_share", "cif_real_tco2_per_mwh", "cif_real_gco2_per_kwh",
    ]]


def validate_against_fixture() -> None:
    """Regenerate SA1 over the fixture's window and compare the scale-invariant quantities
    (renew_share, CIF) the model actually consumes."""
    fixture = pd.read_csv(
        "tests/fixtures/real_aemo_sample_sa1.csv", parse_dates=["hour"]
    )
    start = fixture["hour"].min().strftime("%Y/%m/%d %H:%M")
    end = (fixture["hour"].max() + pd.Timedelta(minutes=55)).strftime("%Y/%m/%d %H:%M")
    gen = fetch_region_hourly("SA1", start, end)

    merged = fixture.merge(gen, on="hour", suffixes=("_fix", "_gen"))
    rs_mae = (merged["renew_share_fix"] - merged["renew_share_gen"]).abs().mean()
    cif_mae = (merged["cif_real_gco2_per_kwh_fix"] - merged["cif_real_gco2_per_kwh_gen"]).abs().mean()
    energy_ratio = (merged["total_energy_so_gen"] / merged["total_energy_so_fix"]).mean()
    print(f"matched hours: {len(merged)} / {len(fixture)}")
    print(f"renew_share MAE vs fixture: {rs_mae:.5f}")
    print(f"CIF (gCO2/kWh) MAE vs fixture: {cif_mae:.4f}")
    print(f"mean energy ratio gen/fixture: {energy_ratio:.4f} (scale-invariant-normalized in loaders)")


def generate_full(year: int) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    for region in REGIONS:
        out_path = os.path.join(OUT_DIR, f"{region}_{year}_hourly.csv")
        if os.path.exists(out_path):
            print(f"skip {region}: {out_path} exists")
            continue
        frames = []
        for month in range(1, 13):
            m_start = f"{year}/{month:02d}/01 00:00"
            nm_year, nm_month = (year + 1, 1) if month == 12 else (year, month + 1)
            m_end = f"{nm_year}/{nm_month:02d}/01 00:00"
            print(f"[{region}] {m_start} .. {m_end}")
            frames.append(fetch_region_hourly(region, m_start, m_end))
        df = pd.concat(frames, ignore_index=True).sort_values("hour").reset_index(drop=True)
        df.to_csv(out_path, index=False)
        print(f"wrote {out_path}: {len(df)} hours")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true", help="check SA1 vs the fixture and exit")
    ap.add_argument("--year", type=int, default=2023)
    args = ap.parse_args()
    if args.validate:
        validate_against_fixture()
    else:
        generate_full(args.year)


if __name__ == "__main__":
    main()
