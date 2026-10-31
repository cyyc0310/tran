#!/usr/bin/env python3
"""Audit hourly EPA CEMS intensity variation around each unit-month mean.

Only rows whose CO2 mass measurement indicator is explicitly Measured are
included. EPA CEMS mass and gross load are rates over the operating interval,
so both are multiplied by operating time before forming unit-month means.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import zipfile

import pandas as pd


COLS = [
    "State",
    "Facility ID",
    "unit_id",
    "Date",
    "Operating Time",
    "Gross Load (MW)",
    "CO2 Mass (short tons)",
    "CO2 Mass Measure Indicator",
    "Primary Fuel Type",
]
MEASURED_CODES = {"MEASURED", "MEASURE"}
SHORT_TON_MWH_TO_G_KWH = 907.18474


def _month(values: pd.Series) -> pd.Series:
    dates = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
    return dates.dt.strftime("%Y-%m")


def _iter_chunks(archive: Path, chunksize: int):
    with zipfile.ZipFile(archive) as zipped:
        members = sorted(n for n in zipped.namelist() if n.endswith(".csv"))
        for member in members:
            with zipped.open(member) as stream:
                for chunk in pd.read_csv(
                    stream,
                    usecols=COLS,
                    chunksize=chunksize,
                    low_memory=False,
                ):
                    chunk["CO2 Mass Measure Indicator"] = (
                        chunk["CO2 Mass Measure Indicator"].astype("string").str.upper().str.strip()
                    )
                    chunk = chunk[chunk["CO2 Mass Measure Indicator"].isin(MEASURED_CODES)].copy()
                    if chunk.empty:
                        continue
                    for col in ("Operating Time", "Gross Load (MW)", "CO2 Mass (short tons)"):
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                    chunk["month"] = _month(chunk["Date"])
                    chunk["fuel"] = chunk["Primary Fuel Type"].astype("string").str.strip()
                    chunk["facility"] = chunk["Facility ID"].astype("string").str.strip()
                    chunk["unit"] = chunk["unit_id"].astype("string").str.strip()
                    valid = (
                        chunk["Operating Time"].gt(0)
                        & chunk["Gross Load (MW)"].gt(0)
                        & chunk["CO2 Mass (short tons)"].gt(0)
                        & chunk["month"].notna()
                        & chunk["fuel"].notna()
                        & chunk["unit"].notna()
                        & chunk["State"].notna()
                    )
                    yield member, chunk.loc[valid].copy()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(archive: Path, outdir: Path, chunksize: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    unit_month = defaultdict(lambda: [0.0, 0.0, 0])
    fuel_quarter = defaultdict(lambda: [0.0, 0.0])
    quarter_counts = defaultdict(int)
    measured_rows = 0
    valid_rows = 0

    for member, frame in _iter_chunks(archive, chunksize):
        measured_rows += len(frame)
        quarter = member.split("q")[-1].split(".")[0]
        quarter_counts[quarter] += len(frame)
        frame["mwh"] = frame["Gross Load (MW)"] * frame["Operating Time"]
        frame["tons"] = frame["CO2 Mass (short tons)"] * frame["Operating Time"]
        valid_rows += len(frame)
        grouped = frame.groupby(
            ["State", "facility", "unit", "month", "fuel"], observed=True
        ).agg(tons=("tons", "sum"), mwh=("mwh", "sum"), n=("mwh", "size"))
        for key, row in grouped.iterrows():
            acc = unit_month[key]
            acc[0] += float(row["tons"])
            acc[1] += float(row["mwh"])
            acc[2] += int(row["n"])
        pooled = frame.groupby("fuel", observed=True)[["tons", "mwh"]].sum()
        for fuel, row in pooled.iterrows():
            acc = fuel_quarter[(fuel, f"2025q{quarter}")]
            acc[0] += float(row["tons"])
            acc[1] += float(row["mwh"])

    unit_means = {key: tons / mwh for key, (tons, mwh, _) in unit_month.items() if mwh > 0}
    pooled_means = {key: tons / mwh for key, (tons, mwh) in fuel_quarter.items() if mwh > 0}
    between_unit = defaultdict(lambda: [0.0, 0.0])
    for key, (tons, mwh, _) in unit_month.items():
        if mwh <= 0:
            continue
        state, facility, unit, month, fuel = key
        quarter = (int(month[-2:]) - 1) // 3 + 1
        period = f"2025q{quarter}"
        pooled = pooled_means.get((fuel, period))
        if pooled and abs((tons / mwh) / pooled - 1.0) > 0.10:
            between_unit[(fuel, period)][0] += mwh
        between_unit[(fuel, period)][1] += mwh
    summary = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # [rows, operating hours, MWh, unit-month hours>10%, unit-month MWh>10%,
    #  fuel-quarter hours>10%, fuel-quarter MWh>10%]
    for member, frame in _iter_chunks(archive, chunksize):
        quarter = member.split("q")[-1].split(".")[0]
        frame["mwh"] = frame["Gross Load (MW)"] * frame["Operating Time"]
        frame["tons"] = frame["CO2 Mass (short tons)"] * frame["Operating Time"]
        frame["rate"] = frame["CO2 Mass (short tons)"] / frame["Gross Load (MW)"]
        keys = list(zip(frame["State"], frame["facility"], frame["unit"], frame["month"], frame["fuel"]))
        frame["unit_month_mean"] = [unit_means.get(k, float("nan")) for k in keys]
        frame["outside_10"] = (frame["rate"] / frame["unit_month_mean"] - 1.0).abs().gt(0.10)
        frame["fuel_quarter_mean"] = [pooled_means.get((f, f"2025q{quarter}"), float("nan")) for f in frame["fuel"]]
        frame["outside_pooled_10"] = (frame["rate"] / frame["fuel_quarter_mean"] - 1.0).abs().gt(0.10)
        for fuel, group in frame.groupby("fuel", observed=True):
            a = summary[(fuel, f"2025q{quarter}")]
            a[0] += len(group)
            a[1] += float(group["Operating Time"].sum())
            a[2] += float(group["mwh"].sum())
            out_unit = group[group["outside_10"]]
            out_pooled = group[group["outside_pooled_10"]]
            a[3] += float(out_unit["Operating Time"].sum())
            a[4] += float(out_unit["mwh"].sum())
            a[5] += float(out_pooled["Operating Time"].sum())
            a[6] += float(out_pooled["mwh"].sum())

    rows = []
    for (fuel, period), (n, hours, mwh, unit_out_hours, unit_out_mwh, pooled_out_hours, pooled_out_mwh) in sorted(summary.items()):
        rows.append({
            "primary_fuel_type": fuel,
            "period": period,
            "measured_unit_hours": n,
            "operating_hours": hours,
            "gross_generation_mwh": mwh,
            "hours_outside_unit_month_mean_10pct": unit_out_hours,
            "mwh_outside_unit_month_mean_10pct": unit_out_mwh,
            "hour_share_outside_unit_month_10pct": unit_out_hours / hours if hours else None,
            "generation_share_outside_unit_month_10pct": unit_out_mwh / mwh if mwh else None,
            "fuel_quarter_pooled_mean_short_ton_mwh": pooled_means.get((fuel, period)),
            "fuel_quarter_pooled_mean_g_kwh": pooled_means.get((fuel, period), 0.0) * SHORT_TON_MWH_TO_G_KWH,
            "hours_outside_fuel_quarter_mean_10pct": pooled_out_hours,
            "mwh_outside_fuel_quarter_mean_10pct": pooled_out_mwh,
            "hour_share_outside_fuel_quarter_10pct": pooled_out_hours / hours if hours else None,
            "generation_share_outside_fuel_quarter_10pct": pooled_out_mwh / mwh if mwh else None,
            "generation_share_unit_month_mean_outside_fuel_quarter_10pct": (
                between_unit[(fuel, period)][0] / between_unit[(fuel, period)][1]
                if between_unit[(fuel, period)][1] else None
            ),
        })
    outpath = outdir / "fuel_hourly_deviation_summary.csv"
    pd.DataFrame(rows).to_csv(outpath, index=False, quoting=csv.QUOTE_MINIMAL)
    metadata = {
        "source_archive": str(archive),
        "source_sha256": sha256(archive),
        "measurement_codes_included": sorted(MEASURED_CODES),
        "validity": "positive operating time, gross load, and CO2 mass; measured CO2 code only",
        "ef_formula": "CO2 short tons / gross load MW; multiplying both hourly rates by operating time cancels in the ratio",
        "normalization": "each unit-hour compared to its own fuel-specific unit-month generation-weighted CO2 intensity",
        "uncertainty_threshold": 0.10,
        "measured_rows_after_validity_filter": measured_rows,
        "measured_rows_by_quarter": dict(sorted(quarter_counts.items())),
        "unit_month_groups": len(unit_means),
        "summary_csv": str(outpath),
    }
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/external-cems/epacems-2025.zip"))
    parser.add_argument("--outdir", type=Path, default=Path("results/v245-cems-ef-audit"))
    parser.add_argument("--chunksize", type=int, default=500_000)
    args = parser.parse_args()
    result = run(args.archive, args.outdir, args.chunksize)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
