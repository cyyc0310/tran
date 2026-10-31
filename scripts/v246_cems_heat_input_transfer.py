#!/usr/bin/env python3
"""Leave-one-balancing-authority-out test of monthly fuel heat-input value."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections import defaultdict
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd


MEASURED = {"MEASURED", "MEASURE"}
COLS = [
    "State", "Facility ID", "Date", "Operating Time", "Gross Load (MW)",
    "CO2 Mass (short tons)", "CO2 Mass Measure Indicator", "Primary Fuel Type",
    "Heat Input (mmBtu)", "Heat Input Measure Indicator",
]


def _clean_id(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype("string")


def _plant_to_ba(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith("Plant_Y2025.xlsx"))
        frame = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=0, header=1,
                              usecols=["Plant Code", "Balancing Authority Code"])
    frame["facility"] = _clean_id(frame["Plant Code"])
    frame["ba"] = frame["Balancing Authority Code"].astype("string").str.strip()
    frame = frame.dropna(subset=["facility", "ba"])
    # Drop any EIA plant ID with conflicting BA assignments rather than guessing.
    counts = frame.groupby("facility", observed=True)["ba"].nunique()
    unambiguous = set(counts[counts == 1].index)
    frame = frame[frame["facility"].isin(unambiguous)].drop_duplicates("facility")
    return dict(zip(frame["facility"].astype(str), frame["ba"].astype(str)))


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _read_measurement(archive: Path, plant_map: dict[str, str], chunksize: int):
    with zipfile.ZipFile(archive) as z:
        for member in sorted(n for n in z.namelist() if n.endswith(".csv")):
            with z.open(member) as stream:
                for chunk in pd.read_csv(stream, usecols=COLS, chunksize=chunksize, low_memory=False):
                    for col in ("CO2 Mass Measure Indicator", "Heat Input Measure Indicator"):
                        chunk[col] = chunk[col].astype("string").str.upper().str.strip()
                    chunk = chunk[
                        chunk["CO2 Mass Measure Indicator"].isin(MEASURED)
                        & chunk["Heat Input Measure Indicator"].isin(MEASURED)
                    ].copy()
                    if chunk.empty:
                        continue
                    for col in ("Operating Time", "Gross Load (MW)", "CO2 Mass (short tons)", "Heat Input (mmBtu)"):
                        chunk[col] = pd.to_numeric(chunk[col], errors="coerce")
                    date = pd.to_datetime(chunk["Date"], format="%Y-%m-%d", errors="coerce")
                    chunk["month"] = date.dt.strftime("%Y-%m")
                    chunk["facility"] = _clean_id(chunk["Facility ID"])
                    chunk["ba"] = chunk["facility"].map(plant_map)
                    chunk["fuel"] = chunk["Primary Fuel Type"].astype("string").str.strip()
                    valid = (
                        chunk["Operating Time"].gt(0)
                        & chunk["Gross Load (MW)"].gt(0)
                        & chunk["CO2 Mass (short tons)"].gt(0)
                        & chunk["Heat Input (mmBtu)"].gt(0)
                        & chunk["month"].notna()
                        & chunk["ba"].notna()
                        & chunk["fuel"].notna()
                        & ~chunk["fuel"].str.contains(",", regex=False)
                    )
                    yield member, chunk.loc[valid].copy()


def run(archive: Path, eia860: Path, outdir: Path, chunksize: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    plant_map = _plant_to_ba(eia860)
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    quarter_rows = defaultdict(int)
    included_rows = 0
    for member, frame in _read_measurement(archive, plant_map, chunksize):
        quarter = member.rsplit("q", 1)[-1].split(".")[0]
        quarter_rows[quarter] += len(frame)
        included_rows += len(frame)
        op = frame["Operating Time"]
        frame["mwh"] = frame["Gross Load (MW)"] * op
        frame["tons"] = frame["CO2 Mass (short tons)"] * op
        frame["mmbtu"] = frame["Heat Input (mmBtu)"] * op
        group = frame.groupby(["ba", "month", "fuel"], observed=True).agg(
            tons=("tons", "sum"), mwh=("mwh", "sum"), mmbtu=("mmbtu", "sum"), n=("mwh", "size")
        )
        for key, row in group.iterrows():
            a = totals[key]
            a[0] += float(row.tons)
            a[1] += float(row.mwh)
            a[2] += float(row.mmbtu)
            a[3] += int(row.n)

    records = [
        {"ba": ba, "month": month, "fuel": fuel, "tons": tons, "mwh": mwh,
         "mmbtu": mmbtu, "unit_hours": n}
        for (ba, month, fuel), (tons, mwh, mmbtu, n) in totals.items()
        if mwh > 0 and mmbtu > 0
    ]
    frame = pd.DataFrame(records)
    frame["true_g_kwh"] = frame["tons"] * 907184.74 / (frame["mwh"] * 1000.0)
    frame["true_ton_mmbtu"] = frame["tons"] / frame["mmbtu"]
    frame["pred_gen_tons"] = np.nan
    frame["pred_heat_tons"] = np.nan
    # Leave one complete BA out. Donor coefficients are estimated from other BAs
    # observed in the same calendar month, one record per BA/fuel/month.
    for idx, row in frame.iterrows():
        donor = frame[(frame.ba != row.ba) & (frame.month == row.month) & (frame.fuel == row.fuel)]
        if donor.empty:
            continue
        gen_ef = donor.tons.sum() / donor.mwh.sum()
        heat_ef = donor.tons.sum() / donor.mmbtu.sum()
        frame.at[idx, "pred_gen_tons"] = row.mwh * gen_ef
        frame.at[idx, "pred_heat_tons"] = row.mmbtu * heat_ef

    # Aggregate only fuel cells for which both fair, paired donor estimates exist.
    valid = frame.dropna(subset=["pred_gen_tons", "pred_heat_tons"])
    ba_month = valid.groupby(["ba", "month"], observed=True).agg(
        true_tons=("tons", "sum"), gen_pred_tons=("pred_gen_tons", "sum"),
        heat_pred_tons=("pred_heat_tons", "sum"), observed_mwh=("mwh", "sum"),
        observed_mmbtu=("mmbtu", "sum"), supported_fuels=("fuel", "nunique"),
        cems_unit_hours=("unit_hours", "sum"),
    ).reset_index()
    ba_month["gen_intensity_error_g_kwh"] = (
        (ba_month.gen_pred_tons - ba_month.true_tons) / ba_month.observed_mwh * 907184.74 / 1000.0
    )
    ba_month["heat_intensity_error_g_kwh"] = (
        (ba_month.heat_pred_tons - ba_month.true_tons) / ba_month.observed_mwh * 907184.74 / 1000.0
    )
    ba_month["gen_abs_error_pct"] = (ba_month.gen_pred_tons / ba_month.true_tons - 1).abs() * 100
    ba_month["heat_abs_error_pct"] = (ba_month.heat_pred_tons / ba_month.true_tons - 1).abs() * 100
    frame.to_csv(outdir / "ba_month_fuel_transfer.csv", index=False)
    ba_month.to_csv(outdir / "ba_month_transfer_metrics.csv", index=False)

    supported_fraction = float(valid.mwh.sum() / frame.mwh.sum()) if len(frame) else 0.0
    metrics = {
        "ba_count": int(frame.ba.nunique()),
        "ba_month_count_with_supported_donors": int(len(ba_month)),
        "ba_fuel_month_rows": int(len(frame)),
        "donor_supported_generation_share": supported_fraction,
        "paired_ba_months": int(len(ba_month)),
        "generation_ef_donor_median_abs_error_pct": float(ba_month.gen_abs_error_pct.median()),
        "heat_input_donor_median_abs_error_pct": float(ba_month.heat_abs_error_pct.median()),
        "generation_ef_donor_mean_abs_error_pct": float(ba_month.gen_abs_error_pct.mean()),
        "heat_input_donor_mean_abs_error_pct": float(ba_month.heat_abs_error_pct.mean()),
        "generation_ef_donor_median_abs_intensity_error_g_kwh": float(ba_month.gen_intensity_error_g_kwh.abs().median()),
        "heat_input_donor_median_abs_intensity_error_g_kwh": float(ba_month.heat_intensity_error_g_kwh.abs().median()),
        "generation_ef_donor_mean_abs_intensity_error_g_kwh": float(ba_month.gen_intensity_error_g_kwh.abs().mean()),
        "heat_input_donor_mean_abs_intensity_error_g_kwh": float(ba_month.heat_intensity_error_g_kwh.abs().mean()),
        "measured_rows_by_quarter_after_mapping_and_filters": dict(sorted(quarter_rows.items())),
        "paired_row_count": int(included_rows),
        "cems_archive_sha256": _sha256(archive),
        "eia860_archive_sha256": _sha256(eia860),
        "cems_measurement_codes": sorted(MEASURED),
        "fuel_scope": "pure primary-fuel labels only; comma-separated multi-fuel labels excluded",
        "estimand": "monthly CO2 intensity of mapped CEMS-covered fossil fleet, not whole-grid CIF",
    }
    (outdir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path("data/external-cems/epacems-2025.zip"))
    p.add_argument("--eia860", type=Path, default=Path("data/research-v87/eia8602025.zip"))
    p.add_argument("--outdir", type=Path, default=Path("results/v246-cems-heat-input-transfer"))
    p.add_argument("--chunksize", type=int, default=500_000)
    a = p.parse_args()
    print(json.dumps(run(a.archive, a.eia860, a.outdir, a.chunksize), indent=2))


if __name__ == "__main__":
    main()
