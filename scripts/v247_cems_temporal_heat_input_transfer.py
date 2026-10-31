#!/usr/bin/env python3
"""Future-year confirmation: 2024 CEMS donor coefficients to 2025 BA targets."""

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


def clean_id(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype("string")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def plant_to_ba(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if "Plant_Y" in n and n.endswith(".xlsx"))
        x = pd.read_excel(io.BytesIO(z.read(name)), sheet_name=0, header=1,
                          usecols=["Plant Code", "Balancing Authority Code"])
    x["facility"] = clean_id(x["Plant Code"])
    x["ba"] = x["Balancing Authority Code"].astype("string").str.strip()
    x = x.dropna(subset=["facility", "ba"])
    counts = x.groupby("facility", observed=True).ba.nunique()
    x = x[x.facility.isin(set(counts[counts == 1].index))].drop_duplicates("facility")
    return dict(zip(x.facility.astype(str), x.ba.astype(str)))


def aggregate(archive: Path, eia860: Path, chunksize: int) -> tuple[pd.DataFrame, dict]:
    mapping = plant_to_ba(eia860)
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    qcounts = defaultdict(int)
    nrows = 0
    with zipfile.ZipFile(archive) as z:
        for member in sorted(n for n in z.namelist() if n.endswith(".csv")):
            with z.open(member) as stream:
                for c in pd.read_csv(stream, usecols=COLS, chunksize=chunksize, low_memory=False):
                    for name in ("CO2 Mass Measure Indicator", "Heat Input Measure Indicator"):
                        c[name] = c[name].astype("string").str.upper().str.strip()
                    c = c[c["CO2 Mass Measure Indicator"].isin(MEASURED)
                          & c["Heat Input Measure Indicator"].isin(MEASURED)].copy()
                    if c.empty:
                        continue
                    for name in ("Operating Time", "Gross Load (MW)", "CO2 Mass (short tons)", "Heat Input (mmBtu)"):
                        c[name] = pd.to_numeric(c[name], errors="coerce")
                    c["month"] = pd.to_datetime(c.Date, format="%Y-%m-%d", errors="coerce").dt.strftime("%m")
                    c["facility"] = clean_id(c["Facility ID"])
                    c["ba"] = c.facility.map(mapping)
                    c["fuel"] = c["Primary Fuel Type"].astype("string").str.strip()
                    ok = (c["Operating Time"].gt(0) & c["Gross Load (MW)"].gt(0)
                          & c["CO2 Mass (short tons)"].gt(0) & c["Heat Input (mmBtu)"].gt(0)
                          & c.month.notna() & c.ba.notna() & c.fuel.notna()
                          & ~c.fuel.str.contains(",", regex=False))
                    c = c.loc[ok].copy()
                    nrows += len(c)
                    qcounts[member.rsplit("q", 1)[-1].split(".")[0]] += len(c)
                    op = c["Operating Time"]
                    c["tons"] = c["CO2 Mass (short tons)"] * op
                    c["mwh"] = c["Gross Load (MW)"] * op
                    c["mmbtu"] = c["Heat Input (mmBtu)"] * op
                    g = c.groupby(["ba", "month", "fuel"], observed=True).agg(
                        tons=("tons", "sum"), mwh=("mwh", "sum"), mmbtu=("mmbtu", "sum"), n=("mwh", "size")
                    )
                    for key, row in g.iterrows():
                        a = totals[key]
                        a[0] += float(row.tons)
                        a[1] += float(row.mwh)
                        a[2] += float(row.mmbtu)
                        a[3] += int(row.n)
    frame = pd.DataFrame([
        {"ba": ba, "month": month, "fuel": fuel, "tons": t, "mwh": g, "mmbtu": h, "unit_hours": n}
        for (ba, month, fuel), (t, g, h, n) in totals.items() if g > 0 and h > 0
    ])
    return frame, {"rows": nrows, "quarter_rows": dict(sorted(qcounts.items())), "ba_count": int(frame.ba.nunique())}


def run(donor_archive: Path, donor_eia860: Path, target_archive: Path,
        target_eia860: Path, outdir: Path, chunksize: int) -> dict:
    outdir.mkdir(parents=True, exist_ok=True)
    donor, donor_meta = aggregate(donor_archive, donor_eia860, chunksize)
    target, target_meta = aggregate(target_archive, target_eia860, chunksize)
    donor_groups = donor.groupby(["ba", "month", "fuel"], observed=True)[["tons", "mwh", "mmbtu"]].sum()
    out = []
    for row in target.itertuples(index=False):
        try:
            pool = donor_groups.loc[(slice(None), row.month, row.fuel), :].reset_index()
        except KeyError:
            continue
        pool = pool[pool.ba != row.ba]
        if pool.empty or pool.mwh.sum() <= 0 or pool.mmbtu.sum() <= 0:
            continue
        gen_ef = pool.tons.sum() / pool.mwh.sum()
        heat_ef = pool.tons.sum() / pool.mmbtu.sum()
        out.append({
            "ba": row.ba, "month": row.month, "fuel": row.fuel,
            "target_tons": row.tons, "target_mwh": row.mwh, "target_mmbtu": row.mmbtu,
            "generation_prediction_tons": row.mwh * gen_ef,
            "heat_prediction_tons": row.mmbtu * heat_ef,
            "donor_ba_count": int(pool.ba.nunique()), "target_unit_hours": row.unit_hours,
        })
    by_fuel = pd.DataFrame(out)
    by_fuel.to_csv(outdir / "target_2025_ba_month_fuel_predictions.csv", index=False)
    bm = by_fuel.groupby(["ba", "month"], observed=True).agg(
        target_tons=("target_tons", "sum"), target_mwh=("target_mwh", "sum"),
        pred_generation_tons=("generation_prediction_tons", "sum"),
        pred_heat_tons=("heat_prediction_tons", "sum"),
        fuels=("fuel", "nunique"), donor_ba_min=("donor_ba_count", "min"),
    ).reset_index()
    bm["generation_ape_pct"] = (bm.pred_generation_tons / bm.target_tons - 1).abs() * 100
    bm["heat_ape_pct"] = (bm.pred_heat_tons / bm.target_tons - 1).abs() * 100
    bm["generation_intensity_error_g_kwh"] = (
        (bm.pred_generation_tons - bm.target_tons) / bm.target_mwh * 907184.74 / 1000
    )
    bm["heat_intensity_error_g_kwh"] = (
        (bm.pred_heat_tons - bm.target_tons) / bm.target_mwh * 907184.74 / 1000
    )
    bm.to_csv(outdir / "target_2025_ba_month_metrics.csv", index=False)
    support_mwh = by_fuel.target_mwh.sum()
    metrics = {
        "design": "2024 donor same calendar month/fuel -> 2025 target same BA/month; BA excluded from donor pool",
        "donor_archive_sha256": sha256(donor_archive), "target_archive_sha256": sha256(target_archive),
        "donor_eia860_sha256": sha256(donor_eia860), "target_eia860_sha256": sha256(target_eia860),
        "donor": donor_meta, "target": target_meta,
        "paired_target_ba_months": int(len(bm)), "paired_ba_count": int(bm.ba.nunique()),
        "paired_fuel_cells": int(len(by_fuel)),
        "target_generation_supported_share": float(support_mwh / target.mwh.sum()),
        "generation_median_ape_pct": float(bm.generation_ape_pct.median()),
        "heat_input_median_ape_pct": float(bm.heat_ape_pct.median()),
        "generation_mean_ape_pct": float(bm.generation_ape_pct.mean()),
        "heat_input_mean_ape_pct": float(bm.heat_ape_pct.mean()),
        "generation_wape_pct": float((bm.pred_generation_tons-bm.target_tons).abs().sum()/bm.target_tons.sum()*100),
        "heat_input_wape_pct": float((bm.pred_heat_tons-bm.target_tons).abs().sum()/bm.target_tons.sum()*100),
        "generation_median_abs_intensity_error_g_kwh": float(bm.generation_intensity_error_g_kwh.abs().median()),
        "heat_input_median_abs_intensity_error_g_kwh": float(bm.heat_intensity_error_g_kwh.abs().median()),
        "generation_mean_abs_intensity_error_g_kwh": float(bm.generation_intensity_error_g_kwh.abs().mean()),
        "heat_input_mean_abs_intensity_error_g_kwh": float(bm.heat_intensity_error_g_kwh.abs().mean()),
        "estimand": "CEMS-covered pure-primary-fuel fossil fleet gross-load monthly intensity; not whole-grid CIF",
    }
    (outdir / "summary.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor-archive", type=Path, default=Path("data/external-cems/epacems-2024.zip"))
    p.add_argument("--donor-eia860", type=Path, default=Path("data/research-v87/eia8602024.zip"))
    p.add_argument("--target-archive", type=Path, default=Path("data/external-cems/epacems-2025.zip"))
    p.add_argument("--target-eia860", type=Path, default=Path("data/research-v87/eia8602025.zip"))
    p.add_argument("--outdir", type=Path, default=Path("results/v247-cems-temporal-heat-transfer"))
    p.add_argument("--chunksize", type=int, default=500_000)
    a = p.parse_args()
    print(json.dumps(run(a.donor_archive, a.donor_eia860, a.target_archive, a.target_eia860,
                         a.outdir, a.chunksize), indent=2))


if __name__ == "__main__":
    main()
