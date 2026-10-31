#!/usr/bin/env python3
"""Test sparse local measured days for calibrating transferable CO2/heat factors."""

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


def clean_id(values):
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype("string")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def map_plants(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if "Plant_Y" in n and n.endswith(".xlsx"))
        d = pd.read_excel(io.BytesIO(z.read(name)), header=1,
                          usecols=["Plant Code", "Balancing Authority Code"])
    d["facility"] = clean_id(d["Plant Code"])
    d["ba"] = d["Balancing Authority Code"].astype("string").str.strip()
    d = d.dropna(subset=["facility", "ba"])
    count = d.groupby("facility", observed=True).ba.nunique()
    d = d[d.facility.isin(set(count[count == 1].index))].drop_duplicates("facility")
    return dict(zip(d.facility.astype(str), d.ba.astype(str)))


def daily_data(archive, eia860, chunksize):
    plant_map = map_plants(eia860)
    totals = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
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
                    dt = pd.to_datetime(c.Date, format="%Y-%m-%d", errors="coerce")
                    c["month"] = dt.dt.strftime("%m")
                    c["day"] = dt.dt.strftime("%Y-%m-%d")
                    c["facility"] = clean_id(c["Facility ID"])
                    c["ba"] = c.facility.map(plant_map)
                    c["fuel"] = c["Primary Fuel Type"].astype("string").str.strip()
                    ok = (c["Operating Time"].gt(0) & c["Gross Load (MW)"].gt(0)
                          & c["CO2 Mass (short tons)"].gt(0) & c["Heat Input (mmBtu)"].gt(0)
                          & c.month.notna() & c.day.notna() & c.ba.notna() & c.fuel.notna()
                          & ~c.fuel.str.contains(",", regex=False))
                    c = c.loc[ok].copy()
                    op = c["Operating Time"]
                    c["tons"] = c["CO2 Mass (short tons)"] * op
                    c["mwh"] = c["Gross Load (MW)"] * op
                    c["mmbtu"] = c["Heat Input (mmBtu)"] * op
                    g = c.groupby(["ba", "month", "fuel", "day"], observed=True).agg(
                        tons=("tons", "sum"), mwh=("mwh", "sum"), mmbtu=("mmbtu", "sum"), n=("mwh", "size")
                    )
                    for key, row in g.iterrows():
                        a = totals[key]
                        a[0] += float(row.tons)
                        a[1] += float(row.mwh)
                        a[2] += float(row.mmbtu)
                        a[3] += int(row.n)
    return pd.DataFrame([
        {"ba": ba, "month": month, "fuel": fuel, "day": day,
         "tons": t, "mwh": g, "mmbtu": h, "unit_hours": n}
        for (ba, month, fuel, day), (t, g, h, n) in totals.items()
    ])


def run(donor_archive, donor_eia860, target_archive, target_eia860, outdir, chunksize, repeats):
    outdir.mkdir(parents=True, exist_ok=True)
    donor = daily_data(donor_archive, donor_eia860, chunksize)
    target = daily_data(target_archive, target_eia860, chunksize)
    pool = donor.groupby(["ba", "month", "fuel"], observed=True)[["tons", "mwh", "mmbtu"]].sum()
    daily_groups = {k: g.reset_index(drop=True) for k, g in target.groupby(["ba", "month", "fuel"], observed=True)}
    monthly = target.groupby(["ba", "month", "fuel"], observed=True)[["tons", "mwh", "mmbtu"]].sum().reset_index()
    rng = np.random.default_rng(2472026)
    results = []
    fuel_results = []
    ba_results = []
    sample_days = [0, 1, 3, 7, 12]
    for k in sample_days:
        base_predictions = []
        sample_predictions = [[] for _ in range(repeats)]
        usable_groups = 0
        locally_calibrated = 0
        for row in monthly.itertuples(index=False):
            key = (row.ba, row.month, row.fuel)
            try:
                d = pool.loc[(slice(None), row.month, row.fuel), :].reset_index()
                d = d[d.ba != row.ba]
            except KeyError:
                continue
            if d.empty or d.mmbtu.sum() <= 0:
                continue
            prior = d.tons.sum() / d.mmbtu.sum()
            base_predictions.append((row.ba, row.month, row.fuel, row.tons, row.mwh, row.mmbtu, row.mmbtu * prior))
            local = daily_groups[key]
            usable_groups += 1
            if k == 0 or len(local) < k:
                for rep in range(repeats):
                    sample_predictions[rep].append((row.ba, row.month, row.fuel, row.tons, row.mwh, row.mmbtu, row.mmbtu * prior))
                continue
            locally_calibrated += 1
            for rep in range(repeats):
                chosen = rng.choice(len(local), size=k, replace=False)
                sample = local.iloc[chosen]
                rate = sample.tons.sum() / sample.mmbtu.sum()
                pred = row.mmbtu * rate
                sample_predictions[rep].append((row.ba, row.month, row.fuel, row.tons, row.mwh, row.mmbtu, pred))

        for rep, preds in enumerate(sample_predictions):
            p = pd.DataFrame(preds, columns=["ba", "month", "fuel", "true_tons", "mwh", "mmbtu", "pred_tons"])
            bm = p.groupby(["ba", "month"], observed=True).agg(true_tons=("true_tons", "sum"),
                pred_tons=("pred_tons", "sum"), mwh=("mwh", "sum"), fuels=("fuel", "nunique")).reset_index()
            bm["ape_pct"] = (bm.pred_tons / bm.true_tons - 1).abs() * 100
            bm["abs_intensity_error_g_kwh"] = (bm.pred_tons - bm.true_tons).abs() / bm.mwh * 907184.74 / 1000
            for fuel, fg in p.groupby("fuel", observed=True):
                fuel_results.append({"sample_days": k, "repeat": rep, "fuel": fuel,
                    "cells": len(fg), "target_twh": float(fg.mwh.sum()/1e6),
                    "wape_pct": float((fg.pred_tons-fg.true_tons).abs().sum()/fg.true_tons.sum()*100),
                    "median_ape_pct": float((fg.pred_tons/fg.true_tons-1).abs().median()*100)})
            for ba, bg in bm.groupby("ba", observed=True):
                ba_results.append({"sample_days": k, "repeat": rep, "ba": ba,
                    "ba_months": len(bg), "wape_pct": float((bg.pred_tons-bg.true_tons).abs().sum()/bg.true_tons.sum()*100),
                    "mean_ape_pct": float(bg.ape_pct.mean()), "median_ape_pct": float(bg.ape_pct.median())})
            results.append({"sample_days": k, "repeat": rep, "ba_months": len(bm),
                "fuel_cells_calibrated": locally_calibrated, "fuel_cells_supported": usable_groups,
                "median_ape_pct": float(bm.ape_pct.median()), "mean_ape_pct": float(bm.ape_pct.mean()),
                "wape_pct": float((bm.pred_tons-bm.true_tons).abs().sum()/bm.true_tons.sum()*100),
                "median_abs_intensity_error_g_kwh": float(bm.abs_intensity_error_g_kwh.median()),
                "mean_abs_intensity_error_g_kwh": float(bm.abs_intensity_error_g_kwh.mean())})
    repdf = pd.DataFrame(results)
    repdf.to_csv(outdir / "sampling_repeats.csv", index=False)
    summary = repdf.groupby("sample_days", observed=True).agg(
        ba_months=("ba_months", "first"), fuel_cells_calibrated=("fuel_cells_calibrated", "first"),
        fuel_cells_supported=("fuel_cells_supported", "first"), median_ape_mean=("median_ape_pct", "mean"),
        median_ape_sd=("median_ape_pct", "std"), mean_ape_mean=("mean_ape_pct", "mean"),
        wape_mean=("wape_pct", "mean"), wape_sd=("wape_pct", "std"),
        median_abs_intensity_mean=("median_abs_intensity_error_g_kwh", "mean"),
        mean_abs_intensity_mean=("mean_abs_intensity_error_g_kwh", "mean"),
    ).reset_index()
    summary.to_csv(outdir / "sampling_curve.csv", index=False)
    fuel_raw = pd.DataFrame(fuel_results)
    fuel_curve = fuel_raw.groupby(["sample_days", "fuel"], observed=True).agg(
        repeats=("repeat", "nunique"), cells=("cells", "first"), target_twh=("target_twh", "first"),
        wape_mean=("wape_pct", "mean"), wape_sd=("wape_pct", "std"),
        median_ape_mean=("median_ape_pct", "mean"),
    ).reset_index()
    fuel_curve.to_csv(outdir / "fuel_sampling_curve.csv", index=False)
    ba_raw = pd.DataFrame(ba_results)
    ba_curve = ba_raw.groupby(["sample_days", "ba"], observed=True).agg(
        repeats=("repeat", "nunique"), ba_months=("ba_months", "first"),
        wape_mean=("wape_pct", "mean"), mean_ape_mean=("mean_ape_pct", "mean"),
        median_ape_mean=("median_ape_pct", "mean"),
    ).reset_index()
    ba_curve.to_csv(outdir / "ba_sampling_curve.csv", index=False)
    metadata = {"design": "2024 same-month donor EF; 2025 local day-sample CO2/heat-input calibration; predicts 2025 monthly direct emissions",
        "donor_archive_sha256": sha256(donor_archive), "target_archive_sha256": sha256(target_archive),
        "donor_eia860_sha256": sha256(donor_eia860), "target_eia860_sha256": sha256(target_eia860),
        "donor_cells": int(len(donor)), "target_day_fuel_cells": int(len(target)),
        "target_ba_count": int(target.ba.nunique()), "repeats": repeats,
        "sampling_days": sample_days, "measurement_codes": sorted(MEASURED),
        "estimand": "CEMS-covered pure-primary-fuel fossil fleet monthly gross-load emissions intensity; not whole-grid CIF",
        "caveat": "Uses target-month heat-input total and sampled same-month measured CO2/heat ratios; retrospective information-value analysis, not pre-month forecast"}
    (outdir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return summary.to_dict(orient="records")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor-archive", type=Path, default=Path("data/external-cems/epacems-2024.zip"))
    p.add_argument("--donor-eia860", type=Path, default=Path("data/research-v87/eia8602024.zip"))
    p.add_argument("--target-archive", type=Path, default=Path("data/external-cems/epacems-2025.zip"))
    p.add_argument("--target-eia860", type=Path, default=Path("data/research-v87/eia8602025.zip"))
    p.add_argument("--outdir", type=Path, default=Path("results/v248-sparse-local-factor-calibration"))
    p.add_argument("--chunksize", type=int, default=500_000)
    p.add_argument("--repeats", type=int, default=50)
    a = p.parse_args()
    print(json.dumps(run(a.donor_archive, a.donor_eia860, a.target_archive,
        a.target_eia860, a.outdir, a.chunksize, a.repeats), indent=2))


if __name__ == "__main__":
    main()
