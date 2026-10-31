#!/usr/bin/env python3
"""Direct CEMS hourly interval test for transferable and locally calibrated anchors."""

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
COLS = ["Facility ID", "Date", "Hour", "Operating Time", "Gross Load (MW)",
        "CO2 Mass (short tons)", "CO2 Mass Measure Indicator", "Primary Fuel Type",
        "Heat Input (mmBtu)", "Heat Input Measure Indicator"]
TARGET_BAS = {"BANC", "CPLE", "ERCO", "NYIS", "SCEG", "SRP"}
SAMPLE_CALIBRATION_DAYS = 12
REPEATS = 20


def clean_id(values):
    return pd.to_numeric(values, errors="coerce").astype("Int64").astype("string")


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def plant_map(path):
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if "Plant_Y" in n and n.endswith(".xlsx"))
        d = pd.read_excel(io.BytesIO(z.read(name)), header=1,
                          usecols=["Plant Code", "Balancing Authority Code"])
    d["facility"] = clean_id(d["Plant Code"])
    d["ba"] = d["Balancing Authority Code"].astype("string").str.strip()
    d = d.dropna(subset=["facility", "ba"])
    counts = d.groupby("facility", observed=True).ba.nunique()
    d = d[d.facility.isin(set(counts[counts == 1].index))].drop_duplicates("facility")
    return dict(zip(d.facility.astype(str), d.ba.astype(str)))


def aggregate(archive, eia860, target=False, chunksize=500_000, target_bas=TARGET_BAS):
    mapping = plant_map(eia860)
    monthly = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    hourly = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
    daily_fuel = defaultdict(lambda: [0.0, 0.0, 0.0, 0])
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
                    for name in ("Operating Time", "Gross Load (MW)", "CO2 Mass (short tons)", "Heat Input (mmBtu)", "Hour"):
                        c[name] = pd.to_numeric(c[name], errors="coerce")
                    date = pd.to_datetime(c.Date, format="%Y-%m-%d", errors="coerce")
                    c["month"] = date.dt.strftime("%m")
                    c["day"] = date.dt.strftime("%Y-%m-%d")
                    c["facility"] = clean_id(c["Facility ID"])
                    c["ba"] = c.facility.map(mapping)
                    c["fuel"] = c["Primary Fuel Type"].astype("string").str.strip()
                    ok = (c["Operating Time"].gt(0) & c["Gross Load (MW)"].gt(0)
                          & c["CO2 Mass (short tons)"].gt(0) & c["Heat Input (mmBtu)"].gt(0)
                          & c.month.notna() & c.day.notna() & c.Hour.between(0, 23)
                          & c.ba.isin(target_bas) & c.fuel.notna()
                          & ~c.fuel.str.contains(",", regex=False))
                    c = c.loc[ok].copy()
                    op = c["Operating Time"]
                    c["tons"] = c["CO2 Mass (short tons)"] * op
                    c["mwh"] = c["Gross Load (MW)"] * op
                    c["mmbtu"] = c["Heat Input (mmBtu)"] * op
                    g = c.groupby(["ba", "month", "fuel"], observed=True).agg(
                        tons=("tons", "sum"), mwh=("mwh", "sum"), mmbtu=("mmbtu", "sum"), n=("mwh", "size")
                    )
                    for key, row in g.iterrows():
                        a = monthly[key]
                        a[0] += float(row.tons); a[1] += float(row.mwh); a[2] += float(row.mmbtu); a[3] += int(row.n)
                    if target:
                        dg = c.groupby(["ba", "month", "fuel", "day"], observed=True).agg(
                            tons=("tons", "sum"), mwh=("mwh", "sum"), mmbtu=("mmbtu", "sum"), n=("mwh", "size")
                        )
                        for key, row in dg.iterrows():
                            a = daily_fuel[key]
                            a[0] += float(row.tons); a[1] += float(row.mwh); a[2] += float(row.mmbtu); a[3] += int(row.n)
                        h = c.groupby(["ba", "month", "day", "Hour"], observed=True).agg(
                            tons=("tons", "sum"), mwh=("mwh", "sum"), mmbtu=("mmbtu", "sum"), n=("mwh", "size")
                        )
                        for key, row in h.iterrows():
                            a = hourly[key]
                            a[0] += float(row.tons); a[1] += float(row.mwh); a[2] += float(row.mmbtu); a[3] += int(row.n)
    mf = pd.DataFrame([{"ba": ba, "month": month, "fuel": fuel, "tons": t, "mwh": g, "mmbtu": h, "n": n}
        for (ba, month, fuel), (t, g, h, n) in monthly.items()])
    hf = pd.DataFrame([{"ba": ba, "month": month, "day": day, "hour": int(hour), "tons": t, "mwh": g, "mmbtu": h, "n": n}
        for (ba, month, day, hour), (t, g, h, n) in hourly.items()])
    df = pd.DataFrame([{"ba": ba, "month": month, "fuel": fuel, "day": day, "tons": t, "mwh": g, "mmbtu": h, "n": n}
        for (ba, month, fuel, day), (t, g, h, n) in daily_fuel.items()])
    return mf, hf, df


def metrics_for(center, y, calibration_days, test_days):
    scores = []
    for day in calibration_days:
        values = y[y.day == day].cif.to_numpy()
        if len(values) != 24:
            continue
        scores.append(float(np.max(np.abs(values - center))))
    if len(scores) != SAMPLE_CALIBRATION_DAYS:
        return None
    q = float(np.sort(scores)[SAMPLE_CALIBRATION_DAYS - 1])
    ys = y[y.day.isin(test_days)].copy()
    if ys.empty:
        return None
    lo, hi = center - q, center + q
    outside = (ys.cif < lo) | (ys.cif > hi)
    winkler = np.full(len(ys), hi-lo)
    winkler[ys.cif < lo] += 20 * (lo - ys.loc[ys.cif < lo, "cif"].to_numpy())
    winkler[ys.cif > hi] += 20 * (ys.loc[ys.cif > hi, "cif"].to_numpy() - hi)
    daily = ys.assign(outside=outside).groupby("day", observed=True).outside.any()
    return {"hours": len(ys), "hour_coverage": float((~outside).mean()),
            "simultaneous_day_coverage": float((~daily).mean()),
            "width": 2*q, "winkler90": float(winkler.mean()), "q90": q}


def run(donor_archive, donor_eia860, target_archive, target_eia860, outdir, chunksize, target_bas):
    outdir.mkdir(parents=True, exist_ok=True)
    donor, _, _ = aggregate(donor_archive, donor_eia860, target=False, chunksize=chunksize, target_bas=target_bas)
    target, hourly, target_daily = aggregate(target_archive, target_eia860, target=True, chunksize=chunksize, target_bas=target_bas)
    donor_g = donor.groupby(["ba", "month", "fuel"], observed=True)[["tons", "mwh", "mmbtu"]].sum()
    donors = {}
    for key in target.groupby(["ba", "month"], observed=True).groups:
        ba, month = key
        rows = target[(target.ba == ba) & (target.month == month)]
        gen_pred = heat_pred = 0.0
        local_heat = 0.0
        supported = True
        heat_efs = {}
        target_heat = {}
        for row in rows.itertuples(index=False):
            try:
                pool = donor_g.loc[(slice(None), month, row.fuel), :].reset_index()
                pool = pool[pool.ba != ba]
            except KeyError:
                supported = False
                break
            if pool.empty:
                supported = False
                break
            gen_pred += row.mwh * pool.tons.sum()/pool.mwh.sum()
            heat_efs[row.fuel] = pool.tons.sum()/pool.mmbtu.sum()
            heat_pred += row.mmbtu * heat_efs[row.fuel]
            target_heat[row.fuel] = row.mmbtu
            local_heat += row.mmbtu
        if supported:
            target_mwh = rows.mwh.sum()
            donors[key] = {"generation": gen_pred/target_mwh*907.18474,
                           "heat": heat_pred/target_mwh*907.18474,
                           "local_heat_total": local_heat,
                           "target_mwh": target_mwh,
                           "heat_efs": heat_efs,
                           "target_heat": target_heat,
                           "fuels": rows.fuel.tolist()}

    hourly["cif"] = hourly.tons/hourly.mwh*907.18474
    # Complete 24-clock-hour local days. Single-time-zone BA cohort avoids UTC/local ambiguity.
    valid_days = []
    for (ba, month, day), g in hourly.groupby(["ba", "month", "day"], observed=True):
        if len(g) == 24 and set(g.hour.astype(int)) == set(range(24)) and (ba, month) in donors:
            valid_days.append((ba, month, day))
    rng = np.random.default_rng(2492026)
    rows = []
    for (ba, month), group in hourly.groupby(["ba", "month"], observed=True):
        if (ba, month) not in donors:
            continue
        full = group[group.day.isin([d for b,m,d in valid_days if b==ba and m==month])].copy()
        days = sorted(full.day.unique())
        if len(days) < SAMPLE_CALIBRATION_DAYS + 2:
            continue
        info = donors[(ba, month)]
        # Target local day factor is sampled independently from the 12 conformal calibration days.
        fuel_month = target[(target.ba == ba) & (target.month == month)]
        for rep in range(REPEATS):
            chosen_factor_day = rng.choice(days)
            remaining = [d for d in days if d != chosen_factor_day]
            cal_days = sorted(rng.choice(remaining, size=SAMPLE_CALIBRATION_DAYS, replace=False).tolist())
            test_days = [d for d in remaining if d not in cal_days]
            # Sampled-day local factor is a CO2/heat ratio by fuel, applied to target month heat input.
            factor_rows = target_daily[(target_daily.ba == ba) & (target_daily.month == month)
                                       & (target_daily.day == chosen_factor_day)]
            local_tons = 0.0
            for fuel, heat_total in info["target_heat"].items():
                observed = factor_rows[factor_rows.fuel == fuel]
                rate = (observed.tons.sum()/observed.mmbtu.sum()
                        if not observed.empty and observed.mmbtu.sum() > 0
                        else info["heat_efs"][fuel])
                local_tons += heat_total * rate
            local_factor = local_tons/info["target_mwh"]*907.18474
            centers = {"generation": info["generation"], "heat": info["heat"], "local_1day": local_factor}
            y = full[["day", "hour", "cif"]]
            for method, center in centers.items():
                m = metrics_for(center, y, cal_days, test_days)
                if m:
                    rows.append({"ba": ba, "month": month, "repeat": rep, "method": method, **m,
                                 "calibration_days": len(cal_days), "factor_sample_day": chosen_factor_day,
                                 "test_days": len(test_days), "fuel_count": len(fuel_month)})
    result = pd.DataFrame(rows)
    result.to_csv(outdir / "hourly_interval_repeats.csv", index=False)
    summary = result.groupby("method", observed=True).agg(
        ba_months=("ba", lambda x: result.loc[x.index, ["ba", "month"]].drop_duplicates().shape[0]),
        hourly_coverage=("hour_coverage", "mean"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean"),
        mean_q90=("q90", "mean"),
    ).reset_index()
    summary.to_csv(outdir / "hourly_interval_summary.csv", index=False)
    meta = {"protocol": "direct CEMS hourly CIF; 2024 donor -> 2025 targets; 12 local daily scores; 1 separate local factor sample day",
        "target_bas": sorted(target_bas), "target_archive_sha256": sha256(target_archive),
        "donor_archive_sha256": sha256(donor_archive), "repeats": REPEATS,
        "estimand": "CEMS-measured, pure-primary-fuel fossil fleet gross-load intensity; not whole-grid CIF",
        "status_note": "Target-hour labels are direct measured CO2/gross-load; local factor is independently estimated from one target day and applied to monthly target heat-input margins"}
    (outdir / "metadata.json").write_text(json.dumps(meta, indent=2)+"\n")
    return summary.to_dict(orient="records")


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--donor-archive",type=Path,default=Path("data/external-cems/epacems-2024.zip"))
    p.add_argument("--donor-eia860",type=Path,default=Path("data/research-v87/eia8602024.zip"))
    p.add_argument("--target-archive",type=Path,default=Path("data/external-cems/epacems-2025.zip"))
    p.add_argument("--target-eia860",type=Path,default=Path("data/research-v87/eia8602025.zip"))
    p.add_argument("--outdir",type=Path,default=Path("results/v249-cems-hourly-interval"))
    p.add_argument("--chunksize",type=int,default=500_000)
    p.add_argument("--target-bas",nargs="+",default=sorted(TARGET_BAS))
    a=p.parse_args()
    print(json.dumps(run(a.donor_archive,a.donor_eia860,a.target_archive,a.target_eia860,
                         a.outdir,a.chunksize,set(a.target_bas)),indent=2))


if __name__=="__main__": main()
