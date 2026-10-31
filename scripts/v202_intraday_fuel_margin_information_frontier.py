#!/usr/bin/env python3
"""V202: partial-identification value of coarse intraday fuel margins."""
from __future__ import annotations

import json
import math
import sys
from fractions import Fraction as F
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v54_unseen_local_ef_confirmation as source_data

OUT = ROOT / "results/v202-intraday-fuel-margin-frontier-r1"
CENSUS = ROOT / "results/eia-ba-quality-census-v52-r1/quality_census.json"
REGIONS = ("AVA", "BANC", "BPAT", "CHPD", "DOPD", "EPE", "FMPP", "FPL", "GCPD", "GRID", "GVL", "IID", "JEA", "LDWP", "MISO", "NWMT", "NYIS", "PSEI", "SC", "SCEG", "SCL", "SEC", "TEC", "TIDC", "TPWR")
START, STOP = pd.Timestamp("2024-07-01", tz="UTC"), pd.Timestamp("2026-01-01", tz="UTC")
FUELS = ("coal", "gas", "oil", "other", "nuclear", "hydro", "wind", "solar")
EF = (1000, 400, 800, 400, 0, 0, 0, 0)
TAUS, TARGETS, BINS, K_VALUES = (100, 300), (0.01, 0.05, 0.10), (1, 2, 4), (1, 3, 7, 14)
REPS, SEED = 50, 202


def fuels_for_rows(rows):
    totals = [F(0) for _ in EF]
    for row in rows:
        for j, fuel in enumerate(FUELS):
            value = F(str(float(row[f"generation_{fuel}"])))
            if value < 0:
                raise ValueError("negative generation")
            totals[j] += value
    return tuple(totals)


def block_bounds(n, fuels, tau):
    g = sum(fuels, F(0))
    if n < 1 or g <= 0:
        raise ValueError("invalid block generation or duration")
    h = sum((x * ef for x, ef in zip(fuels, EF)), F(0))
    support = max((ef for x, ef in zip(fuels, EF) if x > 0), default=0)
    if support <= tau or h == 0:
        return 0, 0
    if h == support * g:
        return n, n
    if h > F(tau) * g:
        return 1, n
    return 0, n - 1


def bounds_for_selection(record, selected_days, bins, tau):
    selected = set(selected_days)
    lower = upper = 0
    remaining = list(record["monthly_fuels"])
    reported_hours = 0
    for day_index in selected:
        for block in record["day_bins"][day_index][bins]:
            lo, hi = block_bounds(24 // bins, block, tau)
            lower += lo
            upper += hi
            reported_hours += 24 // bins
            remaining = [x - y for x, y in zip(remaining, block)]
    if any(x < 0 for x in remaining):
        raise ValueError("reported block fuel exceeds monthly total")
    residual_hours = record["n_hours"] - reported_hours
    if residual_hours:
        lo, hi = block_bounds(residual_hours, tuple(remaining), tau)
        lower += lo
        upper += hi
    return lower, upper


def metric(bounds, actual, threshold, n):
    lo, hi = bounds
    if not lo <= actual <= hi:
        raise ValueError("identified interval misses actual hourly count")
    positive = lo >= threshold
    negative = hi < threshold
    return (hi - lo) / n, int(positive), int(negative), int(not positive and not negative)


def month_records(hourly):
    records, quality = [], []
    for region in REGIONS:
        frame = hourly[region].copy()
        times = pd.to_datetime(frame.time, utc=True)
        frame = frame.loc[times.ge(START) & times.lt(STOP) & frame.valid.astype(bool)].copy()
        frame["time"] = pd.to_datetime(frame.time, utc=True)
        frame["day"] = frame.time.dt.floor("D")
        frame["month"] = pd.to_datetime(frame.time.dt.strftime("%Y-%m-01"), utc=True)
        for month, mg in frame.groupby("month", sort=True):
            days = []
            for day, dg in mg.groupby("day", sort=True):
                dg = dg.sort_values("time")
                if len(dg) != 24 or list(dg.time) != list(pd.date_range(day, periods=24, freq="h")):
                    continue
                hour_rows = [r for _, r in dg.iterrows()]
                day_bins = {}
                for bins in BINS:
                    blocks = []
                    width = 24 // bins
                    for b in range(bins):
                        blocks.append(fuels_for_rows(hour_rows[b * width:(b + 1) * width]))
                    day_bins[bins] = blocks
                actual = {tau: 0 for tau in TAUS}
                for row in hour_rows:
                    fuels = tuple(F(str(float(row[f"generation_{f}"]))) for f in FUELS)
                    g = sum(fuels, F(0))
                    if g <= 0:
                        raise ValueError("nonpositive hourly generation")
                    h = sum((x * ef for x, ef in zip(fuels, EF)), F(0))
                    for tau in TAUS:
                        actual[tau] += int(h > F(tau) * g)
                days.append({"day": day, "bins": day_bins, "actual": actual})
            quality.append({"region": region, "month": month.isoformat(), "complete_days": len(days), "calendar_days": month.days_in_month})
            if not days:
                continue
            monthly_fuels = tuple(sum((day["bins"][1][0][j] for day in days), F(0)) for j in range(len(EF)))
            monthly_sets = {tau: block_bounds(24 * len(days), monthly_fuels, tau) for tau in TAUS}
            actual = {tau: sum(day["actual"][tau] for day in days) for tau in TAUS}
            for tau in TAUS:
                if not monthly_sets[tau][0] <= actual[tau] <= monthly_sets[tau][1]:
                    raise ValueError("monthly interval misses actual")
            records.append({"region": region, "month": month, "days": days, "day_bins": [d["bins"] for d in days],
                            "monthly_fuels": monthly_fuels, "actual": actual, "n_hours": 24 * len(days),
                            "complete_days": len(days), "monthly_sets": monthly_sets})
    return records, quality


def schedules(days, k, rng):
    if k >= days:
        return {}
    even = tuple(int(math.floor((j + 0.5) * days / k)) for j in range(k))
    return {"evenly_spaced": [even], "first_days": [tuple(range(k))],
            "random": [tuple(sorted(rng.choice(days, size=k, replace=False).tolist())) for _ in range(REPS)]}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    census = json.loads(CENSUS.read_text())
    if tuple(census["quality_qualified_regions_alphabetical"]) != REGIONS:
        raise ValueError("region roster differs from frozen census")
    source_data.ALL_REGIONS = tuple(dict.fromkeys(source_data.ALL_REGIONS + REGIONS))
    hourly, _, _, _ = source_data.prepare_regions()
    records, quality = month_records(hourly)
    rng = np.random.default_rng(SEED)
    outputs = []
    for rec in records:
        base = {"region": rec["region"], "month": rec["month"].isoformat(), "n_hours": rec["n_hours"],
                "complete_days": rec["complete_days"]}
        partial_bounds = {}
        for bins in BINS:
            for k in K_VALUES:
                if k >= rec["complete_days"]:
                    continue
                for strategy, draws in schedules(rec["complete_days"], k, rng).items():
                    for tau in TAUS:
                        intervals = [bounds_for_selection(rec, selected, bins, tau) for selected in draws]
                        if any(not lo <= rec["actual"][tau] <= hi for lo, hi in intervals):
                            raise ValueError("sparse interval misses actual count")
                        partial_bounds[(bins, k, strategy, tau)] = intervals
        for tau in TAUS:
            for target in TARGETS:
                threshold = math.ceil(target * rec["n_hours"])
                vals = metric(rec["monthly_sets"][tau], rec["actual"][tau], threshold, rec["n_hours"])
                outputs.append({**base, "tau": tau, "target_frequency": target, "bins_per_day": 0,
                                "strategy": "monthly_only", "sample_days": 0, "daily_fields": 0,
                                "replicates": 1, "span_fraction": vals[0], "positive_certificate": vals[1],
                                "negative_certificate": vals[2], "abstain": vals[3]})
                for bins, label in ((1, "all_days_daily"), (4, "all_days_6h")):
                    full = bounds_for_selection(rec, range(rec["complete_days"]), bins, tau)
                    full_metric = metric(full, rec["actual"][tau], threshold, rec["n_hours"])
                    outputs.append({**base, "tau": tau, "target_frequency": target, "bins_per_day": bins,
                                    "strategy": label, "sample_days": -1, "daily_fields": -1,
                                    "replicates": 1, "span_fraction": full_metric[0],
                                    "positive_certificate": full_metric[1], "negative_certificate": full_metric[2],
                                    "abstain": full_metric[3]})
                for bins in BINS:
                    for k in K_VALUES:
                        if k >= rec["complete_days"]:
                            continue
                        for strategy in ("evenly_spaced", "first_days", "random"):
                            intervals = partial_bounds[(bins, k, strategy, tau)]
                            vals = [metric(interval, rec["actual"][tau], threshold, rec["n_hours"])
                                    for interval in intervals]
                            arr = np.asarray(vals, dtype=float)
                            outputs.append({**base, "tau": tau, "target_frequency": target, "bins_per_day": bins,
                                            "strategy": strategy, "sample_days": k, "daily_fields": 8 * bins * k,
                                            "replicates": len(intervals), "span_fraction": float(arr[:, 0].mean()),
                                            "positive_certificate": float(arr[:, 1].mean()),
                                            "negative_certificate": float(arr[:, 2].mean()), "abstain": float(arr[:, 3].mean())})
    result = pd.DataFrame(outputs)
    key = ["region", "month", "tau", "target_frequency", "bins_per_day", "strategy", "sample_days"]
    if result.duplicated(key).any():
        raise ValueError("duplicate outcome keys in V202 producer")
    OUT.mkdir(parents=True)
    result.to_csv(OUT / "region_month_frontier.csv", index=False, float_format="%.10g")
    pd.DataFrame(quality).to_csv(OUT / "quality_audit.csv", index=False)
    num = ["span_fraction", "positive_certificate", "negative_certificate", "abstain"]
    reg = result.groupby(["region", "tau", "target_frequency", "bins_per_day", "strategy", "sample_days", "daily_fields"], as_index=False)[num].mean()
    macro = reg.groupby(["tau", "target_frequency", "bins_per_day", "strategy", "sample_days", "daily_fields"], as_index=False)[num].mean()
    reg.to_csv(OUT / "region_frontier.csv", index=False, float_format="%.10g")
    macro.to_csv(OUT / "macro_frontier.csv", index=False, float_format="%.10g")
    summary = {"status": "PARTIAL_IDENTIFICATION_INFORMATION_FRONTIER_NOT_FORECAST", "regions": len(REGIONS),
               "region_months": int(result[["region", "month"]].drop_duplicates().shape[0]),
               "rows": len(result), "schedule_replicates": REPS, "random_seed": SEED,
               "aggregation": "equal region-month means within each region, then equal-region means",
               "limitations": ["EIA proxy labels, not China observations", "monthly margins are oracle aggregates",
                               "fixed EFs and no unit/network constraints", "partial-identification certificates are not probabilistic coverage"]}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(macro[(macro.tau == 300) & (macro.target_frequency == 0.05)].to_string(index=False), flush=True)
    print(f"Completed {len(result)} unique region-month outcomes across {summary['region_months']} region-months", flush=True)


if __name__ == "__main__":
    main()
