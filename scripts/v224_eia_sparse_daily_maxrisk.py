"""V224: random-day local calibration for hourly CIF proxy intervals on EIA data.

Calibration units are whole days (the within-day maximum hourly residual), so
the method does not count correlated hourly points as independent ranks.
"""
from __future__ import annotations

import json
import math
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CACHE = Path("/tmp/v219_eia_hourly_series.pkl")
OUT = ROOT / "results/v224-eia-sparse-daily-maxrisk-r3"
ALPHA = .10
BUDGETS = (3, 5, 7, 9, 12, 15)
REPEATS = 100
BOOT = 20_000
REGIONS = (
    "AVA", "BANC", "BPAT", "CHPD", "DOPD", "EPE", "FMPP", "FPL", "GCPD",
    "GRID", "GVL", "IID", "JEA", "LDWP", "MISO", "NWMT", "NYIS", "PSEI",
    "SC", "SCEG", "SCL", "SEC", "TEC", "TIDC", "TPWR",
)


def radius(scores):
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - ALPHA))
    if rank > n:
        return math.inf
    return float(np.partition(np.asarray(scores, float), rank - 1)[rank - 1])


def day_scores(d):
    times = pd.DatetimeIndex(d["time"])
    cif = np.asarray(d["cif"], float)
    center = float(d["monthly_y"])
    frame = pd.DataFrame({"day": times.floor("D"), "score": np.abs(cif - center)})
    return frame.groupby("day", sort=True).score.max(), center


def day_ids(times: pd.DatetimeIndex) -> np.ndarray:
    naive = times.tz_localize(None) if times.tz is not None else times
    return naive.to_numpy(dtype="datetime64[D]").astype(np.int64)


def interval_metrics(d, test_days, center, q):
    times = pd.DatetimeIndex(d["time"])
    y = np.asarray(d["cif"], float)
    day_id = day_ids(times)
    mask = np.isin(day_id, test_days)
    test_y = y[mask]
    if not np.isfinite(q):
        return {"hour_coverage": 1., "hour_width": math.inf, "winkler": math.inf,
                "simultaneous_day_coverage": 1., "n_test_hours": int(mask.sum()),
                "n_test_days": len(test_days)}
    lo, hi = center - q, center + q
    miss = np.maximum(lo - test_y, 0.) + np.maximum(test_y - hi, 0.)
    w = (hi - lo) + (2 / ALPHA) * miss
    test_inside = (test_y >= lo) & (test_y <= hi)
    _, inverse = np.unique(day_id[mask], return_inverse=True)
    all_inside = np.ones(int(inverse.max()) + 1, dtype=bool)
    np.logical_and.at(all_inside, inverse, test_inside)
    day_cov = all_inside.mean()
    return {"hour_coverage": float(np.mean((test_y >= lo) & (test_y <= hi))),
            "hour_width": float(hi - lo), "winkler": float(w.mean()),
            "simultaneous_day_coverage": float(day_cov), "n_test_hours": int(mask.sum()),
            "n_test_days": len(test_days)}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if not CACHE.exists():
        raise FileNotFoundError(CACHE)
    OUT.mkdir(parents=True)
    with CACHE.open("rb") as f:
        series, hashes = pickle.load(f)
    eligible = [r for r in REGIONS
                if sum((r, 2025, m) in series for m in range(1, 13)) >= 9
                and sum((r, 2026, m) in series for m in range(1, 7)) >= 3]
    metrics = []
    rng = np.random.default_rng(224)
    for region in eligible:
        for month in range(1, 7):
            target_key = (region, 2026, month)
            if target_key not in series:
                continue
            d = series[target_key]
            target_day_scores, center = day_scores(d)
            days = day_ids(pd.DatetimeIndex(target_day_scores.index))
            donors = [r for r in eligible if r != region and (r, 2025, month) in series]
            source_score_list = []
            for donor in donors:
                ds, _ = day_scores(series[(donor, 2025, month)])
                source_score_list.extend(ds.to_numpy(float))
            source_q = radius(np.asarray(source_score_list, float))
            for budget in BUDGETS:
                if budget >= len(days):
                    continue
                source_reps, local_reps = [], []
                for _ in range(REPEATS):
                    selected_ix = rng.choice(len(days), size=budget, replace=False)
                    test_days = np.delete(days, selected_ix)
                    local_q = radius(target_day_scores.iloc[selected_ix].to_numpy(float))
                    source_reps.append(interval_metrics(d, test_days, center, source_q))
                    local_reps.append(interval_metrics(d, test_days, center, local_q))
                for method, reps in (("source_loo", source_reps), ("local_day_max", local_reps)):
                    metrics.append({"region": region, "month": month, "budget": budget,
                                    "method": method, "hour_coverage": float(np.mean([x["hour_coverage"] for x in reps])),
                                    "hour_width": float(np.mean([x["hour_width"] for x in reps])),
                                    "winkler": float(np.mean([x["winkler"] for x in reps])),
                                    "simultaneous_day_coverage": float(np.mean([x["simultaneous_day_coverage"] for x in reps])),
                                    "radius_mean": float(np.mean([x["hour_width"] for x in reps]) / 2),
                                    "n_days": len(days), "n_source_regions": len(donors), "n_repeats": REPEATS})
    cm = pd.DataFrame(metrics)
    if cm.empty:
        raise RuntimeError("no target region-months survived")
    cm.to_csv(OUT / "region_month_metrics.csv", index=False)
    region = cm.groupby(["region", "budget", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), hour_width=("hour_width", "mean"),
        winkler=("winkler", "mean"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_months=("month", "count"))
    region.to_csv(OUT / "region_metrics.csv", index=False)
    summary = region.groupby(["budget", "method"], as_index=False).agg(
        macro_region_hour_coverage=("hour_coverage", "mean"), macro_region_width=("hour_width", "mean"),
        macro_region_winkler=("winkler", "mean"), macro_region_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_regions=("region", "nunique"))
    summary.to_csv(OUT / "summary.csv", index=False)
    rng = np.random.default_rng(225)
    comparisons = []
    for budget in BUDGETS:
        p = region.loc[region.budget.eq(budget)].pivot(index="region", columns="method")
        for metric in ("hour_coverage", "hour_width", "winkler", "simultaneous_day_coverage"):
            delta = (p[metric]["local_day_max"] - p[metric]["source_loo"]).to_numpy(float)
            if not np.isfinite(delta).all():
                comparisons.append({"budget": budget, "metric": metric, "delta_local_minus_source": "infinite_or_undefined",
                                    "ci95_low": "infinite_or_undefined", "ci95_high": "infinite_or_undefined"})
                continue
            sims = np.mean(rng.choice(delta, size=(BOOT, len(delta)), replace=True), axis=1)
            comparisons.append({"budget": budget, "metric": metric, "delta_local_minus_source": float(delta.mean()),
                                "ci95_low": float(np.quantile(sims, .025)), "ci95_high": float(np.quantile(sims, .975)),
                                "regions": len(delta)})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_region_bootstrap.csv", index=False)
    metadata = {"input_cache": str(CACHE), "source_hashes": hashes, "source_year": 2025,
                "target_year": 2026, "target_region_count": int(region.region.nunique()),
                "target_region_months": int(cm[["region", "month"]].drop_duplicates().shape[0]),
                "unit": "day; nonconformity is maximum hourly absolute error within day",
                "target_local_observation": "one randomly sampled day provides its complete 24-hour, eight-fuel generation curve",
                "interval_center": "target monthly generation-weighted fixed-EF anchor, frozen",
                "source_calibration": "2025 daily-max residuals from other regions, same calendar month; target region excluded",
                "budgets_days": list(BUDGETS), "repeats_per_region_month_budget": REPEATS,
                "label_caveat": "EIA hourly fuel generation-derived proxy CIF, not direct measured emissions"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
