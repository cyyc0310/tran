"""V225: calibrate hourly CIF risk from partial daily fields using sharp bounds.

For each sampled day, monthly fuel margins and observed aggregate fields define
hourly feasible emissions bounds. The day's nonconformity score is the maximum
possible deviation from the monthly anchor over all feasible compositions.
Whole days are the sampling/calibration units.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import v224_eia_sparse_daily_maxrisk as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v225-partial-snapshot-sharp-risk-r3"
FIELDS = ("full_8", "total_only", "total_plus_coal", "total_plus_thermal",
          "total_thermal_coal", "total_thermal_coal_gas")
BUDGETS = (7, 9, 12, 15)
REPEATS = 100
BOOT = 20_000
EF = np.array([1000., 400., 800., 400., 0., 0., 0., 0.])


def allocation_extreme(indices, amount, lower, upper, ef, maximize):
    """Min/max linear emission over box constraints and a fixed group sum."""
    inds = np.asarray(indices, int)
    lo = lower[..., inds]
    hi = upper[..., inds]
    amount = np.asarray(amount, float)
    base_sum = lo.sum(axis=-1)
    cap = hi - lo
    remaining = np.maximum(amount - base_sum, 0.)
    order = inds[np.argsort(ef[inds])[::-1] if maximize else np.argsort(ef[inds])]
    values = lo.copy()
    for fuel in order:
        col = int(np.where(inds == fuel)[0][0])
        add = np.minimum(remaining, cap[..., col])
        values[..., col] += add
        remaining -= add
    # Small floating residue can arise from monthly/hourly totals.
    if np.any(remaining > 1e-5):
        raise ValueError("observed group total is outside the monthly-margin feasible box")
    return values @ ef[inds]


def hourly_intensity_bounds(d, field):
    totals = np.asarray(d["fuel_totals"], float)
    # V219 stores each hourly profile as a within-month share per fuel.
    # Restore generation units before combining categories or applying margins.
    fuel = np.asarray(d["fuel_profile"], float) * totals[None, :]
    monthly_total = float(totals.sum())
    hourly_total = fuel.sum(axis=1)
    lower = np.maximum(0., totals[None, :] - (monthly_total - hourly_total)[:, None])
    upper = np.minimum(totals[None, :], hourly_total[:, None])
    anchor = float(d["monthly_y"])
    actual = np.asarray(d["cif"], float)
    if field == "full_8":
        return actual.copy(), actual.copy(), actual, anchor

    if field == "total_only":
        lo_emit = allocation_extreme(range(8), hourly_total, lower, upper, EF, maximize=False)
        hi_emit = allocation_extreme(range(8), hourly_total, lower, upper, EF, maximize=True)
    elif field == "total_plus_coal":
        coal = fuel[:, 0]
        noncoal = np.arange(1, 8)
        rem = hourly_total - coal
        lo_emit = coal * EF[0] + allocation_extreme(noncoal, rem, lower, upper, EF, maximize=False)
        hi_emit = coal * EF[0] + allocation_extreme(noncoal, rem, lower, upper, EF, maximize=True)
    elif field == "total_plus_thermal":
        thermal = fuel[:, :4].sum(axis=1)
        fossil = np.arange(0, 4)
        lo_emit = allocation_extreme(fossil, thermal, lower, upper, EF, maximize=False)
        hi_emit = allocation_extreme(fossil, thermal, lower, upper, EF, maximize=True)
    elif field == "total_thermal_coal":
        coal = fuel[:, 0]
        remaining_thermal = fuel[:, :4].sum(axis=1) - coal
        gas_oil_other = np.arange(1, 4)
        lo_emit = coal*EF[0] + allocation_extreme(
            gas_oil_other, remaining_thermal, lower, upper, EF, maximize=False)
        hi_emit = coal*EF[0] + allocation_extreme(
            gas_oil_other, remaining_thermal, lower, upper, EF, maximize=True)
    elif field == "total_thermal_coal_gas":
        coal, gas = fuel[:, 0], fuel[:, 1]
        remaining_thermal = fuel[:, :4].sum(axis=1) - coal - gas
        oil_other = np.arange(2, 4)
        lo_emit = coal*EF[0] + gas*EF[1] + allocation_extreme(
            oil_other, remaining_thermal, lower, upper, EF, maximize=False)
        hi_emit = coal*EF[0] + gas*EF[1] + allocation_extreme(
            oil_other, remaining_thermal, lower, upper, EF, maximize=True)
    else:
        raise ValueError(field)
    lo = lo_emit / np.maximum(hourly_total, 1e-12)
    hi = hi_emit / np.maximum(hourly_total, 1e-12)
    return lo, hi, actual, anchor


def build_day_scores(d, field):
    lo, hi, actual, anchor = hourly_intensity_bounds(d, field)
    times = pd.DatetimeIndex(d["time"])
    day_ids = base.day_ids(times)
    score = np.maximum(np.abs(lo - anchor), np.abs(hi - anchor))
    frame = pd.DataFrame({"day_id": day_ids, "score": score})
    return frame.groupby("day_id", sort=True).score.max(), actual, anchor


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    if not base.CACHE.exists():
        raise FileNotFoundError(base.CACHE)
    OUT.mkdir(parents=True)
    with base.CACHE.open("rb") as stream:
        series, hashes = pickle.load(stream)
    eligible = [r for r in base.REGIONS
                if sum((r, 2025, m) in series for m in range(1, 13)) >= 9
                and sum((r, 2026, m) in series for m in range(1, 7)) >= 3]
    # All score vectors are prepared once; full_8 is the exact local-score
    # comparator and partial fields use monthly-margin sharp upper bounds.
    target_scores = {}
    target_actual = {}
    target_anchor = {}
    source_scores = {}
    for region in eligible:
        for month in range(1, 7):
            key = (region, 2026, month)
            if key not in series:
                continue
            target_scores[key] = {}
            for field in FIELDS:
                scores, actual, anchor = build_day_scores(series[key], field)
                target_scores[key][field] = scores
                target_actual[key] = actual
                target_anchor[key] = anchor
            source_scores[key] = {}
            for field in FIELDS:
                vals = []
                for donor in eligible:
                    if donor == region or (donor, 2025, month) not in series:
                        continue
                    s, _, _ = build_day_scores(series[(donor, 2025, month)], field)
                    vals.extend(s.to_numpy(float))
                source_scores[key][field] = np.asarray(vals, float)

    records = []
    rng = np.random.default_rng(225)
    for (region, year, month), fields in target_scores.items():
        key = (region, year, month)
        actual = target_actual[key]
        anchor = target_anchor[key]
        first_field = FIELDS[0]
        day_index = fields[first_field].index.to_numpy(dtype=np.int64)
        times = pd.DatetimeIndex(series[key]["time"])
        day_id = base.day_ids(times)
        unique_days = day_index
        for budget in BUDGETS:
            if budget >= len(unique_days):
                continue
            replicate = {field: [] for field in FIELDS}
            source_rep = {field: [] for field in FIELDS}
            for _ in range(REPEATS):
                selected_ix = rng.choice(len(unique_days), size=budget, replace=False)
                test_days = np.delete(unique_days, selected_ix)
                for field in FIELDS:
                    local_q = base.radius(fields[field].iloc[selected_ix].to_numpy(float))
                    mask = np.isin(day_id, test_days)
                    y = actual[mask]
                    if np.isfinite(local_q):
                        lo, hi = anchor - local_q, anchor + local_q
                        inside = (y >= lo) & (y <= hi)
                        miss = np.maximum(lo-y, 0.) + np.maximum(y-hi, 0.)
                        winkler = (hi-lo) + (2/base.ALPHA)*miss
                        test_days_hour = day_id[mask]
                        _, inverse = np.unique(test_days_hour, return_inverse=True)
                        day_inside = np.ones(inverse.max()+1, dtype=bool)
                        np.logical_and.at(day_inside, inverse, inside)
                        result = (float(inside.mean()), float(hi-lo), float(winkler.mean()), float(day_inside.mean()))
                    else:
                        result = (1., np.inf, np.inf, 1.)
                    replicate[field].append(result)

                    qsrc = base.radius(source_scores[key][field])
                    if np.isfinite(qsrc):
                        lo, hi = anchor-qsrc, anchor+qsrc
                        inside = (y >= lo) & (y <= hi)
                        miss = np.maximum(lo-y, 0.) + np.maximum(y-hi, 0.)
                        winkler = (hi-lo) + (2/base.ALPHA)*miss
                        test_days_hour = day_id[mask]
                        _, inverse = np.unique(test_days_hour, return_inverse=True)
                        day_inside = np.ones(inverse.max()+1, dtype=bool)
                        np.logical_and.at(day_inside, inverse, inside)
                        result_src = (float(inside.mean()), float(hi-lo), float(winkler.mean()), float(day_inside.mean()))
                    else:
                        result_src = (1., np.inf, np.inf, 1.)
                    source_rep[field].append(result_src)
            for field in FIELDS:
                for method, pool in (("local_bound", replicate[field]), ("source_loo", source_rep[field])):
                    arr = np.asarray(pool, float)
                    records.append({"region": region, "month": month, "budget": budget,
                                    "field": field, "method": method,
                                    "hour_coverage": float(arr[:, 0].mean()),
                                    "width": float(arr[:, 1].mean()),
                                    "winkler": float(arr[:, 2].mean()),
                                    "simultaneous_day_coverage": float(arr[:, 3].mean()),
                                    "n_days": len(unique_days), "n_repeats": REPEATS})
    cm = pd.DataFrame(records)
    if cm.empty:
        raise RuntimeError("no target month metrics")
    cm.to_csv(OUT / "region_month_metrics.csv", index=False)
    region = cm.groupby(["region", "budget", "field", "method"], as_index=False).agg(
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler=("winkler", "mean"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_months=("month", "count"))
    region.to_csv(OUT / "region_metrics.csv", index=False)
    summary = region.groupby(["budget", "field", "method"], as_index=False).agg(
        macro_hour_coverage=("hour_coverage", "mean"), macro_width=("width", "mean"),
        macro_winkler=("winkler", "mean"), macro_simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        n_regions=("region", "nunique"))
    summary.to_csv(OUT / "summary.csv", index=False)
    rng = np.random.default_rng(226)
    comparisons = []
    for budget in BUDGETS:
        local = region.loc[(region.budget.eq(budget)) & region.method.eq("local_bound")]
        wide = local.pivot(index="region", columns="field")
        for field in FIELDS[1:]:
            for metric in ("hour_coverage", "width", "winkler", "simultaneous_day_coverage"):
                delta = (wide[metric][field] - wide[metric]["full_8"]).to_numpy(float)
                if not np.isfinite(delta).all():
                    comparisons.append({"budget": budget, "field": field, "metric": metric,
                                        "delta_vs_full8": "infinite_or_undefined",
                                        "ci95_low": "infinite_or_undefined", "ci95_high": "infinite_or_undefined"})
                    continue
                sims = np.mean(rng.choice(delta, size=(BOOT, len(delta)), replace=True), axis=1)
                comparisons.append({"budget": budget, "field": field, "metric": metric,
                                    "delta_vs_full8": float(delta.mean()),
                                    "ci95_low": float(np.quantile(sims, .025)),
                                    "ci95_high": float(np.quantile(sims, .975)), "regions": len(delta)})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_region_bootstrap.csv", index=False)
    metadata = {"source_hashes": hashes, "source_year": 2025, "target_year": 2026,
                "regions": int(region.region.nunique()),
                "region_months": int(cm[["region", "month"]].drop_duplicates().shape[0]),
                "fields": {"full_8": "all eight hourly fuel categories, exact daily score",
                           "total_only": "hourly total generation; sharp EF bounds under monthly margins",
                           "total_plus_coal": "hourly total plus coal; sharp bounds over remaining categories",
                           "total_plus_thermal": "hourly total plus thermal aggregate; sharp bounds within thermal",
                           "total_thermal_coal": "hourly total, thermal aggregate, and coal; sharp bounds for gas/oil/other",
                           "total_thermal_coal_gas": "hourly total, thermal aggregate, coal, and gas; sharp bounds for oil/other"},
                "monthly_margins": "exact target monthly generation for each of eight fuels",
                "ef_gco2_per_kwh": EF.tolist(), "day_nonconformity": "maximum hourly possible deviation from monthly anchor over the sampled day",
                "sampling": "uniform random days without replacement", "budgets": list(BUDGETS),
                "repeats": REPEATS, "label_caveat": "EIA hourly generation-derived proxy CIF"}
    (OUT / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str) + "\n")
    print(summary.to_string(index=False))
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
