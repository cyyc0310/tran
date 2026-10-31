"""V223: value of sparse local daily fuel snapshots for monthly-anchor risk.

Labels are derived from the Carbon Monitor daily fuel estimates, not metered CIF.
The interval center is always the exact fixed-factor monthly generation anchor;
only the symmetric risk radius is calibrated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import v222_carbon_monitor_country_coldstart as v222

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v223-sparse-daily-risk-snapshots-r1"
ALPHA = .10
BUDGETS = (7, 8, 9, 12, 18, 21)
REPEATS = 100
BOOT = 20_000


def conformal_radius(scores: np.ndarray) -> float:
    n = len(scores)
    rank = math.ceil((n + 1) * (1 - ALPHA))
    if rank > n:
        return math.inf
    return float(np.partition(np.asarray(scores, float), rank - 1)[rank - 1])


def score_interval(actual: np.ndarray, center: float, radius: float):
    if not np.isfinite(radius):
        return {"coverage": 1.0, "width": math.inf, "winkler": math.inf}
    lo, hi = center - radius, center + radius
    miss = np.maximum(lo - actual, 0.) + np.maximum(actual - hi, 0.)
    winkler = (hi - lo) + (2 / ALPHA) * miss
    return {"coverage": float(np.mean((actual >= lo) & (actual <= hi))),
            "width": float(hi - lo), "winkler": float(np.mean(winkler))}


def main():
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    data, raw = v222.load()
    # Source-only leave-country-out score pools. Each score is a daily absolute
    # deviation from the same month's exact generation-weighted anchor.
    source = data.loc[data.year.isin(v222.TRAIN_YEARS)].copy()
    month_mix = source.groupby(["country", "year", "month"])[list(v222.FUELS)].sum()
    month_anchor = (month_mix.to_numpy(float) @ v222.EF) / month_mix.sum(axis=1).to_numpy(float)
    month_mix = month_mix.assign(anchor=month_anchor).reset_index()
    source = source.merge(month_mix[["country", "year", "month", "anchor"]],
                          on=["country", "year", "month"], how="left", validate="many_to_one")
    source["score"] = np.abs(source.y - source.anchor)
    source_scores = {c: g.score.to_numpy(float) for c, g in source.groupby("country", sort=True)}

    targets = data.loc[data.year.isin(v222.TEST_YEARS)].copy()
    metrics = []
    rng = np.random.default_rng(223)
    for (country, year, month), target in targets.groupby(["country", "year", "month"], sort=True):
        expected_days = pd.Period(f"{year}-{month:02d}").days_in_month
        if len(target) != expected_days or target.date.nunique() != expected_days:
            continue
        target = target.sort_values("date")
        totals = target[list(v222.FUELS)].sum().to_numpy(float)
        center = float((totals / totals.sum()) @ v222.EF)
        actual = target.y.to_numpy(float)
        source_pool = np.concatenate([s for c, s in source_scores.items() if c != country])
        source_radius = conformal_radius(source_pool)
        for budget in BUDGETS:
            # A simple random sample without replacement selects local daily
            # fuel snapshots. The remaining days are a random finite-population
            # test set; conformal rank calibration is evaluated over this design.
            replicate_scores = {"source_loo": [], "local_snapshot": []}
            for rep in range(REPEATS):
                sampled = rng.choice(len(target), size=budget, replace=False)
                test_mask = np.ones(len(target), dtype=bool)
                test_mask[sampled] = False
                local_radius = conformal_radius(np.abs(actual[sampled] - center))
                for label, radius in (("source_loo", source_radius), ("local_snapshot", local_radius)):
                    sc = score_interval(actual[test_mask], center, radius)
                    replicate_scores[label].append((sc["coverage"], sc["width"], sc["winkler"], radius))
            for label, vals in replicate_scores.items():
                arr = np.asarray(vals, float)
                metrics.append({"country": country, "year": int(year), "month": int(month),
                                "budget": budget, "method": label,
                                "coverage": float(np.mean(arr[:, 0])), "width": float(np.mean(arr[:, 1])),
                                "winkler": float(np.mean(arr[:, 2])), "radius": float(np.mean(arr[:, 3])),
                                "n_repeats": REPEATS, "n_test_days": int(test_mask.sum()), "center": center})
    result = pd.DataFrame(metrics)
    if result.empty:
        raise RuntimeError("no eligible complete target months")
    # Equal weight: repeat average within country-month -> month -> country.
    # Infinite intervals are kept visible, not silently dropped.
    cm = result.copy()
    cm.to_csv(OUT / "country_month_metrics.csv", index=False)
    country = cm.groupby(["country", "budget", "method"], as_index=False).agg(
        coverage=("coverage", "mean"), width=("width", "mean"), winkler=("winkler", "mean"),
        n_months=("month", "count"))
    country.to_csv(OUT / "country_metrics.csv", index=False)
    summary = country.groupby(["budget", "method"], as_index=False).agg(
        macro_country_coverage=("coverage", "mean"), macro_country_width=("width", "mean"),
        macro_country_winkler=("winkler", "mean"), n_countries=("country", "nunique"))
    summary.to_csv(OUT / "summary.csv", index=False)
    # Country-cluster paired differences; infinities are reported as undefined.
    comparisons = []
    rng = np.random.default_rng(224)
    for budget in BUDGETS:
        p = country.loc[country.budget.eq(budget)].pivot(index="country", columns="method")
        for metric in ("coverage", "width", "winkler"):
            d = (p[metric]["local_snapshot"] - p[metric]["source_loo"]).to_numpy(float)
            if not np.isfinite(d).all():
                comparisons.append({"budget": budget, "metric": metric, "delta_local_minus_source": "infinite_or_undefined",
                                    "ci95_low": "infinite_or_undefined", "ci95_high": "infinite_or_undefined"})
                continue
            sims = np.mean(rng.choice(d, size=(BOOT, len(d)), replace=True), axis=1)
            comparisons.append({"budget": budget, "metric": metric, "delta_local_minus_source": float(d.mean()),
                                "ci95_low": float(np.quantile(sims, .025)), "ci95_high": float(np.quantile(sims, .975)),
                                "countries": len(d)})
    pd.DataFrame(comparisons).to_csv(OUT / "paired_country_bootstrap.csv", index=False)
    meta = {"data_sha256": v222.sha256(v222.INPUT), "source_years": list(v222.TRAIN_YEARS),
            "target_years": list(v222.TEST_YEARS), "target_country_history_excluded_from_source_radius": True,
            "information": "post-month exact category margins plus randomly sampled target-day full category generation vector",
            "center": "generation-weighted monthly fixed-factor anchor; frozen and never updated",
            "risk_rule": "split-conformal absolute anchor residual; finite 90% radius requires >=9 local calibration days",
            "sampling_design": "simple random sample of target days without replacement, per target country-month",
            "budgets": list(BUDGETS), "repeats_per_country_month_budget": REPEATS,
            "complete_country_months": int(cm[["country", "year", "month"]].drop_duplicates().shape[0]),
            "countries": sorted(country.country.unique()), "raw_source_rows": int(len(raw)),
            "label_caveat": "Carbon Monitor daily fuel generation estimates; not independent metered CIF"}
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    print(summary.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))
    print(f"output={OUT}")


if __name__ == "__main__":
    main()
