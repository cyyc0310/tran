#!/usr/bin/env python3
"""Compare symmetric and directional day-score intervals with a rolling 24-day window."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["FMPP", "GVL", "IID", "JEA", "PSCO", "SC", "SEC", "TAL", "TEC", "TIDC"]
N = 24
TONS_TO_G_KWH = 907.18474


def interval_metrics(residual: np.ndarray, lo_q: float, hi_q: float):
    lower, upper = -lo_q, hi_q
    outside_low = residual < lower
    outside_high = residual > upper
    covered_day = not (outside_low.any() or outside_high.any())
    hour_coverage = float((~(outside_low | outside_high)).mean())
    width = lo_q + hi_q
    score = np.full(len(residual), width, dtype=float)
    score[outside_low] += 20 * (lower - residual[outside_low])
    score[outside_high] += 20 * (residual[outside_high] - upper)
    return float(covered_day), hour_coverage, width, float(score.mean())


def main():
    out = ROOT / "results/v259-asymmetric-hourly-interval"
    out.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(ROOT / "results/v253-annual-heat-hourly-interval/target_2025_hourly.csv")
    h = h[h.ba.isin(TARGETS)].copy()
    h["day"] = pd.to_datetime(h.day).dt.strftime("%Y-%m-%d")
    h["cif"] = h.tons / h.mwh * TONS_TO_G_KWH
    mwh = h.groupby(["ba", "month"], observed=True).mwh.sum().rename("hourly_mwh")
    p = pd.read_csv(ROOT / "results/v252-annual-local-heat-rate/target_2025_ba_month_predictions.csv")
    p = p[p.ba.isin(TARGETS)].merge(mwh, on=["ba", "month"], how="inner", validate="one_to_one")
    p["center"] = p.annual_heat_rate_tons / p.hourly_mwh * TONS_TO_G_KWH
    centers = p.set_index(["ba", "month"]).center.to_dict()
    rows = []
    for (ba, month), g in h.groupby(["ba", "month"], observed=True, sort=True):
        center = float(centers[(ba, month)])
        day_residuals = []
        days = []
        for day, x in g.groupby("day", observed=True, sort=True):
            if len(x) == 24 and set(x.hour.astype(int)) == set(range(24)):
                days.append(day)
                day_residuals.append((x.sort_values("hour").cif.to_numpy(dtype=float) - center))
        for i in range(N, len(days)):
            cal = day_residuals[i-N:i]
            test = day_residuals[i]
            lower_scores = np.sort([float(np.max(-x)) for x in cal])
            upper_scores = np.sort([float(np.max(x)) for x in cal])
            abs_scores = np.sort([float(np.max(np.abs(x))) for x in cal])
            # 95% one-sided conformal ranks and 90% two-sided symmetric rank.
            qlo, qhi = float(lower_scores[-1]), float(upper_scores[-1])
            qsym = float(abs_scores[22])  # 23rd of 24
            for method, lo, hi in (("asymmetric_directional", qlo, qhi), ("symmetric_absolute", qsym, qsym)):
                cov, hour_cov, width, score = interval_metrics(test, lo, hi)
                rows.append({"ba": ba, "month": month, "test_day": days[i], "method": method,
                             "day_coverage": cov, "hour_coverage": hour_cov, "width": width,
                             "winkler90": score, "q_lower": lo, "q_upper": hi})
    d = pd.DataFrame(rows)
    d.to_csv(out / "daily_predictions.csv", index=False)
    bm = d.groupby(["method", "ba", "month"], observed=True).agg(
        day_coverage=("day_coverage", "mean"), hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean"), test_days=("test_day", "nunique")
    ).reset_index()
    bm.to_csv(out / "by_ba_month.csv", index=False)
    ba = bm.groupby(["method", "ba"], observed=True).agg(
        day_coverage=("day_coverage", "mean"), hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean"), test_days=("test_days", "mean")
    ).reset_index()
    ba.to_csv(out / "by_ba.csv", index=False)
    summary = ba.groupby("method", observed=True).agg(
        bas=("ba", "nunique"), day_coverage=("day_coverage", "mean"),
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"), mean_test_days_per_ba_month=("test_days", "mean")
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    wide = ba.pivot(index="ba", columns="method", values=["day_coverage", "width", "winkler90"])
    rng = np.random.default_rng(2592026)
    ci = []
    for metric in ["day_coverage", "width", "winkler90"]:
        diff = (wide[(metric, "asymmetric_directional")] - wide[(metric, "symmetric_absolute")]).to_numpy(float)
        samples = diff[rng.integers(0, len(diff), size=(20_000, len(diff)))].mean(axis=1)
        ci.append({"metric": metric, "asymmetric_minus_symmetric": float(diff.mean()),
                   "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975)),
                   "clusters": len(diff), "bootstrap_replicates": 20_000})
    pd.DataFrame(ci).to_csv(out / "paired_ba_cluster_bootstrap.csv", index=False)
    meta = {"protocol": "rolling n=24 previous complete days; asymmetric one-sided max scores vs symmetric absolute score",
            "targets": TARGETS, "ba_months": int(d[["ba", "month"]].drop_duplicates().shape[0]),
            "calibration_days": N, "target_hour_labels_previously_opened": True,
            "estimand": "EPA CEMS measured fossil-fleet gross-load intensity, not BA whole-grid CIF",
            "center_information": "2024 target-BA annual heat rate and 2025 target-month gross MWh",
            "boundary": "target days after day 24 only; within-month retrospective center; not month-start forecast"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nPaired BA bootstrap:")
    print(pd.DataFrame(ci).to_string(index=False))
    print("\nBy BA:")
    print(ba.pivot(index="ba", columns="method", values=["day_coverage", "width", "winkler90"]).round(3).to_string())


if __name__ == "__main__":
    main()
