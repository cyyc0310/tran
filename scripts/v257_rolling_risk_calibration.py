#!/usr/bin/env python3
"""Paired fixed versus rolling recent-day risk calibration for V256 targets."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["FMPP", "GVL", "IID", "JEA", "PSCO", "SC", "SEC", "TAL", "TEC", "TIDC"]
BUDGETS = [9, 12, 18, 24]
TONS_TO_G_KWH = 907.18474


def q_from_scores(scores: np.ndarray, n: int) -> float:
    rank = math.ceil(0.90 * (n + 1))
    return float(np.sort(scores)[rank - 1])


def day_loss(residual: np.ndarray, q: float) -> tuple[float, float, float, float]:
    outside = np.abs(residual) > q
    loss = np.full(len(residual), 2 * q, dtype=float)
    loss[residual < -q] += 20 * (-q - residual[residual < -q])
    loss[residual > q] += 20 * (residual[residual > q] - q)
    return float(not outside.any()), float((~outside).mean()), 2 * q, float(loss.mean())


def main():
    out = ROOT / "results/v257-rolling-risk-calibration"
    out.mkdir(parents=True, exist_ok=True)
    hp = ROOT / "results/v253-annual-heat-hourly-interval/target_2025_hourly.csv"
    pp = ROOT / "results/v252-annual-local-heat-rate/target_2025_ba_month_predictions.csv"
    h = pd.read_csv(hp)
    h = h[h.ba.isin(TARGETS)].copy()
    h["day"] = pd.to_datetime(h.day).dt.strftime("%Y-%m-%d")
    h["cif"] = h.tons / h.mwh * TONS_TO_G_KWH
    mwh = h.groupby(["ba", "month"], observed=True).mwh.sum().rename("hourly_mwh")
    p = pd.read_csv(pp)
    p = p[p.ba.isin(TARGETS)].merge(mwh, on=["ba", "month"], how="inner", validate="one_to_one")
    p["center"] = p.annual_heat_rate_tons / p.hourly_mwh * TONS_TO_G_KWH
    centers = p.set_index(["ba", "month"]).center.to_dict()
    rows = []
    for (ba, month), g in h.groupby(["ba", "month"], observed=True, sort=True):
        center = float(centers[(ba, month)])
        full = g.groupby("day", observed=True).filter(
            lambda z: len(z) == 24 and set(z.hour.astype(int)) == set(range(24))
        )
        dates = sorted(full.day.unique())
        day_scores, day_residuals = [], []
        for day in dates:
            x = full[full.day == day].sort_values("hour")
            residual = (x.cif - center).to_numpy(dtype=float)
            day_residuals.append(residual)
            day_scores.append(float(np.max(np.abs(residual))))
        scores = np.asarray(day_scores)
        for n in BUDGETS:
            if len(dates) <= n:
                continue
            fixed_q = q_from_scores(scores[:n], n)
            for i in range(n, len(dates)):
                rolling_q = q_from_scores(scores[i - n:i], n)
                for method, q in (("fixed_month_start", fixed_q), ("rolling_recent_n", rolling_q)):
                    day_cov, hour_cov, width, winkler = day_loss(day_residuals[i], q)
                    rows.append({"ba": ba, "month": month, "calibration_days": n,
                                 "test_day": dates[i], "test_index": i + 1, "method": method,
                                 "day_coverage": day_cov, "hour_coverage": hour_cov,
                                 "width": width, "winkler90": winkler, "q90": q})
    d = pd.DataFrame(rows)
    d.to_csv(out / "daily_predictions.csv", index=False)
    # First average forecasts within BA-month, then BA, then equally weight BAs.
    bm = d.groupby(["method", "calibration_days", "ba", "month"], observed=True).agg(
        day_coverage=("day_coverage", "mean"), hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean"), test_days=("test_day", "nunique")
    ).reset_index()
    bm.to_csv(out / "by_ba_month.csv", index=False)
    ba = bm.groupby(["method", "calibration_days", "ba"], observed=True).agg(
        day_coverage=("day_coverage", "mean"), hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean"), test_days=("test_days", "mean")
    ).reset_index()
    ba.to_csv(out / "by_ba.csv", index=False)
    summary = ba.groupby(["method", "calibration_days"], observed=True).agg(
        bas=("ba", "nunique"), day_coverage=("day_coverage", "mean"),
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"), mean_test_days_per_ba_month=("test_days", "mean")
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    # Paired BA-cluster bootstrap: rolling minus fixed, same BA-months and forecast days.
    paired = ba.pivot(index=["calibration_days", "ba"], columns="method",
                      values=["day_coverage", "width", "winkler90"]).reset_index()
    rng = np.random.default_rng(2572026)
    ci = []
    for n in BUDGETS:
        z = paired[paired.calibration_days == n].sort_values("ba")
        bas = z.ba.to_numpy()
        for metric in ["day_coverage", "width", "winkler90"]:
            diff = (z[(metric, "rolling_recent_n")] - z[(metric, "fixed_month_start")]).to_numpy(float)
            sample = diff[rng.integers(0, len(diff), size=(20_000, len(diff)))].mean(axis=1)
            ci.append({"calibration_days": n, "metric": metric,
                       "rolling_minus_fixed": float(diff.mean()),
                       "ci_low": float(np.quantile(sample, .025)),
                       "ci_high": float(np.quantile(sample, .975)),
                       "clusters": len(bas), "bootstrap_replicates": 20_000})
    pd.DataFrame(ci).to_csv(out / "paired_ba_cluster_bootstrap.csv", index=False)
    meta = {"protocol": "same target forecast days; fixed first n daily scores versus rolling most recent n",
            "targets": TARGETS, "ba_months": int(d[["ba", "month"]].drop_duplicates().shape[0]),
            "budgets": BUDGETS, "estimand": "EPA CEMS measured fossil-fleet gross-load intensity, not BA whole-grid CIF",
            "target_hour_labels_previously_opened": True,
            "center_information": "2024 target-BA annual heat-rate estimate and 2025 target-month gross MWh",
            "boundary": "within-month daily risk update, retrospective monthly center; not full month-start real-time forecasting"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nPaired rolling - fixed:")
    print(pd.DataFrame(ci).to_string(index=False))


if __name__ == "__main__":
    main()
