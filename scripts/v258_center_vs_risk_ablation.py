#!/usr/bin/env python3
"""Hold rolling risk calibration fixed while ablating center information."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["FMPP", "GVL", "IID", "JEA", "PSCO", "SC", "SEC", "TAL", "TEC", "TIDC"]
N = 18
TONS_TO_G_KWH = 907.18474
CENTERS = {
    "annual_heat_rate": "annual_heat_rate_tons",
    "annual_local_co2_mwh": "annual_local_co2_mwh_tons",
    "monthly_heat_oracle": "monthly_heat_oracle_tons",
}


def interval_loss(residual: np.ndarray, q: float) -> tuple[float, float, float]:
    outside = np.abs(residual) > q
    score = np.full(len(residual), 2 * q, dtype=float)
    score[residual < -q] += 20 * (-q - residual[residual < -q])
    score[residual > q] += 20 * (residual[residual > q] - q)
    return float(not outside.any()), float((~outside).mean()), float(score.mean())


def main():
    out = ROOT / "results/v258-center-vs-risk-ablation"
    out.mkdir(parents=True, exist_ok=True)
    h = pd.read_csv(ROOT / "results/v253-annual-heat-hourly-interval/target_2025_hourly.csv")
    h = h[h.ba.isin(TARGETS)].copy()
    h["day"] = pd.to_datetime(h.day).dt.strftime("%Y-%m-%d")
    h["cif"] = h.tons / h.mwh * TONS_TO_G_KWH
    mwh = h.groupby(["ba", "month"], observed=True).mwh.sum().rename("hourly_mwh")
    pred = pd.read_csv(ROOT / "results/v252-annual-local-heat-rate/target_2025_ba_month_predictions.csv")
    pred = pred[pred.ba.isin(TARGETS)].merge(mwh, on=["ba", "month"], how="inner", validate="one_to_one")
    center_map = {}
    for name, col in CENTERS.items():
        center_map[name] = (pred.set_index(["ba", "month"])[col] / pred.set_index(["ba", "month"]).hourly_mwh * TONS_TO_G_KWH).to_dict()

    rows = []
    for (ba, month), group in h.groupby(["ba", "month"], observed=True, sort=True):
        days = []
        for day, x in group.groupby("day", observed=True, sort=True):
            if len(x) == 24 and set(x.hour.astype(int)) == set(range(24)):
                days.append((day, x.sort_values("hour").cif.to_numpy(dtype=float)))
        for method, mapping in center_map.items():
            center = float(mapping[(ba, month)])
            residuals = [y - center for _, y in days]
            scores = np.asarray([np.max(np.abs(r)) for r in residuals])
            for i in range(N, len(days)):
                q = float(np.sort(scores[i-N:i])[math.ceil(.90*(N+1))-1])
                day_cov, hour_cov, winkler = interval_loss(residuals[i], q)
                rows.append({"ba": ba, "month": month, "center_method": method,
                             "test_day": days[i][0], "day_coverage": day_cov,
                             "hour_coverage": hour_cov, "width": 2*q, "winkler90": winkler,
                             "q90": q, "center_g_kwh": center})
    d = pd.DataFrame(rows)
    d.to_csv(out / "daily_predictions.csv", index=False)
    bm = d.groupby(["center_method", "ba", "month"], observed=True).agg(
        day_coverage=("day_coverage", "mean"), hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean"), test_days=("test_day", "nunique")
    ).reset_index()
    bm.to_csv(out / "by_ba_month.csv", index=False)
    ba = bm.groupby(["center_method", "ba"], observed=True).agg(
        day_coverage=("day_coverage", "mean"), hour_coverage=("hour_coverage", "mean"),
        width=("width", "mean"), winkler90=("winkler90", "mean"), test_days=("test_days", "mean")
    ).reset_index()
    ba.to_csv(out / "by_ba.csv", index=False)
    summary = ba.groupby("center_method", observed=True).agg(
        bas=("ba", "nunique"), day_coverage=("day_coverage", "mean"),
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"), mean_test_days_per_ba_month=("test_days", "mean")
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    wide = ba.pivot(index="ba", columns="center_method", values=["day_coverage", "width", "winkler90"])
    rng = np.random.default_rng(2582026)
    ci = []
    for method in ["annual_local_co2_mwh", "monthly_heat_oracle"]:
        for metric in ["day_coverage", "width", "winkler90"]:
            diff = (wide[(metric, method)] - wide[(metric, "annual_heat_rate")]).to_numpy(dtype=float)
            draws = diff[rng.integers(0, len(diff), size=(20_000, len(diff)))].mean(axis=1)
            ci.append({"comparison": f"{method} minus annual_heat_rate", "metric": metric,
                       "difference": float(diff.mean()), "ci_low": float(np.quantile(draws, .025)),
                       "ci_high": float(np.quantile(draws, .975)), "clusters": int(len(diff)),
                       "bootstrap_replicates": 20_000})
    pd.DataFrame(ci).to_csv(out / "paired_ba_cluster_bootstrap.csv", index=False)
    meta = {"protocol": "same test dates and rolling n=18 calibration; centers ablated",
            "targets": TARGETS, "ba_months": int(d[["ba", "month"]].drop_duplicates().shape[0]),
            "calibration_days": N, "target_hour_labels_previously_opened": True,
            "estimand": "EPA CEMS measured fossil-fleet gross-load intensity, not BA whole-grid CIF",
            "boundary": "annual local CO2/MWh uses historical target emissions; monthly heat oracle uses target-month CEMS heat input"}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))
    print("\nBy BA:")
    print(ba.pivot(index="ba", columns="center_method", values=["day_coverage", "width", "winkler90"]).round(3).to_string())


if __name__ == "__main__":
    main()
