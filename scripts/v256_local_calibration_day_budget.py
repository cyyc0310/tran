#!/usr/bin/env python3
"""Measure the target-local complete-day calibration budget for V253 centers."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["FMPP", "GVL", "IID", "JEA", "PSCO", "SC", "SEC", "TAL", "TEC", "TIDC"]
BUDGETS = [9, 12, 18, 24]
REPEATS = 200
SEED = 2562026
TONS_TO_G_KWH = 907.18474


def interval_metrics(score_by_day: dict[str, float], residual_by_day: dict[str, np.ndarray], calibration_days: list[str],
                     test_days: list[str], n: int):
    scores = np.sort(np.fromiter((score_by_day[d] for d in calibration_days), dtype=float))
    # Finite-sample split-conformal order statistic, 1-indexed in the protocol.
    rank = math.ceil(0.90 * (n + 1))
    if rank > n:
        return None
    q = float(scores[rank - 1])
    if not test_days:
        return None
    widths = 2 * q
    residual = np.concatenate([residual_by_day[d] for d in test_days])
    outside = np.abs(residual) > q
    winkler = np.full(len(residual), widths, dtype=float)
    winkler[residual < -q] += 20 * (-q - residual[residual < -q])
    winkler[residual > q] += 20 * (residual[residual > q] - q)
    day_outside = np.fromiter((score_by_day[d] > q for d in test_days), dtype=bool)
    return {
        "test_hours": int(len(residual)),
        "test_days": int(len(test_days)),
        "hour_coverage": float((~outside).mean()),
        "simultaneous_day_coverage": float((~day_outside).mean()),
        "width": widths,
        "winkler90": float(winkler.mean()),
        "q90": q,
        "calibration_days": n,
        "conformal_rank": rank,
    }


def main():
    out = ROOT / "results/v256-local-calibration-day-budget"
    out.mkdir(parents=True, exist_ok=True)
    hourly_path = ROOT / "results/v253-annual-heat-hourly-interval/target_2025_hourly.csv"
    annual_path = ROOT / "results/v252-annual-local-heat-rate/target_2025_ba_month_predictions.csv"
    h = pd.read_csv(hourly_path)
    p = pd.read_csv(annual_path)
    h = h[h.ba.isin(TARGETS)].copy()
    h["day"] = pd.to_datetime(h.day).dt.strftime("%Y-%m-%d")
    h["cif"] = h.tons / h.mwh * TONS_TO_G_KWH
    monthly_mwh = h.groupby(["ba", "month"], observed=True).mwh.sum().rename("hourly_mwh")
    p = p[p.ba.isin(TARGETS)].merge(monthly_mwh, on=["ba", "month"], how="inner", validate="one_to_one")
    p["center"] = p.annual_heat_rate_tons / p.hourly_mwh * TONS_TO_G_KWH
    centers = p.set_index(["ba", "month"]).center.to_dict()

    rows = []
    split_rows = []
    rng = np.random.default_rng(SEED)
    groups = list(h.groupby(["ba", "month"], observed=True, sort=True))
    for (ba, month), g in groups:
        center = float(centers[(ba, month)])
        daily = g.groupby("day", observed=True).filter(
            lambda x: len(x) == 24 and set(x.hour.astype(int)) == set(range(24))
        )
        days = sorted(daily.day.unique())
        if len(days) < max(BUDGETS) + 2:
            continue
        daily = daily.copy()
        daily["residual"] = daily.cif - center
        daily["residual_abs"] = daily.residual.abs()
        by_day = {d: x["residual"].to_numpy(dtype=float) for d, x in daily.groupby("day", observed=True)}
        score_by_day = {d: float(np.max(np.abs(values))) for d, values in by_day.items()}
        for rep in range(REPEATS):
            perm = rng.permutation(days).tolist()
            for n in BUDGETS:
                cal, test = perm[:n], perm[n:]
                m = interval_metrics(score_by_day, by_day, cal, test, n)
                rows.append({"ba": ba, "month": month, "split": "random_same_month", "repeat": rep, **m})
        ordered = sorted(days)
        for n in BUDGETS:
            cal, test = ordered[:n], ordered[n:]
            m = interval_metrics(score_by_day, by_day, cal, test, n)
            rows.append({"ba": ba, "month": month, "split": "chronological_early_to_late", "repeat": 0, **m})
        split_rows.append({"ba": ba, "month": month, "complete_days": len(days), "center_g_kwh": center})

    result = pd.DataFrame(rows)
    result.to_csv(out / "budget_repeats.csv", index=False)
    pd.DataFrame(split_rows).to_csv(out / "support.csv", index=False)
    # First aggregate within BA, then equally weight BAs. This prevents high-volume BAs dominating.
    ba_mean = result.groupby(["split", "calibration_days", "ba"], observed=True).agg(
        simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"), test_days=("test_days", "mean"),
    ).reset_index()
    summary = ba_mean.groupby(["split", "calibration_days"], observed=True).agg(
        bas=("ba", "nunique"), simultaneous_day_coverage=("simultaneous_day_coverage", "mean"),
        hour_coverage=("hour_coverage", "mean"), width=("width", "mean"),
        winkler90=("winkler90", "mean"), mean_test_days=("test_days", "mean"),
        ba_sd_day_coverage=("simultaneous_day_coverage", "std"),
    ).reset_index()
    summary.to_csv(out / "summary.csv", index=False)
    ba_mean.to_csv(out / "by_ba.csv", index=False)
    boot_rng = np.random.default_rng(SEED + 1)
    ci_rows = []
    for (split, n), group in ba_mean.groupby(["split", "calibration_days"], observed=True):
        group = group.sort_values("ba")
        for metric in ["simultaneous_day_coverage", "width", "winkler90"]:
            values = group[metric].to_numpy(dtype=float)
            samples = values[boot_rng.integers(0, len(values), size=(20_000, len(values)))].mean(axis=1)
            ci_rows.append({"split": split, "calibration_days": int(n), "metric": metric,
                            "estimate": float(values.mean()),
                            "ci_low": float(np.quantile(samples, 0.025)),
                            "ci_high": float(np.quantile(samples, 0.975)),
                            "clusters": int(len(values)), "bootstrap_replicates": 20_000})
    pd.DataFrame(ci_rows).to_csv(out / "ba_cluster_bootstrap_ci.csv", index=False)

    meta = {
        "protocol": "fixed V253 annual-heat-rate center; local day-max residual conformal calibration; n=9,12,18,24",
        "targets": TARGETS, "target_ba_months": int(result[["ba", "month"]].drop_duplicates().shape[0]),
        "complete_day_min": int(pd.DataFrame(split_rows).complete_days.min()),
        "complete_day_max": int(pd.DataFrame(split_rows).complete_days.max()),
        "splits": ["200 repeated random same-month nested day splits", "earliest n days calibrate, later days test"],
        "repeats": REPEATS, "seed": SEED, "finite_sample_rank": "ceil(0.90*(n+1)), 1-indexed",
        "target_hour_labels_previously_opened": True,
        "estimand": "EPA CEMS measured fossil-fleet gross-load intensity, not BA whole-grid CIF",
        "center_information": "2024 target-BA annual heat-rate estimate and 2025 target-month gross MWh",
        "boundary": "Retrospective within-month calibration-budget experiment; not month-start forecasting",
    }
    (out / "metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
