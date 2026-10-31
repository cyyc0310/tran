#!/usr/bin/env python3
"""Audit ex-ante availability of monthly generation mix for threshold identification.

This is a composition-state stability analysis, not an hourly CIF forecast.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def build_panel(source: Path) -> pd.DataFrame:
    mix = pd.read_csv(source)
    keep = ["province", "year", "month", "total_100m_kwh", "thermal_100m_kwh"]
    panel = mix[keep].copy()
    panel["period"] = panel["year"] * 12 + panel["month"]
    panel["local_share"] = panel["thermal_100m_kwh"] / panel["total_100m_kwh"]
    # Match the V379/V392 binary certificate: does the composition-weighted
    # thermal EF support [400*s, 1000*s] straddle tau=300?
    panel["local_state_tau300"] = (
        (400 * panel["local_share"] <= 300) & (300 < 1000 * panel["local_share"])
    ).where(panel["local_share"].notna())

    national = (
        panel.groupby("period", as_index=False)
        .agg(national_thermal=("thermal_100m_kwh", "sum"), national_total=("total_100m_kwh", "sum"))
    )
    national["national_share"] = national["national_thermal"] / national["national_total"]
    national["national_state_tau300"] = (
        (400 * national["national_share"] <= 300) & (300 < 1000 * national["national_share"])
    ).where(national["national_share"].notna())
    panel = panel.merge(national[["period", "national_share", "national_state_tau300"]], on="period", how="left")

    for lag in (1, 2):
        local_lag = panel[["province", "period", "local_share", "local_state_tau300"]].copy()
        local_lag["period"] += lag
        local_lag = local_lag.rename(
            columns={
                "local_share": f"local_share_lag{lag}",
                "local_state_tau300": f"local_state_tau300_lag{lag}",
            }
        )
        panel = panel.merge(local_lag, on=["province", "period"], how="left")

        national_lag = national[["period", "national_share", "national_state_tau300"]].copy()
        national_lag["period"] += lag
        national_lag = national_lag.rename(
            columns={
                "national_share": f"national_share_lag{lag}",
                "national_state_tau300": f"national_state_tau300_lag{lag}",
            }
        )
        panel = panel.merge(national_lag, on="period", how="left")

    return panel.drop(columns="period").sort_values(["province", "year", "month"])


def score(panel: pd.DataFrame, prediction: str) -> dict[str, object]:
    valid = panel.dropna(subset=["local_state_tau300", prediction]).copy()
    y = valid["local_state_tau300"].astype(int)
    pred = valid[prediction].astype(int)
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    return {
        "prediction": prediction,
        "n": int(len(valid)),
        "agreement": float((y == pred).mean()) if len(valid) else None,
        "positive_prevalence": float(y.mean()) if len(valid) else None,
        "sensitivity": float(((y == 1) & (pred == 1)).sum() / positives) if positives else None,
        "specificity": float(((y == 0) & (pred == 0)).sum() / negatives) if negatives else None,
        "state_changes": int((y != pred).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/v392-nbs-official-provincial-long-panel-r1/provincial_mix_and_closure.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/v393-release-lag-identifiability-r1"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    panel = build_panel(args.input)
    panel.to_csv(args.output_dir / "release_lag_panel.csv", index=False)
    scores = [
        score(panel, "local_state_tau300_lag1"),
        score(panel, "national_state_tau300_lag1"),
        score(panel, "local_state_tau300_lag2"),
        score(panel, "national_state_tau300_lag2"),
    ]
    province_scores = []
    for lag in (1, 2):
        prediction = f"local_state_tau300_lag{lag}"
        for province, rows in panel.groupby("province"):
            item = score(rows, prediction)
            item.update({"province": province, "lag_months": lag})
            province_scores.append(item)
    train = panel[(panel["year"] <= 2022) & panel["local_state_tau300"].notna()]
    province_mode = train.groupby("province")["local_state_tau300"].agg(lambda x: int(x.mean() >= 0.5))
    global_mode = int(train["local_state_tau300"].mean() >= 0.5)
    panel["province_static_state_train_through_2022"] = panel["province"].map(province_mode)
    panel["pooled_static_state_train_through_2022"] = global_mode
    forward_predictions = [
        "local_state_tau300_lag2",
        "national_state_tau300_lag2",
        "province_static_state_train_through_2022",
        "pooled_static_state_train_through_2022",
    ]
    forward = panel[panel["year"] >= 2023].dropna(
        subset=["local_state_tau300", *forward_predictions]
    ).copy()
    forward_scores = [score(forward, name) for name in forward_predictions]
    pd.DataFrame(scores).to_csv(args.output_dir / "state_agreement_summary.csv", index=False)
    pd.DataFrame(province_scores).to_csv(args.output_dir / "state_agreement_by_province.csv", index=False)
    pd.DataFrame(forward_scores).to_csv(args.output_dir / "forward_holdout_summary.csv", index=False)
    forward.to_csv(args.output_dir / "forward_holdout_predictions.csv", index=False)
    summary = {
        "input": str(args.input),
        "threshold_g_kwh": 300,
        "thermal_support_lower_bound_g_kwh": 400,
        "crossing_thermal_share_interval": "(0.30, 0.75]",
        "interpretation": "Agreement for the V392 binary mean-support cross-threshold certificate; not hourly CIF forecast accuracy or interval coverage.",
        "release_lag_assumption": "At month-start, use month t-2 as the latest monthly composition; t-1 is shown as a mid-month sensitivity only.",
        "metrics": scores,
        "forward_holdout": {
            "training_cutoff_year": 2022,
            "test_start_year": 2023,
            "test_end": "2026-09",
            "static_baselines_fit_only_on_training_target_states": True,
            "metrics": forward_scores,
        },
    }
    (args.output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
