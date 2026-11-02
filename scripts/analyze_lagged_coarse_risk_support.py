#!/usr/bin/env python3
"""Check whether 2025 BA risk failures coincide with lagged-input OOD."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import v275_coarse_mix_conditional_risk as base  # noqa: E402
from run_lagged_coarse_risk_transfer import cn5_features, shift_month  # noqa: E402

INPUT = ROOT / "results/v272-eia-monitor"
RISK = ROOT / "results/lagged-coarse-risk-transfer-r1"
OUTPUT = ROOT / "results/lagged-coarse-risk-support-r1"
TRAIN_YEARS = range(2019, 2024)
TEST_YEAR = 2025


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    rows = {ba: base.read_ba(INPUT / f"{ba}.xlsx", max_year=TEST_YEAR) for ba in base.BAS}
    result = []
    variants = [("lag2_cn5", False), ("lag2_cn5_plus_net_import", True)]
    perf = {
        "generated": pd.read_csv(RISK / "generated_by_ba_metrics.csv"),
        "consumed": pd.read_csv(RISK / "consumed_by_ba_metrics.csv"),
    }
    predictions = {
        "generated": pd.read_csv(RISK / "generated_monthly_risk_predictions.csv"),
        "consumed": pd.read_csv(RISK / "consumed_monthly_risk_predictions.csv"),
    }

    risk_history = []
    for scope in ("generated", "consumed"):
        risk_info, _ = base.monthly_table(rows, scope)
        grouped: dict[tuple[str, int], list[float]] = {}
        for (ba, year, _month), row in risk_info.items():
            grouped.setdefault((ba, year), []).append(float(row["risk"]))
        for (ba, year), values in sorted(grouped.items()):
            risk_history.append({
                "label_track": scope,
                "ba": ba,
                "year": year,
                "months": len(values),
                "mean_monthly_daily_q90_max_abs_g_per_kwh": float(np.mean(values)),
                "median_monthly_daily_q90_max_abs_g_per_kwh": float(np.median(values)),
            })
    pd.DataFrame(risk_history).to_csv(OUTPUT / "risk_history_by_ba_year.csv", index=False)

    for variant, with_flow in variants:
        for target in base.BAS:
            train: dict[int, list[np.ndarray]] = {month: [] for month in range(1, 13)}
            target_rows: list[tuple[int, np.ndarray]] = []
            for ba, ba_rows in rows.items():
                for year, month in sorted({(r[0], r[1]) for r in ba_rows}):
                    feature_year, feature_month = shift_month(year, month, -2)
                    x = cn5_features(ba_rows, feature_year, feature_month, with_flow)
                    if x is None:
                        continue
                    if ba != target and year in TRAIN_YEARS:
                        train[month].append(np.asarray(x, dtype=float))
                    elif ba == target and year == TEST_YEAR:
                        target_rows.append((month, np.asarray(x, dtype=float)))

            target_support = []
            for month, x in target_rows:
                peers = np.asarray(train[month], dtype=float)
                if len(peers) < 5:
                    continue
                low, high = peers.min(axis=0), peers.max(axis=0)
                mean, scale = peers.mean(axis=0), peers.std(axis=0)
                scale[scale < 1e-8] = 1.0
                standardized_peers = (peers - mean) / scale
                standardized_target = (x - mean) / scale
                target_support.append({
                    "target_month": month,
                    "out_of_range_coordinate_fraction": float(np.mean((x < low) | (x > high))),
                    "nearest_peer_z_distance": float(np.min(np.linalg.norm(standardized_peers - standardized_target, axis=1))),
                    "max_abs_coordinate_z": float(np.max(np.abs(standardized_target))),
                })
            if not target_support:
                continue

            support_summary = {
                "input_variant": variant,
                "ba": target,
                "target_month_count": len(target_support),
                "mean_out_of_range_coordinate_fraction": float(np.mean([x["out_of_range_coordinate_fraction"] for x in target_support])),
                "mean_nearest_peer_z_distance": float(np.mean([x["nearest_peer_z_distance"] for x in target_support])),
                "max_abs_coordinate_z": float(np.max([x["max_abs_coordinate_z"] for x in target_support])),
            }
            for scope in ("generated", "consumed"):
                if scope == "generated" and with_flow:
                    continue
                feature_variant = variant if scope == "consumed" else "lag2_cn5"
                metric_rows = perf[scope]
                chosen = metric_rows[(metric_rows.feature_variant == feature_variant) & (metric_rows.ba == target)]
                for _, metric in chosen.iterrows():
                    if metric.method not in ("conditional", "pooled_source"):
                        continue
                    prediction = predictions[scope]
                    prediction = prediction[(prediction.feature_variant == feature_variant) & (prediction.ba == target)]
                    support_and_perf = {
                        **support_summary,
                        "label_track": scope,
                        "risk_method": metric.method,
                        "daily_coverage": float(metric.daily_simultaneous_coverage),
                        "mean_interval_width_g_per_kwh": float(metric.mean_interval_width_g_per_kwh),
                        "winkler90": float(metric.winkler90),
                        "monthly_risk_mae_g_per_kwh": float(metric.monthly_risk_mae_g_per_kwh),
                    }
                    result.append(support_and_perf)

    df = pd.DataFrame(result)
    df.to_csv(OUTPUT / "support_and_risk_by_ba.csv", index=False)
    metadata = {
        "input": "final EIA BA workbook history, lag-simulated t-2 features",
        "support_reference": "peer BAs, training target years 2019-2023, matched calendar month, target BA excluded",
        "support_metrics": "fraction of five-category-share/log-total/(optional flow) coordinates outside peer min-max; nearest peer Euclidean distance after peer standardization",
        "test_year": TEST_YEAR,
        "source_ba_count": len(base.BAS),
        "output": str((OUTPUT / "support_and_risk_by_ba.csv").relative_to(ROOT)),
    }
    (OUTPUT / "provenance.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(df[df.ba == "BANC"].to_string(index=False))
    print("\nHistorical monthly q90 risk:")
    print(pd.DataFrame(risk_history).query("ba == 'BANC'").to_string(index=False))
    print("\nCoverage/support correlations are descriptive only; target support is only seven BAs.")


if __name__ == "__main__":
    main()
