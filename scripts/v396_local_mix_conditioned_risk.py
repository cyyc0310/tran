#!/usr/bin/env python3
"""V396: condition block-risk calibration on the as-of t-2 local mix feature."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v369_crossfit_risk_intervals import (  # noqa: E402
    ALPHAS,
    REGIONS,
    THRESHOLDS,
    load_hourly,
    interval_score,
    quantile,
)
from v395_temporal_holdout_lagged_mix_cif import (  # noqa: E402
    CALENDAR_FEATURES,
    MIX_FEATURES,
    TRAIN_YEAR,
    CAL_YEAR,
    TEST_YEAR,
    add_features,
    fit_predict,
)

OUT = ROOT / "results/v396-local-mix-conditioned-risk-r1"
BANDWIDTHS = (0.05, 0.10, 0.20, 0.40, 0.80)
METHOD = "calendar_ridge_local_lag2_mix"


def weighted_quantile(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    if not np.isfinite(weights).all() or weights.sum() <= 0:
        raise ValueError("Invalid calibration weights")
    index = min(int(np.searchsorted(np.cumsum(weights), probability * weights.sum(), side="left")), len(values) - 1)
    return float(values[index])


def kernel_weights(source_shares: np.ndarray, target_share: float, bandwidth: float) -> np.ndarray:
    weights = np.exp(-0.5 * np.square((source_shares - target_share) / bandwidth))
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        # Nearest-support fallback is explicit and recorded through the effective sample size.
        weights = np.zeros_like(source_shares, dtype=float)
        weights[int(np.argmin(np.abs(source_shares - target_share)))] = 1.0
    return weights


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    data = add_features(load_hourly())
    rows = []
    for target_region in REGIONS:
        donor_data = data[data["region"] != target_region]
        train = donor_data[donor_data["year"] == TRAIN_YEAR].dropna(subset=["thermal_share_lag2"]).copy()
        calibration = donor_data[donor_data["year"] == CAL_YEAR].dropna(subset=["thermal_share_lag2"]).copy()
        test = data[(data["region"] == target_region) & (data["year"] == TEST_YEAR)].dropna(
            subset=["thermal_share_lag2"]
        ).copy()
        features = [*CALENDAR_FEATURES, *MIX_FEATURES]
        calibration["prediction"] = fit_predict(train, calibration, features)
        test["prediction"] = fit_predict(train, test, features)

        cal_blocks = []
        for (source_region, year, month), block in calibration.groupby(["region", "year", "month"]):
            cal_blocks.append({
                "region": source_region,
                "year": int(year),
                "month": int(month),
                "thermal_share_lag2": float(block["thermal_share_lag2"].iloc[0]),
                "max_hourly_abs_residual": float(np.max(np.abs(block["cif_proxy"] - block["prediction"]))),
                **{
                    f"tail_fraction_error_{int(threshold)}": abs(
                        np.mean(block["cif_proxy"].to_numpy(float) > threshold)
                        - np.mean(block["prediction"].to_numpy(float) > threshold)
                    )
                    for threshold in THRESHOLDS
                },
            })
        cal = pd.DataFrame(cal_blocks)

        for (year, month), target in test.groupby(["year", "month"]):
            share = float(target["thermal_share_lag2"].iloc[0])
            y = target["cif_proxy"].to_numpy(float)
            point = target["prediction"].to_numpy(float)
            for alpha in ALPHAS:
                for bandwidth in BANDWIDTHS:
                    weights = kernel_weights(cal["thermal_share_lag2"].to_numpy(float), share, bandwidth)
                    ess = float(weights.sum() ** 2 / np.square(weights).sum())
                    hourly_radius = weighted_quantile(
                        cal["max_hourly_abs_residual"].to_numpy(float), weights, 1.0 - alpha
                    )
                    lo, hi = np.clip(point - hourly_radius, 0, 1000), np.clip(point + hourly_radius, 0, 1000)
                    for threshold in THRESHOLDS:
                        count_radius_fraction = weighted_quantile(
                            cal[f"tail_fraction_error_{int(threshold)}"].to_numpy(float), weights, 1.0 - alpha
                        )
                        observed = int(np.sum(y > threshold))
                        predicted = int(np.sum(point > threshold))
                        radius_hours = int(np.ceil(count_radius_fraction * len(y)))
                        lower, upper = max(0, predicted - radius_hours), min(len(y), predicted + radius_hours)
                        rows.append({
                            "region": target_region,
                            "year": int(year),
                            "month": int(month),
                            "alpha": alpha,
                            "nominal_coverage": 1.0 - alpha,
                            "threshold_g_kwh": threshold,
                            "method": f"local_mix_kernel_bw_{bandwidth:.2f}",
                            "thermal_share_lag2": share,
                            "effective_calibration_blocks": ess,
                            "hourly_mae_g_kwh": float(np.mean(np.abs(y - point))),
                            "hourly_coverage": float(np.mean((y >= lo) & (y <= hi))),
                            "hourly_width_g_kwh": float(np.mean(hi - lo)),
                            "hourly_interval_score": float(np.mean(interval_score(y, lo, hi, alpha))),
                            "observed_tail_count": observed,
                            "predicted_tail_count": predicted,
                            "tail_count_lower": lower,
                            "tail_count_upper": upper,
                            "tail_count_covered": lower <= observed <= upper,
                            "tail_count_width_hours": upper - lower,
                            "tail_count_interval_score": float(
                                interval_score(np.array([observed]), np.array([lower]), np.array([upper]), alpha)[0]
                            ),
                        })

    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_scores.csv", index=False)

    region = detail.groupby(
        ["method", "alpha", "nominal_coverage", "threshold_g_kwh", "region"], as_index=False
    ).agg(
        region_months=("month", "size"),
        hourly_coverage=("hourly_coverage", "mean"),
        hourly_width_g_kwh=("hourly_width_g_kwh", "mean"),
        hourly_interval_score=("hourly_interval_score", "mean"),
        tail_count_coverage=("tail_count_covered", "mean"),
        tail_count_width_hours=("tail_count_width_hours", "mean"),
        tail_count_interval_score=("tail_count_interval_score", "mean"),
        effective_calibration_blocks=("effective_calibration_blocks", "mean"),
    )
    region.to_csv(OUT / "region_summary.csv", index=False)
    summary = region.groupby(["method", "alpha", "nominal_coverage", "threshold_g_kwh"], as_index=False).agg(
        regions=("region", "nunique"),
        region_equal_hourly_coverage=("hourly_coverage", "mean"),
        worst_region_hourly_coverage=("hourly_coverage", "min"),
        region_equal_hourly_width_g_kwh=("hourly_width_g_kwh", "mean"),
        region_equal_hourly_interval_score=("hourly_interval_score", "mean"),
        region_equal_tail_count_coverage=("tail_count_coverage", "mean"),
        worst_region_tail_count_coverage=("tail_count_coverage", "min"),
        region_equal_tail_count_width_hours=("tail_count_width_hours", "mean"),
        region_equal_tail_count_interval_score=("tail_count_interval_score", "mean"),
        mean_effective_calibration_blocks=("effective_calibration_blocks", "mean"),
    )
    summary.to_csv(OUT / "method_summary.csv", index=False)
    metadata = {
        "experiment": "local-mix-conditioned block-risk calibration using the t-2 feature",
        "frozen_point_model": METHOD,
        "train_year": TRAIN_YEAR,
        "calibration_year": CAL_YEAR,
        "test_year": TEST_YEAR,
        "bandwidths": list(BANDWIDTHS),
        "calibration_unit": "donor-region-month block maxima for hourly absolute residuals; block tail-fraction errors for tail counts",
        "output_rows": int(len(detail)),
        "caveats": [
            "Adaptive follow-up on the already-viewed 2024 EIA proxy holdout; not a fresh confirmation set.",
            "Only seven EIA balancing authorities; source is not Chinese or directly measured emissions.",
            "Weighted calibration may reduce effective support; report ESS and do not claim nominal coverage guarantees.",
            "Hourly data vintages are retrospective; t-2 is a conservative lag convention, not a verified issue date.",
        ],
        "summary_rows": summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
