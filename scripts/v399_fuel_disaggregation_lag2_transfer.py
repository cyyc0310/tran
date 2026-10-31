#!/usr/bin/env python3
"""V399: measure the source-domain value of lagged fuel disaggregation."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v369_crossfit_risk_intervals import (  # noqa: E402
    ALPHAS,
    FUELS,
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
    RIDGE_ALPHA,
    add_features,
)

OUT = ROOT / "results/v399-fuel-disaggregation-lag2-r1"
FULL_SHARE_FEATURES = [f"share_{fuel}_lag2" for fuel in FUELS]
FULL_MIX_INTERACTIONS = [
    f"share_{fuel}_x_{calendar}" for fuel in FUELS
    for calendar in ("hour_sin", "hour_cos", "month_sin", "month_cos")
]
MODELS = {
    "calendar_ridge_no_mix": CALENDAR_FEATURES,
    "calendar_ridge_thermal_lag2": [*CALENDAR_FEATURES, *MIX_FEATURES],
    "calendar_ridge_eight_fuel_lag2": [*CALENDAR_FEATURES, *FULL_SHARE_FEATURES, *FULL_MIX_INTERACTIONS],
}


def add_fuel_share_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["period"] = result["year"] * 12 + result["month"]
    monthly = result.groupby(["region", "period"], as_index=False)[list(FUELS)].sum()
    monthly["total"] = monthly[list(FUELS)].sum(axis=1)
    for fuel in FUELS:
        monthly[f"share_{fuel}_lag2"] = monthly[fuel] / monthly["total"]
    lagged = monthly[["region", "period", *FULL_SHARE_FEATURES]].copy()
    lagged["period"] += 2
    result = result.merge(lagged, on=["region", "period"], how="left")
    for fuel in FUELS:
        share = result[f"share_{fuel}_lag2"]
        for calendar in ("hour_sin", "hour_cos", "month_sin", "month_cos"):
            result[f"share_{fuel}_x_{calendar}"] = share * result[calendar]
    return result


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, feature_names: list[str]) -> np.ndarray:
    usable = train.dropna(subset=[*feature_names, "cif_proxy"])
    if usable.empty or target[feature_names].isna().any().any():
        raise RuntimeError("Missing lagged composition features in train/calibration/test rows")
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(usable[feature_names].to_numpy(float), usable["cif_proxy"].to_numpy(float))
    return np.clip(model.predict(target[feature_names].to_numpy(float)), 0, 1000)


def run_method(method: str, features: list[str], frame: pd.DataFrame, heldout: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = frame[frame.region != heldout]
    train = source[source.year == TRAIN_YEAR].copy()
    calibration = source[source.year == CAL_YEAR].copy()
    test = frame[(frame.region == heldout) & (frame.year == TEST_YEAR)].copy()
    calibration["prediction"] = fit_predict(train, calibration, features)
    test["prediction"] = fit_predict(train, test, features)
    blocks = []
    for (region, year, month), block in calibration.groupby(["region", "year", "month"]):
        record = {
            "region": region, "year": int(year), "month": int(month),
            "share_thermal_lag2": float(block["thermal_share_lag2"].iloc[0]),
            "max_abs_hourly_residual": float(np.max(np.abs(block.cif_proxy - block.prediction))),
        }
        for threshold in THRESHOLDS:
            record[f"tail_fraction_error_{int(threshold)}"] = abs(
                float(np.mean(block.cif_proxy > threshold)) - float(np.mean(block.prediction > threshold))
            )
        blocks.append(record)
    calibration_blocks = pd.DataFrame(blocks)
    scores = []
    for (year, month), block in test.groupby(["year", "month"]):
        actual = block.cif_proxy.to_numpy(float)
        point = block.prediction.to_numpy(float)
        for alpha in ALPHAS:
            radius = quantile(calibration_blocks.max_abs_hourly_residual.to_numpy(float), 1 - alpha)
            lower, upper = np.clip(point - radius, 0, 1000), np.clip(point + radius, 0, 1000)
            for threshold in THRESHOLDS:
                observed = int(np.sum(actual > threshold))
                predicted = int(np.sum(point > threshold))
                tail_radius = quantile(
                    calibration_blocks[f"tail_fraction_error_{int(threshold)}"].to_numpy(float), 1 - alpha
                )
                hours = len(actual)
                radius_hours = int(np.ceil(tail_radius * hours))
                lo_count, hi_count = max(0, predicted - radius_hours), min(hours, predicted + radius_hours)
                scores.append({
                    "region": heldout,
                    "year": int(year),
                    "month": int(month),
                    "method": method,
                    "alpha": alpha,
                    "nominal_coverage": 1 - alpha,
                    "threshold_g_kwh": threshold,
                    "thermal_share_lag2": float(block.thermal_share_lag2.iloc[0]),
                    "n_hours": hours,
                    "hourly_mae": float(np.mean(np.abs(actual - point))),
                    "hourly_coverage": float(np.mean((actual >= lower) & (actual <= upper))),
                    "hourly_width": float(np.mean(upper - lower)),
                    "hourly_interval_score": float(np.mean(interval_score(actual, lower, upper, alpha))),
                    "observed_tail_count": observed,
                    "predicted_tail_count": predicted,
                    "tail_count_lower": lo_count,
                    "tail_count_upper": hi_count,
                    "tail_count_covered": lo_count <= observed <= hi_count,
                    "tail_count_width_hours": hi_count - lo_count,
                    "tail_count_interval_score": float(interval_score(
                        np.array([observed]), np.array([lo_count]), np.array([hi_count]), alpha
                    )[0]),
                })
    return pd.DataFrame(scores), calibration_blocks


def bootstrap_region_gain(paired: pd.DataFrame, column: str) -> dict:
    values = paired.groupby("region")[column].mean().to_numpy(float)
    rng = np.random.default_rng(399)
    draw = rng.choice(values, size=(20_000, len(values)), replace=True).mean(axis=1)
    return {"mean": float(values.mean()), "ci95_low": float(np.quantile(draw, .025)),
            "ci95_high": float(np.quantile(draw, .975)), "regions_positive": int((values > 0).sum()),
            "region_clusters": len(values)}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    frame = add_fuel_share_features(add_features(load_hourly()))
    frame = frame[frame.year.isin([TRAIN_YEAR, CAL_YEAR, TEST_YEAR])].copy()
    # A common-case mask keeps the no-mix, thermal-only, and eight-fuel arms paired.
    frame = frame.dropna(subset=["thermal_share_lag2", *FULL_SHARE_FEATURES]).copy()
    scores, calibration_parts = [], []
    for heldout in REGIONS:
        if frame[(frame.region == heldout) & (frame.year == TEST_YEAR)].empty:
            continue
        for method, features in MODELS.items():
            method_scores, cal_blocks = run_method(method, features, frame, heldout)
            scores.append(method_scores)
            cal_blocks["heldout_region"] = heldout
            cal_blocks["method"] = method
            calibration_parts.append(cal_blocks)
    detail = pd.concat(scores, ignore_index=True)
    calibration = pd.concat(calibration_parts, ignore_index=True)
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_scores.csv", index=False)
    calibration.to_csv(OUT / "calibration_blocks.csv", index=False)

    region = detail.groupby(["method", "alpha", "threshold_g_kwh", "region"], as_index=False).agg(
        months=("month", "nunique"),
        mae=("hourly_mae", "mean"),
        hourly_coverage=("hourly_coverage", "mean"),
        hourly_width=("hourly_width", "mean"),
        interval_score=("hourly_interval_score", "mean"),
        tail_count_coverage=("tail_count_covered", "mean"),
        tail_count_width=("tail_count_width_hours", "mean"),
        tail_count_interval_score=("tail_count_interval_score", "mean"),
    )
    region.to_csv(OUT / "region_summary.csv", index=False)
    summary = region.groupby(["method", "alpha", "threshold_g_kwh"], as_index=False).agg(
        regions=("region", "nunique"),
        region_equal_mae=("mae", "mean"),
        region_equal_hourly_coverage=("hourly_coverage", "mean"),
        worst_region_hourly_coverage=("hourly_coverage", "min"),
        region_equal_width=("hourly_width", "mean"),
        region_equal_interval_score=("interval_score", "mean"),
        region_equal_tail_count_coverage=("tail_count_coverage", "mean"),
        worst_region_tail_count_coverage=("tail_count_coverage", "min"),
        region_equal_tail_count_width_hours=("tail_count_width", "mean"),
        region_equal_tail_count_interval_score=("tail_count_interval_score", "mean"),
    )
    summary.to_csv(OUT / "method_summary.csv", index=False)
    paired = region.pivot(index=["alpha", "threshold_g_kwh", "region"], columns="method",
                          values=["mae", "hourly_width", "interval_score", "hourly_coverage",
                                  "tail_count_width", "tail_count_interval_score"]).reset_index()
    paired.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col)
                      for col in paired.columns]
    base = "calendar_ridge_thermal_lag2"
    full = "calendar_ridge_eight_fuel_lag2"
    for metric in ["mae", "hourly_width", "interval_score", "hourly_coverage",
                   "tail_count_width", "tail_count_interval_score"]:
        paired[f"gain_{metric}_eight_fuel"] = (
            paired[f"{metric}_{base}"] - paired[f"{metric}_{full}"] if metric != "hourly_coverage"
            else paired[f"{metric}_{full}"] - paired[f"{metric}_{base}"]
        )
    paired.to_csv(OUT / "paired_fuel_detail_gains.csv", index=False)
    bootstrap = {}
    for alpha, rows in paired.groupby("alpha"):
        bootstrap[str(alpha)] = {
            col: bootstrap_region_gain(rows, col)
            for col in ["gain_mae_eight_fuel", "gain_hourly_width_eight_fuel", "gain_interval_score_eight_fuel",
                        "gain_hourly_coverage_eight_fuel", "gain_tail_count_width_eight_fuel",
                        "gain_tail_count_interval_score_eight_fuel"]
        }
    metadata = {
        "experiment": "lagged eight-fuel disaggregation information ablation",
        "target": "EIA fixed-factor hourly generation CIF proxy, not measured emissions",
        "train_year": TRAIN_YEAR, "calibration_year": CAL_YEAR, "test_year": TEST_YEAR,
        "regions": list(REGIONS), "ridge_alpha": RIDGE_ALPHA,
        "models": {name: feature_list for name, feature_list in MODELS.items()},
        "input_feature_timing": "each source-region fuel share is derived from t-2 monthly generation only",
        "bootstrap": bootstrap,
        "caveats": [
            "2024 remains an adaptive development holdout; this is not a new confirmation.",
            "Eight fuel shares are an information upper bound unavailable in Chinese NBS monthly data where thermal is aggregated.",
            "Target is a fixed-factor generation proxy, not direct measured CIF.",
            "Only seven region clusters and one test year; calibration is descriptive under temporal dependence.",
        ],
        "summary_rows": summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
