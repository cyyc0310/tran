#!/usr/bin/env python3
"""V397: directly predict monthly hourly tail frequency from legal lagged mix."""

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
from v369_crossfit_risk_intervals import REGIONS, THRESHOLDS, load_hourly, interval_score  # noqa: E402
from v395_temporal_holdout_lagged_mix_cif import TRAIN_YEAR, CAL_YEAR, TEST_YEAR, add_features  # noqa: E402

OUT = ROOT / "results/v397-direct-tail-frequency-r1"
ALPHAS = (0.10, 0.05)
BANDWIDTHS = (None, 0.10, 0.20, 0.40)
RIDGE_ALPHA = 1.0
SEED = 397
BOOTSTRAPS = 20_000


def make_x(frame: pd.DataFrame, use_mix: bool) -> pd.DataFrame:
    x = pd.DataFrame(index=frame.index)
    x["month_sin"] = frame["month_sin"]
    x["month_cos"] = frame["month_cos"]
    if use_mix:
        share = frame["thermal_share_lag2"]
        x["share"] = share
        x["share_month_sin"] = share * frame["month_sin"]
        x["share_month_cos"] = share * frame["month_cos"]
    return x


def weighted_higher(values: np.ndarray, weights: np.ndarray, probability: float) -> float:
    order = np.argsort(values, kind="mergesort")
    ordered_values, ordered_weights = values[order], weights[order]
    cutoff = probability * ordered_weights.sum()
    index = min(int(np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")), len(values) - 1)
    return float(ordered_values[index])


def interval_score_fraction(y: float, lower: float, upper: float, alpha: float) -> float:
    return float((upper - lower) + (2 / alpha) * max(lower - y, 0) + (2 / alpha) * max(y - upper, 0))


def summarize_cluster_gain(paired: pd.DataFrame, value: str) -> dict:
    cluster = paired.groupby("region")[value].mean().to_numpy(float)
    rng = np.random.default_rng(SEED)
    draws = rng.choice(cluster, size=(BOOTSTRAPS, len(cluster)), replace=True).mean(axis=1)
    return {
        "mean": float(cluster.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "regions_positive": int((cluster > 0).sum()),
        "regions": int(len(cluster)),
    }


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    hourly = add_features(load_hourly())
    monthly_rows = []
    for (region, year, month), block in hourly.groupby(["region", "year", "month"]):
        expected = int(pd.Period(year=int(year), month=int(month), freq="M").days_in_month * 24)
        if len(block) / expected < 0.80 or block["thermal_share_lag2"].isna().any():
            continue
        row = {
            "region": region,
            "year": int(year),
            "month": int(month),
            "n_hours": int(len(block)),
            "thermal_share_lag2": float(block["thermal_share_lag2"].iloc[0]),
            "month_sin": float(block["month_sin"].iloc[0]),
            "month_cos": float(block["month_cos"].iloc[0]),
        }
        for threshold in THRESHOLDS:
            row[f"tail_fraction_{int(threshold)}"] = float(np.mean(block["cif_proxy"] > threshold))
            row[f"tail_count_{int(threshold)}"] = int(np.sum(block["cif_proxy"] > threshold))
        monthly_rows.append(row)
    monthly = pd.DataFrame(monthly_rows)
    if monthly.empty:
        raise RuntimeError("No region-months meet the quality and t-2 information contract")

    rows = []
    for heldout in REGIONS:
        source = monthly[monthly.region != heldout]
        train = source[source.year == TRAIN_YEAR].copy()
        cal = source[source.year == CAL_YEAR].copy()
        test = monthly[(monthly.region == heldout) & (monthly.year == TEST_YEAR)].copy()
        if train.empty or cal.empty or test.empty:
            raise RuntimeError(f"missing train/cal/test region-month for {heldout}")

        for threshold in THRESHOLDS:
            target_name = f"tail_fraction_{int(threshold)}"
            for use_mix, model_name in [(False, "seasonal_ridge_no_mix"), (True, "seasonal_ridge_local_lag2_mix")]:
                train_x = make_x(train, use_mix)
                cal_x = make_x(cal, use_mix)
                test_x = make_x(test, use_mix)
                model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
                model.fit(train_x.to_numpy(float), train[target_name].to_numpy(float))
                cal_pred = np.clip(model.predict(cal_x.to_numpy(float)), 0.0, 1.0)
                test_pred = np.clip(model.predict(test_x.to_numpy(float)), 0.0, 1.0)
                cal = cal.copy()
                cal[f"pred_{model_name}_{int(threshold)}"] = cal_pred
                cal[f"abs_error_{model_name}_{int(threshold)}"] = np.abs(cal[target_name].to_numpy(float) - cal_pred)
                test = test.copy()
                test[f"pred_{model_name}_{int(threshold)}"] = test_pred

                cal_error = cal[f"abs_error_{model_name}_{int(threshold)}"].to_numpy(float)
                for idx, (_, target) in enumerate(test.iterrows()):
                    prediction = float(test_pred[idx])
                    share = float(target.thermal_share_lag2)
                    for bandwidth in BANDWIDTHS:
                        if bandwidth is None:
                            radius = float(np.quantile(cal_error, 0.95, method="higher"))
                            ess = float(len(cal_error))
                            calibration = "global"
                        else:
                            source_share = cal.thermal_share_lag2.to_numpy(float)
                            weights = np.exp(-0.5 * np.square((source_share - share) / bandwidth))
                            if weights.sum() <= 1e-12:
                                weights = np.zeros_like(weights)
                                weights[np.argmin(np.abs(source_share - share))] = 1.0
                            radius = weighted_higher(cal_error, weights, 0.95)
                            ess = float(weights.sum() ** 2 / np.square(weights).sum())
                            calibration = f"share_kernel_bw_{bandwidth:.2f}"
                        for alpha in ALPHAS:
                            # Use alpha-specific split-conformal radii, retaining a fixed model.
                            if bandwidth is None:
                                radius_alpha = float(np.quantile(cal_error, 1 - alpha, method="higher"))
                            else:
                                radius_alpha = weighted_higher(cal_error, weights, 1 - alpha)
                            lower, upper = max(0.0, prediction - radius_alpha), min(1.0, prediction + radius_alpha)
                            truth = float(target[target_name])
                            n_hours = int(target.n_hours)
                            pred_count = int(round(prediction * n_hours))
                            true_count = int(target[f"tail_count_{int(threshold)}"])
                            lower_count = max(0, int(np.floor(lower * n_hours)))
                            upper_count = min(n_hours, int(np.ceil(upper * n_hours)))
                            rows.append({
                                "region": heldout,
                                "year": TEST_YEAR,
                                "month": int(target.month),
                                "threshold_g_kwh": threshold,
                                "alpha": alpha,
                                "nominal_coverage": 1 - alpha,
                                "point_model": model_name,
                                "risk_calibration": calibration,
                                "thermal_share_lag2": share,
                                "effective_calibration_blocks": ess,
                                "actual_tail_fraction": truth,
                                "predicted_tail_fraction": prediction,
                                "absolute_fraction_error": abs(truth - prediction),
                                "fraction_coverage": lower <= truth <= upper,
                                "fraction_width": upper - lower,
                                "fraction_interval_score": interval_score_fraction(truth, lower, upper, alpha),
                                "actual_tail_count": true_count,
                                "predicted_tail_count": pred_count,
                                "tail_count_lower": lower_count,
                                "tail_count_upper": upper_count,
                                "tail_count_coverage": lower_count <= true_count <= upper_count,
                                "tail_count_width_hours": upper_count - lower_count,
                                "tail_count_width_fraction": (upper_count - lower_count) / n_hours,
                                "tail_count_interval_score": interval_score_fraction(
                                    true_count / n_hours, lower_count / n_hours, upper_count / n_hours, alpha
                                ),
                            })

    detail = pd.DataFrame(rows)
    OUT.mkdir(parents=True)
    monthly.to_csv(OUT / "monthly_tail_targets.csv", index=False)
    detail.to_csv(OUT / "region_month_scores.csv", index=False)
    region = detail.groupby(
        ["point_model", "risk_calibration", "alpha", "nominal_coverage", "threshold_g_kwh", "region"], as_index=False
    ).agg(
        months=("month", "size"),
        point_absolute_fraction_error=("absolute_fraction_error", "mean"),
        fraction_coverage=("fraction_coverage", "mean"),
        fraction_width=("fraction_width", "mean"),
        fraction_interval_score=("fraction_interval_score", "mean"),
        tail_count_coverage=("tail_count_coverage", "mean"),
        tail_count_width_hours=("tail_count_width_hours", "mean"),
        tail_count_interval_score=("tail_count_interval_score", "mean"),
        effective_calibration_blocks=("effective_calibration_blocks", "mean"),
    )
    region.to_csv(OUT / "region_summary.csv", index=False)
    summary = region.groupby(
        ["point_model", "risk_calibration", "alpha", "nominal_coverage", "threshold_g_kwh"], as_index=False
    ).agg(
        regions=("region", "nunique"),
        region_equal_point_abs_fraction_error=("point_absolute_fraction_error", "mean"),
        region_equal_fraction_coverage=("fraction_coverage", "mean"),
        worst_region_fraction_coverage=("fraction_coverage", "min"),
        region_equal_fraction_width=("fraction_width", "mean"),
        region_equal_fraction_interval_score=("fraction_interval_score", "mean"),
        region_equal_tail_count_coverage=("tail_count_coverage", "mean"),
        worst_region_tail_count_coverage=("tail_count_coverage", "min"),
        region_equal_tail_count_width_hours=("tail_count_width_hours", "mean"),
        region_equal_tail_count_interval_score=("tail_count_interval_score", "mean"),
        mean_effective_calibration_blocks=("effective_calibration_blocks", "mean"),
    )
    summary.to_csv(OUT / "method_summary.csv", index=False)

    meta = {
        "experiment": "direct monthly hourly-tail-frequency transfer from lagged local composition",
        "target": "monthly fraction of EIA fixed-factor hourly CIF proxy above threshold; not measured direct emissions",
        "train_year": TRAIN_YEAR,
        "calibration_year": CAL_YEAR,
        "test_year": TEST_YEAR,
        "heldout_regions": list(REGIONS),
        "monthly_quality_gate": "valid hourly label coverage >=80% and t-2 mix available",
        "point_model": "Ridge on month cyclic terms with/without t-2 thermal share and share-season interactions",
        "ridge_alpha": RIDGE_ALPHA,
        "risk_calibrations": ["global", "share_kernel_bw_0.10", "share_kernel_bw_0.20", "share_kernel_bw_0.40"],
        "caveats": [
            "Same seven previously inspected EIA regions; 2024 is an adaptive development holdout, not untouched confirmation.",
            "Fixed-factor generation proxy, not direct hourly CIF.",
            "Only three annual periods and roughly 72 donor calibration blocks; no formal dependent-data coverage guarantee.",
            "A lag-2 feature is timing-conservative but does not reproduce historical publication vintages.",
            "No result here transfers directly to China without independent local hourly labels and scope harmonization.",
        ],
        "summary_rows": summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records"),
    }
    (OUT / "summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
