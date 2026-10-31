#!/usr/bin/env python3
"""V395: strict time-forward, leave-region-out test of a t-2 local mix feature."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from v369_crossfit_risk_intervals import (
    ALPHAS,
    REGIONS,
    THRESHOLDS,
    load_hourly,
    interval_score,
    quantile,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v395-temporal-heldout-lagged-mix-r1"
TRAIN_YEAR = 2022
CAL_YEAR = 2023
TEST_YEAR = 2024
RIDGE_ALPHA = 10.0
SEED = 395
BOOTSTRAPS = 20_000


def add_features(hourly: pd.DataFrame) -> pd.DataFrame:
    frame = hourly.copy()
    frame["period"] = frame["year"] * 12 + frame["month"]
    monthly = frame.groupby(["region", "period"], as_index=False).agg(
        thermal_mwh=("generation_coal", "sum"), total_mwh=("total_mwh", "sum"), hours=("time", "size")
    )
    monthly["share"] = monthly["thermal_mwh"] / monthly["total_mwh"]
    monthly["source_complete"] = monthly["hours"] / monthly["period"].map(
        lambda p: int(
            pd.Period(year=int((p - 1) // 12), month=int((p - 1) % 12 + 1), freq="M").days_in_month * 24
        )
    )
    # Coal+gas+oil is the proxy's thermal group; add the gas and oil columns.
    thermal = frame.groupby(["region", "period"], as_index=False).agg(
        thermal_mwh=("generation_coal", "sum"),
        gas_mwh=("generation_gas", "sum"),
        oil_mwh=("generation_oil", "sum"),
    )
    monthly = monthly.drop(columns="thermal_mwh").merge(thermal, on=["region", "period"])
    monthly["share"] = (monthly["thermal_mwh"] + monthly["gas_mwh"] + monthly["oil_mwh"]) / monthly["total_mwh"]
    monthly.loc[monthly["source_complete"] < 0.90, "share"] = np.nan
    lagged = monthly[["region", "period", "share"]].copy()
    lagged["period"] += 2
    lagged = lagged.rename(columns={"share": "thermal_share_lag2"})
    frame = frame.merge(lagged, on=["region", "period"], how="left")

    hour = frame["time"].dt.hour.to_numpy(float)
    dow = frame["time"].dt.dayofweek.to_numpy(float)
    month = frame["month"].to_numpy(float)
    frame["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    frame["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    frame["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    frame["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    frame["month_sin"] = np.sin(2 * np.pi * month / 12.0)
    frame["month_cos"] = np.cos(2 * np.pi * month / 12.0)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        frame[col] = frame[col].astype(float)
    frame["share_hour_sin"] = frame["thermal_share_lag2"] * frame["hour_sin"]
    frame["share_hour_cos"] = frame["thermal_share_lag2"] * frame["hour_cos"]
    frame["share_month_sin"] = frame["thermal_share_lag2"] * frame["month_sin"]
    frame["share_month_cos"] = frame["thermal_share_lag2"] * frame["month_cos"]
    return frame


CALENDAR_FEATURES = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"
]
MIX_FEATURES = ["thermal_share_lag2", "share_hour_sin", "share_hour_cos", "share_month_sin", "share_month_cos"]


def fit_predict(train: pd.DataFrame, target: pd.DataFrame, features: list[str]) -> np.ndarray:
    usable = train.dropna(subset=[*features, "cif_proxy"])
    target_values = target[features].to_numpy(float)
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(usable[features].to_numpy(float), usable["cif_proxy"].to_numpy(float))
    return np.clip(model.predict(target_values), 0.0, 1000.0)


def block_interval_calibration(calibration: pd.DataFrame, alpha: float, method: str) -> tuple[float, dict[float, float]]:
    residual_col = f"residual_{method}"
    hourly_radius = quantile(np.abs(calibration[residual_col].to_numpy(float)), 1.0 - alpha)
    count_radii = {}
    for threshold in THRESHOLDS:
        count_col = f"tail_fraction_error_{method}_{int(threshold)}"
        # Residuals are region-month tail-fraction errors, with each block weighted equally.
        count_radii[threshold] = quantile(np.abs(calibration[count_col].to_numpy(float)), 1.0 - alpha)
    return hourly_radius, count_radii


def score_predictions(
    target: pd.DataFrame,
    point: np.ndarray,
    hourly_radius: float,
    count_radii: dict[float, float],
    method: str,
    alpha: float,
) -> list[dict]:
    y = target["cif_proxy"].to_numpy(float)
    lo = np.clip(point - hourly_radius, 0.0, 1000.0)
    hi = np.clip(point + hourly_radius, 0.0, 1000.0)
    rows = []
    common = {
        "region": target["region"].iloc[0],
        "year": int(target["year"].iloc[0]),
        "month": int(target["month"].iloc[0]),
        "method": method,
        "alpha": alpha,
        "nominal_coverage": 1.0 - alpha,
        "n_hours": len(y),
        "thermal_share_lag2": float(target["thermal_share_lag2"].iloc[0]),
        "hourly_mae_g_kwh": float(np.mean(np.abs(y - point))),
        "hourly_coverage": float(np.mean((y >= lo) & (y <= hi))),
        "hourly_width_g_kwh": float(np.mean(hi - lo)),
        "hourly_interval_score": float(np.mean(interval_score(y, lo, hi, alpha))),
    }
    for threshold in THRESHOLDS:
        observed_count = int(np.sum(y > threshold))
        predicted_count = int(np.sum(point > threshold))
        radius = count_radii[threshold]
        count_radius = int(np.ceil(radius * len(y)))
        lower = max(0, predicted_count - count_radius)
        upper = min(len(y), predicted_count + count_radius)
        score = float(interval_score(np.array([observed_count]), np.array([lower]), np.array([upper]), alpha)[0])
        rows.append({
            **common,
            "threshold_g_kwh": threshold,
            "observed_tail_count": observed_count,
            "predicted_tail_count": predicted_count,
            "tail_count_lower": lower,
            "tail_count_upper": upper,
            "tail_count_covered": lower <= observed_count <= upper,
            "tail_count_width_hours": upper - lower,
            "tail_count_width_fraction": (upper - lower) / len(y),
            "tail_count_interval_score": score,
            "calibrated_tail_fraction_radius": radius,
        })
    return rows


def bootstrap_summary(paired: pd.DataFrame, gain_columns: list[str]) -> dict:
    rng = np.random.default_rng(SEED)
    result = {}
    for column in gain_columns:
        region_mean = paired.groupby("region")[column].mean().to_numpy(float)
        draws = rng.choice(region_mean, size=(BOOTSTRAPS, len(region_mean)), replace=True).mean(axis=1)
        result[column] = {
            "mean": float(region_mean.mean()),
            "ci95_low": float(np.quantile(draws, 0.025)),
            "ci95_high": float(np.quantile(draws, 0.975)),
            "regions_positive": int((region_mean > 0).sum()),
            "region_clusters": int(len(region_mean)),
        }
    return result


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    frame = add_features(load_hourly())
    rows = []
    calibration_rows = []
    for heldout in REGIONS:
        donors = frame[frame["region"] != heldout]
        train = donors[donors["year"] == TRAIN_YEAR].dropna(subset=["thermal_share_lag2"])
        calibrate = donors[donors["year"] == CAL_YEAR].dropna(subset=["thermal_share_lag2"])
        test = frame[(frame["region"] == heldout) & (frame["year"] == TEST_YEAR)].dropna(
            subset=["thermal_share_lag2"]
        )
        if train.empty or calibrate.empty or test.empty:
            raise RuntimeError(f"missing train/cal/test data for held-out region {heldout}")

        for method, features in [
            ("calendar_ridge_no_mix", CALENDAR_FEATURES),
            ("calendar_ridge_local_lag2_mix", [*CALENDAR_FEATURES, *MIX_FEATURES]),
        ]:
            train = train.copy()
            calibrate = calibrate.copy()
            test = test.copy()
            calibrate[f"prediction_{method}"] = fit_predict(train, calibrate, features)
            test[f"prediction_{method}"] = fit_predict(train, test, features)
            calibrate[f"residual_{method}"] = calibrate["cif_proxy"] - calibrate[f"prediction_{method}"]

            # Conformal calibration units are donor-region-month blocks, not individual rows.
            cal_blocks = []
            for (region, year, month), block in calibrate.groupby(["region", "year", "month"]):
                row = {"region": region, "year": year, "month": month}
                row["thermal_share_lag2"] = float(block["thermal_share_lag2"].iloc[0])
                residual = block[f"residual_{method}"].to_numpy(float)
                row[f"residual_{method}"] = np.max(np.abs(residual))
                for threshold in THRESHOLDS:
                    actual_count = int(np.sum(block["cif_proxy"].to_numpy(float) > threshold))
                    pred_count = int(np.sum(block[f"prediction_{method}"].to_numpy(float) > threshold))
                    row[f"tail_fraction_error_{method}_{int(threshold)}"] = abs(actual_count - pred_count) / len(block)
                cal_blocks.append(row)
            calibration = pd.DataFrame(cal_blocks)
            for row in cal_blocks:
                calibration_rows.append({"heldout_region": heldout, "method": method, **row})

            for alpha in ALPHAS:
                # Simultaneous within-month max-residual calibration is conservative for hourly coverage.
                hourly_radius = quantile(calibration[f"residual_{method}"].to_numpy(float), 1.0 - alpha)
                count_radii = {
                    threshold: quantile(
                        calibration[f"tail_fraction_error_{method}_{int(threshold)}"].to_numpy(float), 1.0 - alpha
                    )
                    for threshold in THRESHOLDS
                }
                for (_, _, _), block in test.groupby(["region", "year", "month"]):
                    rows.extend(score_predictions(
                        block,
                        block[f"prediction_{method}"].to_numpy(float),
                        hourly_radius,
                        count_radii,
                        method,
                        alpha,
                    ))

    detail = pd.DataFrame(rows)
    calibration_detail = pd.DataFrame(calibration_rows)
    OUT.mkdir(parents=True)
    detail.to_csv(OUT / "region_month_scores.csv", index=False)
    calibration_detail.to_csv(OUT / "calibration_block_scores.csv", index=False)

    region_summary = detail.groupby(
        ["method", "alpha", "nominal_coverage", "threshold_g_kwh", "region"], as_index=False
    ).agg(
        region_months=("month", "size"),
        hourly_mae_g_kwh=("hourly_mae_g_kwh", "mean"),
        hourly_coverage=("hourly_coverage", "mean"),
        hourly_width_g_kwh=("hourly_width_g_kwh", "mean"),
        hourly_interval_score=("hourly_interval_score", "mean"),
        tail_count_coverage=("tail_count_covered", "mean"),
        tail_count_width_hours=("tail_count_width_hours", "mean"),
        tail_count_interval_score=("tail_count_interval_score", "mean"),
    )
    region_summary.to_csv(OUT / "region_summary.csv", index=False)
    summary = region_summary.groupby(["method", "alpha", "nominal_coverage", "threshold_g_kwh"], as_index=False).agg(
        regions=("region", "nunique"),
        region_equal_hourly_mae=("hourly_mae_g_kwh", "mean"),
        region_equal_hourly_coverage=("hourly_coverage", "mean"),
        worst_region_hourly_coverage=("hourly_coverage", "min"),
        region_equal_hourly_width=("hourly_width_g_kwh", "mean"),
        region_equal_hourly_interval_score=("hourly_interval_score", "mean"),
        region_equal_tail_count_coverage=("tail_count_coverage", "mean"),
        worst_region_tail_count_coverage=("tail_count_coverage", "min"),
        region_equal_tail_count_width_hours=("tail_count_width_hours", "mean"),
        region_equal_tail_count_interval_score=("tail_count_interval_score", "mean"),
    )
    summary.to_csv(OUT / "method_summary.csv", index=False)

    paired = detail.pivot_table(
        index=["region", "year", "month", "alpha", "threshold_g_kwh"],
        columns="method",
        values=["hourly_mae_g_kwh", "hourly_coverage", "hourly_width_g_kwh", "hourly_interval_score",
                "tail_count_covered", "tail_count_width_hours", "tail_count_interval_score"],
    ).reset_index()
    paired.columns = ["_".join(str(v) for v in c if str(v)) if isinstance(c, tuple) else str(c) for c in paired.columns]
    metrics = ["hourly_mae_g_kwh", "hourly_width_g_kwh", "hourly_interval_score", "tail_count_width_hours", "tail_count_interval_score"]
    for metric in metrics:
        local_col = f"{metric}_calendar_ridge_local_lag2_mix"
        base_col = f"{metric}_calendar_ridge_no_mix"
        paired[f"gain_{metric}_local_lag2"] = paired[base_col] - paired[local_col]
    paired.to_csv(OUT / "paired_local_mix_vs_no_mix.csv", index=False)

    summary_json = {
        "experiment": "strict time-forward and leave-region-out local mix transfer with split-block calibration",
        "target": "EIA fixed-factor hourly generation CIF proxy, not measured direct emissions",
        "input": "results/research/data/*_hourly.csv",
        "heldout_regions": list(REGIONS),
        "train_year": TRAIN_YEAR,
        "calibration_year": CAL_YEAR,
        "test_year": TEST_YEAR,
        "feature_contract": "held-out local thermal share from t-2 monthly generation only; no held-out hourly target or t/t-1 generation enters model fitting",
        "models": {
            "calendar_ridge_no_mix": "standardized Ridge on hour, weekday and month cyclic terms",
            "calendar_ridge_local_lag2_mix": "same Ridge plus t-2 thermal share and mix-by-hour/month interactions",
        },
        "ridge_alpha": RIDGE_ALPHA,
        "interval_calibration": "2023 donor region-month blocks; hourly interval uses the within-block maximum absolute residual, tail-count interval uses block tail-fraction absolute error quantiles",
        "caveats": [
            "Only seven EIA balancing authorities and one test year; prior exploratory inspection means this is not sealed external confirmation.",
            "Target is an EIA generation-based fixed-factor proxy, not direct measured CIF.",
            "The lagged feature is timing-conservative but source as-published vintages are unavailable.",
            "A 2022/2023/2024 split is descriptive under temporal distribution shift; calibration is not a distribution-free guarantee with serially dependent regions/months.",
            "This source-domain experiment does not establish China-specific transfer or account for NBS above-designated-size scope remainder.",
        ],
        "summary_rows": summary.astype(object).where(pd.notna(summary), None).to_dict(orient="records"),
        "paired_cluster_bootstrap": bootstrap_summary(
            paired,
            [f"gain_{metric}_local_lag2" for metric in metrics],
        ),
    }
    (OUT / "summary.json").write_text(json.dumps(summary_json, indent=2) + "\n")
    print(json.dumps(summary_json, indent=2))


if __name__ == "__main__":
    main()
