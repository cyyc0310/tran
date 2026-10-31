#!/usr/bin/env python3
"""Verify V398 prediction seal, then open and score only external 2026H1 labels."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v398-external-lagged-mix-r1"
DATA_DIR = ROOT / "results/external-v6-data"
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0])
FUELS = [
    "generation_coal", "generation_gas", "generation_oil", "generation_nuclear",
    "generation_hydro", "generation_wind", "generation_solar", "generation_other",
]
METHODS = ("calendar_ridge_no_mix", "calendar_ridge_local_lag2_mix")
ALPHAS = (0.10, 0.05)
THRESHOLDS = (300.0, 500.0, 700.0)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_target_2026h1(path: Path, start: pd.Timestamp, stop: pd.Timestamp) -> pd.DataFrame:
    parts = []
    # The data file spans older source years; only parse/retain the frozen 2026H1 target rows.
    for chunk in pd.read_csv(path, chunksize=100_000):
        times = pd.to_datetime(chunk["time"], utc=True, errors="coerce")
        mask = times.ge(start) & times.lt(stop)
        if not mask.any():
            continue
        selected = chunk.loc[mask, ["time", "valid", *FUELS]].copy()
        selected["time"] = times.loc[mask]
        valid = selected["valid"].astype(str).str.lower().eq("true")
        values = selected[FUELS].apply(pd.to_numeric, errors="coerce").to_numpy(float)
        valid &= np.isfinite(values).all(axis=1) & (values >= -1e-9).all(axis=1)
        selected = selected.loc[valid].copy()
        values = selected[FUELS].to_numpy(float).clip(min=0.0)
        total = values.sum(axis=1)
        positive = total > 0
        selected = selected.loc[positive].copy()
        values, total = values[positive], total[positive]
        selected["actual_cif_proxy"] = values @ EF / total
        parts.append(selected[["time", "actual_cif_proxy"]])
    if not parts:
        return pd.DataFrame(columns=["time", "actual_cif_proxy"])
    result = pd.concat(parts, ignore_index=True)
    if result["time"].duplicated().any():
        raise ValueError(f"Duplicate valid 2026H1 target hours in {path.name}")
    return result


def interval_score(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> np.ndarray:
    return (upper - lower) + (2 / alpha) * np.maximum(lower - y, 0) + (2 / alpha) * np.maximum(y - upper, 0)


def bootstrap_regions(frame: pd.DataFrame, column: str, seed: int) -> dict:
    regions = frame.groupby("region")[column].mean().to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(regions, size=(20_000, len(regions)), replace=True).mean(axis=1)
    return {
        "mean": float(regions.mean()),
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "regions_positive": int((regions > 0).sum()),
        "region_clusters": int(len(regions)),
    }


def main() -> None:
    seal_path = OUT / "prediction_seal.json"
    if not seal_path.exists():
        raise FileNotFoundError("Prediction seal must exist before scoring")
    seal = json.loads(seal_path.read_text())
    if not seal.get("sealed") or seal.get("target_hourly_labels_read") is not False:
        raise RuntimeError("Prediction seal is invalid")
    for rel, expected in seal["prediction_files_sha256"].items():
        if sha(OUT / rel) != expected:
            raise RuntimeError(f"Prediction file changed after sealing: {rel}")
    for file_path, expected in seal["input_and_code_sha256"].items():
        if sha(Path(file_path)) != expected:
            raise RuntimeError(f"Frozen input/code changed after sealing: {file_path}")

    # Only after all predictions and hashes verify do we read external target labels.
    predictions = pd.read_csv(OUT / "hourly_predictions.csv")
    predictions["time"] = pd.to_datetime(predictions["time"], utc=True)
    feature = pd.read_csv(OUT / "monthly_feature_eligibility.csv")
    feature["target_month"] = pd.to_datetime(feature["target_month"], utc=True)
    radii = pd.read_csv(OUT / "calibration_radii.csv")
    start = pd.Timestamp(seal["target_window_start"])
    stop = pd.Timestamp(seal["target_window_stop_exclusive"])

    scored = []
    for region in seal["target_regions"]:
        labels = read_target_2026h1(DATA_DIR / f"{region}_hourly.csv", start, stop)
        pred = predictions[predictions.region == region]
        joined = pred.merge(labels, on="time", how="inner", validate="one_to_one")
        joined["month"] = joined.time.dt.month
        joined["feature_available"] = joined.local_mix_available.astype(bool)
        for method in METHODS:
            point_col = "point_no_mix" if method == METHODS[0] else "point_local_mix"
            for alpha in ALPHAS:
                alabel = int((1 - alpha) * 100)
                lower = joined[f"lo_{method}_a{alabel}"].to_numpy(float)
                upper = joined[f"hi_{method}_a{alabel}"].to_numpy(float)
                actual = joined.actual_cif_proxy.to_numpy(float)
                joined_copy = joined.copy()
                joined_copy[f"coverage_{method}_a{alabel}"] = (actual >= lower) & (actual <= upper)
                joined_copy[f"width_{method}_a{alabel}"] = upper - lower
                joined_copy[f"iscore_{method}_a{alabel}"] = interval_score(actual, lower, upper, alpha)
                for month, block in joined_copy.groupby("month"):
                    feature_row = feature[(feature.region == region) & (feature.target_month.dt.month == int(month))]
                    if len(feature_row) != 1:
                        raise ValueError(f"Missing feature eligibility row for {region} 2026-{int(month):02d}")
                    available = bool(feature_row.available.iloc[0])
                    count_rows = []
                    for threshold in THRESHOLDS:
                        effective_method = method
                        if method == METHODS[1] and not available:
                            effective_method = METHODS[0]
                        radius_row = radii[(radii.method == effective_method) & (radii.alpha == alpha)
                                           & (radii.threshold_g_kwh == threshold)]
                        if len(radius_row) != 1:
                            raise ValueError("Expected one frozen tail calibration radius")
                        radius_fraction = float(radius_row.tail_fraction_radius.iloc[0])
                        values = block[point_col].to_numpy(float)
                        actual_values = block.actual_cif_proxy.to_numpy(float)
                        observed = int(np.sum(actual_values > threshold))
                        predicted = int(np.sum(values > threshold))
                        radius_hours = int(np.ceil(radius_fraction * len(block)))
                        lower_count = max(0, predicted - radius_hours)
                        upper_count = min(len(block), predicted + radius_hours)
                        count_rows.append({
                            "region": region,
                            "month": int(month),
                            "method": method,
                            "alpha": alpha,
                            "threshold_g_kwh": threshold,
                            "n_valid_hours": int(len(block)),
                            "feature_available": available,
                            "observed_tail_count": observed,
                            "predicted_tail_count": predicted,
                            "tail_count_lower": lower_count,
                            "tail_count_upper": upper_count,
                            "tail_count_covered": lower_count <= observed <= upper_count,
                            "tail_count_width_hours": upper_count - lower_count,
                            "tail_count_interval_score": float(interval_score(
                                np.array([observed]), np.array([lower_count]), np.array([upper_count]), alpha
                            )[0]),
                        })
                    scored.extend(count_rows)

    score_rows = pd.DataFrame(scored)
    score_rows.to_csv(OUT / "tail_count_scores.csv", index=False)
    # Rebuild hourly score rows from only valid labels and sealed predictions.
    hourly_rows = []
    for region in seal["target_regions"]:
        labels = read_target_2026h1(DATA_DIR / f"{region}_hourly.csv", start, stop)
        pred = predictions[predictions.region == region]
        joined = pred.merge(labels, on="time", how="inner", validate="one_to_one")
        for method in METHODS:
            point = joined["point_no_mix"].to_numpy(float) if method == METHODS[0] else joined["point_local_mix"].to_numpy(float)
            hourly_mae = np.abs(joined.actual_cif_proxy.to_numpy(float) - point)
            for alpha in ALPHAS:
                alabel = int((1 - alpha) * 100)
                lo = joined[f"lo_{method}_a{alabel}"].to_numpy(float)
                hi = joined[f"hi_{method}_a{alabel}"].to_numpy(float)
                for month, idx in joined.groupby(joined.time.dt.month).groups.items():
                    ii = np.asarray(list(idx), dtype=int)
                    hourly_rows.append({
                        "region": region,
                        "month": int(month),
                        "method": method,
                        "alpha": alpha,
                        "n_valid_hours": int(len(ii)),
                        "feature_available": bool(joined.iloc[ii].local_mix_available.iloc[0]),
                        "hourly_mae_g_kwh": float(np.mean(hourly_mae[ii])),
                        "hourly_coverage": float(np.mean((joined.actual_cif_proxy.to_numpy(float)[ii] >= lo[ii])
                                                          & (joined.actual_cif_proxy.to_numpy(float)[ii] <= hi[ii]))),
                        "hourly_width_g_kwh": float(np.mean(hi[ii] - lo[ii])),
                        "hourly_interval_score": float(np.mean(interval_score(
                            joined.actual_cif_proxy.to_numpy(float)[ii], lo[ii], hi[ii], alpha
                        ))),
                    })
    hourly_scores = pd.DataFrame(hourly_rows)
    hourly_scores.to_csv(OUT / "hourly_scores.csv", index=False)

    grouped_hourly = hourly_scores.groupby(["method", "alpha", "region"], as_index=False).agg(
        valid_hours=("n_valid_hours", "sum"),
        feature_available_months=("feature_available", "sum"),
        mae=("hourly_mae_g_kwh", "mean"),
        coverage=("hourly_coverage", "mean"),
        width=("hourly_width_g_kwh", "mean"),
        interval_score=("hourly_interval_score", "mean"),
    )
    grouped_hourly.to_csv(OUT / "hourly_region_metrics.csv", index=False)
    count_region = score_rows.groupby(
        ["method", "alpha", "threshold_g_kwh", "region"], as_index=False
    ).agg(
        month_coverage=("tail_count_covered", "mean"),
        mean_count_width_hours=("tail_count_width_hours", "mean"),
        mean_count_interval_score=("tail_count_interval_score", "mean"),
    )
    count_region.to_csv(OUT / "tail_count_region_metrics.csv", index=False)

    paired = grouped_hourly.pivot(index=["alpha", "region"], columns="method", values=["mae", "coverage", "width", "interval_score"]).reset_index()
    paired.columns = ["_".join(str(v) for v in col if str(v)) if isinstance(col, tuple) else str(col) for col in paired.columns]
    paired["gain_mae"] = paired["mae_calendar_ridge_no_mix"] - paired["mae_calendar_ridge_local_lag2_mix"]
    paired["gain_width"] = paired["width_calendar_ridge_no_mix"] - paired["width_calendar_ridge_local_lag2_mix"]
    paired["gain_interval_score"] = paired["interval_score_calendar_ridge_no_mix"] - paired["interval_score_calendar_ridge_local_lag2_mix"]
    paired["delta_coverage"] = paired["coverage_calendar_ridge_local_lag2_mix"] - paired["coverage_calendar_ridge_no_mix"]
    paired.to_csv(OUT / "paired_hourly_gains_by_region.csv", index=False)
    gains = {}
    for alpha, subset in paired.groupby("alpha"):
        gains[str(alpha)] = {
            col: bootstrap_regions(subset[["region", col]], col, 398 + int(alpha * 100))
            for col in ["gain_mae", "gain_width", "gain_interval_score", "delta_coverage"]
        }

    summary = {
        "prediction_seal_sha256": sha(seal_path),
        "only_scored_period": "2026H1",
        "valid_target_hours_per_method_alpha": int(
            grouped_hourly[grouped_hourly.method == METHODS[0]].valid_hours.sum()
        ),
        "regions": seal["target_regions"],
        "hourly_region_equal_summary": grouped_hourly.groupby(["method", "alpha"], as_index=False).agg(
            regions=("region", "nunique"), region_equal_mae=("mae", "mean"),
            region_equal_coverage=("coverage", "mean"), worst_region_coverage=("coverage", "min"),
            region_equal_width=("width", "mean"), region_equal_interval_score=("interval_score", "mean")
        ).to_dict(orient="records"),
        "tail_count_region_equal_summary": count_region.groupby(
            ["method", "alpha", "threshold_g_kwh"], as_index=False
        ).agg(regions=("region", "nunique"), region_equal_coverage=("month_coverage", "mean"),
             worst_region_coverage=("month_coverage", "min"), region_equal_width_hours=("mean_count_width_hours", "mean"),
             region_equal_interval_score=("mean_count_interval_score", "mean")).to_dict(orient="records"),
        "paired_bootstrap": gains,
        "feature_eligibility": pd.read_csv(OUT / "monthly_feature_eligibility.csv").to_dict(orient="records"),
        "caveats": [
            "Only six external regions; interval calibration over small dependent calibration blocks is descriptive, not a coverage guarantee.",
            "The hourly target is a generation-derived fixed-factor proxy, not measured emissions.",
            "External regions have been used in other project experiments; predictions were frozen for this model before scoring.",
            "Local arm falls back to no-mix when exact t-2 feature is unavailable.",
        ],
    }
    (OUT / "score_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
