#!/usr/bin/env python3
"""Retrospective rolling budget test for target-local hourly risk calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import run_dual_track_monthly_adaptation as dual
import run_official_intensity_transfer as official
from run_cross_country import ROOT


PREDICTIONS = ROOT / "results/local-monthly-adaptation-pseudotarget-r1"
CALIBRATIONS = PREDICTIONS / "source_calibration.json"
SEALS = PREDICTIONS / "prediction_seal.json"
BA_REGIONS = tuple(official.US_REGIONS)
TRACKS = ("generation_proxy", "consumption_official")
BUDGETS = (18, 24, 28)
METHOD = "fixed_ef_recomposition"
ADAPTER = "local_ridge"
ALPHA = 0.10


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def finite_conformal_radius(scores: np.ndarray, alpha: float = ALPHA) -> float:
    ordered = np.sort(np.asarray(scores, dtype=float))
    rank = math.ceil((len(ordered) + 1) * (1 - alpha))
    if rank > len(ordered):
        return float("inf")
    return float(ordered[rank - 1])


def winkler(y: np.ndarray, lo: np.ndarray, hi: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    width = hi - lo
    below = y < lo
    above = y > hi
    return width + (2 / alpha) * (lo - y) * below + (2 / alpha) * (y - hi) * above


def get_labels(raw: dict[str, pd.DataFrame]) -> dict[str, dict[str, pd.Series]]:
    labels = {
        "generation_proxy": dual.label_series("generation_proxy", raw, (2024,)),
        "consumption_official": {
            region: official.source_labels(region, years=(2024,)) for region in BA_REGIONS
        },
    }
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    capacity = official.capacity_transfer.load_capacity(official.CAPACITY_TABLE_PATH)
    raw, _, _, prep_audit = official.prepare(ROOT, capacity)
    target_labels = get_labels(raw)
    seal_rows = json.loads(SEALS.read_text())
    calibration_rows = json.loads(CALIBRATIONS.read_text())
    calibration_lookup = {
        (row["pseudo_target"], row["adapter"], row["track"]): row["calibration"][METHOD]
        for row in calibration_rows
    }
    seal_lookup = {
        (row["pseudo_target"], row["adapter"], row["track"]): row
        for row in seal_rows
        if row["adapter"] == ADAPTER
    }

    daily_rows: list[dict[str, object]] = []
    for region in BA_REGIONS:
        for track in TRACKS:
            key = (region, ADAPTER, track)
            seal = seal_lookup[key]
            prediction_path = PREDICTIONS / seal["path"]
            if sha256(prediction_path) != seal["sha256"]:
                raise RuntimeError(f"sealed prediction hash mismatch: {prediction_path.name}")
            pred = pd.read_csv(prediction_path)
            pred["time"] = pd.to_datetime(pred["time_utc"], utc=True)
            pred["day"] = pred["time"].dt.floor("D")
            label = target_labels[track][region].copy()
            label.index = pd.to_datetime(label.index, utc=True)
            pred["truth"] = pred["time"].map(label)
            pred = pred.loc[np.isfinite(pred.truth)].copy()
            # Day-level split conformal uses complete UTC days as exchangeable units.
            counts = pred.groupby("day").size()
            complete_days = counts.index[counts.eq(24)]
            pred = pred.loc[pred.day.isin(complete_days)].copy()
            if pred.empty:
                continue
            source_cal = calibration_lookup[key]
            source_bias = float(source_cal["bias"])
            source_radius = float(source_cal["radius90"])
            pred["source_center"] = pred[METHOD].to_numpy(float) + source_bias
            grouped = {day: frame.sort_values("time") for day, frame in pred.groupby("day", sort=True)}

            for budget in BUDGETS:
                fit_days = budget // 2
                risk_days = budget - fit_days
                ordered_days = sorted(grouped)
                for idx, test_day in enumerate(ordered_days):
                    if idx < budget:
                        continue
                    window = list(pd.date_range(
                        end=test_day - pd.Timedelta(days=1), periods=budget, freq="D"
                    ))
                    if any(day not in grouped for day in window):
                        continue
                    location_days = window[:fit_days]
                    risk_calibration_days = window[fit_days:]
                    residuals_by_day: dict[pd.Timestamp, np.ndarray] = {}
                    for day in window:
                        frame = grouped[day]
                        residuals_by_day[day] = (
                            frame.truth.to_numpy(float) - frame.source_center.to_numpy(float)
                        )
                    location_residuals = np.concatenate([residuals_by_day[d] for d in location_days])
                    local_bias = float(np.median(location_residuals))
                    calibration_scores = np.asarray(
                        [np.max(np.abs(residuals_by_day[d] - local_bias)) for d in risk_calibration_days],
                        dtype=float,
                    )
                    local_radius = finite_conformal_radius(calibration_scores)
                    hourly_scores = np.concatenate(
                        [np.abs(residuals_by_day[d] - local_bias) for d in risk_calibration_days]
                    )
                    local_hourly_radius = finite_conformal_radius(hourly_scores)
                    if not np.isfinite(local_radius):
                        raise RuntimeError("frozen protocol unexpectedly produced an infinite radius")
                    if not np.isfinite(local_hourly_radius):
                        raise RuntimeError("frozen protocol unexpectedly produced an infinite hourly radius")

                    test = grouped[test_day]
                    y = test.truth.to_numpy(float)
                    source_center = test.source_center.to_numpy(float)
                    arms = {
                        "source_zero_local": (source_center, source_radius),
                        "local_bias_source_radius": (source_center + local_bias, source_radius),
                        "local_split_conformal_hourly": (source_center + local_bias, local_hourly_radius),
                        "local_split_conformal_daily": (source_center + local_bias, local_radius),
                    }
                    for arm, (center, radius) in arms.items():
                        lo = center - radius
                        hi = center + radius
                        daily_rows.append(
                            {
                                "region": region,
                                "track": track,
                                "budget_days": budget,
                                "test_day_utc": test_day.isoformat(),
                                "arm": arm,
                                "hours": len(test),
                                "local_bias": local_bias if arm != "source_zero_local" else 0.0,
                                "radius": radius,
                                "daily_coverage": bool(np.all((y >= lo) & (y <= hi))),
                                "hourly_coverage": float(np.mean((y >= lo) & (y <= hi))),
                                "width": float(np.mean(hi - lo)),
                                "winkler90": float(np.mean(winkler(y, lo, hi))),
                                "mae": float(np.mean(np.abs(y - center))),
                                "source_radius": source_radius,
                                "calibration_score_days": len(risk_calibration_days),
                                "calibration_score_hours": len(hourly_scores),
                            }
                        )
    daily = pd.DataFrame(daily_rows)
    if daily.empty:
        raise RuntimeError("no complete rolling target days were scored")
    daily.to_csv(out / "daily_scores.csv", index=False)

    per_region = (
        daily.groupby(["track", "budget_days", "region", "arm"], as_index=False)
        .agg(
            score_days=("test_day_utc", "nunique"),
            daily_coverage=("daily_coverage", "mean"),
            hourly_coverage=("hourly_coverage", "mean"),
            width=("width", "mean"),
            winkler90=("winkler90", "mean"),
            mae=("mae", "mean"),
            median_radius=("radius", "median"),
        )
    )
    per_region.to_csv(out / "region_budget_metrics.csv", index=False)
    macro = (
        per_region.groupby(["track", "budget_days", "arm"], as_index=False)
        .agg(
            ba_count=("region", "nunique"),
            macro_score_days=("score_days", "mean"),
            macro_daily_coverage=("daily_coverage", "mean"),
            macro_hourly_coverage=("hourly_coverage", "mean"),
            macro_width=("width", "mean"),
            macro_winkler90=("winkler90", "mean"),
            macro_mae=("mae", "mean"),
        )
    )
    macro.to_csv(out / "macro_metrics.csv", index=False)

    input_paths = [
        CALIBRATIONS,
        SEALS,
        *[PREDICTIONS / row["path"] for row in seal_rows if row["adapter"] == ADAPTER],
        *[ROOT / f"results/research/data/{region}_hourly.csv" for region in BA_REGIONS],
        *[official.TARGET_LABEL_ROOT / f"{region}_consumption_2024.csv.gz" for region in BA_REGIONS],
    ]
    input_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in input_paths}
    code_paths = [Path(__file__), ROOT / "docs/LOCAL_HOURLY_LABEL_BUDGET_PROTOCOL_20261101.md"]
    code_hashes = {str(path.relative_to(ROOT)): sha256(path) for path in code_paths}
    manifest = {
        "status": "retrospective_rolling_budget_diagnostic_on_previously_opened_targets",
        "regions": list(BA_REGIONS),
        "tracks": list(TRACKS),
        "budgets_days": list(BUDGETS),
        "method": METHOD,
        "adapter": ADAPTER,
        "alpha": ALPHA,
        "scored_day_arm_rows": int(len(daily)),
        "daily_score_rows": int(len(daily)),
        "prep_audit": prep_audit,
        "input_sha256": input_hashes,
        "code_sha256": code_hashes,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(macro.to_string(index=False))


if __name__ == "__main__":
    main()
