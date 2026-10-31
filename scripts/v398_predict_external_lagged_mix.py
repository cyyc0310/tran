#!/usr/bin/env python3
"""Freeze source-fitted V395 predictions for six external regions, without reading labels."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v369_crossfit_risk_intervals import ALPHAS, REGIONS, THRESHOLDS, load_hourly, quantile  # noqa: E402
from v395_temporal_holdout_lagged_mix_cif import (  # noqa: E402
    CALENDAR_FEATURES,
    MIX_FEATURES,
    TRAIN_YEAR,
    CAL_YEAR,
    RIDGE_ALPHA,
    add_features,
)

OUT = ROOT / "results/v398-external-lagged-mix-r1"
DATA_DIR = ROOT / "results/external-v6-data"
PROTOCOL = ROOT / "docs/V398_EXTERNAL_LAGGED_MIX_CONFIRMATION_PROTOCOL.md"
EF = np.array([1000.0, 400.0, 800.0, 0.0, 0.0, 0.0, 0.0, 0.0])
FUELS = [
    "generation_coal", "generation_gas", "generation_oil", "generation_nuclear",
    "generation_hydro", "generation_wind", "generation_solar", "generation_other",
]
METHODS = ("calendar_ridge_no_mix", "calendar_ridge_local_lag2_mix")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fit_model(train: pd.DataFrame, method: str):
    features = CALENDAR_FEATURES if method == METHODS[0] else [*CALENDAR_FEATURES, *MIX_FEATURES]
    model = make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))
    model.fit(train[features].to_numpy(float), train["cif_proxy"].to_numpy(float))
    return model, features


def predict(model, features: list[str], frame: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(frame[features].to_numpy(float)), 0.0, 1000.0)


def add_time_features(frame: pd.DataFrame, share: float | None) -> pd.DataFrame:
    out = frame.copy()
    t = pd.to_datetime(out["time"], utc=True)
    hour, dow, month = t.dt.hour.to_numpy(float), t.dt.dayofweek.to_numpy(float), t.dt.month.to_numpy(float)
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["month_sin"] = np.sin(2 * np.pi * month / 12)
    out["month_cos"] = np.cos(2 * np.pi * month / 12)
    out["thermal_share_lag2"] = share
    out["share_hour_sin"] = share * out["hour_sin"] if share is not None else np.nan
    out["share_hour_cos"] = share * out["hour_cos"] if share is not None else np.nan
    out["share_month_sin"] = share * out["month_sin"] if share is not None else np.nan
    out["share_month_cos"] = share * out["month_cos"] if share is not None else np.nan
    return out


def calibration_blocks(source: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for heldout in REGIONS:
        train = source[(source.region != heldout) & (source.year == TRAIN_YEAR)].dropna(
            subset=["thermal_share_lag2"]
        ).copy()
        cal = source[(source.region == heldout) & (source.year == CAL_YEAR)].dropna(
            subset=["thermal_share_lag2"]
        ).copy()
        for method in METHODS:
            model, features = fit_model(train, method)
            cal[f"pred_{method}"] = predict(model, features, cal)
        for (region, year, month), block in cal.groupby(["region", "year", "month"]):
            record = {
                "region": region,
                "year": int(year),
                "month": int(month),
                "thermal_share_lag2": float(block["thermal_share_lag2"].iloc[0]),
            }
            for method in METHODS:
                residual = block["cif_proxy"].to_numpy(float) - block[f"pred_{method}"].to_numpy(float)
                record[f"max_abs_residual_{method}"] = float(np.max(np.abs(residual)))
                for threshold in THRESHOLDS:
                    actual_fraction = float(np.mean(block["cif_proxy"].to_numpy(float) > threshold))
                    pred_fraction = float(np.mean(block[f"pred_{method}"].to_numpy(float) > threshold))
                    record[f"tail_fraction_error_{method}_{int(threshold)}"] = abs(actual_fraction - pred_fraction)
            rows.append(record)
    return pd.DataFrame(rows)


def target_month_feature(monthly: pd.DataFrame, region: str, target_month: pd.Timestamp) -> dict:
    source_month = target_month - pd.DateOffset(months=2)
    rows = monthly[(monthly.region == region) & (monthly.period_start == source_month)]
    if len(rows) != 1:
        return {"available": False, "share": None, "source_period_start": source_month.isoformat(),
                "available_at": None, "coverage": None}
    row = rows.iloc[0]
    available_at = pd.Timestamp(row.available_at)
    period_end = pd.Timestamp(row.period_end)
    coverage = float(row.coverage)
    eligible = period_end <= target_month and available_at <= target_month and coverage >= 0.80
    if not eligible:
        return {"available": False, "share": None, "source_period_start": source_month.isoformat(),
                "available_at": available_at.isoformat(), "coverage": coverage}
    share = float(row.fuelshare_coal + row.fuelshare_gas + row.fuelshare_oil)
    return {"available": True, "share": share, "source_period_start": source_month.isoformat(),
            "available_at": available_at.isoformat(), "coverage": coverage}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    audit_path = DATA_DIR / "audit.json"
    monthly_path = DATA_DIR / "monthly.csv"
    audit = json.loads(audit_path.read_text())
    regions = audit["external_regions"]
    if regions != ["AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP"]:
        raise RuntimeError("Unexpected frozen external region list")
    start = pd.Timestamp(audit["confirmation_start"])
    stop = pd.Timestamp(audit["confirmation_stop_exclusive"])
    monthly = pd.read_csv(monthly_path)
    monthly["period_start"] = pd.to_datetime(monthly["period_start"], utc=True)
    monthly["period_end"] = pd.to_datetime(monthly["period_end"], utc=True)
    monthly["available_at"] = pd.to_datetime(monthly["available_at"], utc=True)

    # Source training and calibration only. The external hourly reference paths are never read here.
    source = add_features(load_hourly())
    source = source[source.year.isin([TRAIN_YEAR, CAL_YEAR])].copy()
    source_blocks = calibration_blocks(source)
    final_train = source[source.year == TRAIN_YEAR].dropna(subset=["thermal_share_lag2"]).copy()
    models = {method: fit_model(final_train, method) for method in METHODS}

    predictions, feature_records, radii_records = [], [], []
    for region in regions:
        for target_month in pd.date_range(start, stop - pd.Timedelta(hours=1), freq="MS", tz="UTC"):
            feature = target_month_feature(monthly, region, target_month)
            feature_records.append({"region": region, "target_month": target_month.isoformat(), **feature})
            hours = pd.DataFrame({"time": pd.date_range(target_month, target_month + pd.offsets.MonthBegin(1),
                                                          freq="h", inclusive="left", tz="UTC")})
            hours["region"] = region
            hours["year"] = hours.time.dt.year
            hours["month"] = hours.time.dt.month
            base_frame = add_time_features(hours, feature["share"] if feature["available"] else None)
            point_no_mix = predict(*models[METHODS[0]], base_frame)
            point_local_raw = predict(*models[METHODS[1]], base_frame) if feature["available"] else point_no_mix.copy()
            point_local = point_local_raw.copy()

            for method in METHODS:
                source_method = method
                for alpha in ALPHAS:
                    q = quantile(source_blocks[f"max_abs_residual_{source_method}"].to_numpy(float), 1 - alpha)
                    for threshold in THRESHOLDS:
                        cq = quantile(
                            source_blocks[f"tail_fraction_error_{source_method}_{int(threshold)}"].to_numpy(float),
                            1 - alpha,
                        )
                        radii_records.append({
                            "method": method, "alpha": alpha, "threshold_g_kwh": threshold,
                            "hourly_block_max_radius_g_kwh": q,
                            "tail_fraction_radius": cq,
                            "calibration_blocks": int(len(source_blocks)),
                        })

            for i, row in hours.iterrows():
                j = i
                record = {
                    "region": region,
                    "time": row.time.isoformat(),
                    "year": int(row.year),
                    "month": int(row.month),
                    "local_mix_available": bool(feature["available"]),
                    "thermal_share_lag2": feature["share"] if feature["available"] else np.nan,
                    "point_no_mix": float(point_no_mix[j]),
                    "point_local_mix": float(point_local[j]),
                }
                for alpha in ALPHAS:
                    for method in METHODS:
                        effective_method = method
                        if method == METHODS[1] and not feature["available"]:
                            effective_method = METHODS[0]
                        q = quantile(source_blocks[f"max_abs_residual_{effective_method}"].to_numpy(float), 1 - alpha)
                        point = point_no_mix[j] if effective_method == METHODS[0] else point_local[j]
                        record[f"lo_{method}_a{int((1-alpha)*100)}"] = float(np.clip(point - q, 0, 1000))
                        record[f"hi_{method}_a{int((1-alpha)*100)}"] = float(np.clip(point + q, 0, 1000))
                predictions.append(record)

    OUT.mkdir(parents=True)
    pred_path = OUT / "hourly_predictions.csv"
    feature_path = OUT / "monthly_feature_eligibility.csv"
    radii_path = OUT / "calibration_radii.csv"
    pd.DataFrame(predictions).to_csv(pred_path, index=False)
    pd.DataFrame(feature_records).to_csv(feature_path, index=False)
    pd.DataFrame(radii_records).drop_duplicates().to_csv(radii_path, index=False)
    used = [PROTOCOL, Path(__file__).resolve(), monthly_path, audit_path]
    used.extend(ROOT / "scripts" / "v369_crossfit_risk_intervals.py" for _ in range(1))
    used.extend([ROOT / "scripts" / "v395_temporal_holdout_lagged_mix_cif.py",
                 ROOT / "results" / "v395-temporal-heldout-lagged-mix-r1" / "summary.json"])
    used.extend(ROOT / "results" / "research" / "data" / f"{region}_hourly.csv" for region in REGIONS)
    manifest = {
        "experiment": "V398 frozen external lagged-mix transfer",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": str(PROTOCOL),
        "target_regions": regions,
        "target_window_start": start.isoformat(),
        "target_window_stop_exclusive": stop.isoformat(),
        "prediction_rows": int(len(predictions)),
        "expected_hourly_grid_rows": int(len(regions) * ((stop - start) / pd.Timedelta(hours=1))),
        "feature_availability_counts": pd.DataFrame(feature_records).groupby("region").available.sum().astype(int).to_dict(),
        "feature_rule": "exact t-2 row only; period_end and available_at <= target month start; coverage >=0.80; otherwise local arm falls back to no-mix forecast",
        "source_training_regions": list(REGIONS),
        "source_train_year": TRAIN_YEAR,
        "source_calibration_year": CAL_YEAR,
        "prediction_files_sha256": {
            pred_path.name: sha(pred_path), feature_path.name: sha(feature_path), radii_path.name: sha(radii_path)
        },
        "input_and_code_sha256": {str(path): sha(path) for path in used},
        "target_hourly_labels_read": False,
        "target_hourly_generation_values_read": False,
        "sealed": True,
        "limitations": [
            "Fixed-factor EIA generation proxy, not measured CIF.",
            "External targets have been used in other project experiments; this is frozen-method confirmation only.",
            "Only six external region clusters and dependent hourly series.",
            "Historical monthly values are controlled lagged inputs, not original published vintages.",
        ],
    }
    seal_path = OUT / "prediction_seal.json"
    seal_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
