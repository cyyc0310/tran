#!/usr/bin/env python3
"""Seal US/GB-source strict-CN5 predictions for Taiwan's public Q2 2026 data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import cross_country as source  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402

PROTOCOL = ROOT / "docs/TAIPOWER_CN5_EXTERNAL_TRANSFER_PROTOCOL_20261102.md"
INPUT_DIR = ROOT / "data/taipower-public-hourly-target-r1"
INPUT_CSV = INPUT_DIR / "target_monthly_cn5_inputs.csv"
LAG_CONTRACT = INPUT_DIR / "target_lag_contract.csv"
INPUT_MANIFEST = INPUT_DIR / "monthly_input_manifest.json"
INPUT_API = INPUT_DIR / "monthly_generation_api_snapshot.json"
SOURCE_CONFIGS = ("US-only", "GB-only", "US+GB")
METHODS = (*model.METHODS, "fuelshare_history")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Create source-side CN5 training features without target information."""
    frame = raw.copy().reset_index(drop=True)
    if "other_renewable" not in frame:
        frame["other_renewable"] = 0.0
    months = model.public_months(frame)
    features = model.public_features(months, frame.clock)
    features["country"] = frame.country.to_numpy()
    features["region"] = frame.region.to_numpy()
    features["year"] = frame.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(frame))
    return features.dropna(subset=list(model.HISTORY)).copy()


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def fit_source_models(
    source_raw: dict[str, pd.DataFrame],
    source_prepared: dict[str, pd.DataFrame],
    config: str,
) -> tuple[dict, dict, dict]:
    allowed_countries = {"US-only": {"US"}, "GB-only": {"GB"}, "US+GB": {"US", "GB"}}[config]
    regions = [region for region, raw in source_raw.items()
               if raw.country.iloc[0] in allowed_countries]
    train_parts = [source_prepared[region].loc[source_prepared[region].year.eq(2022)]
                   for region in regions]
    train = pd.concat(train_parts, ignore_index=True)
    train_fuel = np.concatenate([
        fuel_for(source_raw[region], part)
        for region, part in zip(regions, train_parts)
    ])
    train_coarse = np.concatenate([
        model.coarse_values(source_raw[region].iloc[part.raw_row.to_numpy()])
        for region, part in zip(regions, train_parts)
    ])
    if len(train) != len(train_fuel) or train.empty:
        raise ValueError(f"empty or misaligned source training data for {config}")
    base_models = model.fit(train, train_fuel, train_coarse)
    fuelshare_head = response.fit_fuelshare(train, train_fuel)

    residuals: dict[str, list[np.ndarray]] = {method: [] for method in METHODS}
    calibration_regions = []
    calibration_rows = 0
    for region in regions:
        part = source_prepared[region].loc[
            source_prepared[region].year.eq(2023)
            & source_prepared[region].clock.dt.month.le(6)
        ].copy()
        if part.empty:
            continue
        y = response.truth_for(source_raw[region], part)
        predicted = model.predict(base_models, part)
        predicted["fuelshare_history"] = response.predict_fuelshare(
            fuelshare_head, base_models, part
        )
        for method in METHODS:
            residuals[method].append(y - predicted[method])
        calibration_regions.append(region)
        calibration_rows += len(part)

    calibration = {}
    for method, region_residuals in residuals.items():
        if not region_residuals:
            raise ValueError(f"no source calibration rows for {config}/{method}")
        region_bias = [float(values.mean()) for values in region_residuals]
        bias = float(np.mean(region_bias))
        sigma = float(np.sqrt(np.mean([
            np.mean((values - bias) ** 2) for values in region_residuals
        ])))
        calibration[method] = {"bias": bias, "sigma": max(sigma, 1.0)}

    audit = {
        "source_countries": sorted(allowed_countries),
        "training_regions": regions,
        "calibration_regions": calibration_regions,
        "training_year": 2022,
        "calibration_period": "2023-01..2023-06",
        "training_rows": int(len(train)),
        "calibration_rows": int(calibration_rows),
    }
    return {"base": base_models, "fuelshare": fuelshare_head}, calibration, audit


def target_features() -> pd.DataFrame:
    monthly = pd.read_csv(INPUT_CSV)
    expected_months = {
        "2025/09", "2025/10", "2025/11", "2025/12",
        "2026/01", "2026/02", "2026/03", "2026/04",
    }
    if (len(monthly) != len(expected_months)
            or monthly.month.nunique() != len(expected_months)
            or set(monthly.month) != expected_months):
        raise ValueError("target monthly input snapshot does not match frozen support")
    index = pd.PeriodIndex(monthly.month.str.replace("/", "-"), freq="M")
    groups = pd.DataFrame(
        monthly[[f"{name}_share" for name in model.GROUPS]].to_numpy(float),
        index=index,
        columns=model.GROUPS,
    )
    if not np.allclose(groups.sum(axis=1), 1.0, atol=1e-9):
        raise ValueError("monthly five-group shares do not sum to one")
    clocks = pd.date_range("2026-04-01 00:00", "2026-06-30 23:00", freq="h")
    public = model.public_features(groups, pd.Series(clocks))
    public["observed_month"] = (public.period - 2).astype(str)
    public["country"] = "TW"
    public["region"] = "TW"
    public["year"] = public.clock.dt.year
    if len(public) != 2184 or public[list(model.HISTORY)].isna().any().any():
        raise ValueError("incomplete strict-CN5 target features for Taiwan Q2 2026")
    if not public.observations.eq(1.0).all():
        raise ValueError("target rows do not have all six frozen history months")
    if set(public.observed_month.unique()) != {"2026-02", "2026-03", "2026-04"}:
        raise ValueError("unexpected target-month t-2 input mapping")
    return public


def main() -> None:
    output = ROOT / "results/taipower-cn5-external-transfer-r1"
    output.mkdir(parents=True, exist_ok=False)

    source_raw, source_audit = source.load_inputs(ROOT)
    for frame in source_raw.values():
        frame["other_renewable"] = 0.0
    source_prepared = {
        region: source_features(frame) for region, frame in source_raw.items()
    }
    target = target_features()
    predictions = []
    source_models, source_calibration, source_model_audit = {}, {}, {}
    for config in SOURCE_CONFIGS:
        fitted, calibrated, audit = fit_source_models(
            source_raw, source_prepared, config
        )
        point = model.predict(fitted["base"], target)
        point["fuelshare_history"] = response.predict_fuelshare(
            fitted["fuelshare"], fitted["base"], target
        )
        frame = pd.DataFrame({
            "time_local": target.clock.astype(str),
            "observed_month": target.observed_month,
            "source": config,
            **{f"input_{name}": target[name] for name in model.GROUPS},
            **point,
        })
        for method, values in calibrated.items():
            frame[f"{method}_bias"] = values["bias"]
            frame[f"{method}_sigma"] = values["sigma"]
        predictions.append(frame)
        source_models[config] = audit
        source_calibration[config] = calibrated
        source_model_audit[config] = audit

    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_path = output / "sealed_predictions.csv.gz"
    prediction_frame.to_csv(prediction_path, index=False, compression="gzip")
    calibration_path = output / "source_calibration.json"
    calibration_path.write_text(
        json.dumps(source_calibration, indent=2) + "\n", encoding="utf-8"
    )

    source_files = {row["path"] for row in source_audit.values()}
    source_hashes = {
        str((ROOT / relative).resolve()): sha256(ROOT / relative)
        for relative in sorted(source_files)
    }
    input_hashes = {
        str(path.resolve()): sha256(path)
        for path in (INPUT_API, INPUT_CSV, LAG_CONTRACT, INPUT_MANIFEST)
    }
    code_paths = [
        Path(__file__).resolve(), PROTOCOL, Path(response.__file__).resolve(),
        Path(source.__file__).resolve(), Path(model.__file__).resolve(),
        ROOT / "src/transcif/research/monthly_adaptation.py",
    ]
    code_hashes = {str(path.resolve()): sha256(path) for path in code_paths}
    seal = {
        "status": "SEALED_TAIPOWER_Q2_2026_FROZEN_METHOD_EXTERNAL_STRESS_TEST",
        "prediction_sha256": sha256(prediction_path),
        "prediction_rows": int(len(prediction_frame)),
        "source_calibration_sha256": sha256(calibration_path),
        "protocol_sha256": sha256(PROTOCOL),
        "target_response_sample_values_opened_before_seal": True,
        "full_target_response_series_or_scoring_metrics_opened_before_seal": False,
        "target_period": "2026-04-01..2026-06-30 Asia/Taipei",
        "target_hours_per_source_config": 2184,
        "target_input_contract": "official revised nationwide CN5 monthly shares t-2..t-7 plus local calendar",
        "input_snapshot_sha256": input_hashes,
        "source_input_sha256": source_hashes,
        "code_sha256": code_hashes,
        "source_models": source_models,
        "source_model_audit": source_model_audit,
        "source_calibration": source_calibration,
        "methods": list(METHODS),
        "source_data_audit": source_audit,
    }
    seal_path = output / "prediction_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")
    if sha256(prediction_path) != seal["prediction_sha256"]:
        raise ValueError("prediction seal mismatch")
    print(f"sealed {len(prediction_frame)} predictions across {len(SOURCE_CONFIGS)} source configurations")


if __name__ == "__main__":
    main()
