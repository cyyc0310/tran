#!/usr/bin/env python3
"""Replicate the frozen strict-CN5 fuel-share response head on Italy 2023."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_energy_charts_nine_country_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/ITALY_FUELSHARE_RESPONSE_TRANSFER_PROTOCOL_20261102.md"
TARGET_CODE = "it"
TARGET_YEAR = 2023
SCORE_MONTHS = tuple(range(3, 13))
SOURCE_CODES = nine.COUNTRIES
METHODS = ("anchor700", "ratio_recomposition", "macro_recomposition", "ridge_history",
           "direct_latest", "direct_history", "factor_history", "fuelshare_history")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_features(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().reset_index(drop=True)
    months = model.public_months(frame)
    features = model.public_features(months, frame.clock)
    features["country"] = frame.country.to_numpy()
    features["region"] = frame.region.to_numpy()
    features["year"] = frame.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(frame))
    return features.dropna(subset=list(model.HISTORY)).copy()


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    # Country-specific white-list frozen in the protocol. Missing API series are
    # explicit zeros only when absent from this schema; in-series nulls remain missing.
    nine.base.TYPE_MAP[TARGET_CODE] = {
        "coal": ("fossil_hard_coal",),
        "gas": ("fossil_gas", "fossil_coal_derived_gas"),
        "oil": ("fossil_oil",),
        "other": ("biomass", "waste", "others"),
        "nuclear": (),
        "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
        "wind": ("wind_onshore", "wind_offshore"),
        "solar": ("solar",),
        "other_renewable": ("geothermal",),
    }

    source_raw, source_features, source_audit = {}, {}, {}
    for code in SOURCE_CODES:
        folder, manifest_path, manifest = nine.data_locations(code)
        nine.base.DATA = folder
        raw, audit = nine.base.load_country(code, manifest)
        source_raw[code], source_features[code] = raw.reset_index(drop=True), strict_features(raw)
        source_audit[code] = {"path": str((folder / f"{code}_2022_2024.json").relative_to(ROOT)),
                              "manifest": str(manifest_path.relative_to(ROOT)), **audit}

    target_folder = nine.OLD_DATA
    target_manifest = json.loads((target_folder / "acquisition_manifest.json").read_text())
    nine.base.DATA = target_folder
    target_raw, target_audit = nine.base.load_country(TARGET_CODE, target_manifest)
    target_raw = target_raw.reset_index(drop=True)
    target_features = strict_features(target_raw)
    query = target_features.loc[
        target_features.year.eq(TARGET_YEAR)
        & target_features.clock.dt.month.isin(SCORE_MONTHS)
    ].copy().reset_index(drop=True)
    target_utc = target_raw.iloc[query.raw_row.to_numpy()].time
    if len(query) != 7344 or target_utc.duplicated().any():
        raise ValueError(f"Italy common support changed: {len(query)} rows")

    train_parts = [source_features[c].loc[source_features[c].year.eq(2022)] for c in SOURCE_CODES]
    train = pd.concat(train_parts, ignore_index=True)
    train_fuel = np.concatenate([fuel_for(source_raw[c], part)
                                 for c, part in zip(SOURCE_CODES, train_parts)])
    train_coarse = np.concatenate([
        model.coarse_values(source_raw[c].iloc[part.raw_row.to_numpy()])
        for c, part in zip(SOURCE_CODES, train_parts)
    ])
    baseline_models = model.fit(train, train_fuel, train_coarse)
    candidate_head = response.fit_fuelshare(train, train_fuel)

    calibration_parts = [source_features[c].loc[
        source_features[c].year.eq(2023) & source_features[c].clock.dt.month.le(6)
    ] for c in SOURCE_CODES]
    calibration = pd.concat(calibration_parts, ignore_index=True)
    calibration_fuel = np.concatenate([fuel_for(source_raw[c], part)
                                       for c, part in zip(SOURCE_CODES, calibration_parts)])
    calibration_y = calibration_fuel @ model.FACTORS / calibration_fuel.sum(axis=1)
    calibration_weights = model.weights(calibration)
    calibration_prediction = model.predict(baseline_models, calibration)
    calibration_prediction["fuelshare_history"] = response.predict_fuelshare(
        candidate_head, baseline_models, calibration)
    radii = {method: float(weighted_quantile(
        np.abs(calibration_y - calibration_prediction[method]), calibration_weights, .9
    )) for method in METHODS}

    target_prediction = model.predict(baseline_models, query)
    target_prediction["fuelshare_history"] = response.predict_fuelshare(
        candidate_head, baseline_models, query)
    target_times = target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()
    prediction_frame = pd.DataFrame({"time": target_times, **target_prediction})
    prediction_path = out / "candidate_and_baseline_predictions.csv.gz"
    prediction_frame.to_csv(prediction_path, index=False, compression="gzip")

    input_paths = [PROTOCOL, target_folder / "acquisition_manifest.json",
                   target_folder / "audit.json", target_folder / "it_2022_2024.json"]
    for entry in source_audit.values():
        input_paths.extend([ROOT / entry["path"], ROOT / entry["manifest"]])
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_paths = [Path(__file__), PROTOCOL, Path(nine.__file__), Path(nine.base.__file__),
                  Path(response.__file__), Path(model.__file__),
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    code_hashes = {str(p.resolve()): sha(p) for p in sorted(set(code_paths))}
    prediction_seal = {
        "prediction_sha256": sha(prediction_path), "protocol_sha256": sha(PROTOCOL),
        "code_sha256": code_hashes, "input_sha256": input_hashes,
        "source_countries": list(SOURCE_CODES), "training_year": 2022,
        "source_calibration_period": "2023-01..2023-06",
        "target_country": "IT", "target_period": "2023-03..2023-12",
        "target_hours": int(len(prediction_frame)), "target_cif_scored_before_seal": False,
        "target_input_budget": "strict CN5 t-2..t-7 five-category monthly shares plus local calendar",
        "label_definition": "fixed-factor mapped generation-side proxy; not official/measured CIF",
        "source_audit": source_audit, "target_mapping_audit": target_audit,
        "source_interval_radii": radii,
    }
    (out / "prediction_seal.json").write_text(
        json.dumps(prediction_seal, ensure_ascii=False, indent=2) + "\n")
    if sha(prediction_path) != prediction_seal["prediction_sha256"]:
        raise ValueError("prediction seal mismatch")

    # Target hourly response is scored only after predictions and source radii are sealed.
    target_truth = fuel_for(target_raw, query) @ model.FACTORS
    target_denominator = fuel_for(target_raw, query).sum(axis=1)
    target_truth = target_truth / target_denominator
    rows = []
    for month in SCORE_MONTHS:
        mask = query.clock.dt.month.eq(month).to_numpy()
        times = query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy()
        for method in METHODS:
            prediction = target_prediction[method][mask]
            error = target_truth[mask] - prediction
            daily = pd.DataFrame({"truth": target_truth[mask], "prediction": prediction,
                                  "day": times})
            centered = daily[["truth", "prediction"]] - daily.groupby("day")[
                ["truth", "prediction"]].transform("mean")
            radius = radii[method]
            rows.append({"country": "IT", "method": method, "month": month,
                         "hours": int(mask.sum()), "mae": float(np.mean(np.abs(error))),
                         "shape_mae": float(np.mean(np.abs(centered.truth-centered.prediction))),
                         "coverage90": float(np.mean(np.abs(error) <= radius)),
                         "width90": 2 * radius,
                         "winkler90": float(np.mean(2*radius + 20*np.maximum(np.abs(error)-radius,0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    summary = monthly.groupby("method")[
        ["mae", "shape_mae", "coverage90", "width90", "winkler90"]
    ].mean().reset_index()
    total_hours = monthly.groupby("method").hours.sum().rename("hours")
    summary = summary.merge(total_hours, on="method", validate="one_to_one")
    summary.to_csv(out / "summary_metrics.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "status": "SEQUENTIAL_ITALY_HOURLY_PROXY_METHOD_REPLICATION_NOT_BLIND_COUNTRY",
        "target_country_daily_labels_previously_viewed": True,
        "target_hourly_proxy_previously_scored": False,
        "blind_geographic_confirmation": False,
        "target_hours": int(len(query)), "months": list(SCORE_MONTHS),
        "training_hours": int(len(train)), "calibration_hours": int(len(calibration)),
        "source_countries": list(SOURCE_CODES), "target_hourly_response_scored_after_prediction_seal": True,
        "target_monthly_inputs_reconstructed_from_target_hourly_generation": True,
        "target_vintage": "current revised Energy-Charts hourly values; monthly inputs reconstructed as t-2",
        "target_response": "fixed-factor generation-side proxy, not official/measured CIF",
        "target_mapping": "fixed by protocol; geothermal is zero-EF other_renewable; coal-derived gas maps to gas",
    }, ensure_ascii=False, indent=2) + "\n")
    score_files = ("monthly_metrics.csv", "summary_metrics.csv", "status.json")
    target_raw_sha = next(value for path, value in input_hashes.items()
                          if path.endswith("/it_2022_2024.json"))
    score_seal = {
        "status": "SCORED_AFTER_PREDICTION_SEAL",
        "prediction_sha256": prediction_seal["prediction_sha256"],
        "prediction_seal_sha256": sha(out / "prediction_seal.json"),
        "target_raw_generation_input_sha256": target_raw_sha,
        "score_files_sha256": {name: sha(out / name) for name in score_files},
        "protocol_sha256": prediction_seal["protocol_sha256"],
        "target_hours": int(len(query)),
    }
    (out / "score_seal.json").write_text(
        json.dumps(score_seal, ensure_ascii=False, indent=2) + "\n")
    print(f"sealed IT candidate and baselines for {len(query)} hours", flush=True)
    print(f"scored {len(monthly)} country-month-method rows", flush=True)


if __name__ == "__main__":
    main()
