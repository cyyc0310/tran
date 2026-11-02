#!/usr/bin/env python3
"""Evaluate the frozen strict-CN5 fuel-share response head on AEMO proxy hours."""
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
import run_aemo_cn5_cross_domain_transfer as aemo_runner  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.cross_country import load_inputs  # noqa: E402

PROTOCOL = ROOT / "docs/AEMO_FUELSHARE_RESPONSE_TRANSFER_PROTOCOL_20261102.md"
BASELINE_DIR = ROOT / "results/aemo-cn5-cross-domain-transfer-r2"
FIXED_LABEL_DIR = ROOT / "results/aemo-cn5-common-ef-cross-domain-r5"
SOURCE_CONFIGS = ("US-only", "GB-only", "US+GB")
ELIGIBLE_REGIONS = ("NSW1", "QLD1", "TAS1")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_features(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.copy().reset_index(drop=True)
    if "other_renewable" not in frame:
        frame["other_renewable"] = 0.
    months = model.public_months(frame)
    features = model.public_features(months, frame.clock)
    features["country"] = frame.country.to_numpy()
    features["region"] = frame.region.to_numpy()
    features["year"] = frame.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(frame))
    return features.dropna(subset=list(model.HISTORY)).copy()


def fuel_from(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def coarse_profile(frame: pd.DataFrame, fuel: np.ndarray) -> np.ndarray:
    thermal = fuel[:, :4].sum(axis=1)
    coarse = np.column_stack([thermal, fuel[:, 5], fuel[:, 4], fuel[:, 6], fuel[:, 7]])
    profiles = []
    for region in frame.region.unique():
        mask = frame.region.to_numpy() == region
        local = pd.DataFrame(coarse[mask], columns=model.GROUPS)
        local["hour"] = frame.loc[mask, "clock"].dt.hour.to_numpy()
        by_hour = local.groupby("hour")[list(model.GROUPS)].mean().reindex(range(24)).fillna(0.)
        profiles.append(by_hour.div(by_hour.mean().replace(0., np.nan)).fillna(1.).to_numpy())
    if not profiles:
        raise ValueError("no source profiles for fuel-share model")
    return np.mean(profiles, axis=0)


def fit_candidate(raw_by_region: dict[str, pd.DataFrame], feature_by_region: dict[str, pd.DataFrame],
                  source: str) -> tuple[dict, dict, dict]:
    country = {"US-only": "US", "GB-only": "GB"}.get(source)
    eligible = [r for r, raw in raw_by_region.items()
                if source == "US+GB" or raw.country.iloc[0] == country]
    if not eligible:
        raise ValueError(f"no source regions found for {source}")
    train_parts = [feature_by_region[r].loc[feature_by_region[r].year.eq(2022)] for r in eligible]
    train = pd.concat(train_parts, ignore_index=True)
    fuel = np.concatenate([fuel_from(raw_by_region[r], part)
                           for r, part in zip(eligible, train_parts)])
    profile = coarse_profile(train, fuel)
    head = response.fit_fuelshare(train, fuel)
    calibration_residuals = []
    calibration_regions = []
    for region in eligible:
        part = feature_by_region[region].loc[feature_by_region[region].year.eq(2023)].copy()
        if part.empty:
            continue
        y = response.truth_for(raw_by_region[region], part)
        pred = response.predict_fuelshare(head, {"profile": profile}, part)
        calibration_residuals.append(y - pred)
        calibration_regions.append(region)
    if not calibration_residuals:
        raise ValueError(f"no source calibration hours for {source}")
    region_means = [float(values.mean()) for values in calibration_residuals]
    bias = float(np.mean(region_means))
    sigma = float(np.sqrt(np.mean([np.mean((values - bias) ** 2)
                                   for values in calibration_residuals])))
    audit = {"training_regions": eligible, "calibration_regions": calibration_regions,
             "training_hours": int(len(train)), "training_year": 2022,
             "calibration_year": 2023, "bias": bias, "sigma": max(sigma, 1.)}
    return {"profile": profile, "fuelshare": head}, audit, {"bias": bias, "sigma": max(sigma, 1.)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staging", type=Path,
                        default=Path("/Users/cyyc0310/Desktop/tran/data/staging/aemo-native-2023"))
    parser.add_argument("--preflight", type=Path, default=ROOT / "results/aemo-cn5-fuel-preflight-r1")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    baseline_seal = json.loads((BASELINE_DIR / "prediction_seal.json").read_text())
    if sha(BASELINE_DIR / baseline_seal["prediction_file"]) != baseline_seal["prediction_sha256"]:
        raise ValueError("baseline AEMO predictions do not match their seal")
    fixed_seal = json.loads((FIXED_LABEL_DIR / "target_label_seal.json").read_text())
    source_raw, source_audit = load_inputs(ROOT)
    source_raw = {r: f.copy().reset_index(drop=True) for r, f in source_raw.items()}
    for frame in source_raw.values():
        frame["other_renewable"] = 0.
    source_features = {r: strict_features(f) for r, f in source_raw.items()}

    aemo_raw, aemo_provenance = aemo_runner.make_aemo_inputs(args.staging)
    aemo_features = {}
    aemo_frames = {}
    for region, frame in aemo_raw.groupby("region", sort=True):
        local = frame.copy().reset_index(drop=True)
        local["country"] = "AEMO"
        local["other_renewable"] = 0.
        aemo_frames[region] = local
        aemo_features[region] = strict_features(local)
    support = pd.read_csv(args.preflight / "region_support.csv").set_index("region")
    primary = sorted(support.index[support.eligible_fraction.ge(.95)].tolist())
    if primary != list(ELIGIBLE_REGIONS):
        raise ValueError(f"AEMO support gate changed: {primary}")

    model_audits, candidate_calibration = {}, {}
    predictions = []
    for source in SOURCE_CONFIGS:
        candidate, audit, calibration = fit_candidate(source_raw, source_features, source)
        model_audits[source] = audit
        candidate_calibration[source] = calibration
        for region in sorted(aemo_frames):
            part = aemo_features[region].loc[
                aemo_features[region].year.eq(2023)
                & aemo_features[region].clock.dt.month.between(3, 12)
            ].copy()
            value = response.predict_fuelshare(candidate["fuelshare"], candidate, part)
            predictions.append(pd.DataFrame({
                "time_local": aemo_frames[region].iloc[part.raw_row.to_numpy()].clock.astype(str).to_numpy(),
                "region": region, "source": source, "fuelshare_history": value,
                "fuelshare_bias": calibration["bias"], "fuelshare_sigma": calibration["sigma"],
                "eligible_cn5_hour": aemo_frames[region].iloc[
                    part.raw_row.to_numpy()].eligible_cn5_hour.astype(bool).to_numpy(),
            }))
        print(f"sealed {source}: {audit['training_hours']} source hours", flush=True)
    candidate_predictions = pd.concat(predictions, ignore_index=True)
    prediction_path = out / "candidate_predictions.csv.gz"
    candidate_predictions.to_csv(prediction_path, index=False, compression="gzip")
    seal = {
        "candidate_prediction_sha256": sha(prediction_path),
        "protocol_sha256": sha(PROTOCOL), "runner_sha256": sha(Path(__file__)),
        "source_module_sha256": sha(Path(response.__file__)),
        "feature_module_sha256": sha(Path(model.__file__)),
        "source_file_sha256": {v["path"]: sha(ROOT / v["path"])
                               for v in source_audit.values()},
        "aemo_input_sha256": aemo_provenance,
        "baseline_prediction_sha256": baseline_seal["prediction_sha256"],
        "target_labels_opened_before_prediction_seal": False,
        "candidate_models": model_audits, "candidate_calibration": candidate_calibration,
        "primary_regions": primary,
    }
    (out / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    if sha(prediction_path) != seal["candidate_prediction_sha256"]:
        raise ValueError("candidate prediction seal mismatch")

    # Read target labels only after the candidate prediction and baseline seals pass.
    if sha(FIXED_LABEL_DIR / "target_fixed_ef_labels.csv") != fixed_seal["labels_sha256"]:
        raise ValueError("common-EF target label file does not match its frozen seal")
    labels = pd.read_csv(FIXED_LABEL_DIR / "target_fixed_ef_labels.csv")
    baseline = pd.read_csv(BASELINE_DIR / baseline_seal["prediction_file"])
    merged = candidate_predictions.merge(
        baseline, on=["time_local", "region", "source"], how="inner", validate="one_to_one",
        suffixes=("", "_baseline"))
    merged = merged.merge(labels[["time_local", "region", "fixed_ef_cif_g_kwh",
                                  "fixed_ef_fuel_support"]],
                          on=["time_local", "region"], how="inner", validate="many_to_one")
    merged["target_eligible"] = (merged.eligible_cn5_hour.astype(bool)
                                 & merged.fixed_ef_fuel_support.ge(.99))
    scored = merged.loc[merged.target_eligible & merged.region.isin(primary)].copy()
    if scored.empty:
        raise ValueError("no shared eligible hours after target label join")
    if scored.groupby(["source", "region"]).size().to_dict() != {
            (s, r): n for s in SOURCE_CONFIGS for r, n in
            {"NSW1": 7253, "QLD1": 7076, "TAS1": 7334}.items()}:
        raise ValueError("common EF sample no longer matches frozen 21,663-hour support")

    comparison = []
    base_methods = ["anchor700", "source_thermal_ef", "coarse_ridge",
                    "coarse_boosting", "coarse_recomposition"]
    for (source, region), group in scored.groupby(["source", "region"], sort=True):
        inputs = pd.DataFrame({"clock": pd.to_datetime(group.time_local)})
        y = group.fixed_ef_cif_g_kwh.to_numpy(float)
        for method in base_methods:
            pred = group[method].to_numpy(float)
            bias = float(group[f"{method}_bias"].iloc[0])
            sigma = float(group[f"{method}_sigma"].iloc[0])
            values = aemo_runner.metrics(inputs, y, pred, bias, sigma)
            comparison.append({"source": source, "region": region, "method": method, **values})
        values = aemo_runner.metrics(inputs, y, group.fuelshare_history.to_numpy(float),
                                     float(group.fuelshare_bias.iloc[0]),
                                     float(group.fuelshare_sigma.iloc[0]))
        comparison.append({"source": source, "region": region,
                           "method": "fuelshare_history", **values})
    region_metrics = pd.DataFrame(comparison)
    region_metrics.to_csv(out / "region_metrics.csv", index=False)
    macro = region_metrics.groupby(["source", "method"]).mean(numeric_only=True).reset_index()
    macro.to_csv(out / "macro_metrics.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "status": "SEQUENTIAL_AEMO_METHOD_REPLICATION_NOT_BLIND",
        "source_configs": list(SOURCE_CONFIGS), "primary_regions": primary,
        "score_period": "2023-03..2023-12", "scored_hours_per_source": int(len(scored) / len(SOURCE_CONFIGS)),
        "scored_rows_all_source_configs": int(len(scored)),
        "label": "common fixed-factor generation-side proxy; not official hourly CIF",
        "target_monthly_inputs": "AEMO SCADA retrospective t-2 five-group totals",
        "target_response_opened_after_candidate_prediction_seal": True,
        "blind_confirmation": False,
    }, ensure_ascii=False, indent=2) + "\n")
    print(f"scored {len(scored)} paired target hours after seals", flush=True)


if __name__ == "__main__":
    main()
