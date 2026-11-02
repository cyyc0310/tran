#!/usr/bin/env python3
"""Apply frozen US/GB CN5 generation-proxy models to Japan's nine TSOs."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.research import cross_country as cc  # noqa: E402

PROTOCOL = ROOT / "docs/JAPAN_CN5_EXTERNAL_TRANSFER_PROTOCOL_20261102.md"
README = ROOT / "data/japan-hourly-ci-zenodo-2026-r1/README.md"
AUX = ROOT / "data/japan-hourly-ci-zenodo-2026-r1/auxiliarly.zip"
AREAS = {
    "Hokkaido": "Hokkaido_Electric_Power_Network",
    "Tohoku": "Tohoku_Electric_Power_Network",
    "TEPCO": "TEPCO_Power_Grid",
    "Chubu": "Chubu_Electric_Power_Grid",
    "Hokuriku": "Hokuriku_Electric_Power_Transmission_and_Distribution",
    "Kansai": "Kansai_Transmission_and_Distribution",
    "Chugoku": "Chugoku_Electric_Power_Transmission_and_Distribution",
    "Shikoku": "Shikoku_Electric_Power_Transmission_and_Distribution",
    "Kyushu": "Kyushu_Electric_Power_Transmission_and_Distribution",
}
MIX_COLUMNS = (
    "thermal_coal_MWh", "thermal_lng_MWh", "thermal_oil_MWh",
    "thermal_other_MWh", "hydro_MWh", "nuclear_MWh", "wind_MWh",
    "solar_MWh", "biomass_MWh", "geothermal_MWh", "other_MWh",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_japan_mix() -> tuple[dict[str, pd.DataFrame], dict]:
    frames, audits = {}, {}
    with ZipFile(AUX) as archive:
        for area, suffix in AREAS.items():
            filename = f"auxiliarly/generation_mix_MWh_2024_{suffix}.csv"
            halfhour = pd.read_csv(archive.open(filename))
            stamp = pd.to_datetime(halfhour.timestamp, format="mixed", utc=True)
            local = stamp.dt.tz_convert("Asia/Tokyo").dt.tz_localize(None)
            if local.duplicated().any():
                raise ValueError(f"duplicate Japan timestamp: {area}")
            fuel = halfhour.loc[:, MIX_COLUMNS].apply(pd.to_numeric, errors="coerce")
            if fuel.isna().any().any():
                raise ValueError(f"non-numeric mapped fuel data: {area}")
            fine = pd.DataFrame(index=local)
            fine["coal"] = fuel.thermal_coal_MWh.clip(lower=0).to_numpy()
            fine["gas"] = fuel.thermal_lng_MWh.clip(lower=0).to_numpy()
            fine["oil"] = fuel.thermal_oil_MWh.clip(lower=0).to_numpy()
            fine["other"] = fuel[["thermal_other_MWh", "biomass_MWh",
                                   "geothermal_MWh", "other_MWh"]].clip(lower=0).sum(axis=1).to_numpy()
            fine["hydro"] = fuel.hydro_MWh.clip(lower=0).to_numpy()
            fine["nuclear"] = fuel.nuclear_MWh.clip(lower=0).to_numpy()
            fine["wind"] = fuel.wind_MWh.clip(lower=0).to_numpy()
            fine["solar"] = fuel.solar_MWh.clip(lower=0).to_numpy()
            hourly = fine.resample("h", label="left", closed="left").sum(min_count=2)
            count = fine.resample("h", label="left", closed="left").size()
            hourly = hourly.loc[count.eq(2)].copy()
            hourly.index.name = "clock"
            hourly = hourly.reset_index()
            hourly["time"] = pd.to_datetime(hourly.clock)
            hourly["country"], hourly["region"] = "JP", area
            hourly["expected_hours"] = hourly.clock.dt.days_in_month * 24
            hourly["area"] = area
            frames[area] = hourly

            mapped_original = pd.DataFrame({
                "coal": fuel.thermal_coal_MWh,
                "gas": fuel.thermal_lng_MWh,
                "oil": fuel.thermal_oil_MWh,
                "other": fuel[["thermal_other_MWh", "biomass_MWh",
                                "geothermal_MWh", "other_MWh"]].sum(axis=1),
                "hydro": fuel.hydro_MWh,
                "nuclear": fuel.nuclear_MWh,
                "wind": fuel.wind_MWh,
                "solar": fuel.solar_MWh,
            })
            audits[area] = {
                "raw_30m_rows": int(len(halfhour)),
                "unique_timestamps": int(local.nunique()),
                "hourly_rows": int(len(hourly)),
                "expected_hourly_rows": 8760,
                "negative_mapped_cells_by_fuel": {
                    name: int(mapped_original[name].lt(0).sum()) for name in cc.FUELS
                },
                "negative_mapped_mwh_by_fuel": {
                    name: float(mapped_original[name].clip(upper=0).sum()) for name in cc.FUELS
                },
                "clipped_negative_values_to_zero": True,
                "monthly_raw_rows": {
                    str(period): int(count)
                    for period, count in local.dt.to_period("M").value_counts().sort_index().items()
                },
            }
            if len(hourly) != 8760:
                raise ValueError(f"expected 8760 complete hourly rows for {area}; got {len(hourly)}")
            if (hourly[list(cc.FUELS)].sum(axis=1) <= 0).any():
                raise ValueError(f"nonpositive mapped generation for {area}")
    return frames, audits


def train_and_calibrate() -> tuple[dict, dict, dict]:
    sources, source_audit = cc.load_inputs(ROOT)
    usable = {}
    for region, raw in sources.items():
        features = cc.monthly_inputs(raw)
        features["raw_row"] = features.index
        usable[region] = features.dropna(subset=list(cc.COARSE)).copy()
    training = pd.concat(
        [usable[region].loc[usable[region].year.eq(2022)] for region in sources],
        ignore_index=True,
    )
    fuel = np.concatenate([
        sources[region].loc[usable[region].loc[usable[region].year.eq(2022), "raw_row"],
                           list(cc.FUELS)].to_numpy(float)
        for region in sources
    ])
    if len(training) != len(fuel) or training.empty:
        raise ValueError("source training features and labels do not align")
    models = cc.fit_models(training, fuel)

    residuals = {name: [] for name in cc.METHODS}
    support = {}
    for region, raw in sources.items():
        features = usable[region]
        cal = features.loc[features.year.eq(2023)].copy()
        if cal.empty:
            raise ValueError(f"no frozen 2023 calibration rows for {region}")
        prediction = cc.predict_models(models, cal)
        truth = cc.labels(raw.loc[cal.raw_row])
        support[region] = {
            "training_2022_rows": int(features.year.eq(2022).sum()),
            "calibration_2023_rows": int(len(cal)),
        }
        for name in cc.METHODS:
            residuals[name].append(truth - prediction[name])
    calibration = {}
    for name, parts in residuals.items():
        bias = float(np.mean([part.mean() for part in parts]))
        sigma = max(float(np.sqrt(np.mean([np.mean((part-bias)**2) for part in parts]))), 1.)
        calibration[name] = {"bias": bias, "sigma": sigma}
    return models, calibration, {"source_audit": source_audit, "source_rows": support,
                                 "training_rows": int(len(training))}


def target_inputs(frames: dict[str, pd.DataFrame]) -> tuple[dict[str, pd.DataFrame], dict]:
    output, support = {}, {}
    for area, raw in frames.items():
        features = cc.monthly_inputs(raw)
        period = pd.to_datetime(features.clock).dt.to_period("M")
        features = features.loc[period.between(pd.Period("2024-06", freq="M"),
                                                pd.Period("2025-03", freq="M"))].copy()
        features["area"] = area
        features["raw_row"] = features.index
        if features[list(cc.COARSE)].isna().any().any():
            raise ValueError(f"missing exact t-2 CN5 features for {area}")
        if len(features) != 7296:
            raise ValueError(f"expected 7296 target rows for {area}; got {len(features)}")
        output[area] = features.reset_index(drop=True)
        support[area] = {
            "target_hours": int(len(features)),
            "target_start": str(features.clock.min()),
            "target_end": str(features.clock.max()),
            "target_months": sorted(str(x) for x in period.loc[features.index].unique()),
            "observed_month_lag_example_start": str(features.observed_month.iloc[0]),
            "observed_month_lag_example_end": str(features.observed_month.iloc[-1]),
        }
    return output, support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    source_model, calibration, source_audit = train_and_calibrate()
    japan_frames, feature_audit = read_japan_mix()
    queries, target_support = target_inputs(japan_frames)
    prediction_parts = []
    for area, query in queries.items():
        prediction = cc.predict_models(source_model, query)
        prediction_parts.append(pd.DataFrame({
            "time_local": query.clock,
            "area": area,
            "observed_month": query.observed_month,
            **prediction,
        }))
    predictions = pd.concat(prediction_parts, ignore_index=True)
    prediction_path = out / "sealed_predictions.csv.gz"
    predictions.to_csv(prediction_path, index=False, compression="gzip")
    calibration_path = out / "source_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")

    source_paths = [ROOT / source_audit["source_audit"][r]["path"]
                    for r in sorted(source_audit["source_audit"])]
    inputs_hash = {str(path.resolve()): sha256(path)
                   for path in [README, AUX, *source_paths]}
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(cc.__file__).resolve()]
    code_hashes = {str(path.resolve()): sha256(path) for path in code_paths}
    seal = {
        "prediction_sha256": sha256(prediction_path),
        "prediction_rows": int(len(predictions)),
        "source_calibration_sha256": sha256(calibration_path),
        "target_response_computed_before_seal": False,
        "target_response_files_opened": False,
        "input_contract": "Japanese nine-area generation mix at exact t-2, five CN-compatible groups plus calendar",
        "target_period": "2024-06 through 2025-03 JST",
        "source_years": {"fit": 2022, "calibration": 2023},
        "source_inputs_sha256": inputs_hash,
        "code_sha256": code_hashes,
        "methods": list(cc.METHODS),
    }
    seal_path = out / "prediction_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2) + "\n")
    if sha256(prediction_path) != seal["prediction_sha256"]:
        raise RuntimeError("forecast seal check failed")
    if any(sha256(Path(path)) != value for path, value in {**inputs_hash, **code_hashes}.items()):
        raise RuntimeError("source input or code changed before target response construction")

    # The Japan hourly response is derived only after the sealed predictions exist.
    if (out / "prediction_seal.json").exists() is False:
        raise RuntimeError("prediction seal missing")
    scoring_rows = []
    prediction_check = pd.read_csv(prediction_path)
    prediction_check["time_local"] = pd.to_datetime(prediction_check.time_local, errors="raise")
    if sha256(prediction_path) != seal["prediction_sha256"]:
        raise RuntimeError("forecast changed before scoring")
    for area, raw in japan_frames.items():
        idx = pd.to_datetime(raw.clock).dt.to_period("M")
        selected = raw.loc[idx.between(pd.Period("2024-06", freq="M"),
                                       pd.Period("2025-03", freq="M"))].copy()
        fuel = selected[list(cc.FUELS)].to_numpy(float)
        y = cc.labels(selected)
        selected["y_common_fixed_ef_g_kwh"] = y
        selected["area"] = area
        scoring_rows.append(selected[["clock", "area", "y_common_fixed_ef_g_kwh"]])
    labels = pd.concat(scoring_rows, ignore_index=True)
    labels_path = out / "target_common_fixed_ef_labels.csv.gz"
    labels.to_csv(labels_path, index=False, compression="gzip")
    scored = prediction_check.merge(labels, left_on=["time_local", "area"],
                                    right_on=["clock", "area"], how="inner",
                                    validate="one_to_one")
    if len(scored) != len(predictions) or len(scored) != 9 * 7296:
        raise ValueError(f"target scoring support mismatch: {len(scored)} rows")

    region_rows, month_rows = [], []
    for area, group in scored.groupby("area", sort=True):
        inputs = pd.DataFrame({"clock": pd.to_datetime(group.time_local)})
        truth = group.y_common_fixed_ef_g_kwh.to_numpy(float)
        for method in cc.METHODS:
            values = cc.metrics(inputs, truth, group[method].to_numpy(float),
                                calibration[method]["bias"], calibration[method]["sigma"])
            region_rows.append({"area": area, "method": method, **values})
            for month, part in group.groupby(pd.to_datetime(group.time_local).dt.month):
                m_inputs = pd.DataFrame({"clock": pd.to_datetime(part.time_local)})
                month_rows.append({"area": area, "month": int(month), "method": method,
                                   **cc.metrics(m_inputs, part.y_common_fixed_ef_g_kwh.to_numpy(float),
                                                part[method].to_numpy(float),
                                                calibration[method]["bias"],
                                                calibration[method]["sigma"])})
    regions = pd.DataFrame(region_rows)
    regions.to_csv(out / "region_metrics.csv", index=False)
    pd.DataFrame(month_rows).to_csv(out / "monthly_metrics.csv", index=False)

    pooled_rows = []
    pooled_inputs = pd.DataFrame({"clock": pd.to_datetime(scored.time_local)})
    pooled_truth = scored.y_common_fixed_ef_g_kwh.to_numpy(float)
    for method in cc.METHODS:
        pooled_rows.append({"method": method, **cc.metrics(
            pooled_inputs, pooled_truth, scored[method].to_numpy(float),
            calibration[method]["bias"], calibration[method]["sigma"])})
    macro = regions.groupby("method").mean(numeric_only=True).reset_index()
    macro.to_csv(out / "area_macro_metrics.csv", index=False)
    pooled = pd.DataFrame(pooled_rows)
    # All areas have identical support, so the area macro is the pooled MAE,
    # CRPS, coverage, width, and interval score. Shape must be centered within
    # each area-day, not across nine regions sharing a calendar date.
    shape_by_method = macro.set_index("method")["shape_mae"]
    pooled["shape_mae"] = pooled.method.map(shape_by_method)
    pooled.to_csv(out / "pooled_metrics.csv", index=False)

    audit = {
        "status": "EXTERNAL_NATIONAL_SYSTEM_FIXED_EF_PROXY_TRANSFER",
        "target_country": "Japan",
        "target_areas": list(AREAS),
        "target_label": "hourly common direct fixed-EF generation proxy recomputed from published generation mix",
        "target_response_opened_after_prediction_seal": True,
        "target_response_sha256": sha256(labels_path),
        "target_hours": int(len(scored)),
        "hours_per_area": 7296,
        "feature_data_audit": feature_audit,
        "target_tminus2_support": target_support,
        "source_training_and_calibration": source_audit,
        "source_calibration": calibration,
        "target_label_formula_g_per_kwh": dict(zip(cc.FUELS, cc.EF.tolist())),
        "native_dataset_lifecycle_factor_label_not_scored": True,
        "target_input_vintage": "Zenodo v1 release; reconstructed final generation totals, not as-issued month releases",
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print("Area macro:")
    print(macro.round(4).to_string(index=False))
    print("Pooled:")
    print(pooled.round(4).to_string(index=False))
    print("Saved", out)


if __name__ == "__main__":
    main()
