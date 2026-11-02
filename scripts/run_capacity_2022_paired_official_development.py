#!/usr/bin/env python3
"""Compare a 2022 coal/gas capacity budget in the existing US↔GB task.

This is a development re-score on already-viewed 2024 labels. US EIA-860
2022 is archived as issued; GB 2022 capacity comes from the latest revised
DUKES Table 5.7 because the original 2023 workbook was not publicly recoverable.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from run_cross_country import ROOT
import run_official_intensity_transfer as official

BASE_RESULTS = ROOT / "results/official-intensity-transfer-capacity-r1"
CAPACITY_INPUT = ROOT / "docs/evidence/public-capacity-2022-asof-2024-r1"
CAPACITY_JSON = ROOT / "data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capacity_table() -> dict:
    table = json.loads(CAPACITY_JSON.read_text())
    audit = json.loads((CAPACITY_INPUT / "audit_summary.json").read_text())
    for region, values in audit["eia860_by_ba"].items():
        table["regions"][region].update({
            "coal_mw": values["coal_nameplate_mw"],
            "gas_mw": values["gas_nameplate_mw"],
            "capacity_year": 2022,
            "available_by_utc": "2023-09-19T00:00:00Z",
            "source": "EIA-860 original 2022 archived release, operable generators",
            "scope": "registered BA nameplate by primary energy source",
        })
    dukes = audit["dukes_current_revised_2022_mw"]
    table["regions"]["GB"].update({
        "coal_mw": dukes["coal"], "gas_mw": dukes["gas"],
        "capacity_year": 2022,
        "available_by_utc": "2023-07-27T00:00:00Z",
        "source": "DUKES Table 5.7 2022 historical row, 2026 revised workbook",
        "scope": "UK aggregate including Northern Ireland, revised value proxy for GB",
    })
    table["status"] = "2022_STAT_YEAR_PAIRED_OFFICIAL_DEVELOPMENT_BUDGET"
    return table


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cap = capacity_table()
    raw, prepared, source_label_map, audit = official.prepare(ROOT, cap)
    pred_files, calibration_records = [], {}
    directions = (("US", "GB"), ("GB", "US"))
    prediction_records = []

    # Produce every target forecast and source-only calibration before opening
    # the already-viewed target response files for this development re-score.
    for source_country, target_country in directions:
        sources = [r for r, frame in prepared.items() if frame.country.iloc[0] == source_country]
        targets = [r for r, frame in prepared.items() if frame.country.iloc[0] == target_country]
        train_parts, response_parts, fuel_parts = [], [], []
        for region in sources:
            frame = prepared[region].loc[prepared[region].year.eq(2022)].copy()
            response = official.aligned_labels(frame, source_label_map[region])
            finite = np.isfinite(response)
            frame = frame.loc[finite].copy()
            frame["official_y"] = response[finite]
            train_parts.append(frame)
            response_parts.append(response[finite])
            fuel_parts.append(raw[region].iloc[frame.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float))
        train = pd.concat(train_parts, ignore_index=True)
        y_train = np.concatenate(response_parts)
        model = official.train_official(train, y_train)
        proxy = official.fit_models(train.reset_index(drop=True), np.concatenate(fuel_parts))

        residual_parts = {method: [] for method in official.METHODS}
        for region in sources:
            cal = prepared[region].loc[prepared[region].year.eq(2023)].copy()
            response = official.aligned_labels(cal, source_label_map[region])
            finite = np.isfinite(response)
            cal = cal.loc[finite].copy()
            predictions = official.point_predictions(model, proxy, cal)
            for method in official.METHODS:
                residual_parts[method].append(response[finite] - predictions[method])
        calibration = official.calibrate(residual_parts)
        key = f"{source_country}_to_{target_country}"
        calibration_records[key] = calibration
        (out / f"{source_country}_source_calibration.json").write_text(
            json.dumps(calibration, indent=2) + "\n")

        for target in targets:
            query = prepared[target].loc[prepared[target].year.eq(2024)].copy()
            point = official.point_predictions(model, proxy, query)
            data = {
                "time_utc": pd.to_datetime(query.time, utc=True).astype(str).to_numpy(),
                "clock": query.clock.astype(str).to_numpy(),
                "observed_month": query.observed_month.to_numpy(),
            }
            for method in official.METHODS:
                data[method] = point[method]
                data[f"{method}_source_calibrated"] = point[method] + calibration[method]["bias"]
            path = out / f"{source_country}_to_{target}_predictions_2024.csv.gz"
            pd.DataFrame(data).to_csv(path, index=False, compression="gzip")
            pred_files.append(path)
            prediction_records.append({"source": source_country, "target_country": target_country,
                                       "target": target,
                                       "hours": len(query), "path": path.name,
                                       "sha256": sha(path)})

    # Seal forecast inputs/outputs before reading target response values.
    (out / "prediction_seal.json").write_text(json.dumps(prediction_records, indent=2) + "\n")
    hashes = {
        "code": {str(p): sha(p) for p in [Path(__file__).resolve(), Path(official.__file__),
                  ROOT / "scripts/audit_capacity_2022_asof_2024.py",
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/capacity_transfer.py",
                  ROOT / "docs/OFFICIAL_CONSUMPTION_INTENSITY_TRANSFER_PROTOCOL_20261101.md",
                  ROOT / "docs/OFFICIAL_ANNUAL_COAL_GAS_CAPACITY_PROTOCOL_20261101.md"]},
        "data": {str(p): sha(p) for p in [CAPACITY_INPUT / "audit_summary.json",
                  CAPACITY_INPUT / "capacity_2022_by_ba.csv",
                  CAPACITY_INPUT / "raw/eia8602022.zip",
                  ROOT / "docs/evidence/public-capacity-2025-asof-nov2026-r1/raw/DUKES_5.7.xlsx",
                  *[BASE_RESULTS / f for f in ["US_to_GB_official_predictions.csv.gz",
                    "GB_to_BPAT_official_predictions.csv.gz", "GB_to_CISO_official_predictions.csv.gz",
                    "GB_to_ERCO_official_predictions.csv.gz", "GB_to_FPL_official_predictions.csv.gz",
                    "GB_to_ISNE_official_predictions.csv.gz", "GB_to_MISO_official_predictions.csv.gz",
                    "GB_to_NYIS_official_predictions.csv.gz"]],
                  *[official.SOURCE_LABEL_ROOT / f"{r}_consumption_2022_2023.csv.gz"
                    for r in official.US_REGIONS],
                  official.SOURCE_LABEL_ROOT / "GB_consumption_2022_2023_halfhour.csv.gz",
                  ROOT / "data/neso-gb/df_fuel_ckan.csv",
                  ROOT / "data/eia930-hourly-co2-r1/feature_inputs/GB_generation_2022_2024.csv.gz",
                  ROOT / "data/eia930-hourly-co2-r1/acquisition_manifest.json",
                  ROOT / "data/eia930-hourly-co2-r1/coverage_audit_2022_2024.json",
                  CAPACITY_JSON,
                  *[ROOT / f"results/research/data/{r}_hourly.csv" for r in official.US_REGIONS]]}
    }
    (out / "pre_score_seal.json").write_text(json.dumps({
        "target_labels_read": False, "target_year": 2024,
        "predictions": prediction_records, "input_hashes": hashes,
    }, indent=2) + "\n")

    # Score against the project’s existing 2024 development responses.
    rows = []
    for record, path in zip(prediction_records, pred_files):
        region = record["target"]
        query_clock = pd.to_datetime(pd.read_csv(path).clock)
        forecast = pd.read_csv(path)
        y_series = official.source_labels(region, years=(2024,))
        y = official.aligned_labels(pd.DataFrame({"time": pd.to_datetime(forecast.time_utc, utc=True)}), y_series)
        methods = list(official.METHODS)
        base_path = BASE_RESULTS / f"{record['source']}_to_{region}_official_predictions.csv.gz"
        base = pd.read_csv(base_path)
        if len(base) != len(forecast):
            raise ValueError(f"base and candidate hour counts differ for {region}")
        for method in methods:
            for month in sorted(query_clock.dt.month.unique()):
                mask = query_clock.dt.month.eq(month).to_numpy()
                for vintage in ("2020", "2022"):
                    if vintage == "2020":
                        prediction = base.loc[mask, method].to_numpy(float)
                        # Existing baseline calibration is the previously sealed 2020-vintage source cal.
                        old_cal_path = BASE_RESULTS / f"{record['source']}_source_calibration.json"
                        old_cal = json.loads(old_cal_path.read_text())[method]
                        bias, sigma = old_cal["bias"], old_cal["sigma"]
                    else:
                        prediction = forecast.loc[mask, method].to_numpy(float)
                        cal = calibration_records[
                            f"{record['source']}_to_{record['target_country']}"][method]
                        bias, sigma = cal["bias"], cal["sigma"]
                    scored = official.score(y[mask], prediction, bias, sigma,
                                             query_clock.loc[mask].reset_index(drop=True))
                    rows.append({"source": record["source"], "target": region, "method": method,
                                 "capacity_stat_year": vintage, "month": int(month), **scored})
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "monthly_metrics.csv", index=False)
    metric_columns = ["mae", "shape_mae", "crps_gaussian", "coverage90", "width90", "interval_score90"]
    summary = metrics.groupby(["source", "target", "method", "capacity_stat_year"])[metric_columns].mean().reset_index()
    summary.to_csv(out / "paired_region_metrics.csv", index=False)
    target_paths = [official.TARGET_LABEL_ROOT / f"{r}_consumption_2024.csv.gz"
                    for r in official.US_REGIONS]
    target_paths.append(official.TARGET_LABEL_ROOT / "GB_consumption_2024_halfhour.csv.gz")
    target_hashes = {str(p): sha(p) for p in target_paths}
    (out / "target_scoring_hashes.json").write_text(json.dumps(target_hashes, indent=2) + "\n")
    (out / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_CAPACITY_STAT_YEAR_2022_SCORED_ON_PREVIOUSLY_VIEWED_2024_TARGETS",
        "target_labels_viewed_in_prior_experiment": True,
        "annual_capacity_vintage_note": "DUKES GB values are 2026 revisions for stat-year 2022",
        "source_train_year": 2022, "source_calibration_year": 2023, "target_year": 2024,
    }, indent=2) + "\n")
    print(summary.groupby(["target", "method", "capacity_stat_year"])[["mae", "coverage90", "width90", "interval_score90"]].mean().to_string())


if __name__ == "__main__":
    main()
