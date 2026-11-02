#!/usr/bin/env python3
"""Create a separate 2025-capacity candidate arm for the sealed Nov 2026 task.

The original 2020-capacity forecast remains the primary sealed artifact. This
runner reuses its exact target rows and adds a separately trained, as-of-valid
2025 capacity budget for later paired scoring.
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

BASE = ROOT / "docs/evidence/prospective-gb-to-us-eia923-nov2026-r1"
CAPACITY_AUDIT = ROOT / "docs/evidence/public-capacity-2025-asof-nov2026-r1"
CAPACITY_TABLE_PATH = ROOT / "data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json"
SCRIPT = Path(__file__).resolve()


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_capacity_table() -> dict:
    table = json.loads(CAPACITY_TABLE_PATH.read_text())
    audit = json.loads((CAPACITY_AUDIT / "audit_summary.json").read_text())
    if audit["statistic_year"] != 2025 or audit["availability_cutoff"] != "2026-11-01":
        raise ValueError("2025 capacity audit does not match forecast origin")
    for region, item in audit["eia860_by_ba"].items():
        table["regions"][region].update({
            "coal_mw": item["coal_nameplate_mw"],
            "gas_mw": item["gas_nameplate_mw"],
            "capacity_year": 2025,
            "available_by_utc": "2026-09-10T00:00:00Z",
            "source": "EIA-860 final 2025; registered plant BA; operable generators",
            "scope": "nameplate capacity, primary fuel mapping",
        })
    gb = audit["dukes_2025_all_generating_companies_mw"]
    table["regions"]["GB"].update({
        "coal_mw": gb["coal"], "gas_mw": gb["gas"], "capacity_year": 2025,
        "available_by_utc": "2026-07-30T00:00:00Z",
        "source": "DUKES 2026 Table 5.7, all generating companies",
        "scope": "UK aggregate including Northern Ireland; de-rated capacity proxy for GB",
    })
    table["status"] = "PROSPECTIVE_2025_CAPACITY_CANDIDATE_ASOF_2026-11-01"
    return table


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    table = make_capacity_table()

    # The base target response was never read; independently keep the target
    # response directory absent from this candidate run as well.
    target_files = list(official.TARGET_LABEL_ROOT.glob("*2026*"))
    if target_files:
        raise RuntimeError("future target-response files exist before candidate sealing")

    raw, prepared, labels, _ = official.prepare(ROOT, table)
    train_all = prepared["GB"].loc[prepared["GB"].year.eq(2022)].copy()
    y_all = official.aligned_labels(train_all, labels["GB"])
    keep = np.isfinite(y_all)
    train = train_all.loc[keep].copy().reset_index(drop=True)
    y_train = y_all[keep]
    if train.empty:
        raise ValueError("no GB 2022 source response rows")
    models = official.train_official(train, y_train)
    train_fuel = raw["GB"].iloc[train.raw_row.to_numpy()][list(official.FUELS)].to_numpy(float)
    proxy_models = official.fit_models(train, train_fuel)

    cal_frame = prepared["GB"].loc[prepared["GB"].year.eq(2023)].copy()
    y_cal_all = official.aligned_labels(cal_frame, labels["GB"])
    cal_keep = np.isfinite(y_cal_all)
    cal_frame = cal_frame.loc[cal_keep].copy()
    y_cal = y_cal_all[cal_keep]
    cal_predictions = official.point_predictions(models, proxy_models, cal_frame)
    residual_parts = {method: [y_cal - prediction]
                      for method, prediction in cal_predictions.items()}
    calibration = official.calibrate(residual_parts)
    calibration = {name: calibration[name] for name in ("capacity_ridge", "capacity_boosting")}
    (out / "gb_source_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")

    # Use the exact query rows and local timestamps from the primary seal. Only
    # the capacity fields are regenerated, from the 2025 public budget.
    seals = []
    for base_file in sorted(BASE.glob("GB_to_*_2026-11_predictions.csv.gz")):
        base_frame = pd.read_csv(base_file)
        region = str(base_frame.region.iloc[0])
        query = pd.DataFrame({
            "time": pd.to_datetime(base_frame.time_utc, utc=True),
            "clock": pd.to_datetime(base_frame.local_time),
            "country": "US", "region": region,
        })
        for field in official.COARSE:
            query[field] = base_frame[f"input_{field}_share"].to_numpy(float)
        query["year"] = query.clock.dt.year
        # All time and share interactions match the frozen official runner.
        for name, values, period in (("hour", query.clock.dt.hour, 24),
                                     ("dow", query.clock.dt.dayofweek, 7),
                                     ("month", query.clock.dt.month, 12)):
            query[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
            query[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
        for field in official.COARSE:
            for suffix in ("sin", "cos"):
                query[f"{field}_hour_{suffix}"] = query[field] * query[f"hour_{suffix}"]
        query = official.capacity_transfer.add_capacity(query, table)
        query["log_capacity_coal_gas_mw"] = np.log1p(
            query.capacity_coal_mw + query.capacity_gas_mw)

        predictions = official.point_predictions(models, proxy_models, query)
        for method in ("capacity_ridge", "capacity_boosting"):
            bias, sigma = calibration[method]["bias"], calibration[method]["sigma"]
            base_frame[f"{method}_2025_capacity_raw"] = predictions[method]
            base_frame[f"{method}_2025_capacity_source_calibrated"] = predictions[method] + bias
            radius = 1.6448536269514722 * sigma
            center = predictions[method] + bias
            base_frame[f"{method}_2025_capacity_p05_source_gaussian"] = center - radius
            base_frame[f"{method}_2025_capacity_p95_source_gaussian"] = center + radius
        coal_share = query.capacity_coal_share.to_numpy(float)
        thermal = base_frame.input_thermal_share.to_numpy(float)
        anchor = (400. + 600. * coal_share) * thermal
        base_frame["capacity_2025_simple_coal_gas_share_anchor"] = anchor

        # Preserve all primary M and old-2020 M+C columns byte-value-wise.
        path = out / base_file.name.replace("_predictions.csv.gz", "_2025_capacity_candidate.csv.gz")
        base_frame.to_csv(path, index=False, compression="gzip")
        seals.append({"region": region, "hours": len(base_frame), "path": path.name,
                      "sha256": sha(path), "target_response_read": False})
    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    (out / "status.json").write_text(json.dumps({
        "status": "SEALED_PROSPECTIVE_2025_CAPACITY_CANDIDATE_PENDING_TARGET_LABELS",
        "forecast_origin": "2026-11-01", "capacity_statistic_year": 2025,
        "target_response_read": False, "base_forecast_hashes_preserved": True,
        "regions": [x["region"] for x in seals],
    }, indent=2) + "\n")
    input_paths = [SCRIPT, Path(official.__file__), CAPACITY_TABLE_PATH,
                   CAPACITY_AUDIT / "audit_summary.json",
                   CAPACITY_AUDIT / "annual_capacity_2025_by_ba.csv",
                   *sorted(BASE.glob("GB_to_*_2026-11_predictions.csv.gz"))]
    input_hashes = {str(p): sha(p) for p in input_paths}
    output_hashes = {p.name: sha(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "provenance.json").write_text(json.dumps({
        "input_sha256": input_hashes, "output_sha256_before_provenance": output_hashes,
        "source_train_regions": ["GB"], "source_train_year": 2022,
        "source_calibration_year": 2023,
        "capacity_scope": "EIA-860 2025 seven US BA; DUKES UK aggregate proxy for GB",
    }, indent=2) + "\n")
    print(f"Sealed {len(seals)} separate 2025 capacity candidates; no target response read.")


if __name__ == "__main__":
    main()
