"""Seal a prospective US->GB November 2026 official-intensity forecast.

The query uses only GB fuel-generation data through September 2026 (t-2) and
calendar features. The contemporaneous NESO carbon-intensity field is excluded
from every read; target scoring is a separate post-November activity.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import run_official_intensity_transfer as official
from transcif.research import capacity_transfer
from transcif.research.cross_country import COARSE, FUELS, fit_models, predict_models

OUT = ROOT / "results/prospective-us-to-gb-neso-nov2026-r1"
GB_PANEL = ROOT / "data/neso-gb/df_fuel_ckan.csv"
CAPACITY_TABLE = ROOT / "data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json"
TARGET_START_UTC = pd.Timestamp("2026-11-01T00:00:00Z")
TARGET_STOP_UTC = pd.Timestamp("2026-12-01T00:00:00Z")
OBSERVED_MONTH = pd.Period("2026-09", freq="M")
METHODS = (*official.METHODS, "anchor700")
GEN_COLUMNS = ("DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
               "HYDRO", "BIOMASS", "OTHER", "SOLAR")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def target_features(capacity: dict) -> tuple[pd.DataFrame, dict]:
    # Deliberately do not read the CARBON_INTENSITY target-response column.
    raw = pd.read_csv(GB_PANEL, usecols=GEN_COLUMNS)
    stamp = pd.to_datetime(raw.DATETIME, errors="raise")
    if stamp.dt.tz is not None:
        stamp = stamp.dt.tz_convert("UTC").dt.tz_localize(None)
    if stamp.duplicated().any():
        raise ValueError("duplicate NESO generation timestamps")
    raw["stamp"] = stamp
    raw["coal"], raw["gas"], raw["oil"] = raw.COAL, raw.GAS, 0.
    raw["other"] = raw.BIOMASS + raw.OTHER
    raw["nuclear"], raw["hydro"] = raw.NUCLEAR, raw.HYDRO
    raw["wind"], raw["solar"] = raw.WIND + raw.WIND_EMB, raw.SOLAR
    fuel = [*FUELS]
    vals = raw[fuel].to_numpy(float)
    keep = np.isfinite(vals).all(axis=1) & (vals >= 0).all(axis=1)
    raw = raw.loc[keep].copy()
    raw["utc_hour"] = pd.to_datetime(raw.stamp).dt.floor("h")
    hourly = raw.groupby("utc_hour", sort=True)[fuel].mean()
    counts = raw.groupby("utc_hour", sort=True).size()
    hourly = hourly.loc[counts.eq(2)].copy()
    hourly = hourly.loc[hourly[fuel].sum(axis=1).gt(0)].copy()
    hourly.index = pd.DatetimeIndex(hourly.index).tz_localize("UTC")
    local = hourly.index.tz_convert("Europe/London").tz_localize(None)
    period = local.to_period("M")
    hourly["period"] = period.astype(str)
    hourly["clock"] = local
    hourly["year"] = local.year
    for name, values, cycle in (("hour", local.hour, 24),
                                ("dow", local.dayofweek, 7),
                                ("month", local.month, 12)):
        hourly[f"{name}_sin"] = np.sin(2 * np.pi * values / cycle)
        hourly[f"{name}_cos"] = np.cos(2 * np.pi * values / cycle)

    expected = ((OBSERVED_MONTH + 1).start_time.tz_localize("Europe/London")
                - OBSERVED_MONTH.start_time.tz_localize("Europe/London")).total_seconds() / 3600
    monthly = hourly.loc[hourly.period.eq(str(OBSERVED_MONTH)), fuel].sum()
    observed_hours = int(hourly.period.eq(str(OBSERVED_MONTH)).sum())
    coverage = observed_hours / expected
    if coverage < .99 or monthly.sum() <= 0:
        raise ValueError(f"September 2026 NESO monthly input fails coverage gate: {coverage}")
    shares = {
        "thermal": float(monthly[["coal", "gas", "oil", "other"]].sum() / monthly.sum()),
        "hydro": float(monthly.hydro / monthly.sum()),
        "nuclear": float(monthly.nuclear / monthly.sum()),
        "wind": float(monthly.wind / monthly.sum()),
        "solar": float(monthly.solar / monthly.sum()),
    }
    if abs(sum(shares.values()) - 1.) > 1e-10:
        raise ValueError(f"five-group monthly shares do not close: {shares}")

    local_start = pd.Timestamp("2026-11-01 00:00:00", tz="Europe/London")
    local_stop = pd.Timestamp("2026-12-01 00:00:00", tz="Europe/London")
    times = pd.date_range(local_start.tz_convert("UTC"), local_stop.tz_convert("UTC"),
                          inclusive="left", freq="h")
    clock = times.tz_convert("Europe/London").tz_localize(None)
    query = pd.DataFrame({"time": times, "clock": clock, "year": clock.year})
    for name, values, cycle in (("hour", clock.hour, 24),
                                ("dow", clock.dayofweek, 7),
                                ("month", clock.month, 12)):
        query[f"{name}_sin"] = np.sin(2 * np.pi * values / cycle)
        query[f"{name}_cos"] = np.cos(2 * np.pi * values / cycle)
    for category in COARSE:
        query[category] = shares[category]
        for suffix in ("sin", "cos"):
            query[f"{category}_hour_{suffix}"] = query[category] * query[f"hour_{suffix}"]
    query["observed_month"] = str(OBSERVED_MONTH)
    query["region"] = "GB"
    query = capacity_transfer.add_capacity(query, capacity)
    query["log_capacity_coal_gas_mw"] = np.log1p(query.capacity_coal_mw + query.capacity_gas_mw)
    audit = {
        "target_month": "2026-11", "forecast_origin_utc": "2026-11-01T00:00:00Z",
        "target_hourly_rows": len(query), "observed_month": str(OBSERVED_MONTH),
        "observed_month_hours": observed_hours, "expected_observed_month_hours": int(expected),
        "observed_month_coverage": coverage, "observed_month_generation_total_mwh_proxy": float(monthly.sum()),
        "observed_month_coarse_shares": shares,
        "target_generation_source": str(GB_PANEL.relative_to(ROOT)),
        "columns_read": list(GEN_COLUMNS), "target_response_column_read": False,
        "target_response_column_excluded": "CARBON_INTENSITY",
        "gb_timezone": "Europe/London", "hourly_generation_aggregation": "mean of both NESO half-hour MW rows",
        "target_label_source_for_later_scoring": "NESO CARBON_INTENSITY, current revised series",
    }
    return query, audit


def fit_and_calibrate(capacity: dict) -> tuple[dict, dict, dict, dict]:
    raw, prepared, official_labels, audit = official.prepare(ROOT, capacity)
    sources = [region for region in official.US_REGIONS]
    train_parts, responses = [], []
    for region in sources:
        rows = prepared[region].loc[prepared[region].year.eq(2022)].copy()
        y = official.aligned_labels(rows, official_labels[region])
        good = np.isfinite(y)
        rows = rows.loc[good].copy()
        rows["official_y"] = y[good]
        train_parts.append(rows)
        responses.append(y[good])
    train = pd.concat(train_parts, ignore_index=True)
    y_train = np.concatenate(responses)
    official_models = official.train_official(train, y_train)

    proxy_parts = []
    for region, rows in zip(sources, train_parts):
        proxy_parts.append(raw[region].iloc[rows.raw_row.to_numpy()][list(FUELS)].to_numpy(float))
    proxy_models = fit_models(train.reset_index(drop=True), np.concatenate(proxy_parts))

    residual_parts = {method: [] for method in METHODS}
    for region in sources:
        calibration = prepared[region].loc[prepared[region].year.eq(2023)].copy()
        y = official.aligned_labels(calibration, official_labels[region])
        good = np.isfinite(y)
        calibration = calibration.loc[good].copy()
        y = y[good]
        prediction = official.point_predictions(official_models, proxy_models, calibration)
        prediction["anchor700"] = np.clip(700. * calibration.thermal.to_numpy(float), 0., 1000.)
        for method in METHODS:
            residual_parts[method].append(y - prediction[method])
    calibration = official.calibrate(residual_parts)
    if "anchor700" not in calibration:
        values = residual_parts["anchor700"]
        bias = float(np.mean([v.mean() for v in values]))
        sigma = max(float(np.sqrt(np.mean([np.mean((v - bias) ** 2) for v in values]))), 1.)
        calibration["anchor700"] = {"bias": bias, "sigma": sigma,
                                     "regions": len(values),
                                     "calibration_hours": int(sum(len(v) for v in values))}
    source_audit = {"source_train_regions": sources,
                    "source_train_hours": len(train),
                    "source_calibration_regions": len(official.US_REGIONS),
                    "source_calibration_hours": int(sum(len(x) for x in residual_parts["source_mean"]))}
    return official_models, proxy_models, calibration, {"input_audit": audit, **source_audit}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    OUT.mkdir(parents=True)
    capacity = capacity_transfer.load_capacity(CAPACITY_TABLE)
    for name, expected in capacity["input_hashes"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"capacity source hash mismatch: {name}")
    files = [Path(__file__), ROOT / "scripts/run_official_intensity_transfer.py",
             ROOT / "scripts/prepare_official_intensity_panel.py", ROOT / "scripts/run_cross_country.py",
             ROOT / "src/transcif/research/cross_country.py", ROOT / "src/transcif/research/capacity_transfer.py",
             ROOT / "docs/OFFICIAL_CONSUMPTION_INTENSITY_TRANSFER_PROTOCOL_20261101.md",
             ROOT / "docs/PROSPECTIVE_US_TO_GB_NOV2026_PROTOCOL.md",
             ROOT / "docs/OFFICIAL_ANNUAL_COAL_GAS_CAPACITY_PROTOCOL_20261101.md",
             GB_PANEL, CAPACITY_TABLE, ROOT / "data/eia930-hourly-co2-r1/acquisition_manifest.json",
             ROOT / "data/eia930-hourly-co2-r1/coverage_audit_2022_2024.json"]
    files += [ROOT / name for name in capacity["input_hashes"]]
    source_assets = [official.SOURCE_LABEL_ROOT / f"{r}_consumption_2022_2023.csv.gz"
                     for r in official.US_REGIONS]
    source_assets += [official.SOURCE_LABEL_ROOT / "GB_consumption_2022_2023_halfhour.csv.gz",
                      ROOT / "data/eia930-hourly-co2-r1/feature_inputs/GB_generation_2022_2024.csv.gz"]
    files += source_assets
    files += [ROOT / f"results/research/data/{r}_hourly.csv" for r in official.US_REGIONS]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    startup_hashes = {str(p.relative_to(ROOT)): sha(p) for p in dict.fromkeys(files)}

    official_models, proxy_models, calibration, source_audit = fit_and_calibrate(capacity)
    query, input_audit = target_features(capacity)
    prediction = official.point_predictions(official_models, proxy_models, query)
    prediction["anchor700"] = np.clip(700. * query.thermal.to_numpy(float), 0., 1000.)
    frame = pd.DataFrame({"time_utc": query.time.astype(str),
                          "local_time": query.clock.astype(str),
                          "observed_month": query.observed_month,
                          **{category: query[category] for category in COARSE}})
    for method in METHODS:
        raw_point = np.asarray(prediction[method], dtype=float)
        bias, sigma = calibration[method]["bias"], calibration[method]["sigma"]
        corrected = raw_point + bias
        radius = 1.6448536269514722 * sigma
        frame[method] = raw_point
        frame[f"{method}_calibrated"] = corrected
        frame[f"{method}_lower90_source"] = corrected - radius
        frame[f"{method}_upper90_source"] = corrected + radius
    out_path = OUT / "US_to_GB_Nov2026_predictions.csv"
    frame.to_csv(out_path, index=False)
    seal = {"path": out_path.name, "sha256": sha(out_path),
            "hours": len(frame), "methods": list(METHODS),
            "target_response_field_read_before_seal": False,
            "target_response_field_excluded_from_input_read": "CARBON_INTENSITY"}
    (OUT / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    (OUT / "source_calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
    (OUT / "input_audit.json").write_text(json.dumps(input_audit, indent=2) + "\n")
    (OUT / "source_audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")
    (OUT / "startup_hashes.json").write_text(json.dumps(startup_hashes, indent=2) + "\n")
    if any(sha(ROOT / p) != value for p, value in startup_hashes.items()):
        raise RuntimeError("source inputs/code changed during forecast construction")
    status = {
        "status": "SEALED_PROSPECTIVE_US_TO_GB_NOV2026_FORECAST",
        "source_country": "US", "source_train_year": 2022, "source_calibration_year": 2023,
        "target_country": "GB", "target_month": "2026-11", "forecast_origin_utc": "2026-11-01T00:00:00Z",
        "target_hours_predicted": len(frame), "observed_target_month": "2026-09",
        "target_carbon_intensity_loaded": False, "target_label_scored": False,
        "target_label_source": "NESO current revised CARBON_INTENSITY series",
        "target_generation_input_vintage": "current NESO snapshot; no historical as-issued revision chain",
        "prospective_time_holdout_only": True, "unseen_geography_confirmation": False,
        "source_risk_coverage_guarantee": False,
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(f"sealed {len(frame)} hours, {len(METHODS)} methods -> {OUT}")


if __name__ == "__main__":
    main()
