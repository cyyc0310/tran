#!/usr/bin/env python3
"""Seal a paired US↔GB generation-proxy forecast for the remaining Nov 2026 hours."""
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

import audit_live_eia923_ba_vintage as eia923  # noqa: E402
from transcif.research.cross_country import (  # noqa: E402
    COARSE, FUELS, METHODS, ZONES, fit_models, labels, load_inputs,
    monthly_inputs, predict_models,
)

OUT = ROOT / "results/prospective-generation-us-gb-nov2026-remainder-r1"
PROTOCOL = ROOT / "docs/PROSPECTIVE_GENERATION_US_GB_NOV2026_REMAINDER_PROTOCOL.md"
NESO = ROOT / "data/neso-gb/df_fuel_ckan.csv"
EIA923_ZIP = ROOT / "docs/evidence/eia923-live-monthly-2026-08-r1/f923_2026.zip"
EIA_TARGET_BAS = ("ERCO", "FPL")
OBSERVED_NESO_MONTH = pd.Period("2026-09", freq="M")
OBSERVED_EIA_MONTH = "2026-08"
PREDICTION_START = pd.Timestamp("2026-11-01T15:00:00Z")
PREDICTION_STOP = pd.Timestamp("2026-12-01T00:00:00Z")
FORECAST_ORIGIN = pd.Timestamp("2026-11-01T14:36:00Z")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _calendar_query(region: str, shares: dict[str, float], zone: str) -> pd.DataFrame:
    times = pd.date_range(PREDICTION_START, PREDICTION_STOP, inclusive="left", freq="h")
    clock = times.tz_convert(zone).tz_localize(None)
    frame = pd.DataFrame({"time": times, "clock": clock, "region": region,
                          "year": clock.year, "observed_month": OBSERVED_EIA_MONTH if region in EIA_TARGET_BAS else str(OBSERVED_NESO_MONTH)})
    for name, values, cycle in (("hour", clock.hour, 24),
                                ("dow", clock.dayofweek, 7),
                                ("month", clock.month, 12)):
        frame[f"{name}_sin"] = np.sin(2 * np.pi * values / cycle)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * values / cycle)
    for category in COARSE:
        frame[category] = float(shares[category])
        frame[f"{category}_hour_sin"] = frame[category] * frame.hour_sin
        frame[f"{category}_hour_cos"] = frame[category] * frame.hour_cos
    return frame


def _source_fit(source_names: list[str], raw: dict[str, pd.DataFrame], prepared: dict[str, pd.DataFrame]):
    train_parts, auxiliary = [], []
    for region in source_names:
        usable = prepared[region].dropna(subset=list(COARSE))
        rows = usable.loc[usable.year.eq(2022)].copy()
        rows["raw_row"] = rows.index
        train_parts.append(rows)
        auxiliary.append(raw[region].loc[rows.raw_row, list(FUELS)].to_numpy(float))
    train = pd.concat(train_parts, ignore_index=True)
    fuel = np.concatenate(auxiliary)
    models = fit_models(train.reset_index(drop=True), fuel)

    residuals = {method: [] for method in METHODS}
    calibration_hours = {}
    for region in source_names:
        usable = prepared[region].dropna(subset=list(COARSE))
        rows = usable.loc[usable.year.eq(2023)].copy()
        rows["raw_row"] = rows.index
        prediction = predict_models(models, rows)
        truth = labels(raw[region].loc[rows.raw_row, list(FUELS)])
        calibration_hours[region] = len(rows)
        for method in METHODS:
            residuals[method].append(truth - prediction[method])
    calibration = {}
    for method, parts in residuals.items():
        bias = float(np.mean([part.mean() for part in parts]))
        sigma = max(float(np.sqrt(np.mean([np.mean((part - bias) ** 2) for part in parts]))), 1.0)
        calibration[method] = {"bias": bias, "sigma": sigma,
                               "source_regions": len(parts),
                               "source_calibration_hours": int(sum(map(len, parts)))}
    return models, calibration, {"source_train_regions": source_names,
                                 "source_train_year": 2022,
                                 "source_train_hours": len(train),
                                 "source_calibration_year": 2023,
                                 "source_calibration_hours_by_region": calibration_hours}


def _neso_september_shares() -> tuple[dict[str, float], dict]:
    columns = ["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
               "HYDRO", "BIOMASS", "OTHER", "SOLAR"]
    frame = pd.read_csv(NESO, usecols=columns)
    stamp = pd.to_datetime(frame.DATETIME, errors="raise")
    if stamp.dt.tz is not None:
        stamp = stamp.dt.tz_convert("UTC").dt.tz_localize(None)
    frame["stamp_utc"] = stamp.dt.tz_localize("UTC")
    fuels = pd.DataFrame(index=frame.index)
    fuels["coal"], fuels["gas"], fuels["oil"] = frame.COAL, frame.GAS, 0.0
    fuels["other"] = frame.BIOMASS + frame.OTHER
    fuels["nuclear"], fuels["hydro"] = frame.NUCLEAR, frame.HYDRO
    fuels["wind"], fuels["solar"] = frame.WIND + frame.WIND_EMB, frame.SOLAR
    values = fuels.to_numpy(float)
    keep = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    frame = frame.loc[keep].copy()
    fuels = fuels.loc[keep].copy()
    hourly = fuels.groupby([frame.stamp_utc.dt.floor("h")]).mean()
    counts = frame.groupby(frame.stamp_utc.dt.floor("h")).size()
    hourly = hourly.loc[counts.eq(2)].copy()
    local_month = hourly.index.tz_convert("Europe/London").tz_localize(None).to_period("M")
    monthly = hourly.loc[local_month == OBSERVED_NESO_MONTH].sum()
    expected_hours = int(((OBSERVED_NESO_MONTH + 1).start_time.tz_localize("Europe/London")
                          - OBSERVED_NESO_MONTH.start_time.tz_localize("Europe/London")).total_seconds() / 3600)
    observed_hours = int((local_month == OBSERVED_NESO_MONTH).sum())
    total = float(monthly.sum())
    if observed_hours != expected_hours or total <= 0:
        raise ValueError(f"NESO September input incomplete: {observed_hours}/{expected_hours} hours")
    shares = {
        "thermal": float(monthly[["coal", "gas", "oil", "other"]].sum() / total),
        "hydro": float(monthly.hydro / total),
        "nuclear": float(monthly.nuclear / total),
        "wind": float(monthly.wind / total),
        "solar": float(monthly.solar / total),
    }
    if abs(sum(shares.values()) - 1.0) > 1e-10:
        raise ValueError("NESO five-category shares fail closure")
    return shares, {"source": str(NESO.relative_to(ROOT)), "columns_read": columns,
                    "target_response_column_read": False,
                    "observed_month": str(OBSERVED_NESO_MONTH),
                    "observed_hours": observed_hours, "expected_hours": expected_hours,
                    "monthly_report_vintage": "current revised NESO snapshot; no full as-issued archive",
                    "shares": shares}


def _eia923_target_shares() -> tuple[dict[str, dict[str, float]], dict]:
    rows, audit = eia923.parse_monthly_generation(EIA923_ZIP, 2026, range(1, 9), eia923.DEFAULT_BAS)
    selected = rows.loc[rows.month.eq(OBSERVED_EIA_MONTH) & rows.ba.isin(EIA_TARGET_BAS)].copy()
    if set(selected.ba) != set(EIA_TARGET_BAS) or not selected.eia923_complete.all():
        raise ValueError("selected EIA-923 BA report rows are incomplete")
    out = {}
    for row in selected.itertuples(index=False):
        fuel = {name: float(getattr(row, f"eia923_{name}_mwh")) for name in FUELS}
        if any(not np.isfinite(v) or v < 0 for v in fuel.values()):
            raise ValueError(f"invalid EIA-923 mapped fuel vector for {row.ba}")
        total = sum(fuel.values())
        shares = {
            "thermal": sum(fuel[name] for name in ("coal", "gas", "oil", "other")) / total,
            "hydro": fuel["hydro"] / total, "nuclear": fuel["nuclear"] / total,
            "wind": fuel["wind"] / total, "solar": fuel["solar"] / total,
        }
        if total <= 0 or abs(sum(shares.values()) - 1.0) > 1e-10:
            raise ValueError(f"invalid EIA-923 composition for {row.ba}")
        out[row.ba] = shares
    return out, {"source": str(EIA923_ZIP.relative_to(ROOT)),
                 "observed_month": OBSERVED_EIA_MONTH,
                 "release_date": "2026-10-23", "target_ba": list(EIA_TARGET_BAS),
                 "page1_rows": int(audit["page1_rows"]),
                 "mapped_rows": {r.ba: int(r.eia923_plant_fuel_rows) for r in selected.itertuples(index=False)},
                 "numeric_rows": {r.ba: int(r.eia923_numeric_rows) for r in selected.itertuples(index=False)},
                 "input_scope": "all numeric rows in current monthly plant-frame mapping; not full BA total",
                 "shares": out}


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if not PROTOCOL.is_file() or not NESO.is_file() or not EIA923_ZIP.is_file():
        raise FileNotFoundError("frozen protocol, NESO snapshot, or EIA-923 issue archive is missing")
    if pd.Timestamp.now(tz="UTC") >= PREDICTION_START:
        raise RuntimeError("forecast start has passed; refuse to create a retrospective forecast")

    raw, input_audit = load_inputs(ROOT)
    prepared = {region: monthly_inputs(frame, lag=2) for region, frame in raw.items()}
    neso_shares, neso_audit = _neso_september_shares()
    eia_shares, eia_audit = _eia923_target_shares()

    source_names = {
        "US_to_GB": [region for region in raw if raw[region].country.iloc[0] == "US"],
        "GB_to_US": [region for region in raw if region == "GB"],
    }
    target_queries = {"US_to_GB": {"GB": _calendar_query("GB", neso_shares, "Europe/London")},
                      "GB_to_US": {ba: _calendar_query(ba, eia_shares[ba], ZONES[ba])
                                   for ba in EIA_TARGET_BAS}}

    source_assets = [ROOT / input_audit[region]["path"] for region in raw]
    code_assets = [Path(__file__).resolve(), PROTOCOL, Path(eia923.__file__).resolve(),
                   ROOT / "scripts/run_cross_country.py", ROOT / "src/transcif/research/cross_country.py",
                   ROOT / "src/transcif/data/eia930.py", ROOT / "src/transcif/research/eia_proxy.py"]
    files = [*source_assets, NESO, EIA923_ZIP, *code_assets]
    if any(not path.is_file() for path in files):
        raise FileNotFoundError("one or more source, target-input, or code assets are missing")
    startup_hashes = {str(path.relative_to(ROOT)): sha(path) for path in dict.fromkeys(files)}

    predictions, seals, calibration_audit = [], [], {}
    for direction, targets in target_queries.items():
        models, calibration, fit_audit = _source_fit(source_names[direction], raw, prepared)
        calibration_audit[direction] = {**fit_audit, "methods": calibration}
        for region, query in targets.items():
            points = predict_models(models, query)
            for method in METHODS:
                point = np.asarray(points[method], dtype=float)
                bias = calibration[method]["bias"]
                sigma = calibration[method]["sigma"]
                radius = 1.6448536269514722 * sigma
                query[f"{method}_point"] = point
                query[f"{method}_calibrated"] = point + bias
                query[f"{method}_lower90_source"] = point + bias - radius
                query[f"{method}_upper90_source"] = point + bias + radius
            path_name = f"{direction}_{region}_predictions.csv.gz"
            predictions.append((path_name, query))

    OUT.mkdir(parents=True)
    for name, frame in predictions:
        path = OUT / name
        frame.to_csv(path, index=False, compression="gzip")
        seals.append({"path": name, "sha256": sha(path), "rows": len(frame),
                      "first_hour_utc": str(frame.time.min()), "last_hour_utc": str(frame.time.max()),
                      "target_response_read_before_seal": False})
    if any(sha(ROOT / path) != digest for path, digest in startup_hashes.items()):
        raise RuntimeError("source/input/protocol changed during forecast construction")
    (OUT / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    (OUT / "startup_hashes.json").write_text(json.dumps(startup_hashes, indent=2) + "\n")
    (OUT / "source_calibration.json").write_text(json.dumps(calibration_audit, indent=2) + "\n")
    (OUT / "input_audit.json").write_text(json.dumps({
        "forecast_origin_utc": str(FORECAST_ORIGIN),
        "prediction_start_utc": str(PREDICTION_START),
        "prediction_stop_utc_exclusive": str(PREDICTION_STOP),
        "hours_per_endpoint": int((PREDICTION_STOP - PREDICTION_START) / pd.Timedelta(hours=1)),
        "neso_target_input": neso_audit,
        "eia923_target_input": eia_audit,
        "source_dataset_audit": input_audit,
    }, indent=2) + "\n")
    status = {
        "status": "SEALED_PROSPECTIVE_GENERATION_US_GB_NOV2026_REMAINDER",
        "tracks": ["generation_proxy"],
        "target_month": "2026-11", "target_start_utc": str(PREDICTION_START),
        "target_stop_utc_exclusive": str(PREDICTION_STOP),
        "hours_per_endpoint": int((PREDICTION_STOP - PREDICTION_START) / pd.Timedelta(hours=1)),
        "directions": {"US_to_GB": {"target": "GB", "source_regions": source_names["US_to_GB"]},
                       "GB_to_US": {"targets": list(EIA_TARGET_BAS), "source_regions": ["GB"]}},
        "source_train_year": 2022, "source_calibration_year": 2023,
        "target_response_loaded": False, "target_scored": False,
        "label_definition": "fixed-factor local-generation proxy; not official or measured CIF",
        "eia923_release": "2026-08 report, released 2026-10-23",
        "neso_input_vintage": "current revised NESO generation snapshot; full as-issued chain unavailable",
        "unseen_geography_confirmation": False, "coverage_guarantee": False,
        "score_eligible_after_utc": "2026-12-03T00:00:00Z",
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": status["status"], "seals": seals,
                      "methods": list(METHODS)}, indent=2))


if __name__ == "__main__":
    main()
