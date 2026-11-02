"""Seal a GB-source -> US BA November 2026 forecast on an EIA-923 release."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import audit_live_eia923_ba_vintage as vintage
import run_actual_issue_cross_country as source_model
from transcif.research.cross_country import COARSE, FUELS, METHODS, ZONES, predict_models

OUT = ROOT / "results/prospective-gb-to-us-eia923-nov2026-r1"
EIA923 = ROOT / "docs/evidence/eia923-live-monthly-2026-08-r1/f923_2026.zip"
PROTOCOL = ROOT / "docs/PROSPECTIVE_GB_TO_US_EIA923_STRICT_NOV2026_PROTOCOL.md"
BA_REGIONS = ("ERCO", "FPL")
TARGET_MONTH = pd.Period("2026-11", freq="M")
OBSERVED_MONTH = pd.Period("2026-08", freq="M")
MONTHS = range(1, 9)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_features(ba: str, shares: dict[str, float]) -> pd.DataFrame:
    zone = ZONES[ba]
    local_start = TARGET_MONTH.start_time.tz_localize(zone)
    local_stop = (TARGET_MONTH + 1).start_time.tz_localize(zone)
    utc = pd.date_range(local_start.tz_convert("UTC"), local_stop.tz_convert("UTC"),
                        inclusive="left", freq="h")
    local = utc.tz_convert(zone).tz_localize(None)
    frame = pd.DataFrame({"time": utc, "clock": local, "year": local.year})
    for name, values, cycle in (("hour", local.hour, 24),
                                ("dow", local.dayofweek, 7),
                                ("month", local.month, 12)):
        frame[f"{name}_sin"] = np.sin(2 * np.pi * values / cycle)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * values / cycle)
    for category in COARSE:
        frame[category] = shares[category]
        for suffix in ("sin", "cos"):
            frame[f"{category}_hour_{suffix}"] = frame[category] * frame[f"hour_{suffix}"]
    frame["ba"] = ba
    return frame


def shares_from_record(row: pd.Series) -> dict[str, float]:
    fuel = {name: float(row[f"eia923_{name}_mwh"]) for name in FUELS}
    if any(not np.isfinite(value) or value < 0 for value in fuel.values()):
        raise ValueError(f"invalid reported fuel vector for {row.ba}: {fuel}")
    fuel["thermal"] = sum(fuel[key] for key in ("coal", "gas", "oil", "other"))
    total = sum(fuel[key] for key in FUELS)
    if total <= 0:
        raise ValueError(f"nonpositive EIA-923 report total for {row.ba}")
    shares = {group: fuel[group] / total for group in COARSE}
    if abs(sum(shares.values()) - 1.) > 1e-10:
        raise ValueError(f"coarse input does not close for {row.ba}: {shares}")
    return shares


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite sealed forecast directory {OUT}")
    if not EIA923.is_file() or not PROTOCOL.is_file():
        raise FileNotFoundError("frozen protocol or EIA-923 release archive is missing")
    for path in (ROOT / f"results/research/data/{region}_hourly.csv" for region in ZONES):
        if not path.is_file():
            raise FileNotFoundError(path)

    monthly, mapping_audit = vintage.parse_monthly_generation(EIA923, 2026, MONTHS, vintage.DEFAULT_BAS)
    month_rows = monthly.loc[monthly.month.eq(str(OBSERVED_MONTH))
                             & monthly.ba.isin(BA_REGIONS)].copy()
    if set(month_rows.ba) != set(BA_REGIONS) or not month_rows.eia923_complete.all():
        raise ValueError("frozen BA queue no longer has complete mapped EIA-923 fuel rows")
    if month_rows.eia923_total_mwh.le(0).any():
        raise ValueError("EIA-923 source-set monthly total must be positive")

    # Source-only fit and calibration. No EIA-930 target-year response is read.
    _, source_audit, models, calibration = source_model.fit_source()
    prediction_parts, input_records = [], []
    for _, row in month_rows.set_index("ba").loc[list(BA_REGIONS)].reset_index().iterrows():
        ba = str(row.ba)
        shares = shares_from_record(row)
        features = make_features(ba, shares)
        predictions = predict_models(models, features)
        frame = features[["ba", "time"]].copy().rename(columns={"time": "time_utc"})
        for method in METHODS:
            point = np.asarray(predictions[method], dtype=float)
            bias, sigma = calibration[method]
            radius = 1.6448536269514722 * sigma
            frame[method] = point
            frame[f"{method}_calibrated"] = point + bias
            frame[f"{method}_lower90_source"] = point + bias - radius
            frame[f"{method}_upper90_source"] = point + bias + radius
        prediction_parts.append(frame)
        input_records.append({"ba": ba, "target_month": str(TARGET_MONTH),
                              "observed_month": str(OBSERVED_MONTH),
                              "issue_release_date": "2026-10-23",
                              "issue_release_file": EIA923.relative_to(ROOT).as_posix(),
                              "mapped_plant_fuel_rows": int(row.eia923_plant_fuel_rows),
                              "numeric_plant_fuel_rows": int(row.eia923_numeric_rows),
                              "numeric_row_fraction": float(row.eia923_numeric_row_fraction),
                              "all_reported_rows_complete": bool(row.eia923_complete),
                              "reported_plant_set_netgen_mwh": float(row.eia923_total_mwh),
                              "five_group_shares": shares,
                              "coverage_note": "reported EIA-923 plant set; not certified full-BA total"})

    predictions = pd.concat(prediction_parts, ignore_index=True)
    if len(predictions) != 1442 or predictions.groupby("ba").size().to_dict() != {
            ba: 721 for ba in BA_REGIONS} or predictions.duplicated(["ba", "time_utc"]).any():
        raise ValueError("forecast must contain exactly 721 unique UTC hourly rows per BA")
    if not np.isfinite(predictions[[*METHODS]].to_numpy(float)).all():
        raise ValueError("nonfinite prospective point prediction")

    files = [Path(__file__).resolve(), PROTOCOL, EIA923,
             ROOT / "scripts/score_prospective_gb_to_us_eia923_strict_nov2026.py",
             Path(vintage.__file__).resolve(), Path(source_model.__file__).resolve(),
             ROOT / "scripts/audit_live_eia923_ba_vintage.py",
             ROOT / "src/transcif/data/eia930.py",
             ROOT / "src/transcif/research/cross_country.py",
             ROOT / "src/transcif/research/eia_proxy.py"]
    files += [ROOT / f"results/research/data/{region}_hourly.csv" for region in ZONES]
    if not all(path.is_file() for path in files):
        raise FileNotFoundError("one or more source provenance assets are missing")
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in dict.fromkeys(files)}

    OUT.mkdir(parents=True)
    prediction_path = OUT / "GB_to_US_EIA923_Nov2026_predictions.csv"
    predictions.to_csv(prediction_path, index=False)
    seal = {"path": prediction_path.name, "sha256": sha(prediction_path),
            "target_month": str(TARGET_MONTH), "regions": list(BA_REGIONS),
            "hours_per_region": 721, "methods": list(METHODS),
            "target_response_read_before_seal": False,
            "target_response_source": "EIA-930 2026 November native hourly balance files",
            "input_month_contract": "EIA-923 2026 August report released 2026-10-23; target t-3"}
    (OUT / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    (OUT / "target_input_audit.json").write_text(json.dumps({
        "forecast_origin_utc": "2026-11-01T00:00:00Z", "target_month": str(TARGET_MONTH),
        "target_response_columns_read": False, "input_regions": input_records,
        "eia923_mapping_audit": mapping_audit,
        "target_rows": len(predictions), "rows_per_region": predictions.groupby("ba").size().to_dict(),
    }, indent=2) + "\n")
    (OUT / "source_audit.json").write_text(json.dumps(source_audit, indent=2) + "\n")
    calibration_payload = {method: {"bias": float(values[0]), "sigma": float(values[1])}
                           for method, values in calibration.items()}
    (OUT / "source_calibration.json").write_text(json.dumps(calibration_payload, indent=2) + "\n")
    (OUT / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    if any(sha(ROOT / path) != value for path, value in hashes.items()):
        raise RuntimeError("source inputs or code changed while forecast was constructed")
    status = {"status": "SEALED_PROSPECTIVE_GB_TO_US_EIA923_NOV2026",
              "source_country": "GB", "source_train_year": 2022,
              "source_calibration_year": 2023, "target_bas": list(BA_REGIONS),
              "target_month": str(TARGET_MONTH), "forecast_origin_utc": "2026-11-01T00:00:00Z",
              "target_hours_per_ba": 721,
              "target_response_loaded": False, "target_scored": False,
              "prospective_time_holdout_only": True, "unseen_geography_confirmation": False,
              "input_release_vintage": "EIA-923 2026 August, released 2026-10-23",
              "label_vintage_for_later_scoring": "EIA-930 current revised November 2026 series",
              "coverage_guarantee": False}
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n")
    print(json.dumps({"status": status["status"], "seal": seal,
                      "inputs": input_records}, indent=2))


if __name__ == "__main__":
    main()
