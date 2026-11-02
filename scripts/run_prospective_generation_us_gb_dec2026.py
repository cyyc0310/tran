"""Seal paired US↔GB December 2026 generation-proxy forecasts."""
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

import run_prospective_generation_us_gb_nov2026_remainder as frozen  # noqa: E402

OUT = ROOT / "results/prospective-generation-us-gb-dec2026-r1"
PROTOCOL = ROOT / "docs/PROSPECTIVE_GENERATION_US_GB_DEC2026_PROTOCOL.md"
FORECAST_ORIGIN = pd.Timestamp("2026-11-01T15:33:33Z")
PREDICTION_START = pd.Timestamp("2026-12-01T00:00:00Z")
PREDICTION_STOP = pd.Timestamp("2027-01-01T00:00:00Z")
OBSERVED_NESO_MONTH = pd.Period("2026-09", freq="M")
OBSERVED_EIA_MONTH = "2026-08"
TARGET_MONTH = "2026-12"
EIA_BAS = ("ERCO", "FPL")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    if pd.Timestamp.now(tz="UTC") >= PREDICTION_START:
        raise RuntimeError("December target window has started; refuse retrospective sealing")
    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)

    # Reuse the already sealed November model/data path exactly; only the
    # target calendar changes. These target input values were available at
    # the declared November 1 forecast origin.
    frozen.PROTOCOL = PROTOCOL
    frozen.OBSERVED_NESO_MONTH = OBSERVED_NESO_MONTH
    frozen.OBSERVED_EIA_MONTH = OBSERVED_EIA_MONTH
    frozen.PREDICTION_START = PREDICTION_START
    frozen.PREDICTION_STOP = PREDICTION_STOP
    frozen.FORECAST_ORIGIN = FORECAST_ORIGIN
    frozen.EIA_TARGET_BAS = EIA_BAS

    raw, source_audit = frozen.load_inputs(ROOT)
    neso_shares, neso_audit = frozen._neso_september_shares()
    eia_shares, eia_audit = frozen._eia923_target_shares()
    source_names = {
        "US_to_GB": [region for region, frame in raw.items()
                     if frame.country.iloc[0] == "US"],
        "GB_to_US": ["GB"],
    }
    targets = {
        "US_to_GB": {"GB": frozen._calendar_query(
            "GB", neso_shares, "Europe/London")},
        "GB_to_US": {ba: frozen._calendar_query(
            ba, eia_shares[ba], frozen.ZONES[ba]) for ba in EIA_BAS},
    }

    if (len(targets["US_to_GB"]["GB"]) != 744
            or any(len(targets["GB_to_US"][ba]) != 744 for ba in EIA_BAS)):
        raise ValueError("December queues must contain 744 local/UTC hours per endpoint")
    if any("CARBON_INTENSITY" in frame for group in targets.values()
           for frame in group.values()):
        raise ValueError("target response must not enter the forecast feature tables")

    source_assets = [ROOT / source_audit[region]["path"] for region in raw]
    assets = [Path(__file__).resolve(), PROTOCOL,
              Path(frozen.__file__).resolve(),
              ROOT / "scripts/audit_live_eia923_ba_vintage.py",
              ROOT / "scripts/run_cross_country.py",
              ROOT / "src/transcif/research/cross_country.py",
              ROOT / "src/transcif/data/eia930.py",
              ROOT / "src/transcif/research/eia_proxy.py",
              frozen.NESO, frozen.EIA923_ZIP, *source_assets]
    assets = list(dict.fromkeys(assets))
    if any(not path.is_file() for path in assets):
        raise FileNotFoundError("source, issue-input, protocol, or model asset missing")
    hashes = {str(path.relative_to(ROOT)): sha(path) for path in assets}

    output_rows, seals, calibration_audit = [], [], {}
    for direction, endpoint_queries in targets.items():
        models, calibration, fit_audit = frozen._source_fit(
            source_names[direction], raw,
            {region: frozen.monthly_inputs(frame, lag=2)
             for region, frame in raw.items()})
        calibration_audit[direction] = {**fit_audit, "methods": calibration}
        for region, query in endpoint_queries.items():
            points = frozen.predict_models(models, query)
            frame = query.copy()
            for method in frozen.METHODS:
                point = np.asarray(points[method], dtype=float)
                bias = calibration[method]["bias"]
                sigma = calibration[method]["sigma"]
                radius = 1.6448536269514722 * sigma
                frame[f"{method}_point"] = point
                frame[f"{method}_calibrated"] = point + bias
                frame[f"{method}_lower90_source"] = point + bias - radius
                frame[f"{method}_upper90_source"] = point + bias + radius
            frame["direction"] = direction
            frame["endpoint"] = region
            name = f"{direction}_{region}_predictions.csv.gz"
            output_rows.append((name, frame))

    OUT.mkdir(parents=True)
    for name, frame in output_rows:
        path = OUT / name
        frame.to_csv(path, index=False, compression="gzip")
        seals.append({"path": name, "sha256": sha(path), "rows": len(frame),
                      "first_time_utc": frame.time.min().isoformat(),
                      "last_time_utc": frame.time.max().isoformat(),
                      "direction": frame.direction.iloc[0],
                      "endpoint": frame.endpoint.iloc[0],
                      "target_response_read_before_seal": False})
    (OUT / "prediction_seal.json").write_text(
        json.dumps(seals, indent=2) + "\n", encoding="utf-8")
    (OUT / "source_calibration.json").write_text(
        json.dumps(calibration_audit, indent=2) + "\n", encoding="utf-8")
    (OUT / "startup_hashes.json").write_text(
        json.dumps(hashes, indent=2) + "\n", encoding="utf-8")
    (OUT / "input_audit.json").write_text(json.dumps({
        "forecast_origin_utc": FORECAST_ORIGIN.isoformat(),
        "target_month": TARGET_MONTH,
        "prediction_start_utc": PREDICTION_START.isoformat(),
        "prediction_stop_utc_exclusive": PREDICTION_STOP.isoformat(),
        "target_hours_per_endpoint": 744,
        "source_countries": source_names,
        "neso_target_input": neso_audit,
        "eia923_target_input": eia_audit,
        "source_dataset_audit": source_audit,
        "target_response_loaded": False,
    }, indent=2) + "\n", encoding="utf-8")
    if any(sha(ROOT / name) != digest for name, digest in hashes.items()):
        raise RuntimeError("input, source code, or protocol changed during sealing")
    if any(sha(OUT / seal["path"]) != seal["sha256"] for seal in seals):
        raise RuntimeError("prediction seal hash mismatch")

    status = {
        "status": "SEALED_PROSPECTIVE_GENERATION_US_GB_DEC2026",
        "tracks": ["generation_proxy"], "target_month": TARGET_MONTH,
        "forecast_origin_utc": FORECAST_ORIGIN.isoformat(),
        "target_start_utc": PREDICTION_START.isoformat(),
        "target_stop_utc_exclusive": PREDICTION_STOP.isoformat(),
        "target_hours_per_endpoint": 744,
        "directions": {key: {"source_regions": value,
                              "target_regions": list(targets[key])}
                       for key, value in source_names.items()},
        "source_train_year": 2022, "source_calibration_year": 2023,
        "target_response_loaded": False, "target_scored": False,
        "label_definition": "fixed-factor local-generation proxy; not official or measured CIF",
        "target_inputs": {"GB": {"source_month": "2026-09", "lag": "t-3",
                                  "vintage": "current revised NESO snapshot"},
                          "US": {"source_month": "2026-08", "lag": "t-4",
                                  "vintage": "EIA-923 issue released 2026-10-23"}},
        "prospective_time_holdout_only": True,
        "unseen_geography_confirmation": False,
        "coverage_guarantee": False,
        "score_eligible_after_utc": "2027-02-03T00:00:00Z",
    }
    (OUT / "status.json").write_text(json.dumps(status, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({"status": status["status"], "seals": seals}, indent=2))


if __name__ == "__main__":
    main()
