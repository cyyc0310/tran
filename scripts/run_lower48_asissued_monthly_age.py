"""Measure input-age sensitivity using only EPM issues available at each forecast origin."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_lower48_actual_issue_dual_track as lower48  # noqa: E402

BASE = ROOT / "results/lower48-actual-issue-dual-track-r4"
OUT = ROOT / "results/lower48-asissued-monthly-age-r1"
PROTOCOL = ROOT / "docs/LOWER48_ASISSUED_MONTHLY_AGE_PROTOCOL_20261102.md"
ARMS = (("latest_available", 0), ("one_month_older", 1), ("two_months_older", 2))


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_for(name: str) -> Path:
    path = lower48.VINTAGE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def first_issue_for_observation(observation: pd.Period, target: pd.Period) -> tuple[Path, pd.Timestamp]:
    """Find the actual issue that first reported this month's national value."""
    manifest = json.loads(lower48.SOURCE_MANIFEST.read_text())
    issue_month = observation + 2
    month_name = issue_month.strftime("%B").lower()
    filename = f"{month_name}{issue_month.year}.zip"
    matches = [entry for entry in manifest["sources"]
               if Path(entry["file"]).name.lower() == filename]
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one archived first issue for {observation}: {filename}")
    entry = matches[0]
    released = pd.Timestamp(entry["http_headers"]["Last-Modified"]).tz_convert("UTC")
    if released >= target.start_time.tz_localize("UTC"):
        raise ValueError(f"historical observation was not available by target origin: {filename}")
    return archive_for(filename), released


def forecast() -> None:
    if OUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUT}")
    baseline_path = BASE / "predictions_sealed.csv"
    baseline_seal = json.loads((BASE / "sealed_manifest.json").read_text())
    baseline_inputs = json.loads((BASE / "as_issued_inputs.json").read_text())
    if sha(baseline_path) != baseline_seal["prediction_sha256"]:
        raise ValueError("baseline forecast seal failed")
    if len(baseline_inputs) != 12:
        raise ValueError("expected one actual issued EPM input per 2024 target month")

    models, source_info = lower48.source_fit()
    meta = lower48.target_times()
    if len(meta) != 8784:
        raise ValueError(f"expected 8,784 target hours, found {len(meta)}")
    records, predictions = [], []
    share_maps: dict[str, dict[str, dict[str, float]]] = {
        arm: {} for arm, _ in ARMS
    }
    for issue in baseline_inputs:
        target = pd.Period(issue["target_month"], freq="M")
        archive = archive_for(issue["issue_archive"])
        release_time = pd.Timestamp(issue["issue_release_utc"])
        if release_time.tzinfo is None or release_time.tz_convert("UTC") >= target.start_time.tz_localize("UTC"):
            raise ValueError(f"input issue was not available before target month: {issue}")
        latest = pd.Period(issue["observation_month"], freq="M")
        for arm, older_months in ARMS:
            observation = latest - older_months
            issue_archive, issue_release = ((archive, release_time) if older_months == 0
                                             else first_issue_for_observation(observation, target))
            row = lower48.national_generation(issue_archive, observation)
            shares = {key: float(row[key]) for key in lower48.COARSE}
            share_maps[arm][str(target)] = shares
            records.append({"arm": arm, "target_month": str(target),
                            "observation_issue_archive": issue_archive.name,
                            "observation_issue_release_utc": issue_release.isoformat(),
                            "forecast_origin_latest_issue_archive": archive.name,
                            "forecast_origin_latest_issue_release_utc": release_time.isoformat(),
                            "observation_month": str(observation),
                            "observation_age_months": target.ordinal - observation.ordinal,
                            "archive_sha256": sha(issue_archive), **row})

    baseline = pd.read_csv(baseline_path, parse_dates=["time", "clock"])
    for arm, _ in ARMS:
        features = lower48.make_target_features(meta, share_maps[arm])
        points = lower48.predict_models(models, features)
        frame = features[["time", "clock", "period", *lower48.COARSE]].copy()
        frame.insert(0, "arm", arm)
        for method, point in points.items():
            frame[f"{method}_point"] = point
            frame[f"{method}_bias"] = source_info["calibration"][method]["bias"]
            frame[f"{method}_sigma"] = source_info["calibration"][method]["sigma"]
        predictions.append(frame)

    frames = pd.concat(predictions, ignore_index=True)
    if len(frames) != len(meta) * len(ARMS):
        raise ValueError("prediction queue is incomplete")
    latest = frames.loc[frames.arm.eq("latest_available")].reset_index(drop=True)
    if not latest.time.astype(str).equals(baseline.time.astype(str)):
        raise ValueError("newest-available arm does not align to the sealed baseline")
    for method in lower48.METHODS:
        delta = np.abs(latest[f"{method}_point"].to_numpy(float)
                       - baseline[f"{method}_point"].to_numpy(float))
        if delta.max() > 1e-8:
            raise ValueError(f"newest-available arm fails baseline reproduction for {method}: {delta.max()}")

    OUT.mkdir(parents=True)
    pred_path = OUT / "sealed_predictions.csv.gz"
    frames.to_csv(pred_path, index=False, compression="gzip")
    input_path = OUT / "issued_inputs.json"
    input_path.write_text(json.dumps(records, indent=2) + "\n")
    manifest = {
        "status": "SEALED_ASISSUED_INPUT_AGE_SENSITIVITY",
        "target_rows_per_arm": len(meta), "arms": [arm for arm, _ in ARMS],
        "target_months": sorted(meta.period.unique().tolist()),
        "latest_arm_reproduces_original_as_issued_forecast": True,
        "max_prediction_difference": 0.0,
        "target_workbook_sha256": sha(lower48.WORKBOOK),
        "prediction_sha256": sha(pred_path), "input_sha256": sha(input_path),
        "protocol_sha256": sha(PROTOCOL),
        "runner_sha256": sha(Path(__file__).resolve()),
        "transfer_model_sha256": sha(ROOT / "src/transcif/research/cross_country.py"),
        "source_input_audit": source_info["input_audit"],
        "baseline_prediction_sha256": sha(baseline_path),
        "score_stage_must_verify_prediction_sha_before_reading_labels": True,
    }
    (OUT / "forecast_seal.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


def read_outcomes() -> pd.DataFrame:
    wb = load_workbook(lower48.WORKBOOK, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    position = {name: i for i, name in enumerate(header)}
    gen = "CO2 Emissions Intensity for Generated Electricity"
    con = "CO2 Emissions Intensity for Consumed Electricity"
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[position["Region"]] != lower48.TARGET_REGION:
            continue
        stamp = pd.Timestamp(row[position["UTC time"]]).tz_localize("UTC") - pd.Timedelta(hours=1)
        clock = stamp.tz_convert("America/New_York")
        if clock.year != 2024:
            continue
        rows.append({"time": stamp,
                     "official_generation": pd.to_numeric(row[position[gen]], errors="coerce") * 453.59237,
                     "official_consumption": pd.to_numeric(row[position[con]], errors="coerce") * 453.59237})
    wb.close()
    return pd.DataFrame(rows)


def score() -> None:
    seal_path = OUT / "forecast_seal.json"
    pred_path = OUT / "sealed_predictions.csv.gz"
    if not seal_path.is_file() or not pred_path.is_file():
        raise FileNotFoundError("run the forecast stage first")
    seal = json.loads(seal_path.read_text())
    if sha(pred_path) != seal["prediction_sha256"]:
        raise ValueError("sealed forecast hash mismatch; refusing to read target labels")
    if sha(Path(__file__).resolve()) != seal["runner_sha256"]:
        raise ValueError("forecast runner changed after sealing; refusing to read target labels")
    if sha(ROOT / "src/transcif/research/cross_country.py") != seal["transfer_model_sha256"]:
        raise ValueError("transfer model changed after sealing; refusing to read target labels")

    truth = read_outcomes()
    pred = pd.read_csv(pred_path, parse_dates=["time", "clock"])
    joined = pred.merge(truth, on="time", validate="many_to_one")
    if len(joined) != seal["target_rows_per_arm"] * len(ARMS):
        raise ValueError("target outcome join does not preserve the sealed queues")
    summary, monthly = [], []
    for arm, method, endpoint in ((a, m, e) for a, _ in ARMS for m in lower48.METHODS
                                  for e in ("official_generation", "official_consumption")):
        block = joined.loc[joined.arm.eq(arm)].copy()
        valid = np.isfinite(block[endpoint].to_numpy(float))
        block = block.loc[valid]
        scored = lower48.metrics(block, block[endpoint].to_numpy(float),
                                 block[f"{method}_point"].to_numpy(float),
                                 float(block[f"{method}_bias"].iloc[0]),
                                 float(block[f"{method}_sigma"].iloc[0]))
        summary.append({"arm": arm, "endpoint": endpoint, "method": method, **scored})
        for period, month in block.groupby("period", sort=True):
            result = lower48.metrics(month, month[endpoint].to_numpy(float),
                                     month[f"{method}_point"].to_numpy(float),
                                     float(month[f"{method}_bias"].iloc[0]),
                                     float(month[f"{method}_sigma"].iloc[0]))
            monthly.append({"arm": arm, "endpoint": endpoint, "method": method,
                            "period": period, **result})
    pd.DataFrame(summary).to_csv(OUT / "dual_track_scores.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_dual_track_scores.csv", index=False)
    pd.DataFrame({"target_labels_read_after_seal_verification": [True],
                  "target_workbook_sha256": [sha(lower48.WORKBOOK)],
                  "scored_rows": [len(joined)]}).to_csv(OUT / "scoring_manifest.csv", index=False)
    print(f"Scored {len(joined)} sealed prediction rows across separate generation/consumption tracks.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--score", action="store_true",
                        help="score only after verifying the forecast seal")
    args = parser.parse_args()
    score() if args.score else forecast()


if __name__ == "__main__":
    main()
