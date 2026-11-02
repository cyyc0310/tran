"""Compare as-issued EPM monthly inputs with the same observations one year later."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_lower48_actual_issue_dual_track as lower48  # noqa: E402

BASE_RUN = ROOT / "results/lower48-actual-issue-dual-track-r4"
LATER_ISSUES = ROOT / "data/eia-epm-vintages-2025-r1"
PROTOCOL = ROOT / "docs/LOWER48_MONTHLY_VINTAGE_SENSITIVITY_PROTOCOL_20261102.md"
MONTHS = {name: number for number, name in enumerate(
    ("january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"), start=1)}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def archive_path(issue_name: str) -> Path:
    match = re.fullmatch(r"([a-z]+)(\d{4})\.zip", issue_name.lower())
    if not match or match.group(1) not in MONTHS:
        raise ValueError(f"unexpected EPM archive name: {issue_name}")
    month, year = match.group(1), int(match.group(2)) + 1
    filename = f"{month}{year}.zip"
    candidates = [LATER_ISSUES / filename,
                  ROOT / "data/eia-epm-vintages-r1" / filename,
                  lower48.VINTAGE_DIR / filename]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"later EPM issue missing: {filename}")


def header_fields(path: Path) -> dict:
    header_path = path.with_suffix(".headers")
    fields = {}
    if header_path.is_file():
        for line in header_path.read_text(errors="replace").splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                if key.lower() in {"last-modified", "content-length", "content-type", "etag"}:
                    fields[key.lower()] = value.strip()
    return fields


def read_outcomes() -> pd.DataFrame:
    """Read target rows only after forecasts are sealed."""
    workbook = load_workbook(lower48.WORKBOOK, read_only=True, data_only=True)
    ws = workbook["Published Hourly Data"]
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    position = {name: i for i, name in enumerate(header)}
    columns = ("Region", "UTC time", "CO2 Emissions Intensity for Generated Electricity",
               "CO2 Emissions Intensity for Consumed Electricity")
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[position["Region"]] != lower48.TARGET_REGION:
            continue
        timestamp = pd.Timestamp(row[position["UTC time"]]).tz_localize("UTC") - pd.Timedelta(hours=1)
        local = timestamp.tz_convert("America/New_York")
        if local.year != 2024:
            continue
        rows.append({"time": timestamp,
                     "official_generation": pd.to_numeric(
                         row[position[columns[2]]], errors="coerce") * 453.59237,
                     "official_consumption": pd.to_numeric(
                         row[position[columns[3]]], errors="coerce") * 453.59237})
    workbook.close()
    return pd.DataFrame(rows)


def score(output: Path) -> None:
    seal_path = output / "prediction_seal.json"
    if not seal_path.is_file():
        raise FileNotFoundError("run forecast stage first")
    seal = json.loads(seal_path.read_text())
    for artifact, key in (("predictions_as_issued.csv", "as_issued_sha256"),
                          ("predictions_revised.csv", "revised_sha256")):
        if sha(output / artifact) != seal[key]:
            raise ValueError(f"sealed forecast changed: {artifact}")
    # Target values are first read after both prediction files pass SHA checks.
    truth = read_outcomes()
    score_rows, hourly_rows = [], []
    methods = lower48.METHODS
    for vintage, filename in (("as_issued", "predictions_as_issued.csv"),
                              ("one_year_later_revision", "predictions_revised.csv")):
        predictions = pd.read_csv(output / filename, parse_dates=["time", "clock"])
        joined = predictions.merge(truth, on="time", validate="one_to_one")
        if len(joined) != seal["prediction_rows"]:
            raise ValueError(f"{vintage}: outcome alignment lost rows")
        for endpoint in ("official_generation", "official_consumption"):
            valid = np.isfinite(joined[endpoint].to_numpy(float))
            data = joined.loc[valid].copy()
            for method in methods:
                result = lower48.metrics(
                    data, data[endpoint].to_numpy(float), data[f"{method}_point"].to_numpy(float),
                    float(data[f"{method}_bias"].iloc[0]), float(data[f"{method}_sigma"].iloc[0]))
                score_rows.append({"vintage": vintage, "endpoint": endpoint,
                                   "method": method, **result})
                for period, block in data.groupby("period", sort=True):
                    period_result = lower48.metrics(
                        block, block[endpoint].to_numpy(float), block[f"{method}_point"].to_numpy(float),
                        float(block[f"{method}_bias"].iloc[0]), float(block[f"{method}_sigma"].iloc[0]))
                    score_rows.append({"vintage": vintage, "endpoint": endpoint,
                                       "method": method, "period": period, **period_result})
            hourly = data[["time", "period", endpoint]].copy()
            hourly["vintage"] = vintage
            for method in methods:
                hourly[method] = data[f"{method}_point"].to_numpy(float)
            hourly_rows.append(hourly)
    scores = pd.DataFrame(score_rows)
    scores.to_csv(output / "scores.csv", index=False)
    hourly = pd.concat(hourly_rows, ignore_index=True)
    hourly.to_csv(output / "scored_hourly.csv.gz", index=False, compression="gzip")
    summary = []
    overall = scores.loc[~scores.period.notna()].copy()
    for endpoint in overall.endpoint.unique():
        block = overall.loc[overall.endpoint.eq(endpoint)].pivot(
            index="method", columns="vintage", values=["mae", "shape_mae", "crps_gaussian",
                                                        "coverage90", "width90", "interval_score90"])
        for method in block.index:
            row = {"endpoint": endpoint, "method": method}
            for metric in ("mae", "shape_mae", "crps_gaussian", "coverage90", "width90", "interval_score90"):
                before = float(block.loc[method, (metric, "as_issued")])
                after = float(block.loc[method, (metric, "one_year_later_revision")])
                row[f"{metric}_as_issued"] = before
                row[f"{metric}_revision"] = after
                row[f"{metric}_delta_revision_minus_as_issued"] = after - before
            summary.append(row)
    pd.DataFrame(summary).to_csv(output / "revision_score_deltas.csv", index=False)
    (output / "scoring_manifest.json").write_text(json.dumps({
        "target_hours": int(len(truth)), "target_workbook_sha256": sha(lower48.WORKBOOK),
        "forecast_seal_verified_before_target_read": True,
        "target_labels": ["official_generation", "official_consumption"],
        "score_semantics": "single-GB source calibration; one Lower48 target domain; no coverage guarantee",
    }, indent=2) + "\n")


def forecast(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    baseline_path = BASE_RUN / "predictions_sealed.csv"
    baseline_manifest = json.loads((BASE_RUN / "sealed_manifest.json").read_text())
    if sha(baseline_path) != baseline_manifest["prediction_sha256"]:
        raise ValueError("original as-issued forecast seal failed")
    as_issued = pd.read_csv(baseline_path, parse_dates=["time", "clock"])
    issues = json.loads((BASE_RUN / "as_issued_inputs.json").read_text())
    if len(issues) != 12 or len(as_issued) != 8784:
        raise ValueError("expected complete 2024 as-issued base forecast")

    revised_shares, issue_records = {}, []
    revised_hashes = {}
    for issue in issues:
        target = pd.Period(issue["target_month"], freq="M")
        observation = pd.Period(issue["observation_month"], freq="M")
        revised_archive = archive_path(issue["issue_archive"])
        row = lower48.national_generation(revised_archive, observation)
        revised_shares[str(target)] = {key: float(row[key]) for key in lower48.COARSE}
        headers = header_fields(revised_archive)
        revised_hashes[str(revised_archive.resolve())] = sha(revised_archive)
        issue_records.append({"target_month": str(target), "observation_month": str(observation),
                              "as_issued_archive": issue["issue_archive"],
                              "revision_archive": revised_archive.name,
                              "revision_archive_sha256": sha(revised_archive),
                              "revision_http_headers": headers,
                              "as_issued_share": {key: float(issue[key]) for key in lower48.COARSE},
                              "revision_share": revised_shares[str(target)],
                              "share_change": {key: revised_shares[str(target)][key] - float(issue[key])
                                               for key in lower48.COARSE}})

    target_meta = lower48.target_times()
    revised_features = lower48.make_target_features(target_meta, revised_shares)
    models, source_info = lower48.source_fit()
    revised_points = lower48.predict_models(models, revised_features)
    revised = revised_features[["time", "clock", "period", *lower48.COARSE]].copy()
    for method, point in revised_points.items():
        revised[f"{method}_point"] = point
        revised[f"{method}_bias"] = source_info["calibration"][method]["bias"]
        revised[f"{method}_sigma"] = source_info["calibration"][method]["sigma"]
    if not as_issued.time.astype(str).reset_index(drop=True).equals(
            revised.time.astype(str).reset_index(drop=True)):
        raise ValueError("vintage predictions do not share exact target hours")
    # Confirm the frozen source models reproduce the old release-input forecast.
    baseline_features = lower48.make_target_features(target_meta, {
        issue["target_month"]: {key: float(issue[key]) for key in lower48.COARSE}
        for issue in issues})
    check = lower48.predict_models(models, baseline_features)
    max_diff = {}
    for method in lower48.METHODS:
        delta = np.abs(as_issued[f"{method}_point"].to_numpy(float) - check[method])
        max_diff[method] = float(delta.max())
        if delta.max() > 1e-8:
            raise ValueError(f"cannot reproduce sealed baseline for {method}: {delta.max()}")

    as_path, revised_path = output / "predictions_as_issued.csv", output / "predictions_revised.csv"
    as_issued.to_csv(as_path, index=False)
    revised.to_csv(revised_path, index=False)
    (output / "vintage_inputs.json").write_text(json.dumps(issue_records, indent=2) + "\n")
    model_code = Path(inspect.getsourcefile(lower48.fit_models)).resolve()
    input_hashes = {str(path.resolve()): sha(path) for path in (
        lower48.WORKBOOK, lower48.SOURCE_MANIFEST, PROTOCOL, Path(__file__).resolve(),
        Path(lower48.__file__).resolve(), model_code)}
    # Include the exact source generation series used for fitting and calibration.
    for source in source_info["input_audit"].values():
        path = ROOT / source["path"]
        input_hashes[str(path.resolve())] = sha(path)
    # Hash original-release archives as well as the later release archives.
    original_manifest = json.loads(lower48.SOURCE_MANIFEST.read_text())
    for item in original_manifest["sources"]:
        path = ROOT / item["file"]
        if path.suffix == ".zip" and path.is_file():
            input_hashes[str(path.resolve())] = sha(path)
    input_hashes.update(revised_hashes)
    seal = {"as_issued_sha256": sha(as_path), "revised_sha256": sha(revised_path),
            "baseline_original_sha256": baseline_manifest["prediction_sha256"],
            "target_workbook_sha256": sha(lower48.WORKBOOK), "prediction_rows": len(revised),
            "target_periods": sorted(revised.period.unique().tolist()),
            "source_model_reproduction_max_abs_error": max_diff,
            "code_and_input_sha256": input_hashes,
            "forecast_stage_read_target_features_only": True,
            "target_outcomes_read": False}
    (output / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    print(json.dumps(seal, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if args.score:
        score(output)
    else:
        forecast(output)


if __name__ == "__main__":
    main()
