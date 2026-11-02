"""UK to US Lower 48 transfer with a sealed, real-issue EPM input budget.

The forecast stage reads only the first six columns of the EIA workbook.
Target outcome columns are read only by the explicit ``--score`` stage.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_actual_monthly_vintages import as_number, read_sheet
from transcif.research.cross_country import (
    COARSE, FEATURES, FUELS, METHODS, CALENDAR, fit_models, labels,
    load_inputs, metrics, monthly_inputs, predict_models,
)

DATA = ROOT / "data/eia930-lower48-asissued-r1"
WORKBOOK = DATA / "lower48_eia930.xlsx"
VINTAGE_DIR = DATA / "epm_vintages"
SOURCE_MANIFEST = DATA / "source_manifest.json"
DEFAULT_OUT = ROOT / "results/lower48-actual-issue-dual-track-r1"
MONTH_START = pd.Period("2024-01", freq="M")
MONTH_STOP = pd.Period("2025-01", freq="M")
TARGET_REGION = "US48"

TABLES = {
    "total": "Table_1_03_A.xlsx", "coal": "Table_1_04_A.xlsx",
    "oil": "Table_1_05_A.xlsx", "coke": "Table_1_06_A.xlsx",
    "gas": "Table_1_07_A.xlsx", "other_fossil_gas": "Table_1_08_A.xlsx",
    "nuclear": "Table_1_09_A.xlsx", "hydro": "Table_1_10_A.xlsx",
    "wind": "Table_1_14_A.xlsx", "solar_pv": "Table_1_17_A.xlsx",
    "solar_thermal": "Table_1_18_A.xlsx",
}
EPM_NATIONAL_ROW = "U.S. Total"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def period_label(p: pd.Period) -> str:
    return f"{calendar.month_name[p.month]} {p.year}"


def national_generation(archive_path: Path, observed: pd.Period) -> dict[str, float]:
    """Read published U.S. total as a documented proxy for the Lower 48 input."""
    wanted = period_label(observed)
    values = {key: 0. for key in TABLES}
    found = {key: set() for key in TABLES}
    with zipfile.ZipFile(archive_path) as archive:
        for key, member in TABLES.items():
            rows = read_sheet(archive, member)
            hi = next(i for i, row in enumerate(rows)
                      if row and row[0] == "Census Division\nand State")
            headers = rows[hi]
            candidates = [i for i, value in enumerate(headers) if value == wanted]
            if not candidates:
                raise ValueError(f"{member}: observation month {wanted} absent")
            col = candidates[1] if key == "solar_pv" else candidates[0]
            for row in rows[hi + 1:]:
                if not row or row[0] != EPM_NATIONAL_ROW:
                    continue
                amount = as_number(row[col] if col < len(row) else None)
                if amount is None:
                    raise ValueError(f"missing {key} national value for {wanted}")
                values[key] += amount
                found[key].add(row[0])
    missing = [key for key in TABLES if not found[key]]
    if missing:
        raise ValueError(f"EPM national rows are missing: {missing}")
    named = sum(values[k] for k in TABLES if k != "total")
    residual = values["total"] - named
    if values["total"] <= 0 or residual < -max(2., .002 * values["total"]):
        raise ValueError(f"invalid EPM closure for {wanted}: residual={residual}")
    groups = {
        "thermal": values["coal"] + values["oil"] + values["coke"] + values["gas"]
                   + values["other_fossil_gas"] + residual,
        "hydro": values["hydro"], "nuclear": values["nuclear"], "wind": values["wind"],
        "solar": values["solar_pv"] + values["solar_thermal"],
    }
    denominator = sum(groups.values())
    return {**{k: v / denominator for k, v in groups.items()},
            "epm_total_gwh": values["total"], "epm_residual_gwh": residual}


def issue_months() -> tuple[dict[str, dict], list[dict]]:
    manifest = json.loads(SOURCE_MANIFEST.read_text())
    issues = []
    for source in manifest["sources"]:
        path = ROOT / source["file"]
        if path.suffix != ".zip":
            continue
        released = pd.Timestamp(source["http_headers"]["Last-Modified"]).tz_convert("UTC")
        issues.append((released, path))
    values, audit = {}, []
    for target in pd.period_range(MONTH_START, MONTH_STOP - 1, freq="M"):
        cutoff = target.start_time.tz_localize("UTC")
        eligible = [(released, path) for released, path in issues if released < cutoff]
        if not eligible:
            raise ValueError(f"no archived EPM issue before {target}")
        released, path = max(eligible, key=lambda x: x[0])
        issue_month = pd.Period(path.stem.replace("january", "01").replace("february", "02")
                                .replace("march", "03").replace("april", "04")
                                .replace("may", "05").replace("june", "06")
                                .replace("july", "07").replace("august", "08")
                                .replace("september", "09").replace("october", "10")
                                .replace("november", "11").replace("december", "12"), freq="M")
        observed = issue_month - 2
        row = national_generation(path, observed)
        values[str(target)] = {k: row[k] for k in COARSE}
        audit.append({"target_month": str(target), "issue_archive": path.name,
                      "issue_release_utc": released.isoformat(),
                      "observation_month": str(observed), "archive_sha256": sha(path), **row})
    return values, audit


def target_times() -> pd.DataFrame:
    """Read only A:F metadata; do not touch labels or target fuel columns."""
    wb = load_workbook(WORKBOOK, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]
    rows = ws.iter_rows(min_row=2, min_col=1, max_col=6, values_only=True)
    out = []
    for region, utc_end, local_date, hour, local_time, zone in rows:
        if region != TARGET_REGION or utc_end is None:
            continue
        stamp = pd.Timestamp(utc_end)
        time = stamp.tz_localize("UTC") - pd.Timedelta(hours=1)
        clock = time.tz_convert("America/New_York").tz_localize(None)
        if clock.year != 2024:
            continue
        out.append({"time": time, "clock": clock, "local_time": local_time, "zone": zone})
    wb.close()
    frame = pd.DataFrame(out).sort_values("time").reset_index(drop=True)
    frame["period"] = frame.clock.dt.to_period("M").astype(str)
    return frame


def make_target_features(meta: pd.DataFrame, shares: dict[str, dict]) -> pd.DataFrame:
    frame = meta.copy()
    clock = frame.clock
    for name, vals, period in [("hour", clock.dt.hour, 24), ("dow", clock.dt.dayofweek, 7),
                               ("month", clock.dt.month, 12)]:
        frame[f"{name}_sin"] = np.sin(2 * np.pi * vals / period)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * vals / period)
    for category in COARSE:
        frame[category] = frame.period.map(lambda p: shares[p][category])
        for suffix in ("sin", "cos"):
            frame[f"{category}_hour_{suffix}"] = frame[category] * frame[f"hour_{suffix}"]
    frame["year"] = clock.dt.year
    return frame


def source_fit() -> tuple[dict, dict]:
    frames, audit = load_inputs(ROOT)
    source = frames["GB"]
    prepared = monthly_inputs(source, lag=3).dropna(subset=list(COARSE)).copy()
    prepared["source_index"] = prepared.index
    train = prepared.loc[prepared.year.eq(2022)]
    fuel = source.loc[train.source_index, list(FUELS)].to_numpy(float)
    models = fit_models(train, fuel)
    calibration_data = prepared.loc[prepared.year.eq(2023)]
    cal_pred = predict_models(models, calibration_data)
    cal_y = labels(source.loc[calibration_data.source_index])
    cal = {}
    for name in METHODS:
        residual = cal_y - cal_pred[name]
        cal[name] = {"bias": float(residual.mean()),
                     "sigma": max(float(np.sqrt(np.mean((residual - residual.mean()) ** 2))), 1.)}
    return models, {"input_audit": audit, "calibration": cal}


def forecast(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    shares, issue_audit = issue_months()
    meta = target_times()
    target = make_target_features(meta, shares)
    models, source_info = source_fit()
    predictions = predict_models(models, target)
    frame = target[["time", "clock", "period", *COARSE]].copy()
    for name, point in predictions.items():
        frame[f"{name}_point"] = point
        frame[f"{name}_bias"] = source_info["calibration"][name]["bias"]
        frame[f"{name}_sigma"] = source_info["calibration"][name]["sigma"]
    sealed = output / "predictions_sealed.csv"
    frame.to_csv(sealed, index=False)
    (output / "as_issued_inputs.json").write_text(json.dumps(issue_audit, indent=2))
    manifest = {"prediction_sha256": sha(sealed), "rows": len(frame),
                "periods": sorted(frame.period.unique().tolist()),
                "target_workbook_sha256": sha(WORKBOOK),
                "target_features_read": "Published Hourly Data columns A:F only",
                "source_audit": source_info["input_audit"],
                "source_calibration": source_info["calibration"]}
    (output / "sealed_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"stage": "forecast_sealed", **manifest}, indent=2))


def score(output: Path) -> None:
    sealed = output / "predictions_sealed.csv"
    manifest_path = output / "sealed_manifest.json"
    if not sealed.exists() or not manifest_path.exists():
        raise FileNotFoundError("run the forecast stage first")
    manifest = json.loads(manifest_path.read_text())
    if sha(sealed) != manifest["prediction_sha256"]:
        raise ValueError("sealed predictions changed; refusing to score")
    pred = pd.read_csv(sealed, parse_dates=["time", "clock"])
    # Load outcome columns only after checking the sealed forecast hash.
    wb = load_workbook(WORKBOOK, read_only=True, data_only=True)
    ws = wb["Published Hourly Data"]
    header = next(ws.iter_rows(min_row=1, max_row=1, values_only=True))
    pos = {name: i + 1 for i, name in enumerate(header)}
    use = ["Region", "UTC time", "NG: COL", "NG: NG", "NG: OIL", "NG: GEO", "NG: WAT",
           "NG: SUN", "NG: SNB", "NG: WND", "NG: WNB", "NG: BAT", "NG: OES", "NG: UES",
           "NG: OTH", "NG: UNK", "CO2 Emissions Intensity for Generated Electricity",
           "CO2 Emissions Intensity for Consumed Electricity"]
    outcome = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if row[pos["Region"] - 1] != TARGET_REGION:
            continue
        stamp = pd.Timestamp(row[pos["UTC time"] - 1])
        time = stamp.tz_localize("UTC") - pd.Timedelta(hours=1)
        if time.tz_convert("America/New_York").year != 2024:
            continue
        yrow = {k: row[pos[k] - 1] for k in use if k not in {"Region", "UTC time"}}
        yrow["time"] = time
        outcome.append(yrow)
    wb.close()
    y = pd.DataFrame(outcome)
    y["official_generation"] = pd.to_numeric(y["CO2 Emissions Intensity for Generated Electricity"], errors="coerce") * 453.59237
    y["official_consumption"] = pd.to_numeric(y["CO2 Emissions Intensity for Consumed Electricity"], errors="coerce") * 453.59237
    joined = pred.merge(y, on="time", how="inner", validate="one_to_one")
    joined["clock"] = pd.to_datetime(joined.clock)
    outcomes = ["official_generation", "official_consumption"]
    summary = {"hours": len(joined), "workbook_rows_2024": len(y), "prediction_rows": len(pred),
               "unit": "gCO2/kWh; official EIA lbs/kWh converted by 453.59237",
               "tracks": {}}
    month_rows = []
    for endpoint in outcomes:
        valid = np.isfinite(joined[endpoint].to_numpy(float))
        frame = joined.loc[valid].copy()
        rows = {}
        for method in METHODS:
            point = frame[f"{method}_point"].to_numpy(float)
            bias = float(frame[f"{method}_bias"].iloc[0])
            sigma = float(frame[f"{method}_sigma"].iloc[0])
            score_result = metrics(frame, frame[endpoint].to_numpy(float), point, bias, sigma)
            rows[method] = score_result
        summary["tracks"][endpoint] = rows
        for month, part in frame.groupby("period"):
            for method in METHODS:
                point = part[f"{method}_point"].to_numpy(float)
                bias = float(part[f"{method}_bias"].iloc[0])
                sigma = float(part[f"{method}_sigma"].iloc[0])
                m = metrics(part, part[endpoint].to_numpy(float), point, bias, sigma)
                month_rows.append({"endpoint": endpoint, "period": month, "method": method, **m})
    (output / "dual_track_scores.json").write_text(json.dumps(summary, indent=2))
    pd.DataFrame(month_rows).to_csv(output / "monthly_scores.csv", index=False)
    joined.to_csv(output / "scored_hourly.csv", index=False)
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--score", action="store_true",
                        help="read outcome columns after checking the sealed forecast hash")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.score:
        score(output)
    else:
        forecast(output)


if __name__ == "__main__":
    main()
