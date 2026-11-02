"""Rolling GB->NYIS transfer using actual EIA EPM monthly issue archives.

Monthly inputs are read from the release issued in the month before each target
month. Target-hour predictions are sealed before current target fuel columns
are read. NY state monthly inputs and NYIS hourly outcomes are not identical
statistical geographies; the output therefore labels this a state-to-BA proxy.
"""
from __future__ import annotations

import calendar
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import zipfile

import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from audit_actual_monthly_vintages import as_number, read_sheet
from transcif.data.eia930 import normalize_balance
from transcif.research.cross_country import (
    COARSE, FUELS, METHODS, fit_models, labels, load_inputs, metrics,
    monthly_inputs, predict_models,
)
from transcif.research.eia_proxy import _proxy_rows

OUT = ROOT / "results/actual-issue-gb-to-nyis-r3"
ARCHIVES = ROOT / "data/eia-epm-vintages-r1"
START = pd.Period("2025-01", freq="M")
STOP = pd.Period("2026-10", freq="M")  # exclusive; Aug-2026 issue informs Sep-2026
TARGET_REGION = "NYIS"
TABLES = {
    "total": "Table_1_03_A.xlsx", "coal": "Table_1_04_A.xlsx",
    "oil": "Table_1_05_A.xlsx", "coke": "Table_1_06_A.xlsx",
    "gas": "Table_1_07_A.xlsx", "other_fossil_gas": "Table_1_08_A.xlsx",
    "nuclear": "Table_1_09_A.xlsx", "hydro": "Table_1_10_A.xlsx",
    "wind": "Table_1_14_A.xlsx", "solar_pv": "Table_1_17_A.xlsx",
    "solar_thermal": "Table_1_18_A.xlsx",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def archive_name(issue: pd.Period) -> str:
    return f"{calendar.month_name[issue.month].lower()}{issue.year}.zip"


def state_generation(path: Path, observed: pd.Period) -> dict[str, float]:
    label = f"{calendar.month_name[observed.month]} {observed.year}"
    values: dict[str, float] = {}
    nonnumeric: list[str] = []
    with zipfile.ZipFile(path) as archive:
        for key, member in TABLES.items():
            rows = read_sheet(archive, member)
            hi = next(i for i, row in enumerate(rows)
                      if row and row[0] == "Census Division\nand State")
            headers = rows[hi]
            # Solar-PV Table 1.17's first All-Sectors column includes estimated
            # small-scale generation. Select its separate utility-scale column
            # so it matches the utility-scale denominator in Table 1.03.
            candidates = [i for i, value in enumerate(headers) if value == label]
            col = candidates[1] if key == "solar_pv" else candidates[0]
            row = next(r for r in rows[hi + 1:] if r and r[0] == "New York")
            value = as_number(row[col] if col < len(row) else None)
            if value is None:
                if key in {"coal", "oil", "coke", "gas", "other_fossil_gas"}:
                    # EIA's NM fuel cells are not treated as observed zeros;
                    # the unallocated total remains in the explicit residual.
                    nonnumeric.append(key)
                    value = 0.
                else:
                    raise ValueError(f"missing NY {key} for {label} in {path.name}")
            values[key] = value
    # EIA EPM values are GWh. The residual retains biomass and other sources
    # in the same broad other/thermal proxy bucket used by the source model.
    named = sum(values[k] for k in TABLES if k != "total")
    residual = values["total"] - named
    if values["total"] <= 0 or residual < -max(2., .002 * values["total"]):
        raise ValueError(f"invalid EPM closure for {label}: residual={residual}")
    groups = {
        "thermal": values["coal"] + values["oil"] + values["coke"]
                   + values["gas"] + values["other_fossil_gas"] + residual,
        "hydro": values["hydro"], "nuclear": values["nuclear"],
        "wind": values["wind"],
        "solar": values["solar_pv"] + values["solar_thermal"],
    }
    if any(v < 0 for v in groups.values()):
        raise ValueError(f"negative broad category for {label}: {groups}")
    total = sum(groups.values())
    return {**{f"{k}_gwh": v for k, v in groups.items()},
            **{k: v / total for k, v in groups.items()},
            "epm_reported_total_gwh": values["total"],
            "epm_residual_other_gwh": residual,
            "epm_nonnumeric_fossil_cells": ";".join(nonnumeric)}


def issue_inputs() -> tuple[dict[str, dict], list[dict]]:
    rows: dict[str, dict] = {}
    manifest = pd.read_csv(ARCHIVES / "manifest.csv").set_index("issue_archive")
    available = []
    for name, record in manifest.iterrows():
        released = pd.Timestamp(record["last_modified_http"]).tz_convert("UTC")
        available.append((released, name))
    audit = []
    for target in pd.period_range(START, STOP - 1, freq="M"):
        target_start = target.start_time.tz_localize("UTC")
        eligible = [(released, name) for released, name in available if released < target_start]
        if not eligible:
            raise ValueError(f"no archived EPM issue available before {target}")
        released, name = max(eligible)
        match = re.fullmatch(r"([a-z]+)(\d{4})\.zip", name)
        if match is None:
            raise ValueError(f"unexpected archive name: {name}")
        month = list(calendar.month_name).index(match.group(1).capitalize())
        issue = pd.Period(f"{match.group(2)}-{month:02d}", freq="M")
        # An EPM issue typically reports data through issue-month minus two;
        # parse the actual observation month in each workbook rather than
        # assuming a fixed release delay.
        path = ARCHIVES / name
        observed = issue - 2
        shares = state_generation(path, observed)
        rows[str(target)] = shares
        audit.append({"target_month": str(target), "issue_archive": name,
                      "issue_last_modified_http": str(released),
                      "observation_month_in_issue": str(observed),
                      "issue_sha256": sha(path), **shares})
    return rows, audit


def target_metadata() -> pd.DataFrame:
    paths = [ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jan_Jun.csv",
             ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jul_Dec.csv",
             ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jan_Jun.csv",
             ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"]
    frames = []
    for path in paths:
        raw = pd.read_csv(path, usecols=["Balancing Authority", "UTC Time at End of Hour"])
        raw = raw.loc[raw["Balancing Authority"].eq(TARGET_REGION)].copy()
        raw["time"] = pd.to_datetime(raw["UTC Time at End of Hour"], utc=True) - pd.Timedelta(hours=1)
        frames.append(raw[["time"]])
    frame = pd.concat(frames, ignore_index=True).drop_duplicates("time")
    local = pd.DatetimeIndex(frame.time).tz_convert("America/New_York").tz_localize(None)
    periods = local.to_period("M")
    frame["period"] = periods.astype(str)
    frame["clock"] = local
    frame["year"] = local.year
    for name, vals, period in [("hour", local.hour, 24), ("dow", local.dayofweek, 7),
                               ("month", local.month, 12)]:
        frame[f"{name}_sin"] = np.sin(2 * np.pi * vals / period)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * vals / period)
    for target_month in pd.period_range(START, STOP - 1, freq="M"):
        mask = frame.period.eq(str(target_month))
        shares = issue_rows[str(target_month)]
        for category in COARSE:
            frame.loc[mask, category] = shares[category]
    for category in COARSE:
        for suffix in ("sin", "cos"):
            frame[f"{category}_hour_{suffix}"] = frame[category] * frame[f"hour_{suffix}"]
    frame = frame.loc[frame.period.ge(str(START)) & frame.period.lt(str(STOP))].copy()
    return frame.sort_values("time").reset_index(drop=True)


def fit_source():
    raw, audit = load_inputs(ROOT)
    source = raw["GB"]
    prepared = monthly_inputs(source)
    usable = prepared.dropna(subset=list(COARSE)).copy()
    usable["raw_row"] = usable.index
    train = usable.loc[usable.year.eq(2022)].copy()
    train_fuel = source.loc[train.raw_row, list(FUELS)].to_numpy(float)
    models = fit_models(train, train_fuel)
    cal = usable.loc[usable.year.eq(2023)]
    pred = predict_models(models, cal)
    y = labels(source.loc[cal.raw_row])
    residuals = {method: y - pred[method] for method in METHODS}
    calibration = {}
    for method, values in residuals.items():
        bias = float(values.mean())
        sigma = max(float(np.sqrt(np.mean((values - bias) ** 2))), 1.)
        calibration[method] = (bias, sigma)
    return raw, audit, models, calibration


def score_labels(metadata: pd.DataFrame) -> pd.DataFrame:
    paths = [ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jan_Jun.csv",
             ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jul_Dec.csv",
             ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jan_Jun.csv",
             ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"]
    parts, schemas = [], {}
    for snapshot, path in enumerate(paths):
        native = pd.read_csv(path)
        native = native.loc[native["Balancing Authority"].eq(TARGET_REGION)]
        normalized = normalize_balance(native)
        normalized["source_snapshot"] = snapshot
        schemas[str(snapshot)] = {"fuel_columns": [c for c in normalized if c.startswith("generation_")]}
        parts.append(normalized)
    combined = pd.concat(parts, ignore_index=True)
    combined = combined.loc[(combined.valid_at >= "2025-01-01")
                            & (combined.valid_at < "2026-10-01")].copy()
    combined.index = pd.DatetimeIndex(combined.valid_at)
    combined = combined.sort_index()
    hourly, qa = _proxy_rows(combined, TARGET_REGION, schemas)
    hourly = hourly.loc[hourly.valid.astype(bool)].copy()
    hourly["time"] = pd.to_datetime(hourly.time, utc=True)
    hourly = hourly.merge(metadata[["time", "period", "clock"]], on="time", how="inner", validate="one_to_one")
    labels_frame = hourly.rename(columns={f"generation_{fuel}": fuel for fuel in FUELS})
    hourly["y"] = labels(labels_frame)
    return hourly, qa


def audit_october_2024_geography() -> dict:
    archive = ARCHIVES / "december2024.zip"
    state = state_generation(archive, pd.Period("2024-10", freq="M"))
    path = ROOT / "results/research/data/NYIS_hourly.csv"
    hourly = pd.read_csv(path)
    hourly["time"] = (pd.to_datetime(hourly.time, utc=True)
                       .dt.tz_convert("America/New_York").dt.tz_localize(None))
    hourly = hourly.loc[hourly.valid.astype(str).str.lower().eq("true")
                        & hourly.time.dt.to_period("M").eq(pd.Period("2024-10", freq="M"))]
    sums = {fuel: float(hourly[f"generation_{fuel}"].sum()) for fuel in FUELS}
    total = sum(sums.values())
    ba = {"thermal": sum(sums[f] for f in ("coal", "gas", "oil", "other")) / total,
          "hydro": sums["hydro"] / total, "nuclear": sums["nuclear"] / total,
          "wind": sums["wind"] / total, "solar": sums["solar"] / total}
    fields = {key: {"epm_state_share": float(state[key]), "nyis_ba_share": float(ba[key]),
                    "difference_percentage_points": 100 * float(state[key] - ba[key])}
              for key in COARSE}
    return {"observation_month": "2024-10", "issue_archive": archive.name,
            "issue_sha256": sha(archive), "nyis_hourly_path": str(path.relative_to(ROOT)),
            "nyis_hourly_sha256": sha(path), "epm_state_total_gwh": state["epm_reported_total_gwh"],
            "nyis_ba_total_gwh": total / 1000,
            "nyis_total_difference_pct": 100 * (state["epm_reported_total_gwh"] * 1000 - total) / total,
            "five_group_share_l1_percentage_points": 100 * sum(abs(state[k] - ba[k]) for k in COARSE),
            "groups": fields}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT,
                        help="fresh output directory (defaults to results/actual-issue-gb-to-nyis-r3)")
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else ROOT / args.output
    global issue_rows
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    issue_rows, input_audit = issue_inputs()
    source, source_audit, models, calibration = fit_source()
    meta = target_metadata()
    pred = predict_models(models, meta)
    prediction_path = output / "predictions_sealed.csv"
    predictions = pd.DataFrame({"time": meta.time.astype(str), "period": meta.period,
                                 "observed_month": (pd.PeriodIndex(meta.period, freq="M") - 3).astype(str),
                                 **pred})
    predictions.to_csv(prediction_path, index=False)
    seal = {"path": prediction_path.name, "sha256": sha(prediction_path),
            "prediction_rows": len(predictions), "target_label_columns_read_before_seal": False}
    (output / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")

    # Load target generation and labels only after the immutable prediction file exists.
    target, qa = score_labels(meta)
    predictions["time"] = pd.to_datetime(predictions.time, utc=True)
    scored = target.merge(predictions, on=["time", "period"], how="inner", validate="one_to_one")
    rows = []
    for method in METHODS:
        for period, group in scored.groupby("period", sort=True):
            rows.append({"method": method, "period": period,
                         **metrics(group, group.y.to_numpy(), group[method].to_numpy(),
                                   *calibration[method])})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    pooled = []
    for method in METHODS:
        pooled.append({"method": method, **metrics(scored, scored.y.to_numpy(),
                                                    scored[method].to_numpy(),
                                                    *calibration[method])})
    pd.DataFrame(pooled).to_csv(output / "pooled_metrics.csv", index=False)
    pd.DataFrame(input_audit).to_csv(output / "as_issued_inputs.csv", index=False)
    geography = audit_october_2024_geography()
    (output / "geography_alignment_audit.json").write_text(json.dumps(geography, indent=2) + "\n")
    provenance_paths = [Path(__file__), ROOT / "src/transcif/research/cross_country.py",
                        ROOT / "src/transcif/data/eia930.py", ROOT / "src/transcif/research/eia_proxy.py"]
    provenance = {str(p.relative_to(ROOT)): sha(p) for p in provenance_paths}
    provenance.update({f"archive:{r['issue_archive']}": r["issue_sha256"] for r in input_audit})
    provenance.update({str(p.relative_to(ROOT)): sha(p) for p in [
        ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jan_Jun.csv",
        ROOT / "data/native-eia-2025/EIA930_BALANCE_2025_Jul_Dec.csv",
        ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jan_Jun.csv",
        ROOT / "data/native-eia-2026/EIA930_BALANCE_2026_Jul_Dec.csv"]})
    for record in source_audit.values():
        source_path = Path(record["path"])
        provenance[str(source_path)] = sha(ROOT / source_path)
    provenance[str(Path(geography["nyis_hourly_path"]))] = geography["nyis_hourly_sha256"]
    (output / "provenance.json").write_text(json.dumps({"sha256": provenance,
        "target_label_quality": qa, "source_audit": source_audit,
        "source_calibration": calibration, "target_hours_scored": len(scored),
        "geography_alignment_audit": geography,
        "target_scope": "NY state monthly issue -> NYIS BA generation proxy",
        "target_information_release_lag": "target t uses issue t-1 containing observation t-3",
        "label_vintage": "current EIA-930 archive, not historical as-issued labels"}, indent=2) + "\n")
    print(pd.DataFrame(pooled).to_string(index=False))
    print(f"\nScored hours: {len(scored)}; months: {scored.period.nunique()}; outputs: {output}")


if __name__ == "__main__":
    main()
