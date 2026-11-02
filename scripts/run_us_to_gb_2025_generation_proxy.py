#!/usr/bin/env python3
"""US-trained cross-country forecast of GB 2025 generation-side proxy."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from transcif.research import cross_country as cc  # noqa: E402

PROTOCOL = ROOT / "docs/US_TO_GB_2025_GENERATION_PROXY_PROTOCOL_20261102.md"
NESO = ROOT / "data/neso-gb/df_fuel_ckan.csv"
FUELS_FILE_COLUMNS = ("DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
                      "HYDRO", "BIOMASS", "OTHER", "SOLAR")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_us_sources() -> tuple[dict[str, pd.DataFrame], dict[str, str]]:
    frames, paths = {}, {}
    for region, zone in cc.ZONES.items():
        path = ROOT / f"results/research/data/{region}_hourly.csv"
        columns = ["time", *(f"generation_{f}" for f in cc.FUELS), "valid"]
        raw = pd.read_csv(path, usecols=columns)
        stamp = pd.to_datetime(raw.time, utc=True, errors="raise")
        values = raw[[f"generation_{f}" for f in cc.FUELS]].to_numpy(float)
        valid = (raw.valid.astype(str).str.lower().eq("true").to_numpy()
                 & np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
                 & (values.sum(axis=1) > 0))
        frame = pd.DataFrame(values[valid], columns=cc.FUELS)
        frame["time"] = stamp[valid].reset_index(drop=True)
        frame["clock"] = frame.time.dt.tz_convert(zone).dt.tz_localize(None)
        frame["expected_hours"] = frame.clock.map(
            lambda t: ((t.to_period("M") + 1).start_time.tz_localize(zone)
                       - t.to_period("M").start_time.tz_localize(zone)).total_seconds()/3600)
        frame["country"], frame["region"] = "US", region
        frames[region] = frame
        paths[region] = str(path.relative_to(ROOT))
    return frames, paths


def load_gb_target() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.read_csv(NESO, usecols=FUELS_FILE_COLUMNS)
    raw["stamp"] = pd.to_datetime(raw.DATETIME, errors="raise")
    if raw.stamp.duplicated().any():
        raise ValueError("duplicate NESO timestamps")
    raw = raw.loc[raw.stamp.dt.year.between(2022, 2025)].copy()
    raw["coal"], raw["gas"], raw["oil"] = raw.COAL, raw.GAS, 0.
    raw["other"] = raw.BIOMASS + raw.OTHER
    raw["nuclear"], raw["hydro"] = raw.NUCLEAR, raw.HYDRO
    raw["wind"], raw["solar"] = raw.WIND + raw.WIND_EMB, raw.SOLAR
    values = raw[list(cc.FUELS)].to_numpy(float)
    valid = np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
    raw = raw.loc[valid].copy()
    raw["clock"] = raw.stamp.dt.floor("h")
    hourly = raw.groupby("clock")[list(cc.FUELS)].mean()
    counts = raw.groupby("clock").size()
    hourly = hourly.loc[counts.eq(2)].reset_index()
    hourly["time"] = hourly.clock
    hourly["expected_hours"] = hourly.clock.dt.days_in_month * 24
    hourly["country"], hourly["region"] = "GB", "GB"
    input_history = hourly.loc[hourly.clock.dt.year.le(2025)].copy().reset_index(drop=True)
    target = hourly.loc[hourly.clock.dt.year.eq(2025)].copy().reset_index(drop=True)
    expected = 365 * 24
    if len(target) != expected or target.clock.duplicated().any():
        raise ValueError(f"GB 2025 target cohort invalid: {len(target)} hours")
    if not (target[list(cc.FUELS)].to_numpy(float) >= 0).all():
        raise ValueError("negative target fuel output")
    return input_history, target


def target_public_inputs(history: pd.DataFrame, target_clock: pd.Series) -> pd.DataFrame:
    """Build each M-only query from its exact t-2 month, without later months."""
    history = history.copy()
    out = pd.DataFrame({"time": target_clock, "clock": target_clock})
    out["period"] = out.clock.dt.to_period("M")
    month_records = []
    history["source_period"] = history.clock.dt.to_period("M")
    for target_period in out.period.drop_duplicates():
        observed_period = target_period - 2
        available = history.loc[history.source_period.le(observed_period)]
        exact_month = available.loc[available.source_period.eq(observed_period)]
        if exact_month.empty:
            continue
        count = len(exact_month)
        expected = float(exact_month.expected_hours.iloc[0])
        if count / expected < .99:
            continue
        totals = exact_month[list(cc.FUELS)].sum(axis=0)
        total = float(totals.sum())
        if total <= 0 or not np.isfinite(total):
            continue
        thermal = float(totals[["coal", "gas", "oil", "other"]].sum())
        values = {name: float(value/total) for name, value in zip(cc.COARSE,
                   [thermal, totals.hydro, totals.nuclear, totals.wind, totals.solar])}
        values.update({"period": target_period, "observed_month": str(observed_period)})
        month_records.append(values)
    if month_records:
        shares = pd.DataFrame(month_records).set_index("period")
        out = out.join(shares, on="period")
    else:
        out[[*cc.COARSE, "observed_month"]] = np.nan
    out["year"] = out.clock.dt.year
    for name, values, n in [("hour", out.clock.dt.hour, 24),
                            ("dow", out.clock.dt.dayofweek, 7),
                            ("month", out.clock.dt.month, 12)]:
        out[f"{name}_sin"] = np.sin(2*np.pi*values/n)
        out[f"{name}_cos"] = np.cos(2*np.pi*values/n)
    for fuel in cc.COARSE:
        for suffix in ("sin", "cos"):
            out[f"{fuel}_hour_{suffix}"] = out[fuel] * out[f"hour_{suffix}"]
    return out


def source_calibration(models: dict, sources: dict[str, pd.DataFrame]) -> dict:
    residuals = {m: [] for m in cc.METHODS}
    for region, raw in sources.items():
        features = cc.monthly_inputs(raw)
        features["raw_row"] = features.index
        cal = features.loc[features.year.eq(2023)].dropna(subset=list(cc.COARSE)).copy()
        if cal.empty:
            raise ValueError(f"no 2023 source calibration rows for {region}")
        prediction = cc.predict_models(models, cal)
        truth = cc.labels(raw.loc[cal.raw_row])
        for method in cc.METHODS:
            residuals[method].append(truth-prediction[method])
    result = {}
    for method in cc.METHODS:
        parts = residuals[method]
        bias = float(np.mean([part.mean() for part in parts]))
        sigma = max(float(np.sqrt(np.mean([np.mean((part-bias)**2) for part in parts]))), 1.)
        result[method] = {"bias": bias, "sigma": sigma}
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    us, source_paths = load_us_sources()
    gb_history, gb_target = load_gb_target()
    usable = {}
    for region, raw in us.items():
        features = cc.monthly_inputs(raw)
        features["raw_row"] = features.index
        usable[region] = features.dropna(subset=list(cc.COARSE)).copy()
    train = pd.concat([usable[r].loc[usable[r].year.eq(2022)] for r in us], ignore_index=True)
    fuel = np.concatenate([us[r].loc[usable[r].loc[usable[r].year.eq(2022), "raw_row"],
                                   list(cc.FUELS)].to_numpy(float) for r in us])
    if len(train) != len(fuel) or train.empty:
        raise ValueError("US source training rows and hourly labels do not align")
    models = cc.fit_models(train, fuel)
    calibration = source_calibration(models, us)

    public = target_public_inputs(gb_history, gb_target.clock)
    if public[list(cc.COARSE)].isna().any().any():
        raise ValueError("missing t-2 public features for a 2025 target hour")
    prediction = cc.predict_models(models, public)
    calibration_payload = {method: calibration[method] for method in cc.METHODS}
    (output/"source_calibration.json").write_text(json.dumps(calibration_payload, indent=2)+"\n")
    pred_path = output/"US_to_GB_2025_generation_proxy_predictions.csv.gz"
    pd.DataFrame({"time": public.time.astype(str), "observed_month": public.observed_month,
                  **prediction}).to_csv(pred_path, index=False, compression="gzip")

    source_inputs = [ROOT/source_paths[r] for r in sorted(source_paths)]
    input_hashes = {str(p.resolve()): sha(p) for p in [*source_inputs, NESO]}
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(cc.__file__).resolve()]
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    # Freeze the target prediction and source calibration before deriving the target response.
    prediction_seal = {"path": pred_path.name, "sha256": sha(pred_path),
                       "rows": len(public), "source_calibration_path": "source_calibration.json",
                       "source_calibration_sha256": sha(output/"source_calibration.json")}
    (output/"prediction_seal.json").write_text(json.dumps(prediction_seal, indent=2)+"\n")
    (output/"startup_hashes.json").write_text(json.dumps({"input_sha256": input_hashes,
        "code_sha256": code_hashes}, indent=2)+"\n")
    if sha(pred_path) != prediction_seal["sha256"]:
        raise RuntimeError("prediction seal mismatch")
    if any(sha(Path(path)) != value for path, value in {**input_hashes, **code_hashes}.items()):
        raise RuntimeError("input or code changed during run")

    # Target labels are first computed below, after predictions have been sealed.
    target_y = cc.labels(gb_target)
    sealed = pd.read_csv(pred_path)
    if len(target_y) != len(sealed):
        raise ValueError("sealed prediction and target response row counts differ")
    monthly_rows = []
    for month in range(1, 13):
        mask = public.clock.dt.month.eq(month).to_numpy()
        for method in cc.METHODS:
            score = cc.metrics(public.loc[mask], target_y[mask],
                               sealed.loc[mask, method].to_numpy(float),
                               calibration[method]["bias"], calibration[method]["sigma"])
            monthly_rows.append({"month": month, "method": method, **score,
                                 "input_observed_month": str(public.loc[mask, "observed_month"].iloc[0])})
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(output/"monthly_metrics.csv", index=False)
    overall = []
    for method in cc.METHODS:
        overall.append({"method": method, **cc.metrics(
            public, target_y, sealed[method].to_numpy(float),
            calibration[method]["bias"], calibration[method]["sigma"])})
    pd.DataFrame(overall).to_csv(output/"overall_metrics.csv", index=False)
    audit = {"target_hours": len(public), "target_months": sorted(int(x) for x in public.clock.dt.month.unique()),
             "features_cut_off_exactly_at_each_target_t_minus_2_month": True,
             "target_inputs": "t-2 five-group shares (per-month as-of cutoff) and calendar",
             "target_labels_opened_after_prediction_seal": True,
             "target_label": "fixed-EF domestic generation proxy; not measured CIF",
             "gb_hourly_coverage_by_month": {str(m): int(public.clock.dt.month.eq(m).sum()) for m in range(1,13)},
             "methods": list(cc.METHODS)}
    (output/"audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(pd.DataFrame(overall).to_string(index=False))


if __name__ == "__main__":
    main()
