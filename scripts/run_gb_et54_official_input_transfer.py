"""Paired US-to-GB transfer with official ET 5.4 vs NESO-derived lagged shares."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.cross_country import (
    CALENDAR, COARSE, EF, FEATURES, FUELS, METHODS, ZONES, fit_models,
    labels, metrics, monthly_inputs, predict_models,
)

ROOT = Path(__file__).resolve().parents[1]
ET54_COLUMNS = {
    "thermal": ("Coal\n[note 7]", "Oil", "Gas", "Bioenergy\n[note 10]",
                "Other fuels \n[note 10]"),
    "hydro": (" Hydro",), "nuclear": (" Nuclear",),
    "wind": ("Total wind \n[note 8]\n[note 9]",), "solar": ("Solar",),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_us_source() -> dict[str, pd.DataFrame]:
    """Load only US source supervision; never open target-year hourly fuel labels."""
    frames = {}
    for region, zone in ZONES.items():
        if region == "GB":
            continue
        path = ROOT / f"results/research/data/{region}_hourly.csv"
        raw = pd.read_csv(path, usecols=["time", *(f"generation_{f}" for f in FUELS), "valid"])
        stamp = pd.to_datetime(raw.time, utc=True, errors="raise")
        values = raw[[f"generation_{f}" for f in FUELS]].to_numpy(float)
        valid = (raw.valid.astype(str).str.lower().eq("true").to_numpy()
                 & np.isfinite(values).all(axis=1) & (values >= 0).all(axis=1)
                 & (values.sum(axis=1) > 0))
        frame = pd.DataFrame(values[valid], columns=FUELS)
        frame["time"] = stamp[valid].reset_index(drop=True)
        frame["clock"] = frame.time.dt.tz_convert(zone).dt.tz_localize(None)
        frame["expected_hours"] = frame.clock.map(
            lambda t: ((t.to_period("M") + 1).start_time.tz_localize(zone)
                       - t.to_period("M").start_time.tz_localize(zone)).total_seconds() / 3600)
        frame["country"], frame["region"] = "US", region
        frames[region] = frame
    return frames


def official_et54_shares(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name="Month", header=None, engine="openpyxl")
    raw.columns = range(raw.shape[1])
    table = raw.iloc[7:].copy()
    headers = raw.iloc[6].tolist()
    table.columns = headers
    month = pd.to_datetime(table["Month"], format="%B %Y", errors="coerce")
    table = table.loc[month.notna()].copy()
    table["period"] = month.loc[table.index].dt.to_period("M")
    values = pd.DataFrame(index=table.index)
    for group, columns in ET54_COLUMNS.items():
        values[group] = sum(pd.to_numeric(table[col], errors="coerce") for col in columns)
    values["period"] = table.period
    values = values.groupby("period", sort=True)[list(COARSE)].sum(min_count=1)
    valid = values.notna().all(axis=1) & values.ge(0).all(axis=1) & values.sum(axis=1).gt(0)
    return values.loc[valid].div(values.loc[valid].sum(axis=1), axis=0)


def neso_monthly_shares(path: Path) -> pd.DataFrame:
    """Construct only the legally lagged target input months, without CIF labels."""
    columns = ["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
               "HYDRO", "BIOMASS", "OTHER", "SOLAR"]
    # Scan timestamps alone to stop the generation read at the last legal t-2 month.
    stamps = pd.to_datetime(pd.read_csv(path, usecols=["DATETIME"])["DATETIME"], errors="raise")
    cutoff = pd.Period("2024-10", freq="M")
    first_after_cutoff = np.flatnonzero(stamps.dt.to_period("M").gt(cutoff).to_numpy())
    nrows = int(first_after_cutoff[0]) if len(first_after_cutoff) else len(stamps)
    if not stamps.iloc[:nrows].is_monotonic_increasing:
        raise ValueError("NESO legal-input rows must be chronological up to the cutoff")
    raw = pd.read_csv(path, usecols=columns, nrows=nrows)
    raw["stamp"] = pd.to_datetime(raw.DATETIME, errors="raise")
    # For 2024 targets, t-2 inputs span November 2023 through October 2024.
    raw = raw.loc[raw.stamp.dt.to_period("M").between("2023-11", "2024-10")].copy()
    raw["thermal"] = raw.GAS + raw.COAL + raw.BIOMASS + raw.OTHER
    raw["wind"] = raw.WIND + raw.WIND_EMB
    raw["nuclear"], raw["hydro"], raw["solar"] = raw.NUCLEAR, raw.HYDRO, raw.SOLAR
    raw["period"] = raw.stamp.dt.to_period("M")
    hourly = raw.groupby(["period", raw.stamp.dt.floor("h")])[list(COARSE)].mean()
    monthly = hourly.groupby(level=0)[list(COARSE)].sum(min_count=1)
    counts = hourly.groupby(level=0).size()
    expected = pd.Series({period: period.days_in_month * 24 for period in monthly.index})
    monthly = monthly.loc[counts.div(expected).ge(.99)]
    if not monthly.index.equals(pd.period_range("2023-11", "2024-10", freq="M")):
        raise ValueError("NESO t-2 input months are incomplete")
    return monthly.div(monthly.sum(axis=1), axis=0)


def calendar_target(path: Path) -> pd.DataFrame:
    """Read target timestamps only; fuel columns and CIF are not opened."""
    stamps = pd.read_csv(path, usecols=["DATETIME"])["DATETIME"]
    stamps = pd.to_datetime(stamps, errors="raise")
    stamps = stamps.loc[stamps.dt.year.eq(2024)].dt.floor("h").drop_duplicates().reset_index(drop=True)
    if len(stamps) != 8784:
        raise ValueError(f"expected 8,784 target timestamps, got {len(stamps)}")
    frame = pd.DataFrame({"time": stamps, "clock": stamps, "year": 2024})
    clock = frame.clock
    for name, values, period in [("hour", clock.dt.hour, 24),
                                 ("dow", clock.dt.dayofweek, 7),
                                 ("month", clock.dt.month, 12)]:
        frame[f"{name}_sin"] = np.sin(2 * np.pi * values / period)
        frame[f"{name}_cos"] = np.cos(2 * np.pi * values / period)
    frame["period"] = clock.dt.to_period("M")
    return frame


def attach_input(base: pd.DataFrame, monthly_shares: pd.DataFrame) -> pd.DataFrame:
    by_target = monthly_shares.copy()
    by_target.index = by_target.index + 2
    target = base.loc[base.period.dt.year.eq(2024)].copy()
    mapped = target.period.map(by_target.index.to_series())
    if mapped.isna().any():
        raise ValueError("missing lagged monthly input for target hours")
    target["observed_month"] = (target.period - 2).astype(str)
    for group in COARSE:
        target[group] = mapped.map(by_target[group]).to_numpy(float)
        target[f"{group}_hour_sin"] = target[group] * target.hour_sin
        target[f"{group}_hour_cos"] = target[group] * target.hour_cos
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--et54", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)

    raw = load_us_source()
    prepared = {region: monthly_inputs(frame) for region, frame in raw.items()}
    usable = {region: frame.dropna(subset=list(COARSE)).copy()
              for region, frame in prepared.items()}
    source_regions = list(raw)
    train_parts = [usable[r].loc[usable[r].year.eq(2022)] for r in source_regions]
    train = pd.concat(train_parts)
    fuel = np.concatenate([raw[r].loc[part.index, list(FUELS)].to_numpy(float)
                           for r, part in zip(source_regions, train_parts)])
    models = fit_models(train, fuel)

    # Source-only 2023 equal-region residual calibration is identical for both arms.
    residuals = {method: [] for method in METHODS}
    for region in source_regions:
        cal = usable[region].loc[usable[region].year.eq(2023)]
        prediction = predict_models(models, cal)
        y = labels(raw[region].loc[cal.index])
        for method in METHODS:
            residuals[method].append(y - prediction[method])
    calibration = {}
    for method, parts in residuals.items():
        bias = float(np.mean([part.mean() for part in parts]))
        sigma = float(np.sqrt(np.mean([np.mean((part - bias) ** 2) for part in parts])))
        calibration[method] = (bias, max(sigma, 1.))
    (args.output / "source_calibration.json").write_text(json.dumps(
        {key: list(value) for key, value in calibration.items()}, indent=2) + "\n")

    neso_path = ROOT / "data/neso-gb/df_fuel_ckan.csv"
    base = calendar_target(neso_path)
    neso_shares = neso_monthly_shares(neso_path)
    reconstructed = attach_input(base, neso_shares)
    reconstructed["input_arm"] = "NESO_reconstructed"
    official_monthly = official_et54_shares(args.et54)
    official = attach_input(base, official_monthly)
    official["input_arm"] = "DESNZ_ET54_MPP"
    common_idx = reconstructed.index.intersection(official.index)
    reconstructed = reconstructed.loc[common_idx].copy()
    official = official.loc[common_idx].copy()
    if len(common_idx) == 0:
        raise ValueError("the two input arms have no common target hours")
    # Persist the exact legal target information and its hashes before prediction.
    base[["time"]].to_csv(args.output / "target_timestamps.csv", index=False)
    input_months = pd.period_range("2023-11", "2024-10", freq="M")
    neso_shares.loc[input_months].to_csv(args.output / "neso_lagged_input_shares.csv")
    official_monthly.loc[input_months].to_csv(args.output / "et54_lagged_input_shares.csv")
    inputs = [args.et54,
              ROOT / "docs/GB_OFFICIAL_MONTHLY_INPUT_TRANSFER_PROTOCOL_20261101.md",
              ROOT / "scripts/run_gb_et54_official_input_transfer.py",
              ROOT / "src/transcif/research/cross_country.py",
              args.output / "target_timestamps.csv",
              args.output / "neso_lagged_input_shares.csv",
              args.output / "et54_lagged_input_shares.csv"]
    provenance = {str(path): digest(path) for path in inputs}
    provenance.update({str(ROOT / f"results/research/data/{region}_hourly.csv"):
                       digest(ROOT / f"results/research/data/{region}_hourly.csv")
                       for region in source_regions})
    (args.output / "startup_hashes.json").write_text(json.dumps(provenance, indent=2) + "\n")
    rows, seals = [], []
    predictions_by_arm = {}
    for arm, target in (("NESO_reconstructed", reconstructed),
                        ("DESNZ_ET54_MPP", official)):
        prediction = predict_models(models, target)
        predictions_by_arm[arm] = prediction
        table = pd.DataFrame({"time": target.time.astype(str),
                              "observed_month": target.observed_month,
                              **prediction})
        output_path = args.output / f"{arm}_predictions.csv"
        table.to_csv(output_path, index=False)
        seals.append({"path": output_path.name, "sha256": digest(output_path),
                      "rows": len(table)})
    (args.output / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    # Held-out target responses are read only after both input-arm predictions are sealed.
    # Only after the immutable prediction hashes exist do we open target-year fuel labels.
    label_columns = ["DATETIME", "GAS", "COAL", "NUCLEAR", "WIND", "WIND_EMB",
                     "HYDRO", "BIOMASS", "OTHER", "SOLAR"]
    label_frame = pd.read_csv(neso_path, usecols=label_columns)
    label_frame["stamp"] = pd.to_datetime(label_frame.DATETIME, errors="raise")
    label_frame = label_frame.loc[label_frame.stamp.dt.year.eq(2024)].reset_index(drop=True)
    (args.output / "postscore_target_label_sha256.json").write_text(json.dumps({
        "path": str(neso_path), "sha256": digest(neso_path),
        "read_after_prediction_seal": True,
        "label_columns": label_columns,
    }, indent=2) + "\n")
    label_frame["coal"], label_frame["gas"], label_frame["oil"] = label_frame.COAL, label_frame.GAS, 0.
    label_frame["other"] = label_frame.BIOMASS + label_frame.OTHER
    label_frame["nuclear"], label_frame["hydro"] = label_frame.NUCLEAR, label_frame.HYDRO
    label_frame["wind"], label_frame["solar"] = label_frame.WIND + label_frame.WIND_EMB, label_frame.SOLAR
    label_frame["hour"] = label_frame.stamp.dt.floor("h")
    hourly_labels = label_frame.groupby("hour")[list(FUELS)].mean()
    if len(hourly_labels) != len(common_idx):
        raise ValueError("sealed target timestamps and post-seal hourly labels differ")
    fuel = hourly_labels.to_numpy(float)
    y = fuel @ EF / fuel.sum(axis=1)
    for arm, target in (("NESO_reconstructed", reconstructed),
                        ("DESNZ_ET54_MPP", official)):
        prediction = predictions_by_arm[arm]
        for method in METHODS:
            rows.append({"input_arm": arm, "method": method,
                         **metrics(target, y, prediction[method], *calibration[method])})
    metrics_frame = pd.DataFrame(rows)
    metrics_frame.to_csv(args.output / "method_metrics.csv", index=False)
    paired = metrics_frame.pivot(index="method", columns="input_arm")
    deltas = pd.DataFrame(index=paired.index)
    for metric in ("mae", "shape_mae", "crps_gaussian", "coverage90", "width90",
                   "interval_score90"):
        deltas[f"{metric}_ET54_minus_NESO"] = (
            paired[(metric, "DESNZ_ET54_MPP")] - paired[(metric, "NESO_reconstructed")])
    deltas.to_csv(args.output / "paired_input_arm_deltas.csv")
    summary = {
        "status": "DEVELOPMENT_INPUT_SOURCE_ABLATION",
        "target_hours": int(len(common_idx)),
        "target_months": sorted(set(official.loc[common_idx, "observed_month"].astype(str))),
        "target_label": "NESO GB local-generation fixed-EF proxy; target labels previously viewed",
        "source_training": "US six BA, 2022 fit, 2023 source-only calibration",
        "ET54_vintage": "current revised 2026-10-29 workbook; not historical as-issued",
        "prediction_hashes": seals,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(metrics_frame.to_string(index=False))
    print("\nPaired input-arm changes (ET54 minus NESO):")
    print(deltas.to_string())


if __name__ == "__main__":
    main()
