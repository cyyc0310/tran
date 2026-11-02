"""Run a frozen US-to-Denmark hourly transfer stress test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.cross_country import (
    COARSE, FUELS, METHODS, fit_models, labels, load_inputs, metrics,
    monthly_inputs, predict_models,
)

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/DENMARK_HOURLY_TRANSFER_STRESS_PROTOCOL_20261102.md"
TARGET_RAW = ROOT / "data/energy-charts-flow-audit-r1/dk_public_power_query_2022-01-01_to_2025-01-01.json"
OUT = ROOT / "docs/evidence/denmark-hourly-transfer-stress-r1"
EF = np.array([1000., 400., 800., 400., 0., 0., 0., 0.])
TRANSFER_METHODS = ("anchor700", "source_thermal_ef", "coarse_boosting", "coarse_recomposition")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def energy_charts_target(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text())
    rows = pd.DataFrame(payload["data"])
    values = pd.json_normalize(rows.pop("values"))
    raw = pd.concat([rows, values], axis=1)
    stamp = pd.to_datetime(raw["timestamp"], utc=True)
    local = stamp.dt.tz_convert("Europe/Copenhagen")
    raw["time"] = stamp
    raw["clock"] = local.dt.tz_localize(None)

    def get(name: str) -> pd.Series:
        if name not in raw:
            return pd.Series(0., index=raw.index)
        return pd.to_numeric(raw[name], errors="coerce")

    raw["coal"] = get("fossil_hard_coal") + get("fossil_brown_coal_lignite")
    raw["gas"] = get("fossil_gas")
    raw["oil"] = get("fossil_oil")
    raw["other"] = get("biomass") + get("waste") + get("others")
    raw["nuclear"] = get("nuclear")
    raw["hydro"] = get("hydro_run_of_river") + get("hydro_water_reservoir")
    raw["wind"] = get("wind_onshore") + get("wind_offshore")
    raw["solar"] = get("solar")
    fuel = raw[list(FUELS)].to_numpy(float)
    valid = np.isfinite(fuel).all(axis=1) & (fuel >= 0).all(axis=1) & (fuel.sum(axis=1) > 0)
    raw = raw.loc[valid].copy()
    raw["country"], raw["region"] = "DK", "DK"
    local_clock = pd.to_datetime(raw["clock"])
    raw["expected_hours"] = local_clock.map(
        lambda t: ((t.to_period("M") + 1).start_time.tz_localize("Europe/Copenhagen")
                   - t.to_period("M").start_time.tz_localize("Europe/Copenhagen"))
        .total_seconds() / 3600
    )
    return raw[["time", "clock", "country", "region", "expected_hours", *FUELS]]


def source_fit(raw: dict[str, pd.DataFrame], usable: dict[str, pd.DataFrame]) -> dict:
    regions = [r for r in raw if raw[r].country.iloc[0] == "US"]
    training = pd.concat([usable[r].loc[usable[r].year.eq(2022)] for r in regions])
    fuel = np.concatenate([
        raw[r].loc[usable[r].loc[usable[r].year.eq(2022), "raw_row"], list(FUELS)].to_numpy(float)
        for r in regions
    ])
    return fit_models(training, fuel)


def source_calibration(raw: dict[str, pd.DataFrame], usable: dict[str, pd.DataFrame], models: dict) -> dict:
    residuals = {method: [] for method in METHODS}
    for region in raw:
        if raw[region].country.iloc[0] != "US":
            continue
        frame = usable[region].loc[usable[region].year.eq(2023)]
        prediction = predict_models(models, frame)
        actual = labels(raw[region].loc[frame.raw_row])
        for method in METHODS:
            residuals[method].append(actual - prediction[method])
    result = {}
    for method, parts in residuals.items():
        bias = float(np.mean([part.mean() for part in parts]))
        sigma = float(np.sqrt(np.mean([np.mean((part - bias) ** 2) for part in parts])))
        result[method] = [bias, max(sigma, 1.)]
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=False)
    raw, audit = load_inputs(ROOT)
    prepared = {region: monthly_inputs(frame) for region, frame in raw.items()}
    usable = {region: frame.dropna(subset=list(COARSE)).copy()
              for region, frame in prepared.items()}
    for region, frame in usable.items():
        frame["raw_row"] = frame.index

    target_raw = energy_charts_target(TARGET_RAW)
    target_inputs = monthly_inputs(target_raw).dropna(subset=list(COARSE)).copy()
    target = target_inputs.loc[target_inputs.year.eq(2024)].copy()
    if len(target) != 8779:
        raise ValueError(f"expected 8,779 complete target features, found {len(target)}")
    models = source_fit(raw, usable)
    calibration = source_calibration(raw, usable, models)
    predictions = predict_models(models, target)
    prediction_frame = pd.DataFrame({
        "time_utc": target.time.astype(str),
        "observed_month": target.observed_month,
        **{name: predictions[name] for name in TRANSFER_METHODS},
        **{name: target[name].to_numpy() for name in COARSE},
    })
    prediction_path = OUT / "predictions_2024.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    source_hashes = {
        audit_item["path"]: digest(ROOT / audit_item["path"])
        for audit_item in audit.values()
    }
    seal = {
        "status": "SEALED_PREDICTIONS_AWAIT_SCORE",
        "protocol_sha256": digest(PROTOCOL),
        "runner_sha256": digest(Path(__file__)),
        "model_sha256": digest(ROOT / "src/transcif/research/cross_country.py"),
        "target_input_sha256": digest(TARGET_RAW),
        "source_input_sha256": source_hashes,
        "predictions_file": prediction_path.name,
        "predictions_sha256": digest(prediction_path),
        "prediction_hours": len(prediction_frame),
        "prediction_days_local": int(target.clock.dt.strftime("%Y-%m-%d").nunique()),
        "source": "US seven EIA balancing areas; 2022 training, 2023 calibration",
        "target_features": "Denmark t-2 five-group generation shares and local calendar only",
        "target_2024_label_scored_after_seal": True,
        "prior_target_data_exposure": "2024 Denmark hourly series had been inspected in a prior flow/balance audit; not a blind country confirmation",
    }
    (OUT / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")

    # Scoring starts only after the prediction artifact and its digest are persisted.
    if digest(prediction_path) != seal["predictions_sha256"]:
        raise ValueError("prediction hash mismatch before scoring")
    target_y = target_raw.set_index("time").loc[target.time, list(FUELS)]
    y = target_y.to_numpy(float) @ EF / target_y.sum(axis=1).to_numpy(float)
    pred_read = pd.read_csv(prediction_path, parse_dates=["time_utc"])
    if not np.array_equal(pd.to_datetime(pred_read.time_utc, utc=True).to_numpy(),
                          pd.to_datetime(target.time, utc=True).to_numpy()):
        raise ValueError("target/prediction UTC timestamp mismatch")

    rows = []
    monthly = []
    for method in TRANSFER_METHODS:
        bias, sigma = calibration[method]
        score = metrics(target, y, pred_read[method].to_numpy(float), bias, sigma)
        radius = 1.6448536269514722 * sigma
        point = pred_read[method].to_numpy(float) + bias
        error = y - point
        winkler = float(np.mean(2 * radius + 20 * np.maximum(np.abs(error) - radius, 0)))
        rows.append({"method": method, **score, "winkler90": winkler,
                     "source_bias": bias, "source_sigma": sigma,
                     "coverage90": float((np.abs(error) <= radius).mean()),
                     "width90": float(2 * radius)})
        for month in range(1, 13):
            mask = target.clock.dt.month.eq(month).to_numpy()
            month_score = metrics(target.loc[mask], y[mask], pred_read.loc[mask, method].to_numpy(float),
                                  bias, sigma)
            month_error = error[mask]
            monthly.append({"method": method, "month": month, **month_score,
                            "winkler90": float(np.mean(2 * radius + 20 * np.maximum(np.abs(month_error) - radius, 0))),
                            "coverage90": float((np.abs(month_error) <= radius).mean()),
                            "width90": float(2 * radius)})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "scored_summary.csv", index=False)
    pd.DataFrame(monthly).to_csv(OUT / "monthly_metrics.csv", index=False)

    # Paired daily bootstrap for the primary structured method against the source-factor anchor.
    day = target.clock.dt.strftime("%Y-%m-%d").to_numpy()
    pair = pd.DataFrame({
        "day": day,
        "recomposition": np.abs(y - (pred_read.coarse_recomposition.to_numpy(float)
                                      + calibration["coarse_recomposition"][0])),
        "source_factor": np.abs(y - (pred_read.source_thermal_ef.to_numpy(float)
                                      + calibration["source_thermal_ef"][0])),
    }).groupby("day").mean()
    delta = (pair.recomposition - pair.source_factor).to_numpy()
    rng = np.random.default_rng(20261102)
    boot = np.mean(rng.choice(delta, size=(10000, len(delta)), replace=True), axis=1)
    comparison = {
        "primary": "coarse_recomposition_vs_source_thermal_ef",
        "delta_mae_day_paired_g_per_kwh": float(delta.mean()),
        "bootstrap_ci95": [float(v) for v in np.quantile(boot, [.025, .975])],
        "days_recomposition_wins": int((pair.recomposition < pair.source_factor).sum()),
        "days": int(len(pair)),
    }
    result = {
        "target": "Denmark 2024 hourly fixed-EF generation proxy; development stress test, previously viewed domain",
        "hours": len(target),
        "local_days": int(len(pair)),
        "source_calibration": calibration,
        "metrics": summary.set_index("method").to_dict(orient="index"),
        "paired_daily_bootstrap": comparison,
        "label_mapping": "coal 1000, gas 400, oil 800, biomass+waste 400, nuclear/hydro/wind/solar 0; unreported hydro set to zero as explicit proxy limitation",
        "official_hydro_context": "2024 Denmark generation share ~0.1%; omission sensitivity is small in annual energy but hourly hydro allocation is unavailable",
        "target_labels_accessed_after_prediction_seal": True,
    }
    (OUT / "scored_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(summary[["method", "mae", "shape_mae", "coverage90", "width90", "winkler90"]].to_string(index=False))
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
