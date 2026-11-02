"""Strict-CN5 source-supervised response models with coarse-mix trajectory features."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_energy_charts_nine_country_cn5_loco as loader  # noqa: E402
import run_cn5_fuel_resolved_response as fuel_model  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_LAGGED_TRAJECTORY_RESPONSE_PROTOCOL_20261101.md"
COUNTRIES = list(loader.COUNTRIES)
NONTHERMAL = ("hydro", "nuclear", "wind", "solar")
TRAJECTORY = tuple(f"{group}_{feature}" for group in NONTHERMAL
                   for feature in ("slope6", "recent_shift3"))
METRICS = ("mae", "shape_mae", "coverage90", "width90", "interval_score90")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked_files() -> tuple[list[Path], list[Path]]:
    inputs = [PROTOCOL]
    for country in COUNTRIES:
        folder, manifest, _ = loader.six_group.data_locations(country)
        inputs.extend([manifest, folder / f"{country}_2022_2024.json"])
        if country in loader.six_group.OLD_COUNTRIES:
            inputs.append(folder / "audit.json")
    code = [Path(__file__).resolve(), PROTOCOL, Path(loader.__file__).resolve(),
            Path(loader.six_group.__file__).resolve(), Path(loader.six_group.base.__file__).resolve(),
            Path(fuel_model.__file__).resolve(), Path(model.__file__).resolve(),
            ROOT / "src/transcif/research/cross_country.py",
            ROOT / "src/transcif/research/monthly_adaptation.py"]
    return sorted(set(inputs)), sorted(set(code))


def augment_trajectory(raw: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    months = model.public_months(raw)
    shares = months.div(months.sum(axis=1), axis=0)
    by_period: dict[pd.Period, dict[str, float]] = {}
    periods = features.clock.dt.to_period("M")
    for period in periods.unique():
        history = shares.loc[(shares.index >= period - 7) & (shares.index <= period - 2)]
        if len(history) != 6 or not history.index.is_monotonic_increasing:
            continue
        values: dict[str, float] = {}
        x = np.arange(6, dtype=float)
        for group in NONTHERMAL:
            y = history[group].to_numpy(float)
            values[f"{group}_slope6"] = float(np.polyfit(x, y, 1)[0])
            values[f"{group}_recent_shift3"] = float(y[3:].mean() - y[:3].mean())
        by_period[period] = values
    out = features.copy()
    for name in TRAJECTORY:
        out[name] = periods.map({period: values[name] for period, values in by_period.items()}).to_numpy()
    return out


def input_rows(prepared: dict, countries: list[str], year: int,
               months: range | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    parts = []
    for country in countries:
        frame = prepared[country][1].loc[prepared[country][1].year.eq(year)]
        frame = frame.loc[frame[list(TRAJECTORY)].notna().all(axis=1)]
        if months is not None:
            frame = frame.loc[frame.clock.dt.month.isin(months)]
        if len(frame):
            parts.append((country, frame))
    combined = pd.concat([frame for _, frame in parts], ignore_index=True)
    fine = np.concatenate([fuel_model.fuel_for(prepared[country][0], frame)
                           for country, frame in parts])
    coarse = np.concatenate([model.coarse_values(
        prepared[country][0].iloc[frame.raw_row.to_numpy()]) for country, frame in parts])
    return combined, fine, coarse


def fit_trajectory_factor(frame: pd.DataFrame, fine: np.ndarray) -> dict:
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    target = np.divide(fine @ model.FACTORS, thermal,
                       out=np.full(len(fine), 400.), where=active)
    train = frame.loc[active].reset_index(drop=True)
    weights = model.weights(train) * len(train)
    learner = HistGradientBoostingRegressor(max_iter=80, max_leaf_nodes=15,
        min_samples_leaf=100, l2_regularization=10., random_state=416, early_stopping=False)
    learner.fit(train[[*model.HISTORY, *TRAJECTORY]], target[active], sample_weight=weights)
    return learner


def fit_trajectory_shares(frame: pd.DataFrame, fine: np.ndarray) -> dict:
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    target = fine[active, :4] / thermal[active, None]
    train = frame.loc[active].reset_index(drop=True)
    weights = model.weights(train)
    weights /= weights.mean()
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    heads = []
    for index in range(4):
        learner = HistGradientBoostingRegressor(**options)
        learner.fit(train[[*model.HISTORY, *TRAJECTORY]], target[:, index],
                     sample_weight=weights)
        heads.append(learner)
    prior = np.average(target, axis=0, weights=weights)
    prior /= prior.sum()
    return {"heads": heads, "prior": prior}


def predict_trajectory_shares(head: dict, frame: pd.DataFrame) -> np.ndarray:
    x = frame[[*model.HISTORY, *TRAJECTORY]]
    raw = np.column_stack([learner.predict(x) for learner in head["heads"]])
    raw = np.maximum(raw, 0.)
    total = raw.sum(axis=1)
    return np.divide(raw, total[:, None], out=np.tile(head["prior"], (len(raw), 1)),
                     where=total[:, None] > 0.)


def hourly_thermal_share(frame: pd.DataFrame, profile: np.ndarray) -> np.ndarray:
    q = frame[list(model.GROUPS)].to_numpy(float) * profile[frame.clock.dt.hour.to_numpy()]
    return np.divide(q[:, 0], q.sum(axis=1), out=np.zeros(len(q)), where=q.sum(axis=1) > 0.)


def run_fold(target: str, prepared: dict, output: Path) -> dict:
    sources = [country for country in COUNTRIES if country != target]
    train, fine, coarse = input_rows(prepared, sources, 2022)
    base = model.fit(train, fine, coarse)
    base_shares = fuel_model.fit_fuelshare(train, fine)
    trajectory_factor = fit_trajectory_factor(train, fine)
    trajectory_shares = fit_trajectory_shares(train, fine)

    calibration, cal_fine, _ = input_rows(prepared, sources, 2023, range(7, 13))
    cal_truth = cal_fine @ model.FACTORS / cal_fine.sum(axis=1)
    cal_weights = model.weights(calibration)
    cal_points = model.predict(base, calibration)
    base_fuel_cif = fuel_model.predict_fuelshare(base_shares, base, calibration)
    q_cal = hourly_thermal_share(calibration, base["profile"])
    cal_factor = np.clip(trajectory_factor.predict(calibration[[*model.HISTORY, *TRAJECTORY]]),
                         400., 1000.) * q_cal
    trajectory_share_cif = (predict_trajectory_shares(trajectory_shares, calibration) @
                            fuel_model.FUEL_FACTORS) * q_cal
    cal_points.update({"fuelshare_hgb": base_fuel_cif,
                       "factor_trajectory_hgb": cal_factor,
                       "fuelshare_trajectory_hgb": trajectory_share_cif})
    radii = {name: float(weighted_quantile(np.abs(cal_truth - pred), cal_weights, .9))
             for name, pred in cal_points.items()}

    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024) &
                         features[list(TRAJECTORY)].notna().all(axis=1)].copy()
    points = model.predict(base, query)
    base_fuel_cif = fuel_model.predict_fuelshare(base_shares, base, query)
    q_query = hourly_thermal_share(query, base["profile"])
    points.update({
        "fuelshare_hgb": base_fuel_cif,
        "factor_trajectory_hgb": np.clip(
            trajectory_factor.predict(query[[*model.HISTORY, *TRAJECTORY]]), 400., 1000.) * q_query,
        "fuelshare_trajectory_hgb": (predict_trajectory_shares(trajectory_shares, query) @
                                      fuel_model.FUEL_FACTORS) * q_query,
    })
    base_share_values = np.column_stack([
        np.maximum(head.predict(query[list(model.HISTORY)]), 0.)
        for head in base_shares["heads"]])
    base_share_values = np.divide(base_share_values, base_share_values.sum(axis=1)[:, None],
        out=np.tile(base_shares["prior"], (len(query), 1)),
        where=base_share_values.sum(axis=1)[:, None] > 0.)
    trajectory_share_values = predict_trajectory_shares(trajectory_shares, query)
    for i, fuel in enumerate(("coal", "gas", "oil", "other")):
        points[f"hgb_share_{fuel}"] = base_share_values[:, i]
        points[f"trajectory_share_{fuel}"] = trajectory_share_values[:, i]
    prediction_path = output / f"{target}_predictions.csv.gz"
    pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  **points}).to_csv(prediction_path, index=False, compression="gzip")
    radius_path = output / f"{target}_source_radii.json"
    radius_path.write_text(json.dumps(radii, indent=2) + "\n")
    return {"target": target, "prediction_file": prediction_path.name,
            "radius_file": radius_path.name, "sources": sources,
            "scored_hours": len(query),
            "available_hours": int(features.year.eq(2024).sum()),
            "scored_fraction": float(len(query) / features.year.eq(2024).sum())}


def score_fold(target: str, prepared: dict, output: Path, selection: dict):
    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024) &
                         features[list(TRAJECTORY)].notna().all(axis=1)].copy()
    stored = pd.read_csv(output / selection["prediction_file"])
    radii = json.loads((output / selection["radius_file"]).read_text())
    fine = fuel_model.fuel_for(raw, query)
    truth = fuel_model.truth_for(raw, query)
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    target_shares = np.divide(fine[:, :4], thermal[:, None], out=np.zeros((len(fine), 4)),
                              where=thermal[:, None] > 0.)
    metric_rows, mechanism_rows = [], []
    for method, radius in radii.items():
        prediction = stored[method].to_numpy(float)
        error = truth - prediction
        daily = pd.DataFrame({"truth": truth, "prediction": prediction,
                              "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
        centered = daily[["truth", "prediction"]] - daily.groupby("day")[[
            "truth", "prediction"]].transform("mean")
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            absolute = np.abs(error[mask])
            metric_rows.append({"target_country": target.upper(), "method": method,
                "month": int(month), "hours": int(mask.sum()), "mae": float(absolute.mean()),
                "shape_mae": float(np.abs(centered.truth.to_numpy()[mask] -
                                           centered.prediction.to_numpy()[mask]).mean()),
                "coverage90": float((absolute <= radius).mean()), "width90": 2 * radius,
                "interval_score90": float(np.mean(2 * radius +
                    20 * np.maximum(absolute - radius, 0)))})
    base_share = stored[[f"hgb_share_{name}" for name in ("coal", "gas", "oil", "other")]].to_numpy(float)
    trend_share = stored[[f"trajectory_share_{name}" for name in (
        "coal", "gas", "oil", "other")]].to_numpy(float)
    for model_name, predicted in (("fuelshare_hgb", base_share), ("fuelshare_trajectory_hgb", trend_share)):
        for i, fuel in enumerate(("coal", "gas", "oil", "other")):
            mechanism_rows.append({"target_country": target.upper(), "model": model_name,
                "target": fuel, "mae": float(np.abs(target_shares[active, i] -
                                                       predicted[active, i]).mean())})
        mechanism_rows.append({"target_country": target.upper(), "model": model_name,
            "target": "thermal_effective_ef", "mae": float(np.abs(
                target_shares[active] @ fuel_model.FUEL_FACTORS -
                predicted[active] @ fuel_model.FUEL_FACTORS).mean())})
    return metric_rows, mechanism_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_files, code_files = tracked_files()
    hashes = {"inputs": {str(p.resolve()): sha(p) for p in input_files},
              "code": {str(p.resolve()): sha(p) for p in code_files}}
    (output / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    prepared = {}
    for country in COUNTRIES:
        raw, features, _ = loader.prepare_country(country)
        enriched = augment_trajectory(raw, features)
        required = [*model.HISTORY, *TRAJECTORY]
        if enriched[list(model.HISTORY)].isna().any().any():
            raise ValueError(f"incomplete base features for {country}")
        prepared[country] = raw, enriched

    selections = []
    for target in COUNTRIES:
        selection = run_fold(target, prepared, output)
        selections.append(selection)
        print(f"sealed {target.upper()}", flush=True)
    seals = []
    for selection in selections:
        for key in ("prediction_file", "radius_file"):
            path = output / selection[key]
            seals.append({"target": selection["target"], "file": path.name,
                          "sha256": sha(path), "kind": key})
    (output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    (output / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    for group in ("inputs", "code"):
        for path, expected in hashes[group].items():
            if sha(Path(path)) != expected:
                raise RuntimeError(f"{group} changed during run: {path}")
    for seal in seals:
        if sha(output / seal["file"]) != seal["sha256"]:
            raise RuntimeError(f"prediction/radius seal mismatch: {seal['file']}")

    metric_rows, mechanism_rows = [], []
    for target, selection in zip(COUNTRIES, selections):
        current, mechanisms = score_fold(target, prepared, output, selection)
        metric_rows.extend(current)
        mechanism_rows.extend(mechanisms)
    monthly = pd.DataFrame(metric_rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target_country", "method"])[list(METRICS)].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    country.groupby("method")[list(METRICS)].mean().reset_index().to_csv(
        output / "macro_metrics.csv", index=False)
    pd.DataFrame(mechanism_rows).to_csv(output / "mechanism_metrics.csv", index=False)
    summary = {"status": "DEVELOPMENT_STRICT_CN5_LAGGED_TRAJECTORY_RESPONSE",
        "countries": [c.upper() for c in COUNTRIES], "trajectory_features": list(TRAJECTORY),
        "method_macro": country.groupby("method")[list(METRICS)].mean().to_dict(orient="index"),
        "limitations": ["All countries and target years were previously viewed; development only.",
            "Monthly inputs are reconstructed from revised hourly series to emulate t-2 through t-7, not as-issued.",
            "Labels are fixed-factor generation-side proxies, not measured CIF.",
            "Trajectory features are deterministic summaries of existing legal coarse monthly inputs.",
            "Target fuel composition is read only after prediction and radius sealing."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.read_csv(output / "macro_metrics.csv").to_string(index=False), flush=True)
    print(f"completed CN5 lagged trajectory response transfer: {output}", flush=True)


if __name__ == "__main__":
    main()
