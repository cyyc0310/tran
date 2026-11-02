"""Strict-CN5 LOCO nonlinear log-ratio fuel-composition transfer baseline."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_energy_charts_nine_country_cn5_loco as loader  # noqa: E402
import run_cn5_fuel_resolved_response as fuel_model  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_LOGRATIO_HGB_TRANSFER_PROTOCOL_20261101.md"
COUNTRIES = list(loader.COUNTRIES)
EPSILON = 1e-4
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


def rows(prepared: dict, countries: list[str], year: int,
         months: range | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    parts = []
    for country in countries:
        frame = prepared[country][1].loc[prepared[country][1].year.eq(year)]
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


def fit_logratio(frame: pd.DataFrame, fine: np.ndarray) -> dict:
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    x = frame.loc[active, list(model.HISTORY)]
    shares = fine[active, :4] / thermal[active, None]
    shares = (1.0 - 4.0 * EPSILON) * shares + EPSILON
    targets = np.log(shares[:, :3] / shares[:, [3]])
    weights = model.weights(frame.loc[active].reset_index(drop=True))
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10.0, random_state=416, early_stopping=False)
    heads = []
    for index in range(3):
        learner = HistGradientBoostingRegressor(**options)
        learner.fit(x, targets[:, index], sample_weight=weights)
        heads.append(learner)
    return {"heads": heads, "prior": np.average(shares, axis=0, weights=weights)}


def predict_logratio(head: dict, frame: pd.DataFrame) -> np.ndarray:
    raw = np.column_stack([learner.predict(frame[list(model.HISTORY)])
                           for learner in head["heads"]])
    logits = np.column_stack([np.clip(raw, -12., 12.), np.zeros(len(raw))])
    return softmax(logits, axis=1)


def hourly_thermal_share(frame: pd.DataFrame, profile: np.ndarray) -> np.ndarray:
    groups = frame[list(model.GROUPS)].to_numpy(float)
    q = groups * profile[frame.clock.dt.hour.to_numpy()]
    return np.divide(q[:, 0], q.sum(axis=1), out=np.zeros(len(q)), where=q.sum(axis=1) > 0.)


def predict_hgb_shares(head: dict, frame: pd.DataFrame) -> np.ndarray:
    raw = np.column_stack([learner.predict(frame[list(model.HISTORY)])
                           for learner in head["heads"]])
    raw = np.maximum(raw, 0.)
    total = raw.sum(axis=1)
    return np.divide(raw, total[:, None], out=np.tile(head["prior"], (len(raw), 1)),
                     where=total[:, None] > 0.)


def run_fold(target: str, prepared: dict, output: Path) -> dict:
    sources = [country for country in COUNTRIES if country != target]
    train, fine, coarse = rows(prepared, sources, 2022)
    reference = model.fit(train, fine, coarse)
    hgb_head = fuel_model.fit_fuelshare(train, fine)
    logratio_head = fit_logratio(train, fine)

    calibration, calibration_fine, _ = rows(prepared, sources, 2023, range(7, 13))
    calibration_truth = calibration_fine @ model.FACTORS / calibration_fine.sum(axis=1)
    weights = model.weights(calibration)
    cal_points = model.predict(reference, calibration)
    hgb_shares = predict_hgb_shares(hgb_head, calibration)
    log_shares = predict_logratio(logratio_head, calibration)
    thermal_share = hourly_thermal_share(calibration, reference["profile"])
    cal_points["fuelshare_hgb"] = hgb_shares @ fuel_model.FUEL_FACTORS * thermal_share
    cal_points["fuelshare_logratio_hgb"] = log_shares @ fuel_model.FUEL_FACTORS * thermal_share
    radii = {name: float(weighted_quantile(np.abs(calibration_truth - pred), weights, .9))
             for name, pred in cal_points.items()}

    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024)].copy()
    points = model.predict(reference, query)
    hgb_query_shares = predict_hgb_shares(hgb_head, query)
    log_query_shares = predict_logratio(logratio_head, query)
    target_thermal_share = hourly_thermal_share(query, reference["profile"])
    points["fuelshare_hgb"] = hgb_query_shares @ fuel_model.FUEL_FACTORS * target_thermal_share
    points["fuelshare_logratio_hgb"] = log_query_shares @ fuel_model.FUEL_FACTORS * target_thermal_share
    names = ("coal", "gas", "oil", "other")
    for i, name in enumerate(names):
        points[f"hgb_share_{name}"] = hgb_query_shares[:, i]
        points[f"logratio_share_{name}"] = log_query_shares[:, i]
    prediction_path = output / f"{target}_predictions.csv.gz"
    pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  **points}).to_csv(prediction_path, index=False, compression="gzip")
    radius_path = output / f"{target}_source_radii.json"
    radius_path.write_text(json.dumps(radii, indent=2) + "\n")
    return {"target": target, "prediction_file": prediction_path.name,
            "radius_file": radius_path.name, "sources": sources,
            "optimizer": "HistGradientBoostingRegressor, fixed source-only parameters",
            "epsilon": EPSILON}


def score_fold(target: str, prepared: dict, output: Path, selection: dict):
    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024)].copy()
    stored = pd.read_csv(output / selection["prediction_file"])
    radii = json.loads((output / selection["radius_file"]).read_text())
    fine = fuel_model.fuel_for(raw, query)
    truth = fuel_model.truth_for(raw, query)
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    true_shares = np.divide(fine[:, :4], thermal[:, None], out=np.zeros((len(fine), 4)),
                            where=thermal[:, None] > 0)
    rows_out, composition = [], []
    for method, radius in radii.items():
        pred = stored[method].to_numpy(float)
        error = truth - pred
        daily = pd.DataFrame({"truth": truth, "prediction": pred,
                              "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
        centered = daily[["truth", "prediction"]] - daily.groupby("day")[[
            "truth", "prediction"]].transform("mean")
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            absolute = np.abs(error[mask])
            rows_out.append({"target_country": target.upper(), "method": method,
                "month": int(month), "hours": int(mask.sum()), "mae": float(absolute.mean()),
                "shape_mae": float(np.abs(centered.truth.to_numpy()[mask] -
                                           centered.prediction.to_numpy()[mask]).mean()),
                "coverage90": float((absolute <= radius).mean()), "width90": 2 * radius,
                "interval_score90": float(np.mean(2 * radius +
                    20 * np.maximum(absolute - radius, 0)))})
    for method, prefix in (("hgb", "hgb_share_"), ("logratio_hgb", "logratio_share_")):
        predicted = stored[[prefix + name for name in ("coal", "gas", "oil", "other")]].to_numpy(float)
        for i, fuel in enumerate(("coal", "gas", "oil", "other")):
            composition.append({"target_country": target.upper(), "method": method,
                "fuel": fuel, "mae": float(np.abs(true_shares[active, i] - predicted[active, i]).mean())})
        composition.append({"target_country": target.upper(), "method": method,
            "fuel": "thermal_effective_ef", "mae": float(np.abs(
                true_shares[active] @ fuel_model.FUEL_FACTORS -
                predicted[active] @ fuel_model.FUEL_FACTORS).mean())})
    return rows_out, composition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_files, code_files = tracked_files()
    hashes = {"inputs": {str(path.resolve()): sha(path) for path in input_files},
              "code": {str(path.resolve()): sha(path) for path in code_files}}
    (output / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    prepared = {}
    for country in COUNTRIES:
        raw, features, _ = loader.prepare_country(country)
        prepared[country] = raw, features

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

    metric_rows, composition_rows = [], []
    for target, selection in zip(COUNTRIES, selections):
        rows_out, comp = score_fold(target, prepared, output, selection)
        metric_rows.extend(rows_out)
        composition_rows.extend(comp)
    monthly = pd.DataFrame(metric_rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target_country", "method"])[list(METRICS)].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    country.groupby("method")[list(METRICS)].mean().reset_index().to_csv(
        output / "macro_metrics.csv", index=False)
    pd.DataFrame(composition_rows).to_csv(output / "mechanism_metrics.csv", index=False)
    summary = {"status": "DEVELOPMENT_STRICT_CN5_LOGRATIO_HGB_TRANSFER",
        "countries": [c.upper() for c in COUNTRIES],
        "method_macro": country.groupby("method")[list(METRICS)].mean().to_dict(orient="index"),
        "limitations": ["All target countries were previously viewed; development only.",
            "Monthly input is reconstructed from revised hourly data to emulate t-2, not as-issued.",
            "Labels are fixed-factor generation-side proxies, not measured CIF.",
            "Log-ratio and softmax composition regression are established techniques.",
            "Target composition diagnostics use labels only after prediction sealing."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.read_csv(output / "macro_metrics.csv").to_string(index=False), flush=True)
    print(f"completed strict CN5 log-ratio HGB transfer: {output}", flush=True)


if __name__ == "__main__":
    main()
