"""Nested softmax fuel-composition regression under strict CN5 hourly transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp, softmax
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_energy_charts_nine_country_cn5_loco as loader  # noqa: E402
import run_cn5_fuel_resolved_response as hgb_fuel  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_SOFTMAX_COMPOSITION_TRANSFER_PROTOCOL_20261101.md"
COUNTRIES = list(loader.COUNTRIES)
ALPHAS = (0.01, 0.1, 1.0, 10.0)
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
    code = [Path(__file__), PROTOCOL, Path(loader.__file__).resolve(),
            Path(loader.six_group.__file__).resolve(), Path(loader.six_group.base.__file__).resolve(),
            Path(model.__file__).resolve(), Path(hgb_fuel.__file__).resolve(),
            ROOT / "src/transcif/research/cross_country.py",
            ROOT / "src/transcif/research/monthly_adaptation.py"]
    return sorted(set(inputs)), sorted(set(code))


def input_rows(prepared: dict, countries: list[str], year: int,
               months: range | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    pairs = []
    for country in countries:
        part = prepared[country][1].loc[prepared[country][1].year.eq(year)]
        if months is not None:
            part = part.loc[part.clock.dt.month.isin(months)]
        if len(part):
            pairs.append((country, part))
    frames = [part for _, part in pairs]
    frame = pd.concat(frames, ignore_index=True)
    fuel = np.concatenate([hgb_fuel.fuel_for(prepared[c][0], part) for c, part in pairs])
    coarse = np.concatenate([model.coarse_values(
        prepared[c][0].iloc[part.raw_row.to_numpy()]) for c, part in pairs])
    return frame, fuel, coarse


def fit_softmax_composition(frame: pd.DataFrame, fuel: np.ndarray, alpha: float) -> dict:
    thermal = fuel[:, :4].sum(axis=1)
    active = thermal > 0
    active_frame = frame.loc[active]
    shares = fuel[active, :4] / thermal[active, None]
    scaler = StandardScaler().fit(active_frame[list(model.HISTORY)],
                                   sample_weight=model.weights(active_frame))
    x = scaler.transform(active_frame[list(model.HISTORY)])
    design = np.column_stack([np.ones(len(x)), x])
    weights = model.weights(active_frame)
    weights = weights / weights.sum()
    n_features = design.shape[1]

    def objective(flat):
        coef = flat.reshape(n_features, 4)
        logits = design @ coef
        log_norm = logsumexp(logits, axis=1)
        loss = np.sum(weights * (log_norm - np.sum(shares * logits, axis=1)))
        loss += 0.5 * alpha * np.square(coef[1:]).sum()
        residual = softmax(logits, axis=1) - shares
        gradient = design.T @ (weights[:, None] * residual)
        gradient[1:] += alpha * coef[1:]
        return float(loss), gradient.ravel()

    result = minimize(objective, np.zeros(n_features * 4), jac=True, method="L-BFGS-B",
                      options={"maxiter": 500, "ftol": 1e-10, "gtol": 1e-7})
    if not result.success and result.status != 1:
        raise RuntimeError(f"softmax composition optimization failed: {result.message}")
    return {"scaler": scaler, "coef": result.x.reshape(n_features, 4),
            "alpha": alpha, "optimization": {"success": bool(result.success),
            "status": int(result.status), "iterations": int(result.nit),
            "objective": float(result.fun)}}


def predict_composition(fitted: dict, frame: pd.DataFrame) -> np.ndarray:
    x = fitted["scaler"].transform(frame[list(model.HISTORY)])
    return softmax(np.column_stack([np.ones(len(x)), x]) @ fitted["coef"], axis=1)


def hourly_thermal_share(frame: pd.DataFrame, profile: np.ndarray) -> np.ndarray:
    coarse = frame[list(model.GROUPS)].to_numpy(float) * profile[frame.clock.dt.hour.to_numpy()]
    return np.divide(coarse[:, 0], coarse.sum(axis=1), out=np.zeros(len(coarse)),
                     where=coarse.sum(axis=1) > 0)


def predict_composition_cif(fitted: dict, frame: pd.DataFrame,
                             profile: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shares = predict_composition(fitted, frame)
    ef = shares @ hgb_fuel.FUEL_FACTORS
    return np.clip(ef * hourly_thermal_share(frame, profile), 0., 1000.), shares


def predict_hgb_shares(head: dict, frame: pd.DataFrame) -> np.ndarray:
    raw = np.column_stack([learner.predict(frame[list(model.HISTORY)]) for learner in head["heads"]])
    raw = np.maximum(raw, 0.)
    total = raw.sum(axis=1)
    return np.divide(raw, total[:, None], out=np.tile(head["prior"], (len(raw), 1)),
                     where=total[:, None] > 0.)


def train_reference(train: pd.DataFrame, fuel: np.ndarray, coarse: np.ndarray):
    return model.fit(train, fuel, coarse)


def choose_alpha(outer_target: str, sources: list[str], prepared: dict) -> tuple[float, list[dict]]:
    scores = []
    for pseudo in sources:
        train_countries = [c for c in sources if c != pseudo]
        train, train_fuel, _ = input_rows(prepared, train_countries, 2022)
        validation, _, _ = input_rows(prepared, [pseudo], 2023, range(1, 7))
        truth = hgb_fuel.truth_for(prepared[pseudo][0], validation)
        profile_fit = train_reference(train, train_fuel,
            np.concatenate([model.coarse_values(prepared[c][0].iloc[
                prepared[c][1].loc[prepared[c][1].year.eq(2022)].raw_row.to_numpy()])
                for c in train_countries]))
        profile = profile_fit["profile"]
        for alpha in ALPHAS:
            fitted = fit_softmax_composition(train, train_fuel, alpha)
            prediction, _ = predict_composition_cif(fitted, validation, profile)
            scores.append({"outer_target": outer_target, "pseudo_target": pseudo,
                           "alpha": alpha,
                           "hourly_cif_mae": float(np.mean(np.abs(truth - prediction))),
                           "optimizer_iterations": fitted["optimization"]["iterations"]})
    table = pd.DataFrame(scores)
    means = table.groupby("alpha").hourly_cif_mae.mean()
    selected = float(means.idxmin())
    return selected, scores


def fit_for_countries(countries: list[str], year: int, prepared: dict):
    train, fuel, coarse = input_rows(prepared, countries, year)
    reference = train_reference(train, fuel, coarse)
    hgb_head = hgb_fuel.fit_fuelshare(train, fuel)
    return train, fuel, reference, hgb_head


def predict_outer(target: str, prepared: dict, output: Path) -> tuple[list[dict], dict]:
    sources = [c for c in COUNTRIES if c != target]
    alpha, inner = choose_alpha(target, sources, prepared)
    train, train_fuel, reference, hgb_head = fit_for_countries(sources, 2022, prepared)
    softmax_head = fit_softmax_composition(train, train_fuel, alpha)
    calibration, cal_fuel, _ = input_rows(prepared, sources, 2023, range(7, 13))
    cal_truth = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
    cal_weights = model.weights(calibration)
    cal_predictions = model.predict(reference, calibration)
    hgb_shares = predict_hgb_shares(hgb_head, calibration)
    hgb_cif = np.clip(hgb_shares @ hgb_fuel.FUEL_FACTORS, 0., 1000.) * hourly_thermal_share(
        calibration, reference["profile"])
    soft_cif, _ = predict_composition_cif(softmax_head, calibration, reference["profile"])
    cal_predictions.update({"fuelshare_hgb": hgb_cif, "fuelshare_softmax": soft_cif})
    radii = {name: float(weighted_quantile(np.abs(cal_truth - prediction), cal_weights, .9))
             for name, prediction in cal_predictions.items()}

    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024)].copy()
    predictions = model.predict(reference, query)
    hgb_pred_shares = predict_hgb_shares(hgb_head, query)
    hgb_cif = np.clip(hgb_pred_shares @ hgb_fuel.FUEL_FACTORS, 0., 1000.) * hourly_thermal_share(
        query, reference["profile"])
    soft_cif, soft_shares = predict_composition_cif(softmax_head, query, reference["profile"])
    predictions.update({"fuelshare_hgb": hgb_cif, "fuelshare_softmax": soft_cif})
    for i, name in enumerate(("coal", "gas", "oil", "other")):
        predictions[f"hgb_share_{name}"] = hgb_pred_shares[:, i]
        predictions[f"softmax_share_{name}"] = soft_shares[:, i]
    prediction_path = output / f"{target}_predictions.csv.gz"
    pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  **predictions}).to_csv(prediction_path, index=False, compression="gzip")
    radius_path = output / f"{target}_source_radii.json"
    radius_path.write_text(json.dumps(radii, indent=2) + "\n")
    return inner, {"target": target, "selected_alpha": alpha,
                   "inner_score_by_alpha": pd.DataFrame(inner).groupby("alpha").hourly_cif_mae.mean().to_dict(),
                   "optimization": softmax_head["optimization"],
                   "prediction_file": prediction_path.name, "radius_file": radius_path.name}


def score_outer(target: str, prepared: dict, output: Path, selection: dict):
    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024)].copy()
    stored = pd.read_csv(output / selection["prediction_file"])
    radii = json.loads((output / selection["radius_file"]).read_text())
    fuel = hgb_fuel.fuel_for(raw, query)
    truth = hgb_fuel.truth_for(raw, query)
    thermal = fuel[:, :4].sum(axis=1)
    active = thermal > 0
    true_shares = np.divide(fuel[:, :4], thermal[:, None], out=np.zeros((len(fuel), 4)),
                            where=thermal[:, None] > 0)
    rows, share_rows = [], []
    for method, radius in radii.items():
        prediction = stored[method].to_numpy(float)
        error = truth - prediction
        daily = pd.DataFrame({"truth": truth, "prediction": prediction,
                              "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
        centered = daily[["truth", "prediction"]] - daily.groupby("day")[
            ["truth", "prediction"]].transform("mean")
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            absolute = np.abs(error[mask])
            rows.append({"target_country": target.upper(), "selected_alpha": selection["selected_alpha"],
                         "method": method, "month": int(month), "hours": int(mask.sum()),
                         "mae": float(absolute.mean()),
                         "shape_mae": float(np.abs(centered.truth.to_numpy()[mask] -
                                                   centered.prediction.to_numpy()[mask]).mean()),
                         "coverage90": float((absolute <= radius).mean()), "width90": 2 * radius,
                         "interval_score90": float(np.mean(2 * radius +
                             20 * np.maximum(absolute - radius, 0)))})
    for name, prefix in (("hgb", "hgb_share_"), ("softmax", "softmax_share_")):
        cols = [prefix + fuel_name for fuel_name in ("coal", "gas", "oil", "other")]
        predicted = stored[cols].to_numpy(float)
        for idx, fuel_name in enumerate(("coal", "gas", "oil", "other")):
            share_rows.append({"target_country": target.upper(), "model": name,
                               "fuel": fuel_name, "mae": float(np.abs(
                                   true_shares[active, idx] - predicted[active, idx]).mean()),
                               "active_hours": int(active.sum())})
        true_ef = true_shares[active] @ hgb_fuel.FUEL_FACTORS
        pred_ef = predicted[active] @ hgb_fuel.FUEL_FACTORS
        share_rows.append({"target_country": target.upper(), "model": name,
                           "fuel": "thermal_effective_ef", "mae": float(np.abs(
                               true_ef - pred_ef).mean()), "active_hours": int(active.sum())})
    return rows, share_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs, code = tracked_files()
    hashes = {"inputs": {str(p.resolve()): sha(p) for p in inputs},
              "code": {str(p.resolve()): sha(p) for p in code}}
    (output / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    prepared = {}
    for country in COUNTRIES:
        raw, features, _ = loader.prepare_country(country)
        prepared[country] = raw, features

    inner_rows, selections = [], []
    for target in COUNTRIES:
        inner, selection = predict_outer(target, prepared, output)
        inner_rows.extend(inner)
        selections.append(selection)
        print(f"sealed {target.upper()}: alpha={selection['selected_alpha']}", flush=True)
    pd.DataFrame(inner_rows).to_csv(output / "inner_pseudotarget_scores.csv", index=False)
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
            raise RuntimeError(f"prediction/calibration seal mismatch: {seal['file']}")

    monthly_rows, share_rows = [], []
    by_target = {selection["target"]: selection for selection in selections}
    for target in COUNTRIES:
        rows, mechanism = score_outer(target, prepared, output, by_target[target])
        monthly_rows.extend(rows)
        share_rows.extend(mechanism)
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target_country", "method"])[list(METRICS)].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    country.groupby("method")[list(METRICS)].mean().reset_index().to_csv(
        output / "macro_metrics.csv", index=False)
    pd.DataFrame(share_rows).to_csv(output / "mechanism_fuel_share_metrics.csv", index=False)
    summary = {"status": "DEVELOPMENT_STRICT_CN5_SOFTMAX_COMPOSITION_TRANSFER",
               "countries": [c.upper() for c in COUNTRIES],
               "selected_alpha": {s["target"].upper(): s["selected_alpha"] for s in selections},
               "method_macro": country.groupby("method")[list(METRICS)].mean().to_dict(orient="index"),
               "limitations": ["All nine target countries were previously viewed; development only.",
                               "Monthly inputs are reconstructed from revised hourly series to emulate t-2, not as-issued releases.",
                               "Labels are fixed-factor generation-side proxies, not measured CIF.",
                               "Softmax/log-ratio compositional regression is an established model family, not by itself a novel method.",
                               "Target composition and effective-factor diagnostics use labels only after prediction sealing."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(pd.read_csv(output / "macro_metrics.csv").to_string(index=False), flush=True)
    print(f"completed strict CN5 compositional transfer: {output}", flush=True)


if __name__ == "__main__":
    main()
