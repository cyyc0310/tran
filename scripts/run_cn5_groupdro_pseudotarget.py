"""Nested country-group robust training under the strict five-group CIF input budget."""
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
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_GROUPDRO_PSEUDOTARGET_PROTOCOL_20261101.md"
COUNTRIES = list(loader.COUNTRIES)
ETA = 0.5
N_UPDATES = 5
METRICS = ("mae", "shape_mae", "coverage90", "width90", "interval_score90")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inputs_and_code() -> tuple[list[Path], list[Path]]:
    inputs = [PROTOCOL]
    for country in COUNTRIES:
        folder, manifest, _ = loader.six_group.data_locations(country)
        inputs.extend([manifest, folder / f"{country}_2022_2024.json"])
        if country in loader.six_group.OLD_COUNTRIES:
            inputs.append(folder / "audit.json")
    code = [Path(__file__), PROTOCOL, Path(loader.__file__).resolve(),
            Path(loader.six_group.__file__).resolve(), Path(loader.six_group.base.__file__).resolve(),
            Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
            ROOT / "src/transcif/research/monthly_adaptation.py"]
    return sorted(set(inputs)), sorted(set(code))


def write_hashes(output: Path, inputs: list[Path], code: list[Path]) -> dict:
    hashes = {"inputs": {str(path.resolve()): sha(path) for path in inputs},
              "code": {str(path.resolve()): sha(path) for path in code}}
    (output / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    return hashes


def verify_hashes(hashes: dict) -> None:
    for group in ("inputs", "code"):
        for name, expected in hashes[group].items():
            if sha(Path(name)) != expected:
                raise RuntimeError(f"{group} changed during run: {name}")


def cif_y(fuel: np.ndarray) -> np.ndarray:
    return fuel @ model.FACTORS / fuel.sum(axis=1)


def thermal_ef(fuel: np.ndarray) -> np.ndarray:
    thermal = fuel[:, :4].sum(axis=1)
    return np.divide(fuel @ model.FACTORS, thermal, out=np.full(len(fuel), 400.),
                     where=thermal > 0)


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def prepare_all() -> dict:
    prepared = {}
    for country in COUNTRIES:
        raw, features, _ = loader.prepare_country(country)
        prepared[country] = raw, features
    return prepared


def pieces(prepared: dict, countries: list[str], year: int,
           months: range | None = None) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    parts = [prepared[country][1].loc[prepared[country][1].year.eq(year)]
             for country in countries]
    if months is not None:
        parts = [part.loc[part.clock.dt.month.isin(months)] for part in parts]
    keep = [(country, part) for country, part in zip(countries, parts) if len(part)]
    frames = [part for _, part in keep]
    frame = pd.concat(frames, ignore_index=True)
    fuel = np.concatenate([fuel_for(prepared[country][0], part) for country, part in keep])
    coarse = np.concatenate([model.coarse_values(
        prepared[country][0].iloc[part.raw_row.to_numpy()]) for country, part in keep])
    return frame, fuel, coarse


def fit_factor(frame: pd.DataFrame, fuel: np.ndarray, dro: bool) -> tuple:
    """Fit an active-sample country-balanced ERM or fixed GroupDRO estimator."""
    active = fuel[:, :4].sum(axis=1) > 0
    rows = frame.loc[active].copy()
    target = thermal_ef(fuel[active])
    group = rows.country.astype(str).to_numpy()
    countries = sorted(set(group))
    q = {country: 1 / len(countries) for country in countries}
    region_n = rows.groupby(["country", "region"]).region.transform("size").to_numpy(float)
    regions_per_country = rows.groupby("country").region.transform("nunique").to_numpy(float)
    base = 1 / (region_n * regions_per_country)
    base /= base.sum()
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    fitted = None
    for _ in range(N_UPDATES if dro else 1):
        sample_weight = base * np.array([q[country] * len(countries) for country in group])
        fitted = HistGradientBoostingRegressor(**options).fit(
            rows[list(model.HISTORY)], target, sample_weight=sample_weight * len(target))
        if dro:
            prediction = fitted.predict(rows[list(model.HISTORY)])
            loss = {country: float(np.mean(np.abs(
                target[group == country] - prediction[group == country])))
                    for country in countries}
            values = np.array([loss[country] for country in countries])
            scale = max(float(values.mean()), 1e-8)
            logits = np.log(np.array([q[country] for country in countries])) + ETA * values / scale
            logits -= logits.max()
            updated = np.exp(logits)
            updated /= updated.sum()
            q = dict(zip(countries, updated.tolist()))
    return fitted, q


def predict_factor(fitted, frame: pd.DataFrame, profile: np.ndarray) -> np.ndarray:
    factor = np.clip(fitted.predict(frame[list(model.HISTORY)]), 400., 1000.)
    coarse = frame[list(model.GROUPS)].to_numpy() * profile[frame.clock.dt.hour.to_numpy()]
    thermal_share = coarse[:, 0] / coarse.sum(axis=1)
    return factor * thermal_share


def fit_reference(frame: pd.DataFrame, fuel: np.ndarray, coarse: np.ndarray) -> dict:
    return model.fit(frame, fuel, coarse)


def nested_select(target: str, prepared: dict) -> tuple[str, dict, list[dict]]:
    sources = [country for country in COUNTRIES if country != target]
    records = []
    for pseudo in sources:
        training = [country for country in sources if country != pseudo]
        train, train_fuel, train_coarse = pieces(prepared, training, 2022)
        reference = fit_reference(train, train_fuel, train_coarse)
        erm, _ = fit_factor(train, train_fuel, dro=False)
        dro, _ = fit_factor(train, train_fuel, dro=True)
        validation, validation_fuel, _ = pieces(prepared, [pseudo], 2023, range(1, 7))
        truth = cif_y(validation_fuel)
        predictions = {"erm": predict_factor(erm, validation, reference["profile"]),
                       "groupdro": predict_factor(dro, validation, reference["profile"])}
        for candidate, prediction in predictions.items():
            records.append({"outer_target": target, "pseudo_target": pseudo,
                            "candidate": candidate,
                            "hourly_cif_mae": float(np.mean(np.abs(truth - prediction)))})
    table = pd.DataFrame(records)
    means = table.groupby("candidate").hourly_cif_mae.mean().to_dict()
    selected = "groupdro" if means["groupdro"] < means["erm"] else "erm"
    return selected, means, records


def run_outer(target: str, prepared: dict, output: Path) -> tuple[list[dict], dict]:
    sources = [country for country in COUNTRIES if country != target]
    selected, inner_means, inner_records = nested_select(target, prepared)
    train, train_fuel, train_coarse = pieces(prepared, sources, 2022)
    reference = fit_reference(train, train_fuel, train_coarse)
    erm, _ = fit_factor(train, train_fuel, dro=False)
    dro, final_q = fit_factor(train, train_fuel, dro=True)

    calibration, calibration_fuel, _ = pieces(prepared, sources, 2023, range(7, 13))
    calibration_truth = cif_y(calibration_fuel)
    calibration_predictions = {
        "anchor700": 700. * calibration.thermal.to_numpy(),
        "ratio_recomposition": model.predict(reference, calibration)["ratio_recomposition"],
        "factor_history_erm": predict_factor(erm, calibration, reference["profile"]),
        "factor_history_groupdro": predict_factor(dro, calibration, reference["profile"]),
    }
    radii = {name: float(weighted_quantile(
        np.abs(calibration_truth - prediction), model.weights(calibration), .9))
        for name, prediction in calibration_predictions.items()}

    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024)].copy()
    predictions = {
        "anchor700": 700. * query.thermal.to_numpy(),
        "ratio_recomposition": model.predict(reference, query)["ratio_recomposition"],
        "factor_history_erm": predict_factor(erm, query, reference["profile"]),
        "factor_history_groupdro": predict_factor(dro, query, reference["profile"]),
    }
    prediction_path = output / f"{target}_predictions.csv.gz"
    pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  **predictions}).to_csv(prediction_path, index=False, compression="gzip")
    radius_path = output / f"{target}_source_radii.json"
    radius_path.write_text(json.dumps(radii, indent=2) + "\n")
    return inner_records, {"target": target, "selected": selected,
                           "inner_h1_mae": inner_means, "final_group_weights": final_q,
                           "prediction_file": prediction_path.name,
                           "radius_file": radius_path.name}


def score_outer(target: str, prepared: dict, output: Path, selection: dict) -> list[dict]:
    raw, features = prepared[target]
    query = features.loc[features.year.eq(2024)].copy()
    prediction_path = output / selection["prediction_file"]
    stored = pd.read_csv(prediction_path)
    radii = json.loads((output / selection["radius_file"]).read_text())
    fuel = fuel_for(raw, query)
    truth = cif_y(fuel)
    rows = []
    for method in radii:
        prediction = stored[method].to_numpy(float)
        error = truth - prediction
        daily = pd.DataFrame({"truth": truth, "prediction": prediction,
                              "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
        centered = daily[["truth", "prediction"]] - daily.groupby("day")[
            ["truth", "prediction"]].transform("mean")
        radius = float(radii[method])
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            absolute = np.abs(error[mask])
            rows.append({"target": target, "selected_candidate": selection["selected"],
                         "method": method, "month": int(month), "hours": int(mask.sum()),
                         "mae": float(absolute.mean()),
                         "shape_mae": float(np.abs(centered.truth.to_numpy()[mask] -
                                                   centered.prediction.to_numpy()[mask]).mean()),
                         "coverage90": float((absolute <= radius).mean()),
                         "width90": 2 * radius,
                         "interval_score90": float(np.mean(
                             2 * radius + 20 * np.maximum(absolute - radius, 0)))})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    inputs, code = inputs_and_code()
    hashes = write_hashes(output, inputs, code)
    prepared = prepare_all()

    inner_records, selections = [], []
    for target in COUNTRIES:
        inner, selection = run_outer(target, prepared, output)
        inner_records.extend(inner)
        selections.append(selection)
        print(f"sealed {target.upper()} with nested choice {selection['selected']}", flush=True)
    pd.DataFrame(inner_records).to_csv(output / "inner_pseudotarget_scores.csv", index=False)
    prediction_seal = []
    for selection in selections:
        for key, hash_key in (("prediction_file", "prediction_sha256"),
                              ("radius_file", "radius_sha256")):
            path = output / selection[key]
            prediction_seal.append({"target": selection["target"], "file": path.name,
                                    "sha256": sha(path), "kind": key})
    (output / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    (output / "prediction_seal.json").write_text(json.dumps(prediction_seal, indent=2) + "\n")
    verify_hashes(hashes)
    for seal in prediction_seal:
        if sha(output / seal["file"]) != seal["sha256"]:
            raise RuntimeError(f"prediction or calibration seal mismatch: {seal['file']}")

    rows = []
    by_target = {selection["target"]: selection for selection in selections}
    for target in COUNTRIES:
        rows.extend(score_outer(target, prepared, output, by_target[target]))
    monthly = pd.DataFrame(rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target", "method"])[list(METRICS)].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    country.groupby("method")[list(METRICS)].mean().reset_index().to_csv(
        output / "macro_metrics.csv", index=False)
    selection_map = {row["target"]: row["selected"] for row in selections}
    selected_rows = []
    for target, selected in selection_map.items():
        method = "factor_history_groupdro" if selected == "groupdro" else "factor_history_erm"
        selected_rows.append(country[(country.target == target) &
                                     (country.method == method)].iloc[0])
    selected_country = pd.DataFrame(selected_rows)
    selected_country.to_csv(output / "nested_selected_policy_country_metrics.csv", index=False)
    summary = {"status": "DEVELOPMENT_STRICT_CN5_GROUPDRO_NESTED_PSEUDOTARGET",
               "countries": [country.upper() for country in COUNTRIES],
               "selected_policy_by_outer_country": selection_map,
               "selected_policy_macro": {metric: float(selected_country[metric].mean())
                                         for metric in METRICS},
               "method_macro": country.groupby("method")[list(METRICS)].mean().to_dict(orient="index"),
               "limitations": ["All nine countries were previously viewed; development only.",
                               "Monthly features are reconstructed from revised hourly series and emulate t-2, not as-issued official monthly releases.",
                               "Labels are fixed-factor generation-side proxies, not measured CIF.",
                               "GroupDRO is a standard baseline, not a novel method.",
                               "Prediction interval coverage is empirical on the held-out countries, not a target-domain guarantee."]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    verify_hashes(hashes)
    print(json.dumps(summary["selected_policy_macro"], indent=2), flush=True)
    print(f"completed strict CN5 GroupDRO nested pseudotarget experiment: {output}", flush=True)


if __name__ == "__main__":
    main()
