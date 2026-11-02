#!/usr/bin/env python3
"""Strict-CN5 LOCO mechanism ablations for source-trained thermal fuel shares."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = nine.COUNTRIES
FUELS = model.FINE[:4]
PROTOCOL = ROOT / "docs/CN5_FUELSHARE_COMPONENT_ABLATION_PROTOCOL_20261102.md"
METHODS = ("anchor700", "ratio_recomposition", "factor_history", "direct_history",
           "fuelshare_fixed_prior", "fuelshare_history",
           *(f"ablate_{fuel}" for fuel in FUELS))


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def fit_head(frame: pd.DataFrame, fuel: np.ndarray) -> dict:
    thermal = fuel[:, :4].sum(axis=1)
    active = thermal > 0
    if not active.any():
        raise ValueError("source training fold has no thermal generation")
    truth = fuel[active, :4] / thermal[active, None]
    active_frame = frame.loc[active].reset_index(drop=True)
    weights = model.weights(active_frame)
    weights = weights / weights.mean()
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    heads = []
    for index in range(4):
        head = HistGradientBoostingRegressor(**options)
        head.fit(active_frame[list(model.HISTORY)], truth[:, index], sample_weight=weights)
        heads.append(head)
    prior = np.average(truth, axis=0, weights=weights)
    prior = np.maximum(prior, 0.)
    prior /= prior.sum()
    return {"heads": heads, "prior": prior}


def predict_composition(head: dict, public: pd.DataFrame) -> dict[str, np.ndarray]:
    raw = np.column_stack([learner.predict(public[list(model.HISTORY)])
                           for learner in head["heads"]])
    raw = np.maximum(raw, 0.)
    total = raw.sum(axis=1)
    full = np.divide(raw, total[:, None], out=np.tile(head["prior"], (len(raw), 1)),
                     where=total[:, None] > 0.)
    result = {"fuelshare_history": full}
    result["fuelshare_fixed_prior"] = np.tile(head["prior"], (len(raw), 1))
    for index, fuel in enumerate(FUELS):
        ablated = raw.copy()
        ablated[:, index] = head["prior"][index]
        denom = ablated.sum(axis=1)
        result[f"ablate_{fuel}"] = np.divide(
            ablated, denom[:, None], out=np.tile(head["prior"], (len(raw), 1)),
            where=denom[:, None] > 0.)
    return result


def predict_points(base: dict, head: dict, public: pd.DataFrame) -> tuple[dict, dict]:
    points = model.predict(base, public)
    compositions = predict_composition(head, public)
    q = public[list(model.GROUPS)].to_numpy(float) * base["profile"][
        public.clock.dt.hour.to_numpy()]
    total = q.sum(axis=1)
    thermal_share = np.divide(q[:, 0], total, out=np.zeros(len(q)), where=total > 0.)
    for name, shares in compositions.items():
        points[name] = np.clip((shares @ model.FACTORS[:4]) * thermal_share, 0., 1000.)
    return points, compositions


def score_interval(y: np.ndarray, pred: np.ndarray, radius: float) -> dict:
    error = y - pred
    abs_error = np.abs(error)
    return {"mae": float(abs_error.mean()),
            "coverage90": float((abs_error <= radius).mean()),
            "width90": float(2 * radius),
            "winkler90": float(np.mean(2 * radius + 20 * np.maximum(abs_error - radius, 0.)))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths = sorted(set(input_paths))
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(p.resolve()): sha(p) for p in input_paths}
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (output / "startup_hashes.json").write_text(json.dumps(
        {"input_sha256": input_hashes, "code_sha256": code_hashes}, indent=2) + "\n")

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = raw, features
    (output / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    radii_by_target, composition_predictions, seals = {}, {}, []
    for target in COUNTRIES:
        sources = [code for code in COUNTRIES if code != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, train_parts)])
        train_coarse = np.concatenate([model.coarse_values(prepared[c][0].iloc[
            part.raw_row.to_numpy()]) for c, part in zip(sources, train_parts)])
        if not len(train) or len(train) != len(train_fuel) or len(train) != len(train_coarse):
            raise ValueError(f"empty or misaligned source training data for {target}")
        base = model.fit(train, train_fuel, train_coarse)
        head = fit_head(train, train_fuel)

        cal_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2023)
                                      & prepared[c][1].clock.dt.month.le(6)] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                   for c, part in zip(sources, cal_parts)])
        cal_y = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
        cal_weights = model.weights(calibration)
        cal_points, cal_compositions = predict_points(base, head, calibration)
        radii = {name: float(weighted_quantile(np.abs(cal_y - cal_points[name]),
                                               cal_weights, .9)) for name in METHODS}

        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        points, compositions = predict_points(base, head, query)
        arrays = {f"prediction_{method}": points[method] for method in METHODS}
        for method, shares in compositions.items():
            for index, fuel_name in enumerate(FUELS):
                arrays[f"share_{method}_{fuel_name}"] = shares[:, index]
        pred_path = output / f"{target.upper()}_2024_predictions.csv.gz"
        pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **arrays}).to_csv(pred_path, index=False, compression="gzip")
        radius_path = output / f"{target.upper()}_source_radii.json"
        radius_path.write_text(json.dumps(radii, indent=2) + "\n")
        comp_path = output / f"{target.upper()}_composition_audit.json"
        comp_path.write_text(json.dumps({
            "training_prior": head["prior"].tolist(),
            "calibration_composition_row_sum_max_error": float(
                np.max(np.abs(cal_compositions["fuelshare_history"].sum(axis=1) - 1.))),
            "target_composition_row_sum_max_error": float(
                np.max(np.abs(compositions["fuelshare_history"].sum(axis=1) - 1.))),
            "target_composition_minimum": float(compositions["fuelshare_history"].min()),
        }, indent=2) + "\n")
        seals.append({"target": target.upper(), "prediction_file": pred_path.name,
                      "prediction_sha256": sha(pred_path), "radius_file": radius_path.name,
                      "radius_sha256": sha(radius_path), "composition_audit_file": comp_path.name,
                      "composition_audit_sha256": sha(comp_path), "hours": len(query),
                      "source_countries": [x.upper() for x in sources]})
        radii_by_target[target] = radii
        composition_predictions[target] = compositions
        print(f"sealed {target.upper()}: {len(query)} hours", flush=True)

    seal_path = output / "prediction_seal.json"
    seal_path.write_text(json.dumps(seals, indent=2) + "\n")
    if any(sha(Path(path)) != value for path, value in {**input_hashes, **code_hashes}.items()):
        raise RuntimeError("input, protocol, or code changed during experiment")
    for entry in seals:
        for fkey, hkey in (("prediction_file", "prediction_sha256"),
                           ("radius_file", "radius_sha256"),
                           ("composition_audit_file", "composition_audit_sha256")):
            if sha(output / entry[fkey]) != entry[hkey]:
                raise RuntimeError(f"sealed file hash mismatch: {entry[fkey]}")

    monthly_rows, component_rows = [], []
    for target in COUNTRIES:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        fine = fuel_for(raw, query)
        truth = fine @ model.FACTORS / fine.sum(axis=1)
        thermal = fine[:, :4].sum(axis=1)
        active = thermal > 0
        truth_share = np.divide(fine[:, :4], thermal[:, None], out=np.zeros_like(fine[:, :4]),
                                where=active[:, None])
        source_frame = pd.read_csv(output / f"{target.upper()}_2024_predictions.csv.gz")
        for method in METHODS:
            radius = radii_by_target[target][method]
            pred = source_frame[f"prediction_{method}"].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                daily = pd.DataFrame({"truth": truth[mask], "prediction": pred[mask],
                                      "day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy()})
                centered = daily[["truth", "prediction"]] - daily.groupby("day")[
                    ["truth", "prediction"]].transform("mean")
                scored = score_interval(truth[mask], pred[mask], radius)
                monthly_rows.append({"target_country": target.upper(), "method": method,
                                     "month": int(month), "hours": int(mask.sum()),
                                     **scored,
                                     "shape_mae": float(np.abs(centered.truth - centered.prediction).mean())})
                if method in composition_predictions[target]:
                    predicted_share = np.column_stack([
                        source_frame[f"share_{method}_{fuel_name}"].to_numpy(float)
                        for fuel_name in FUELS])
                    target_active = mask & active
                    if target_active.any():
                        idx = np.flatnonzero(target_active)
                        w = fine[idx, :4].sum(axis=1)
                        comp_err = np.average(np.abs(truth_share[idx] - predicted_share[idx]),
                                              axis=0, weights=w)
                        true_ef = np.divide(fine[idx, :4] @ model.FACTORS[:4], w)
                        pred_ef = predicted_share[idx] @ model.FACTORS[:4]
                        component_rows.append({"target_country": target.upper(), "method": method,
                            "month": int(month), "thermal_hours": int(len(idx)),
                            **{f"{fuel_name}_thermal_weighted_share_mae": float(comp_err[j])
                               for j, fuel_name in enumerate(FUELS)},
                            "thermal_ef_mae": float(np.average(np.abs(true_ef - pred_ef), weights=w))})

    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "winkler90"]
    country = monthly.groupby(["target_country", "method"])[metrics].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    country.groupby("method")[metrics].mean().reset_index().to_csv(
        output / "macro_metrics.csv", index=False)
    components = pd.DataFrame(component_rows)
    components.to_csv(output / "component_metrics.csv", index=False)
    comp_country = components.groupby(["target_country", "method"])[
        [f"{f}_thermal_weighted_share_mae" for f in FUELS] + ["thermal_ef_mae"]].mean().reset_index()
    comp_country.to_csv(output / "component_country_metrics.csv", index=False)
    comp_country.groupby("method").mean(numeric_only=True).reset_index().to_csv(
        output / "component_macro_metrics.csv", index=False)
    (output / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_ONLY_STRICT_CN5_FUEL_COMPONENT_ABLATION",
        "target_year": 2024, "train_year": 2022, "source_calibration": "2023H1",
        "target_budget": "strict CN5 lagged t-2..t-7 monthly shares plus calendar",
        "target_label": "fixed-factor generation-side hourly proxy",
        "all_target_predictions_and_radii_sealed_before_scoring": True,
        "outer_target_labels_previously_viewed": True,
        "confirmation": False,
        "fuel_ablation_definition": "replace exactly one dynamic source-trained thermal fuel share by outer-fold source prior, then renormalize four-part simplex",
    }, indent=2) + "\n")
    print(country.groupby("method")[metrics].mean().to_string())
    print("\nFuel-composition errors (country/month macro):")
    print(comp_country.groupby("method").mean(numeric_only=True).to_string())


if __name__ == "__main__":
    main()
