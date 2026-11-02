"""Strict-CN5 LOCO: predict thermal fuel shares from lagged coarse monthly inputs."""
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
PROTOCOL = ROOT / "docs/CN5_FUEL_RESOLVED_RESPONSE_PROTOCOL_20261101.md"
FUEL_FACTORS = model.FACTORS[:4]
METHODS = ("anchor700", "ratio_recomposition", "macro_recomposition",
           "factor_history", "direct_history", "fuelshare_history")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_fuelshare(frame: pd.DataFrame, fuel: np.ndarray) -> dict:
    """Fit four source-only fuel-share heads and return their simplex prior."""
    thermal = fuel[:, :4].sum(axis=1)
    active = thermal > 0
    if not active.any():
        raise ValueError("fuel-share head requires positive source thermal generation")
    shares = fuel[active, :4] / thermal[active, None]
    active_frame = frame.loc[active].reset_index(drop=True)
    weights = model.weights(active_frame)
    weights = weights / weights.mean()
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    heads = []
    for index in range(4):
        head = HistGradientBoostingRegressor(**options)
        head.fit(frame.loc[active, list(model.HISTORY)], shares[:, index],
                 sample_weight=weights)
        heads.append(head)
    prior = np.average(shares, axis=0, weights=weights)
    prior = prior / prior.sum()
    return {"heads": heads, "prior": prior}


def predict_fuelshare(head: dict, base_models: dict, public: pd.DataFrame) -> np.ndarray:
    """Map predicted thermal fuel composition to CIF using the shared profile."""
    raw = np.column_stack([learner.predict(public[list(model.HISTORY)])
                           for learner in head["heads"]])
    raw = np.maximum(raw, 0.)
    total = raw.sum(axis=1)
    shares = np.divide(raw, total[:, None], out=np.tile(head["prior"], (len(raw), 1)),
                       where=total[:, None] > 0.)
    thermal_ef = shares @ FUEL_FACTORS
    q = public[list(model.GROUPS)].to_numpy(float) * base_models["profile"][
        public.clock.dt.hour.to_numpy()]
    total_coarse = q.sum(axis=1)
    thermal_share = np.divide(q[:, 0], total_coarse,
                              out=np.zeros(len(q), dtype=float), where=total_coarse > 0.)
    return np.clip(thermal_ef * thermal_share, 0., 1000.)


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def truth_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    fuel = fuel_for(raw, part)
    return fuel @ model.FACTORS / fuel.sum(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths = sorted(set(input_paths))
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(),
                  Path(nine.six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(path.resolve()): sha(path) for path in input_paths}
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps({"code": code_hashes,
        "inputs": input_hashes}, indent=2) + "\n", encoding="utf-8")

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False,
        indent=2) + "\n", encoding="utf-8")

    predictions_by_target, radii_by_target, seals = {}, {}, []
    for target in COUNTRIES:
        sources = [country for country in COUNTRIES if country != target]
        train_parts = [prepared[country][1].loc[prepared[country][1].year.eq(2022)]
                       for country in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([fuel_for(prepared[country][0], part)
                                     for country, part in zip(sources, train_parts)])
        train_coarse = np.concatenate([
            model.coarse_values(prepared[country][0].iloc[part.raw_row.to_numpy()])
            for country, part in zip(sources, train_parts)
        ])
        base_models = model.fit(train, train_fuel, train_coarse)
        fuel_head = fit_fuelshare(train, train_fuel)

        cal_parts = [prepared[country][1].loc[
            prepared[country][1].year.eq(2023) & prepared[country][1].clock.dt.month.le(6)
        ] for country in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([fuel_for(prepared[country][0], part)
                                   for country, part in zip(sources, cal_parts)])
        cal_y = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
        cal_weights = model.weights(calibration)
        base_cal_point = model.predict(base_models, calibration)
        cal_point = {name: base_cal_point[name] for name in METHODS
                     if name != "fuelshare_history"}
        cal_point["fuelshare_history"] = predict_fuelshare(fuel_head, base_models, calibration)
        radii = {name: float(weighted_quantile(np.abs(cal_y - pred), cal_weights, .9))
                 for name, pred in cal_point.items()}

        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        base_point = model.predict(base_models, query)
        point = {name: base_point[name] for name in METHODS
                 if name != "fuelshare_history"}
        point["fuelshare_history"] = predict_fuelshare(fuel_head, base_models, query)
        point_path = out / f"{target.upper()}_heldout_predictions.csv.gz"
        pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **point}).to_csv(point_path, index=False, compression="gzip")
        radius_path = out / f"{target.upper()}_source_radii.json"
        radius_path.write_text(json.dumps(radii, indent=2) + "\n", encoding="utf-8")
        seals.append({"target": target.upper(), "path": point_path.name,
                      "sha256": sha(point_path), "radii_path": radius_path.name,
                      "radii_sha256": sha(radius_path), "hours": len(query),
                      "source_countries": [country.upper() for country in sources]})
        predictions_by_target[target] = (query, point)
        radii_by_target[target] = radii
        print(f"sealed {target.upper()}: {len(query)} hours", flush=True)

    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n",
                                                encoding="utf-8")
    if any(sha(Path(path)) != value for path, value in input_hashes.items()):
        raise RuntimeError("input changed during experiment")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("code or protocol changed during experiment")
    if any(sha(out / row[artifact]) != row[hash_field] for row in seals
           for artifact, hash_field in (("path", "sha256"),
                                        ("radii_path", "radii_sha256"))):
        raise RuntimeError("prediction or calibration-radius hash mismatch")

    rows = []
    for target in COUNTRIES:
        raw = prepared[target][0]
        query, point = predictions_by_target[target]
        truth = truth_for(raw, query)
        for method, radius in radii_by_target[target].items():
            prediction = point[method]
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask] - prediction[mask]
                daily = pd.DataFrame({
                    "truth": truth[mask], "prediction": prediction[mask],
                    "day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy(),
                })
                centered = daily[["truth", "prediction"]] - daily.groupby("day")[
                    ["truth", "prediction"]].transform("mean")
                rows.append({
                    "target_country": target.upper(), "method": method, "month": int(month),
                    "hours": int(mask.sum()), "mae": float(np.abs(error).mean()),
                    "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
                    "coverage90": float((np.abs(error) <= radius).mean()),
                    "width90": float(2 * radius),
                    "interval_score90": float(np.mean(2 * radius +
                        20 * np.maximum(np.abs(error) - radius, 0))),
                })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = monthly.groupby(["target_country", "method"])[metrics].mean().reset_index()
    country.to_csv(out / "country_metrics.csv", index=False)
    country.groupby("method")[metrics].mean().reset_index().to_csv(
        out / "macro_metrics.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_STRICT_CN5_FUEL_RESOLVED_RESPONSE",
        "countries": [country.upper() for country in COUNTRIES], "target_year": 2024,
        "training_year": 2022, "source_calibration": "2023H1 country-equal pooled absolute residual q90",
        "target_budget": "lagged t-2..t-7 thermal, hydro, nuclear, wind and solar shares plus calendar",
        "other_renewable": "excluded from target features; retained only in fixed-factor proxy denominator",
        "response": "fixed-factor generation-side proxy, not measured CIF",
        "target_labels_used_before_prediction_seal": False,
        "blind_confirmation": False,
        "fuelshare_head": "four source-trained HGB thermal fuel-share heads projected to nonnegative simplex",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"completed strict-CN5 fuel-resolved LOCO: {len(monthly)} monthly rows", flush=True)


if __name__ == "__main__":
    main()
