"""Strict-CN5 country-held-out test of level anchoring plus shape transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_energy_charts_nine_country_cn5_loco as loco
from transcif.research import multicountry_cn5 as model
from transcif.research.monthly_adaptation import weighted_quantile

PROTOCOL = ROOT / "docs/CN5_PSEUDOTARGET_LEVEL_SHAPE_PROTOCOL_20261101.md"
METHODS = ("anchor700", "ratio_recomposition", "level_anchor_shape")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def centered_shape(recomposition: np.ndarray, clock: pd.Series) -> np.ndarray:
    day = clock.dt.strftime("%Y-%m-%d").to_numpy()
    return recomposition - pd.Series(recomposition).groupby(day).transform("mean").to_numpy()


def prepare() -> tuple[dict, dict]:
    prepared, audits = {}, {}
    for country in loco.COUNTRIES:
        raw, features, audits[country] = loco.prepare_country(country)
        prepared[country] = raw, features
    return prepared, audits


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else ROOT / args.output
    if out.exists():
        raise FileExistsError(out)
    out.mkdir(parents=True)

    # Freeze audit identity before reading labels for each outer score.
    code_inputs = [Path(__file__).resolve(), PROTOCOL, Path(loco.__file__).resolve(),
                   Path(loco.six_group.__file__).resolve(), Path(loco.six_group.base.__file__).resolve(),
                   Path(loco.model.__file__).resolve(),
                   ROOT / "src/transcif/research/cross_country.py",
                   ROOT / "src/transcif/research/monthly_adaptation.py"]
    code_hashes = {str(p.resolve()): sha(p) for p in code_inputs}
    input_paths = []
    for country in loco.COUNTRIES:
        folder, manifest, _ = loco.six_group.data_locations(country)
        input_paths.extend([manifest, folder / f"{country}_2022_2024.json"])
        if country in loco.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    prepared, audits = prepare()
    (out / "input_audit.json").write_text(json.dumps(audits, indent=2) + "\n")
    results, seals = [], []

    for target in loco.COUNTRIES:
        source_countries = [c for c in loco.COUNTRIES if c != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in source_countries]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([fuel_for(prepared[c][0], p)
                                     for c, p in zip(source_countries, train_parts)])
        train_coarse = np.concatenate([
            model.coarse_values(prepared[c][0].iloc[p.raw_row.to_numpy()])
            for c, p in zip(source_countries, train_parts)])
        fitted = model.fit(train, train_fuel, train_coarse)

        cal_parts = [prepared[c][1].loc[
            prepared[c][1].year.eq(2023) & prepared[c][1].clock.dt.month.le(6)]
            for c in source_countries]
        cal = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([fuel_for(prepared[c][0], p)
                                   for c, p in zip(source_countries, cal_parts)])
        cal_truth = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
        cal_prediction = model.predict(fitted, cal)
        cal_hybrid = cal_prediction["anchor700"] + centered_shape(
            cal_prediction["ratio_recomposition"], cal.clock)
        cal_prediction["level_anchor_shape"] = cal_hybrid
        cal_weights = model.weights(cal)
        radii = {name: float(weighted_quantile(np.abs(cal_truth - cal_prediction[name]),
                                              cal_weights, .9)) for name in METHODS}

        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        point = model.predict(fitted, query)
        point["level_anchor_shape"] = point["anchor700"] + centered_shape(
            point["ratio_recomposition"], query.clock)
        predicted = pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                                  **{k: point[k] for k in METHODS}})
        prediction_path = out / f"{target.upper()}_predictions.csv.gz"
        predicted.to_csv(prediction_path, index=False, compression="gzip")
        seal = {"target": target.upper(), "source_countries": [c.upper() for c in source_countries],
                "prediction_file": prediction_path.name, "prediction_sha256": sha(prediction_path),
                "rows": len(predicted), "calibration_radii": radii}

        # Outer labels are read only after the prediction file hash has been recorded.
        if sha(prediction_path) != seal["prediction_sha256"]:
            raise RuntimeError("prediction seal mismatch")
        truth_fuel = fuel_for(raw, query)
        truth = truth_fuel @ model.FACTORS / truth_fuel.sum(axis=1)
        for method in METHODS:
            error = truth - point[method]
            daily = pd.DataFrame({"truth": truth, "prediction": point[method],
                                  "day": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
            centered = daily[["truth", "prediction"]] - daily.groupby("day")[
                ["truth", "prediction"]].transform("mean")
            radius = radii[method]
            rows = {"target": target.upper(), "method": method, "hours": len(query),
                    "mae": float(np.abs(error).mean()),
                    "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
                    "coverage90": float((np.abs(error) <= radius).mean()),
                    "width90": float(2 * radius),
                    "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(np.abs(error) - radius, 0)))}
            results.append(rows)
        seals.append(seal)
        print(f"scored {target.upper()} with {len(source_countries)} source countries", flush=True)

    if any(sha(Path(p)) != digest for p, digest in code_hashes.items()):
        raise RuntimeError("code or frozen protocol changed during run")
    if any(sha(Path(p)) != digest for p, digest in input_hashes.items()):
        raise RuntimeError("source data changed during run")
    pd.DataFrame(results).to_csv(out / "country_metrics.csv", index=False)
    by_country = pd.DataFrame(results)
    by_country.groupby("method")[[
        "mae", "shape_mae", "coverage90", "width90", "interval_score90"]].mean().to_csv(out / "macro_metrics.csv")
    (out / "prediction_seals.json").write_text(json.dumps(seals, indent=2) + "\n")
    (out / "run_manifest.json").write_text(json.dumps({
        "status": "CN5_PSEUDOTARGET_LEVEL_SHAPE_DEVELOPMENT",
        "protocol_sha256": sha(PROTOCOL), "code_sha256": code_hashes,
        "input_sha256": input_hashes,
        "target_labels_used_before_prediction": False, "target_labels_viewed_after_seal": True,
        "outer_targets": [c.upper() for c in loco.COUNTRIES], "target_year": 2024,
        "train_year": 2022, "calibration": "other-country 2023H1 weighted abs-error q90",
        "budget": "strict five-group t-2 through t-7 monthly shares and calendar",
        "metrics": "fixed-factor generation-side proxy, not measured CIF",
    }, indent=2) + "\n")
    print(pd.read_csv(out / "macro_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
