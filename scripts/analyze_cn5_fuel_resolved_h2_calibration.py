"""Post-hoc H2 cross-fitted source calibration for sealed strict-CN5 points."""
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
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as candidate  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = nine.COUNTRIES
PROTOCOL = ROOT / "docs/CN5_FUEL_RESOLVED_H2_CALIBRATION_PROTOCOL_20261101.md"
SEALED = ROOT / "results/cn5-fuel-resolved-response-r1"
METHODS = candidate.METHODS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    if not (SEALED / "prediction_seal.json").is_file():
        raise FileNotFoundError("strict-CN5 sealed point predictions not found")

    input_paths = [PROTOCOL, SEALED / "prediction_seal.json",
                   SEALED / "startup_hashes.json"]
    for code in COUNTRIES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
        input_paths.append(SEALED / f"{code.upper()}_heldout_predictions.csv.gz")
    input_paths = sorted(set(input_paths))
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(candidate.__file__).resolve(),
                  Path(nine.__file__).resolve(), Path(nine.six_group.__file__).resolve(),
                  Path(nine.six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(path.resolve()): sha(path) for path in input_paths}
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps({"code": code_hashes,
        "inputs": input_hashes}, indent=2) + "\n", encoding="utf-8")

    prepared = {}
    for code in COUNTRIES:
        raw, features, _audit = nine.prepare_country(code)
        prepared[code] = raw, features

    radii_by_target, calibration_records = {}, []
    for target in COUNTRIES:
        sources = [country for country in COUNTRIES if country != target]
        residuals = {method: [] for method in METHODS}
        calibration_frames = []
        for pseudo in sources:
            pseudo_sources = [country for country in sources if country != pseudo]
            train_parts = [prepared[country][1].loc[
                prepared[country][1].year.eq(2022)] for country in pseudo_sources]
            train = pd.concat(train_parts, ignore_index=True)
            train_fuel = np.concatenate([candidate.fuel_for(prepared[country][0], part)
                                         for country, part in zip(pseudo_sources, train_parts)])
            train_coarse = np.concatenate([
                model.coarse_values(prepared[country][0].iloc[part.raw_row.to_numpy()])
                for country, part in zip(pseudo_sources, train_parts)
            ])
            base_models = model.fit(train, train_fuel, train_coarse)
            fuel_head = candidate.fit_fuelshare(train, train_fuel)
            query = prepared[pseudo][1].loc[
                prepared[pseudo][1].year.eq(2023) &
                prepared[pseudo][1].clock.dt.month.gt(6)
            ].copy()
            if query.empty:
                raise ValueError(f"empty H2 calibration cohort for {pseudo}")
            truth = candidate.truth_for(prepared[pseudo][0], query)
            point = model.predict(base_models, query)
            point["fuelshare_history"] = candidate.predict_fuelshare(
                fuel_head, base_models, query)
            for method in METHODS:
                residuals[method].append(np.abs(truth - point[method]))
            calibration_frames.append(query)

        calibration = pd.concat(calibration_frames, ignore_index=True)
        weights = model.weights(calibration)
        radii = {method: float(weighted_quantile(np.concatenate(values), weights, .9))
                 for method, values in residuals.items()}
        radii_by_target[target] = radii
        calibration_records.append({"outer_target": target.upper(),
                                    "pseudo_targets": [c.upper() for c in sources],
                                    "hours": int(len(calibration)), "radii90": radii})

    (out / "crossfit_h2_radii.json").write_text(json.dumps(calibration_records, indent=2) + "\n",
                                                 encoding="utf-8")
    rows = []
    for target in COUNTRIES:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        truth = candidate.truth_for(raw, query)
        point = pd.read_csv(SEALED / f"{target.upper()}_heldout_predictions.csv.gz")
        expected_time = raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()
        if len(point) != len(query) or not np.array_equal(
                point["time"].astype(str).to_numpy(), expected_time):
            raise ValueError(f"sealed target timestamp/row mismatch for {target}")
        for method in METHODS:
            radius = radii_by_target[target][method]
            pred = point[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask] - pred[mask]
                daily = pd.DataFrame({
                    "truth": truth[mask], "prediction": pred[mask],
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
        "status": "POSTHOC_STRICT_CN5_FUEL_RESOLVED_H2_CROSSFIT_CALIBRATION",
        "point_predictions": "r1 sealed and unchanged",
        "calibration": "2023H2 cross-fitted by source country; 2022 models exclude pseudo target",
        "target_year": 2024, "target_labels_previously_viewed": True,
        "not_blind_confirmation": True,
    }, indent=2) + "\n", encoding="utf-8")
    print(pd.read_csv(out / "macro_metrics.csv").to_string(index=False))


if __name__ == "__main__":
    main()
