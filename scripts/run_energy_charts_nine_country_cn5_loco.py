"""Strict five-group, China-budget nine-country held-out benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_loco as six_group  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = six_group.COUNTRIES
PROTOCOL = ROOT / "docs/ENERGY_CHARTS_NINE_COUNTRY_CN5_PROTOCOL_20261101.md"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_country(code: str):
    folder, _, manifest = six_group.data_locations(code)
    six_group.base.DATA = folder
    raw, audit = six_group.base.load_country(code, manifest)
    months = model.public_months(raw)
    features = model.public_features(months, raw.clock)
    features["country"], features["region"] = code.upper(), code.upper()
    features["year"] = raw.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(raw))
    usable = features.dropna(subset=list(model.HISTORY)).copy()
    audit["public_input_groups"] = list(model.GROUPS)
    audit["other_renewable_public_input"] = "excluded; retained only in the hourly proxy label denominator"
    audit["complete_public_months_by_year"] = {
        str(int(y)): int(n) for y, n in months.groupby(months.index.year).size().items()
    }
    audit["usable_feature_hours_by_year"] = {
        str(int(y)): int(n) for y, n in usable.groupby("year").size().items()
    }
    return raw.reset_index(drop=True), usable, audit


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest_path, _ = six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(six_group.__file__).resolve(),
                  Path(six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (output / "startup_hashes.json").write_text(
        json.dumps({"code": code_hashes, "inputs": input_hashes}, indent=2) + "\n",
        encoding="utf-8",
    )

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = prepare_country(code)
        prepared[code] = raw, features
    (output / "input_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    seals, radii_by_target = [], {}
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, train_parts)])
        train_coarse = np.concatenate([
            model.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
            for c, part in zip(sources, train_parts)
        ])
        if not len(train) or len(train) != len(train_fuel) or len(train) != len(train_coarse):
            raise ValueError(f"empty or misaligned five-group training cohort for {target}")
        models = model.fit(train, train_fuel, train_coarse)

        cal_parts = [prepared[c][1].loc[
            prepared[c][1].year.eq(2023) & prepared[c][1].clock.dt.month.le(6)
        ] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                   for c, part in zip(sources, cal_parts)])
        cal_y = cal_fuel @ model.FACTORS / cal_fuel.sum(axis=1)
        cal_weights = model.weights(calibration)
        cal_predictions = model.predict(models, calibration)
        radii = {name: float(weighted_quantile(np.abs(cal_y - value), cal_weights, .9))
                 for name, value in cal_predictions.items()}

        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        point = model.predict(models, query)
        point_path = output / f"{target.upper()}_heldout_predictions.csv.gz"
        pd.DataFrame({
            "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(), **point,
        }).to_csv(point_path, index=False, compression="gzip")
        radius_path = output / f"{target.upper()}_source_radii.json"
        radius_path.write_text(json.dumps(radii, indent=2) + "\n", encoding="utf-8")
        seals.append({"target": target.upper(), "path": point_path.name,
                      "sha256": sha(point_path), "radii_path": radius_path.name,
                      "radii_sha256": sha(radius_path), "hours": len(query),
                      "source_countries": [c.upper() for c in sources]})
        radii_by_target[target] = radii
        print(f"sealed {target.upper()}: {len(query)} hours; strict CN5 inputs", flush=True)

    (output / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n", encoding="utf-8")
    if any(sha(Path(p)) != value for p, value in input_hashes.items()):
        raise RuntimeError("input changed during experiment")
    if any(sha(Path(p)) != value for p, value in code_hashes.items()):
        raise RuntimeError("code or protocol changed during experiment")
    if any(sha(output / row[artifact]) != row[hash_field] for row in seals
           for artifact, hash_field in (("path", "sha256"),
                                        ("radii_path", "radii_sha256"))):
        raise RuntimeError("sealed prediction or calibration-radius hash mismatch")

    rows = []
    for target in COUNTRIES:
        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        fuel = fuel_for(raw, query)
        truth = fuel @ model.FACTORS / fuel.sum(axis=1)
        stored = pd.read_csv(output / f"{target.upper()}_heldout_predictions.csv.gz")
        for method, radius in radii_by_target[target].items():
            prediction = stored[method].to_numpy(float)
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
                    "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(np.abs(error) - radius, 0))),
                })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = monthly.groupby(["target_country", "method"])[metrics].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    country.groupby("method")[metrics].mean().reset_index().to_csv(
        output / "macro_metrics.csv", index=False
    )
    (output / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_NINE_COUNTRY_STRICT_CN5_LOCO",
        "countries": [c.upper() for c in COUNTRIES], "target_year": 2024,
        "training_year": 2022, "source_calibration": "2023H1",
        "target_budget": "thermal, hydro, nuclear, wind, solar monthly shares t-2 through t-7 plus calendar",
        "other_renewable": "excluded from public input; retained in fixed-factor hourly proxy denominator",
        "labels": "fixed-factor generation-side proxy, not measured CIF",
        "target_labels_used_before_prediction_seal": False,
        "final_blind_confirmation": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"completed strict five-group nine-country LOCO: {len(monthly)} monthly rows", flush=True)


if __name__ == "__main__":
    main()
