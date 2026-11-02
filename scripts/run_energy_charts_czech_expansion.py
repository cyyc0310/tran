"""Run a six-country LOCO development benchmark with Czechia as a source."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_loco as base  # noqa: E402
from transcif.research.multicountry import FACTORS, FINE, HISTORY, predict, weights  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = (*base.COUNTRIES, "cz")
OLD_DATA = ROOT / "data/energy-charts-v2-panel-r1"
NEW_DATA = ROOT / "data/energy-charts-v2-panel-r2"
OLD_MANIFEST = OLD_DATA / "acquisition_manifest.json"
OLD_AUDIT = OLD_DATA / "audit.json"
CZ_RAW = NEW_DATA / "cz_2022_2024.json"
CZ_MANIFEST = NEW_DATA / "cz_acquisition_manifest.json"
PROTOCOL = ROOT / "docs/ENERGY_CHARTS_CZECH_EXPANSION_PROTOCOL_20261101.md"
BASELINE = ROOT / "docs/evidence/energy-charts-loco-capacity-r3/results"

base.TYPE_MAP["cz"] = {
    "coal": ("fossil_brown_coal_lignite", "fossil_hard_coal"),
    "gas": ("fossil_gas", "fossil_coal_derived_gas"),
    "oil": ("fossil_oil",),
    "other": ("biomass", "waste", "others"),
    "nuclear": ("nuclear",),
    "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
    "wind": ("wind_onshore",),
    "solar": ("solar",),
    "other_renewable": ("other_renewables",),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_country(code: str, manifest: dict):
    base.DATA = NEW_DATA if code == "cz" else OLD_DATA
    raw, audit = base.load_country(code, manifest)
    months = base.public_months(raw)
    features = base.public_features(months, raw.clock)
    features["country"], features["region"] = code.upper(), code.upper()
    features["year"] = raw.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(raw))
    usable = features.dropna(subset=list(HISTORY)).copy()
    audit["complete_public_months_by_year"] = {
        str(int(year)): int(count)
        for year, count in months.groupby(months.index.year).size().items()
    }
    audit["usable_feature_hours_by_year"] = {
        str(int(year)): int(count)
        for year, count in usable.groupby("year").size().items()
    }
    return raw.reset_index(drop=True), usable, audit


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(FINE)].to_numpy(float)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_dir = output / "input"
    input_dir.mkdir()
    shutil.copy2(CZ_RAW, input_dir / CZ_RAW.name)
    shutil.copy2(CZ_MANIFEST, input_dir / CZ_MANIFEST.name)

    old_manifest = json.loads(OLD_MANIFEST.read_text(encoding="utf-8"))
    cz_manifest = json.loads(CZ_MANIFEST.read_text(encoding="utf-8"))
    manifests = {
        code: old_manifest for code in base.COUNTRIES
    }
    manifests["cz"] = {"countries": [{"country": "cz", "sha256": cz_manifest["sha256"]}]}
    input_paths = [OLD_MANIFEST, OLD_AUDIT, CZ_RAW, CZ_MANIFEST]
    input_paths += [OLD_DATA / f"{code}_2022_2024.json" for code in base.COUNTRIES]
    input_hashes = {str(path.resolve()): sha(path) for path in input_paths}
    for code in base.COUNTRIES:
        expected = next(row["sha256"] for row in old_manifest["countries"] if row["country"] == code)
        if input_hashes[str((OLD_DATA / f"{code}_2022_2024.json").resolve())] != expected:
            raise ValueError(f"old source SHA differs from acquisition manifest: {code}")
    if input_hashes[str(CZ_RAW.resolve())] != cz_manifest["sha256"]:
        raise ValueError("Czech raw response differs from acquisition manifest")

    code_paths = [
        Path(__file__).resolve(), Path(base.__file__).resolve(),
        ROOT / "src/transcif/research/multicountry.py",
        ROOT / "src/transcif/research/cross_country.py",
        ROOT / "src/transcif/research/monthly_adaptation.py", PROTOCOL,
    ]
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    (output / "startup_hashes.json").write_text(
        json.dumps({"code": code_hashes, "inputs": input_hashes}, indent=2) + "\n",
        encoding="utf-8",
    )

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = prepare_country(code, manifests[code])
        prepared[code] = (raw, features)
    (output / "input_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    predictions, seals = {}, []
    for target in COUNTRIES:
        sources = [code for code in COUNTRIES if code != target]
        training_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        training = pd.concat(training_parts, ignore_index=True)
        train_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, training_parts)])
        train_coarse = np.concatenate([base.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
                                       for c, part in zip(sources, training_parts)])
        if not len(training) or len(training) != len(train_fuel) or len(train_fuel) != len(train_coarse):
            raise ValueError(f"empty or misaligned source training cohort for {target}")
        models = base.fit(training, train_fuel, train_coarse)
        cal_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2023)
                                      & prepared[c][1].clock.dt.month.le(6)] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                   for c, part in zip(sources, cal_parts)])
        cal_y = cal_fuel @ FACTORS / cal_fuel.sum(axis=1)
        cal_weights = weights(calibration)
        cal_predictions = predict(models, calibration)
        radii = {name: weighted_quantile(np.abs(cal_y - value), cal_weights, 0.9)
                 for name, value in cal_predictions.items()}

        target_raw, target_features = prepared[target]
        query = target_features.loc[target_features.year.eq(2024)].copy()
        point = predict(models, query)
        if not len(query) or any(not np.isfinite(value).all() for value in point.values()):
            raise ValueError(f"empty or nonfinite target prediction for {target}")
        path = output / f"{target.upper()}_heldout_predictions.csv.gz"
        pd.DataFrame({"time": target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                      **point}).to_csv(path, index=False, compression="gzip")
        seals.append({"target": target.upper(), "file": path.name,
                      "sha256": sha(path), "hours": len(query)})
        predictions[target] = (query, path, radii)
        print(f"{target.upper()} predictions sealed: {len(query)} hours", flush=True)
    (output / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n", encoding="utf-8")
    if any(sha(Path(path)) != value for path, value in input_hashes.items()):
        raise RuntimeError("frozen input changed during run")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("frozen code/protocol changed during run")

    rows = []
    for target, (query, path, radii) in predictions.items():
        truth_fuel = fuel_for(prepared[target][0], query)
        truth = truth_fuel @ FACTORS / truth_fuel.sum(axis=1)
        if sha(path) != next(item["sha256"] for item in seals if item["target"] == target.upper()):
            raise RuntimeError(f"prediction file changed before scoring: {target}")
        forecast = pd.read_csv(path)
        for method, radius in radii.items():
            point = forecast[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                daily = pd.DataFrame({"day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy(),
                                      "truth": truth[mask], "prediction": point[mask]})
                centered = daily[["truth", "prediction"]] - daily.groupby("day")[["truth", "prediction"]].transform("mean")
                err = truth[mask] - point[mask]
                rows.append({"target_country": target.upper(), "method": method, "month": int(month),
                             "hours": int(mask.sum()), "mae": float(np.abs(err).mean()),
                             "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
                             "coverage90": float((np.abs(err) <= radius).mean()), "width90": float(2 * radius),
                             "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(np.abs(err) - radius, 0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = monthly.groupby(["target_country", "method"])[metrics].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    country.groupby("method")[metrics].mean().reset_index().to_csv(output / "macro_metrics_all_six.csv", index=False)
    old = country.loc[country.target_country.isin([c.upper() for c in base.COUNTRIES])]
    old.groupby("method")[metrics].mean().reset_index().to_csv(
        output / "macro_metrics_original_five_targets.csv", index=False
    )
    baseline = pd.read_csv(BASELINE / "country_metrics.csv")
    baseline = baseline.loc[baseline.target_country.isin([c.upper() for c in base.COUNTRIES])]
    paired = old.merge(baseline, on=["target_country", "method"], suffixes=("_with_cz", "_five_source"), validate="one_to_one")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[f"{metric}_with_cz"] - paired[f"{metric}_five_source"]
    paired.to_csv(output / "paired_original_five_targets.csv", index=False)
    status = {
        "status": "DEVELOPMENT_SIX_COUNTRY_LOCO_WITH_CZECHIA",
        "countries": [c.upper() for c in COUNTRIES],
        "training_year": 2022, "calibration_period": "2023-01 through 2023-06",
        "target_year": 2024, "source_attribution": "Energy-Charts.info, CC BY 4.0",
        "target_hourly_inputs_used": False,
        "target_monthly_inputs": "current revised hourly aggregates with simulated t-2 delay",
        "historical_as_issued_vintages": False, "blind_confirmation": False,
        "labels": "fixed-factor generation proxy, not measured CIF",
        "comparison": "six-source LOCO; original five target countries paired with frozen five-source baseline",
    }
    (output / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_files = [path for path in output.iterdir() if path.is_file()]
    provenance = {
        "input_sha256": input_hashes, "code_sha256": code_hashes,
        "output_sha256": {path.name: sha(path) for path in output_files},
        "comparison_baseline_sha256": {path.name: sha(path) for path in [
            BASELINE / "country_metrics.csv", *[BASELINE / f"{c.upper()}_heldout_predictions.csv.gz" for c in base.COUNTRIES]]},
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print("Six-country macro:")
    print(country.groupby("method")[metrics].mean().to_string())
    print("Paired delta on original five targets (with CZ minus baseline):")
    print(paired.groupby("method")[[f"delta_{m}" for m in metrics]].mean().to_string())


if __name__ == "__main__":
    main()
