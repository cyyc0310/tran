"""Nine-country LOCO transfer benchmark under lagged five-category month inputs."""
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
import run_energy_charts_loco as base  # noqa: E402
from transcif.research.multicountry import FACTORS, FINE, HISTORY, predict, weights  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = ("nl", "es", "be", "at", "se", "cz", "sk", "hu", "si")
OLD_COUNTRIES = ("nl", "es", "be", "at", "se")
PROTOCOL = ROOT / "docs/ENERGY_CHARTS_NINE_COUNTRY_LOCO_PROTOCOL_20261101.md"
OLD_DATA = ROOT / "data/energy-charts-v2-panel-r1"
CZ_DATA = ROOT / "data/energy-charts-v2-panel-r2"
NEW_DATA = ROOT / "data/energy-charts-v2-panel-r3"
OLD_RESULTS = ROOT / "results/energy-charts-loco-r1"

base.TYPE_MAP.update({
    "cz": {
        "coal": ("fossil_brown_coal_lignite", "fossil_hard_coal"),
        "gas": ("fossil_gas", "fossil_coal_derived_gas"), "oil": ("fossil_oil",),
        "other": ("biomass", "waste", "others"), "nuclear": ("nuclear",),
        "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
        "wind": ("wind_onshore",), "solar": ("solar",),
        "other_renewable": ("other_renewables",),
    },
    "sk": {
        "coal": ("fossil_brown_coal_lignite", "fossil_hard_coal"),
        "gas": ("fossil_gas",), "oil": ("fossil_oil",),
        "other": ("biomass", "others"), "nuclear": ("nuclear",),
        "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
        "wind": ("wind_onshore",), "solar": ("solar",),
        "other_renewable": ("other_renewables",),
    },
    "hu": {
        "coal": ("fossil_brown_coal_lignite", "fossil_hard_coal"),
        "gas": ("fossil_gas",), "oil": ("fossil_oil",),
        "other": ("biomass", "waste", "others"), "nuclear": ("nuclear",),
        "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
        "wind": ("wind_onshore",), "solar": ("solar",),
        "other_renewable": ("other_renewables", "geothermal"),
    },
    "si": {
        "coal": ("fossil_brown_coal_lignite",), "gas": ("fossil_gas",),
        "oil": ("fossil_oil",), "other": ("biomass", "waste"),
        "nuclear": ("nuclear",), "hydro": ("hydro_run_of_river",),
        "wind": ("wind_onshore",), "solar": ("solar",),
        "other_renewable": (),
    },
})


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def data_locations(code: str) -> tuple[Path, Path, dict]:
    if code in OLD_COUNTRIES:
        folder = OLD_DATA
        manifest_path = folder / "acquisition_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif code == "cz":
        folder = CZ_DATA
        manifest_path = folder / "cz_acquisition_manifest.json"
        row = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest = {"countries": [{"country": code, "sha256": row["sha256"]}]}
    else:
        folder = NEW_DATA
        manifest_path = folder / "acquisition_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return folder, manifest_path, manifest


def prepare_country(code: str) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    folder, _, manifest = data_locations(code)
    base.DATA = folder
    raw, audit = base.load_country(code, manifest)
    audit["known_source_schema_gaps"] = {
        "sk": ["waste is not separately reported; omitted from the mapped other-combustion proxy"],
        "si": ["other_renewables is not separately reported; mapped as zero"],
    }.get(code, [])
    months = base.public_months(raw)
    public = base.public_features(months, raw.clock)
    public["country"], public["region"] = code.upper(), code.upper()
    public["year"] = raw.clock.dt.year.to_numpy()
    public["raw_row"] = np.arange(len(raw))
    usable = public.dropna(subset=list(HISTORY)).copy()
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    manifests = {}
    for code in COUNTRIES:
        folder, manifest_path, manifest = data_locations(code)
        manifests[code] = manifest
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths.append(OLD_RESULTS / "prediction_seal.json")
    input_paths.extend(OLD_RESULTS / f"{code.upper()}_heldout_predictions.csv.gz"
                       for code in OLD_COUNTRIES)
    input_hashes = {str(path.resolve()): sha(path) for path in sorted(set(input_paths))}
    code_paths = [
        Path(__file__).resolve(), Path(base.__file__).resolve(), PROTOCOL,
        ROOT / "src/transcif/research/multicountry.py",
        ROOT / "src/transcif/research/cross_country.py",
        ROOT / "src/transcif/research/monthly_adaptation.py",
    ]
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    (output / "startup_hashes.json").write_text(
        json.dumps({"code": code_hashes, "inputs": input_hashes}, indent=2) + "\n",
        encoding="utf-8",
    )

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = prepare_country(code)
        prepared[code] = (raw, features)
    (output / "input_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    seals, query_by_target, radii_by_target = [], {}, {}
    for target in COUNTRIES:
        sources = [code for code in COUNTRIES if code != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, train_parts)])
        train_coarse = np.concatenate([
            base.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
            for c, part in zip(sources, train_parts)
        ])
        if not len(train) or len(train) != len(train_fuel) or len(train_fuel) != len(train_coarse):
            raise ValueError(f"empty or misaligned training cohort for {target}")
        models = base.fit(train, train_fuel, train_coarse)

        cal_parts = [prepared[c][1].loc[
            prepared[c][1].year.eq(2023) & prepared[c][1].clock.dt.month.le(6)
        ] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([fuel_for(prepared[c][0], part)
                                   for c, part in zip(sources, cal_parts)])
        cal_y = cal_fuel @ FACTORS / cal_fuel.sum(axis=1)
        cal_weights = weights(calibration)
        cal_prediction = predict(models, calibration)
        radii = {name: float(weighted_quantile(np.abs(cal_y - value), cal_weights, .9))
                 for name, value in cal_prediction.items()}

        target_raw, target_features = prepared[target]
        query = target_features.loc[target_features.year.eq(2024)].copy()
        point = predict(models, query)
        if not len(query) or any(not np.isfinite(value).all() for value in point.values()):
            raise ValueError(f"empty or nonfinite prediction for {target}")
        path = output / f"{target.upper()}_heldout_predictions.csv.gz"
        pd.DataFrame({
            "time": target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
            **point,
        }).to_csv(path, index=False, compression="gzip")
        seals.append({"target": target.upper(), "path": path.name,
                      "sha256": sha(path), "hours": len(query),
                      "source_countries": [c.upper() for c in sources]})
        query_by_target[target] = query
        radii_by_target[target] = radii
        radii_path = output / f"{target.upper()}_source_radii.json"
        radii_path.write_text(
            json.dumps(radii, indent=2) + "\n", encoding="utf-8"
        )
        seals[-1]["radii_path"] = radii_path.name
        seals[-1]["radii_sha256"] = sha(radii_path)
        print(f"sealed {target.upper()}: {len(query)} hours from {len(sources)} source countries",
              flush=True)

    (output / "prediction_seal.json").write_text(
        json.dumps(seals, indent=2) + "\n", encoding="utf-8"
    )
    if any(sha(Path(path)) != expected for path, expected in input_hashes.items()):
        raise RuntimeError("input changed during experiment")
    if any(sha(Path(path)) != expected for path, expected in code_hashes.items()):
        raise RuntimeError("code or protocol changed during experiment")

    rows = []
    for target in COUNTRIES:
        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        truth_fuel = raw.iloc[query.raw_row.to_numpy()][list(FINE)].to_numpy(float)
        truth = truth_fuel @ FACTORS / truth_fuel.sum(axis=1)
        stored = pd.read_csv(output / f"{target.upper()}_heldout_predictions.csv.gz")
        for method, radius in radii_by_target[target].items():
            prediction = stored[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask] - prediction[mask]
                daily = pd.DataFrame({"truth": truth[mask], "prediction": prediction[mask],
                                      "day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy()})
                center = daily[["truth", "prediction"]] - daily.groupby("day")[
                    ["truth", "prediction"]].transform("mean")
                rows.append({
                    "target_country": target.upper(), "method": method, "month": int(month),
                    "hours": int(mask.sum()), "mae": float(np.abs(error).mean()),
                    "shape_mae": float(np.abs(center.truth - center.prediction).mean()),
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

    # Pair the five original targets against their already-sealed five-source runs.
    paired = []
    for target in OLD_COUNTRIES:
        old_path = OLD_RESULTS / f"{target.upper()}_heldout_predictions.csv.gz"
        old = pd.read_csv(old_path)
        new = pd.read_csv(output / f"{target.upper()}_heldout_predictions.csv.gz")
        raw, all_features = prepared[target]
        query = all_features.loc[all_features.year.eq(2024)].copy()
        query_rows = raw.iloc[query.raw_row.to_numpy()]
        truth_fuel = query_rows[list(FINE)].to_numpy(float)
        truth = pd.DataFrame({"time": query_rows.time.astype(str).to_numpy(),
                              "truth": truth_fuel @ FACTORS / truth_fuel.sum(axis=1)})
        merged = old.merge(new, on="time", suffixes=("_old", "_new"), validate="one_to_one")
        merged = merged.merge(truth, on="time", validate="one_to_one")
        for method in old.columns.drop("time"):
            eo = np.abs(merged[f"{method}_old"].to_numpy() - merged.truth.to_numpy())
            en = np.abs(merged[f"{method}_new"].to_numpy() - merged.truth.to_numpy())
            paired.append({"target_country": target.upper(), "method": method,
                           "common_hours": len(merged), "old_five_source_mae": float(eo.mean()),
                           "nine_country_source_mae": float(en.mean()),
                           "mae_delta": float(en.mean() - eo.mean())})
    pd.DataFrame(paired).to_csv(output / "old_target_source_expansion_pair.csv", index=False)
    (output / "status.json").write_text(json.dumps({
        "status": "DEVELOPMENT_NINE_COUNTRY_ENERGY_CHARTS_LOCO",
        "countries": [c.upper() for c in COUNTRIES], "target_year": 2024,
        "training_year": 2022, "source_calibration": "2023H1",
        "target_budget": "five coarse monthly generation shares t-2 through t-7 plus calendar",
        "labels": "fixed-factor generation-side proxy, not measured CIF",
        "monthly_inputs": "current revised hourly generation aggregated to months; lag simulation, not as-issued",
        "target_labels_used_before_prediction_seal": False,
        "all_countries_previously_scored": False,
        "final_blind_confirmation": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"completed nine-country LOCO: {len(monthly)} monthly rows", flush=True)


if __name__ == "__main__":
    main()
