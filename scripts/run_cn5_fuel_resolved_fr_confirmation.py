"""Frozen strict-CN5 France confirmation for the source-trained fuel-share head."""
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
import run_cn5_fuel_resolved_response as candidate  # noqa: E402
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_FUEL_RESOLVED_FR_CONFIRMATION_PROTOCOL_20261101.md"
FR_DATA = ROOT / "data/energy-charts-v2-panel-r1"
TARGET = "fr"
TARGET_MONTHS = tuple(range(3, 13))
METHODS = candidate.METHODS

# Frozen Energy-Charts schema mapping; missing API series are declared zero by
# the source protocol, while null values in present series remain missing.
nine.six_group.base.TYPE_MAP[TARGET] = {
    "coal": ("fossil_hard_coal",), "gas": ("fossil_gas",),
    "oil": ("fossil_oil",), "other": ("biomass", "waste"),
    "nuclear": ("nuclear",),
    "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
    "wind": ("wind_onshore", "wind_offshore"), "solar": ("solar",),
    "other_renewable": (),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_fr() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = FR_DATA / "acquisition_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = FR_DATA / "fr_2022_2024.json"
    expected_sha = next(row["sha256"] for row in manifest["countries"]
                        if row["country"] == TARGET)
    if sha(path) != expected_sha:
        raise ValueError("raw Energy-Charts SHA mismatch for FR")
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = [row for row in payload["data"]
            if row["timestamp"][:4] in {"2023", "2024"}]
    if not rows:
        raise ValueError("no FR feature/target rows in the frozen 2023–2024 window")
    values = pd.DataFrame([row["values"] for row in rows])
    mapping = nine.six_group.base.TYPE_MAP[TARGET]
    present = set(values.columns)
    missing = sorted({key for group in mapping.values() for key in group} - present)
    if missing:
        raise ValueError(f"mapped FR production series absent: {missing}")
    fine = pd.DataFrame(index=values.index)
    null_audit = {}
    for fuel in model.FINE:
        keys = mapping[fuel]
        if not keys:
            fine[fuel] = 0.
            continue
        block = values[list(keys)].apply(pd.to_numeric, errors="coerce")
        null_audit[fuel] = int(block.isna().sum().sum())
        fine[fuel] = block.sum(axis=1, min_count=len(keys))
    fine["__complete"] = fine[list(model.FINE)].notna().all(axis=1)
    stamps = pd.DatetimeIndex([pd.Timestamp(row["timestamp"]).tz_convert("UTC")
                               for row in rows])
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError("duplicate or unordered FR timestamps")
    if len(stamps) < 2 or not np.all(np.diff(stamps) == pd.Timedelta(hours=1)):
        raise ValueError("FR external target requires a continuous hourly source series")
    frame = fine.loc[fine["__complete"]].drop(columns="__complete").copy()
    frame.index = stamps[fine["__complete"].to_numpy()]
    if (frame[list(model.FINE)] < 0).any().any() or not np.isfinite(
            frame[list(model.FINE)].to_numpy(float)).all():
        raise ValueError("negative or nonfinite mapped FR production")
    frame = frame.loc[frame[list(model.FINE)].sum(axis=1).gt(0)].copy()
    if frame.empty:
        raise ValueError("no complete positive-generation FR hours in 2023–2024")
    frame["time"] = frame.index
    frame["clock"] = frame.index.tz_convert(payload["timezone"]).tz_localize(None)
    frame["country"], frame["region"] = "FR", "FR"
    periods = frame.clock.dt.to_period("M")
    expected = {period: nine.six_group.base.expected_local_hours(
        period, payload["timezone"]) for period in periods.unique()}
    frame["expected_hours"] = periods.map(expected).to_numpy()
    raw = frame.reset_index(drop=True)
    raw_2024 = raw.loc[raw.clock.dt.year.eq(2024)].copy()
    raw_period_2024 = raw_2024.clock.dt.to_period("M")
    observed_hours = raw_2024.groupby(raw_period_2024).size()
    expected_hours = raw_2024.groupby(raw_period_2024).expected_hours.first()
    expected_months = {pd.Period(f"2024-{month:02d}") for month in TARGET_MONTHS}
    observed_hours = observed_hours.loc[observed_hours.index.isin(expected_months)]
    expected_hours = expected_hours.loc[expected_hours.index.isin(expected_months)]
    label_coverage = observed_hours.div(expected_hours).to_dict()
    if set(label_coverage) != expected_months or min(label_coverage.values()) < .99:
        raise ValueError("FR target-label completeness fails the frozen per-month 99% gate")
    audit = {
        "raw_sha256": expected_sha, "timezone": payload["timezone"],
        "unit": payload.get("unit"), "reported_resolution": payload.get("resolution"),
        "feature_target_rows_read": len(rows),
        "complete_positive_hours_retained": len(raw),
        "2024_target_label_coverage_by_month": {
            str(month): float(value) for month, value in sorted(label_coverage.items())
        },
        "null_cells_by_mapped_fuel": null_audit,
        "schema_absent_categories_assumed_zero": [
            key for key, series_keys in mapping.items() if not series_keys
        ],
        "excluded_source_types": sorted(present - {
            key for series_keys in mapping.values() for key in series_keys
        }),
        "excluded_2022_target_rows": True,
    }
    months = model.public_months(raw)
    features = model.public_features(months, raw.clock)
    features["country"], features["region"] = "FR", "FR"
    features["year"] = raw.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(raw))
    usable = features.dropna(subset=list(model.HISTORY)).copy()
    audit["public_input_groups"] = list(model.GROUPS)
    audit["mapped_production_types"] = nine.six_group.base.TYPE_MAP[TARGET]
    audit["other_renewable_public_input"] = "excluded; zero because no FR API series is listed"
    audit["complete_public_months_by_year"] = {
        str(int(year)): int(count)
        for year, count in months.groupby(months.index.year).size().items()
    }
    audit["usable_feature_hours_by_year"] = {
        str(int(year)): int(count)
        for year, count in usable.groupby("year").size().items()
    }
    audit["usable_target_months_2024"] = sorted(
        int(month) for month in usable.loc[usable.year.eq(2024), "clock"].dt.month.unique()
    )
    return raw.reset_index(drop=True), usable, audit


def truth_for(raw: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    fuel = raw.iloc[query.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)
    total = fuel.sum(axis=1)
    if np.any(total <= 0.) or not np.isfinite(fuel).all():
        raise ValueError("France target labels must be positive, finite and complete")
    return fuel @ model.FACTORS / total


def fit_source_models(prepared: dict, sources: list[str]) -> tuple[dict, dict]:
    parts = [prepared[code][1].loc[prepared[code][1].year.eq(2022)] for code in sources]
    train = pd.concat(parts, ignore_index=True)
    train_fuel = np.concatenate([
        candidate.fuel_for(prepared[code][0], part)
        for code, part in zip(sources, parts)
    ])
    train_coarse = np.concatenate([
        model.coarse_values(prepared[code][0].iloc[part.raw_row.to_numpy()])
        for code, part in zip(sources, parts)
    ])
    if len(train) == 0 or len(train) != len(train_fuel) or len(train) != len(train_coarse):
        raise ValueError("empty or misaligned source training cohort")
    return model.fit(train, train_fuel, train_coarse), candidate.fit_fuelshare(train, train_fuel)


def point_predictions(base: dict, fuelshare: dict, query: pd.DataFrame) -> dict[str, np.ndarray]:
    point = model.predict(base, query)
    point["fuelshare_history"] = candidate.predict_fuelshare(fuelshare, base, query)
    return {name: np.asarray(point[name], dtype=float) for name in METHODS}


def source_crossfit_radii(prepared: dict, sources: list[str]) -> dict[str, float]:
    residuals = {method: [] for method in METHODS}
    weights = {method: [] for method in METHODS}
    for pseudo_target in sources:
        pseudo_sources = [code for code in sources if code != pseudo_target]
        base, fuelshare = fit_source_models(prepared, pseudo_sources)
        _, features, _ = prepared[pseudo_target]
        query = features.loc[
            features.year.eq(2023) & features.clock.dt.month.ge(7)
        ].copy()
        if query.empty:
            raise ValueError(f"no H2 pseudo-target calibration hours for {pseudo_target}")
        truth = truth_for(prepared[pseudo_target][0], query)
        predictions = point_predictions(base, fuelshare, query)
        cohort_weights = model.weights(query)
        for method in METHODS:
            residuals[method].append(np.abs(truth - predictions[method]))
            weights[method].append(cohort_weights)

    return {
        method: float(weighted_quantile(np.concatenate(residuals[method]),
                                        np.concatenate(weights[method]), .9))
        for method in METHODS
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    sources = list(nine.COUNTRIES)
    source_prepared = {code: nine.prepare_country(code) for code in sources}
    target_prepared = prepare_fr()
    prepared = dict(source_prepared)
    prepared[TARGET] = target_prepared

    target_raw, target_features, target_audit = target_prepared
    query = target_features.loc[
        target_features.year.eq(2024)
        & target_features.clock.dt.month.isin(TARGET_MONTHS)
    ].copy()
    actual_months = tuple(sorted(int(value) for value in query.clock.dt.month.unique()))
    if actual_months != TARGET_MONTHS:
        raise ValueError(f"France predeclared target months unavailable: {actual_months}")
    expected_hours = int(query.groupby(query.clock.dt.month).size().sum())
    if expected_hours < 7000:
        raise ValueError(f"France confirmation queue too small: {expected_hours} hours")

    input_paths = [FR_DATA / "acquisition_manifest.json",
                   FR_DATA / "fr_2022_2024.json"]
    for code in sources:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths = sorted(set(input_paths))
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(candidate.__file__).resolve(),
                  Path(nine.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(path.resolve()): sha(path) for path in input_paths}
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps({"code": code_hashes,
        "inputs": input_hashes}, indent=2) + "\n", encoding="utf-8")
    (out / "preflight_audit.json").write_text(json.dumps({
        "fr": target_audit,
        "source_countries": [code.upper() for code in sources],
        "target_months": list(actual_months),
        "target_hours": expected_hours,
        "target_labels_scored_before_seal": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    radii = source_crossfit_radii(prepared, sources)
    final_base, final_fuelshare = fit_source_models(prepared, sources)
    predictions = point_predictions(final_base, final_fuelshare, query)
    prediction_path = out / "FR_heldout_predictions.csv.gz"
    pd.DataFrame({
        "time": target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
        **predictions,
    }).to_csv(prediction_path, index=False, compression="gzip")
    radii_path = out / "source_crossfit_radii.json"
    radii_path.write_text(json.dumps(radii, indent=2) + "\n", encoding="utf-8")
    seal = {"target": "FR", "prediction_path": prediction_path.name,
            "prediction_sha256": sha(prediction_path), "radii_path": radii_path.name,
            "radii_sha256": sha(radii_path), "rows": len(query),
            "months": list(actual_months), "source_countries": [x.upper() for x in sources]}
    seal_path = out / "prediction_seal.json"
    seal_path.write_text(json.dumps(seal, indent=2) + "\n", encoding="utf-8")

    if any(sha(Path(path)) != value for path, value in input_hashes.items()):
        raise RuntimeError("source or target input changed before prediction scoring")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("script, protocol or dependency code changed before scoring")
    if sha(prediction_path) != seal["prediction_sha256"] or sha(radii_path) != seal["radii_sha256"]:
        raise RuntimeError("prediction or radius hash mismatch before scoring")

    truth = truth_for(target_raw, query)
    stored = pd.read_csv(prediction_path)
    expected_time = target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()
    if not np.array_equal(stored.time.astype(str).to_numpy(), expected_time):
        raise RuntimeError("sealed France prediction timestamps do not exactly align")
    rows = []
    for method in METHODS:
        prediction = stored[method].to_numpy(float)
        radius = radii[method]
        for month in actual_months:
            mask = query.clock.dt.month.eq(month).to_numpy()
            error = truth[mask] - prediction[mask]
            daily = pd.DataFrame({
                "truth": truth[mask], "prediction": prediction[mask],
                "day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy(),
            })
            centered = daily[["truth", "prediction"]] - daily.groupby("day")[
                ["truth", "prediction"]].transform("mean")
            rows.append({
                "target_country": "FR", "method": method, "month": month,
                "hours": int(mask.sum()), "mae": float(np.abs(error).mean()),
                "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
                "coverage90": float((np.abs(error) <= radius).mean()),
                "width90": float(2. * radius),
                "interval_score90": float(np.mean(2. * radius +
                    20. * np.maximum(np.abs(error) - radius, 0.))),
            })
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    monthly.groupby("method")[metrics].mean().reset_index().to_csv(
        out / "macro_metrics.csv", index=False)
    pooled_rows = []
    for method, group in monthly.groupby("method", sort=True):
        pooled_rows.append({
            "method": method,
            **{metric: float(np.average(group[metric], weights=group.hours))
               for metric in metrics},
            "hours": int(group.hours.sum()),
        })
    pd.DataFrame(pooled_rows).to_csv(out / "hourly_pooled_metrics.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "status": "FROZEN_CN5_FUEL_RESOLVED_FR_EXTERNAL_CONFIRMATION",
        "target": "FR", "target_months": list(actual_months),
        "target_hours": len(query), "target_labels_used_before_seal": False,
        "scored_after_prediction_and_input_hash_verification": True,
        "blind_confirmation": True,
        "response": "fixed-factor generation-side proxy, not official CIF",
        "source_training": "2022 hours from nine pre-existing development countries",
        "calibration": "2023H2 country-held-out source pseudo-target residual q90",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(monthly.groupby("method")[metrics].mean().to_string(), flush=True)
    print(f"scored {len(query)} sealed FR hours over {len(actual_months)} months", flush=True)


if __name__ == "__main__":
    main()
