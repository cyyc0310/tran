"""Test whether a Serbian source domain changes six-country cold-start transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_loco as base  # noqa: E402
from transcif.research.multicountry import FACTORS, FINE, HISTORY, predict, weights  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

OLD_COUNTRIES = base.COUNTRIES
COUNTRIES = (*OLD_COUNTRIES, "rs")
RS_DATA = ROOT / "data/energy-charts-v2-panel-r2/rs_2022_2024.json"
RS_MANIFEST = ROOT / "data/energy-charts-v2-panel-r2/acquisition_manifest.json"
BASELINE = ROOT / "docs/evidence/energy-charts-loco-capacity-r3/results"
PROTOCOL = ROOT / "docs/ENERGY_CHARTS_SERBIA_EXPANSION_PROTOCOL_20261101.md"
RS_MAP = {
    "coal": ("fossil_brown_coal_lignite",),
    "gas": ("fossil_gas",),
    "oil": (),
    "other": ("biomass", "others"),
    "nuclear": (),
    "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
    "wind": ("wind_onshore",),
    "solar": (),
    "other_renewable": (),
}


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_local_hours(period: pd.Period, timezone: str) -> int:
    zone = ZoneInfo(timezone)
    start = pd.Timestamp(period.start_time).tz_localize(zone)
    end = pd.Timestamp((period + 1).start_time).tz_localize(zone)
    return len(pd.date_range(start, end, freq="h", inclusive="left"))


def load_serbia(expected_sha: str) -> tuple[pd.DataFrame, dict]:
    if sha(RS_DATA) != expected_sha:
        raise ValueError("Serbia raw response SHA mismatch")
    payload = json.loads(RS_DATA.read_text(encoding="utf-8"))
    if payload.get("country") != "rs" or payload.get("unit") != "MW":
        raise ValueError("unexpected Serbia response country or unit")
    timezone = payload["timezone"]
    selected = [row for row in payload["data"] if row["timestamp"][:4] in {"2022", "2023", "2024"}]
    if len(selected) != 26304:
        raise ValueError(f"expected 2022–2024 hourly window, got {len(selected)} rows")

    stamps = pd.DatetimeIndex([pd.Timestamp(row["timestamp"]).tz_convert("UTC") for row in selected])
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError("duplicate or unordered Serbia timestamps")
    if not np.all(np.diff(stamps.as_unit("ns").asi8) == 3_600_000_000_000):
        raise ValueError("Serbia response is not a continuous UTC hourly sequence")

    values = pd.DataFrame([row["values"] for row in selected])
    present = set(values.columns)
    required = {key for group in RS_MAP.values() for key in group}
    if not required.issubset(present):
        raise ValueError(f"mapped Serbia series absent: {sorted(required - present)}")
    fine = pd.DataFrame(index=values.index)
    null_audit: dict[str, int] = {}
    for fuel in FINE:
        keys = RS_MAP[fuel]
        if not keys:
            fine[fuel] = 0.0
            continue
        block = values[list(keys)].apply(pd.to_numeric, errors="coerce")
        null_audit[fuel] = int(block.isna().sum().sum())
        fine[fuel] = block.sum(axis=1, min_count=len(keys))

    fine["__complete"] = fine[list(FINE)].notna().all(axis=1)
    rows = pd.DataFrame({"time": stamps, "fine_row": np.arange(len(selected))})
    local_year = pd.Series([pd.Timestamp(row["timestamp"]).year for row in selected])
    rows["local_year"] = local_year.to_numpy()
    hourly_parts = []
    for year, positions in rows.groupby("local_year", sort=True).groups.items():
        idx = np.asarray(list(positions), dtype=int)
        utc = stamps[idx]
        expected = 8760 + int(pd.Timestamp(f"{year}-12-31").is_leap_year) * 24
        if len(idx) != expected:
            raise ValueError(f"unexpected Serbia hours in {year}: {len(idx)} vs {expected}")
        block = fine.iloc[idx].copy()
        complete = block["__complete"]
        group = block.loc[complete, list(FINE)].copy()
        if (group[list(FINE)] < 0).any().any() or not np.isfinite(group.to_numpy(float)).all():
            raise ValueError(f"negative or nonfinite mapped Serbia generation in {year}")
        group["time"] = utc[complete.to_numpy()]
        group["clock"] = group.time.dt.tz_convert(timezone).dt.tz_localize(None)
        group["country"], group["region"] = "RS", "RS"
        periods = group.clock.dt.to_period("M")
        group["expected_hours"] = [expected_local_hours(period, timezone) for period in periods]
        group = group.loc[group[list(FINE)].sum(axis=1).gt(0)].copy()
        hourly_parts.append(group)
    frame = pd.concat(hourly_parts, ignore_index=True)
    audit = {
        "raw_sha256": expected_sha,
        "timezone": timezone,
        "unit": payload.get("unit"),
        "reported_resolution": payload.get("resolution"),
        "api_generated_at": payload.get("generated_at"),
        "source_rows_2022_2024": len(selected),
        "hourly_rows_after_complete_mapping": len(frame),
        "null_cells_by_mapped_fuel": null_audit,
        "schema_absent_categories_assumed_zero": [fuel for fuel, keys in RS_MAP.items() if not keys],
        "excluded_source_types": sorted(present - required),
        "labels": "fixed-factor generation proxy over listed mapped categories; not measured CIF",
    }
    return frame.reset_index(drop=True), audit


def make_country(code: str, manifest: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if code == "rs":
        entry = json.loads(RS_MANIFEST.read_text(encoding="utf-8"))
        raw, audit = load_serbia(entry["sha256"])
    else:
        raw, audit = base.load_country(code, manifest)
    months = base.public_months(raw)
    features = base.public_features(months, raw.clock)
    features["country"], features["region"] = code.upper(), code.upper()
    features["year"] = raw.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(raw))
    usable = features.dropna(subset=list(HISTORY)).copy()
    audit["complete_public_months_by_year"] = {
        str(int(year)): int(count) for year, count in months.groupby(months.index.year).size().items()
    }
    audit["usable_feature_hours_by_year"] = {
        str(int(year)): int(count) for year, count in usable.groupby("year").size().items()
    }
    return raw.reset_index(drop=True), usable, audit


def fuel_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    return raw.iloc[part.raw_row.to_numpy()][list(FINE)].to_numpy(float)


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    input_dir = output / "input"
    input_dir.mkdir()
    shutil.copy2(RS_DATA, input_dir / RS_DATA.name)
    shutil.copy2(RS_MANIFEST, input_dir / RS_MANIFEST.name)

    manifest_path = base.DATA / "acquisition_manifest.json"
    audit_path = base.DATA / "audit.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rs_manifest = json.loads(RS_MANIFEST.read_text(encoding="utf-8"))
    inputs = {
        str(path.resolve()): sha(path)
        for path in [manifest_path, audit_path, RS_DATA, RS_MANIFEST]
    }
    for code in OLD_COUNTRIES:
        path = base.DATA / f"{code}_2022_2024.json"
        inputs[str(path.resolve())] = sha(path)
    for code in OLD_COUNTRIES:
        expected = next(row["sha256"] for row in manifest["countries"] if row["country"] == code)
        if inputs[str((base.DATA / f"{code}_2022_2024.json").resolve())] != expected:
            raise ValueError(f"old source SHA differs from acquisition manifest: {code}")
    if inputs[str(RS_DATA.resolve())] != rs_manifest["sha256"]:
        raise ValueError("Serbia raw response differs from acquisition manifest")

    code_paths = [
        Path(__file__).resolve(),
        Path(base.__file__).resolve(),
        ROOT / "src/transcif/research/multicountry.py",
        ROOT / "src/transcif/research/cross_country.py",
        ROOT / "src/transcif/research/monthly_adaptation.py",
        PROTOCOL,
    ]
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    if any(path.read_bytes() == b"" for path in code_paths):
        raise ValueError("empty code or protocol input")
    (output / "startup_hashes.json").write_text(
        json.dumps({"code": code_hashes, "inputs": inputs}, indent=2) + "\n",
        encoding="utf-8",
    )

    prepared: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    audits: dict[str, dict] = {}
    for code in COUNTRIES:
        raw, features, audits[code] = make_country(code, manifest)
        prepared[code] = (raw, features)
    (output / "input_audit.json").write_text(
        json.dumps(audits, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    prediction_seals: list[dict] = []
    predictions: dict[str, tuple[pd.DataFrame, Path, dict]] = {}
    for target in COUNTRIES:
        sources = [code for code in COUNTRIES if code != target]
        training_parts = [
            prepared[code][1].loc[prepared[code][1].year.eq(2022)] for code in sources
        ]
        training = pd.concat(training_parts, ignore_index=True)
        train_fuel = np.concatenate([
            fuel_for(prepared[code][0], part) for code, part in zip(sources, training_parts)
        ])
        train_coarse = np.concatenate([
            base.coarse_values(prepared[code][0].iloc[part.raw_row.to_numpy()])
            for code, part in zip(sources, training_parts)
        ])
        if training.empty or len(training) != len(train_fuel) or len(train_fuel) != len(train_coarse):
            raise ValueError(f"empty or misaligned training cohort for target {target}")
        models = base.fit(training, train_fuel, train_coarse)

        calibration_parts = [
            prepared[code][1].loc[
                prepared[code][1].year.eq(2023)
                & prepared[code][1].clock.dt.month.le(6)
            ]
            for code in sources
        ]
        calibration = pd.concat(calibration_parts, ignore_index=True)
        cal_fuel = np.concatenate([
            fuel_for(prepared[code][0], part)
            for code, part in zip(sources, calibration_parts)
        ])
        cal_y = cal_fuel @ FACTORS / cal_fuel.sum(axis=1)
        cal_weights = weights(calibration)
        cal_predictions = base.predict(models, calibration)
        radii = {
            name: weighted_quantile(np.abs(cal_y - value), cal_weights, 0.9)
            for name, value in cal_predictions.items()
        }

        target_raw, target_features = prepared[target]
        query = target_features.loc[target_features.year.eq(2024)].copy()
        point = base.predict(models, query)
        if not len(query) or any(not np.isfinite(value).all() for value in point.values()):
            raise ValueError(f"empty or nonfinite prediction for target {target}")
        path = output / f"{target.upper()}_heldout_predictions.csv.gz"
        pd.DataFrame({
            "time": target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
            **point,
        }).to_csv(path, index=False, compression="gzip")
        prediction_seals.append({
            "target": target.upper(),
            "file": path.name,
            "sha256": sha(path),
            "hours": len(query),
        })
        predictions[target] = (query, path, radii)
        print(f"{target.upper()} predictions sealed: {len(query)} hours", flush=True)

    seal_path = output / "prediction_seal.json"
    seal_path.write_text(json.dumps(prediction_seals, indent=2) + "\n", encoding="utf-8")
    if any(sha(Path(path)) != value for path, value in inputs.items()):
        raise RuntimeError("frozen data input changed during fit")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("frozen code/protocol changed during fit")

    monthly_rows: list[dict] = []
    for target, (query, path, radii) in predictions.items():
        raw = prepared[target][0]
        truth_fuel = fuel_for(raw, query)
        truth = truth_fuel @ FACTORS / truth_fuel.sum(axis=1)
        forecast = pd.read_csv(path)
        if sha(path) != next(item["sha256"] for item in prediction_seals if item["target"] == target.upper()):
            raise RuntimeError(f"prediction file changed before scoring: {target}")
        for method, radius in radii.items():
            point = forecast[method].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask] - point[mask]
                daily = pd.DataFrame({
                    "day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy(),
                    "truth": truth[mask],
                    "prediction": point[mask],
                })
                centered = daily[["truth", "prediction"]] - daily.groupby("day")[[
                    "truth", "prediction"
                ]].transform("mean")
                monthly_rows.append({
                    "target_country": target.upper(),
                    "method": method,
                    "month": int(month),
                    "hours": int(mask.sum()),
                    "mae": float(np.abs(error).mean()),
                    "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
                    "coverage90": float((np.abs(error) <= radius).mean()),
                    "width90": float(2 * radius),
                    "interval_score90": float(np.mean(
                        2 * radius + 20 * np.maximum(np.abs(error) - radius, 0)
                    )),
                })
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(output / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country = monthly.groupby(["target_country", "method"])[metrics].mean().reset_index()
    country.to_csv(output / "country_metrics.csv", index=False)
    macro = country.groupby("method")[metrics].mean().reset_index()
    macro.to_csv(output / "macro_metrics_all_six.csv", index=False)
    country.loc[country.target_country.isin([code.upper() for code in OLD_COUNTRIES])].groupby(
        "method"
    )[metrics].mean().reset_index().to_csv(output / "macro_metrics_original_five_targets.csv", index=False)

    old_metrics = pd.read_csv(BASELINE / "country_metrics.csv")
    old_metrics = old_metrics.loc[old_metrics.target_country.isin([code.upper() for code in OLD_COUNTRIES])]
    new_metrics = country.loc[country.target_country.isin([code.upper() for code in OLD_COUNTRIES])]
    paired = new_metrics.merge(old_metrics, on=["target_country", "method"], suffixes=("_six_source", "_five_source"), validate="one_to_one")
    for metric in metrics:
        paired[f"delta_{metric}"] = paired[f"{metric}_six_source"] - paired[f"{metric}_five_source"]
    paired.to_csv(output / "paired_existing_target_delta.csv", index=False)

    status = {
        "status": "DEVELOPMENT_SIX_COUNTRY_LOCO_WITH_SERBIA",
        "countries": [code.upper() for code in COUNTRIES],
        "source_training_year": 2022,
        "source_calibration_period": "2023-01 through 2023-06",
        "target_year": 2024,
        "source_attribution": "Energy-Charts.info, CC BY 4.0",
        "target_hourly_inputs_used": False,
        "target_monthly_inputs": "current revised hourly aggregates with simulated t−2 delay",
        "historical_as_issued_vintages": False,
        "new_final_confirmation_country": False,
        "serbia_mapping_limit": "API-absent categories are zero placeholders, not verified zero generation",
        "comparison": "six-source predictions on original five targets paired against frozen five-country r3",
    }
    (output / "status.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    outputs = [path for path in output.iterdir() if path.is_file()]
    provenance = {
        "input_sha256": inputs,
        "code_sha256": code_hashes,
        "output_sha256": {path.name: sha(path) for path in outputs},
        "comparison_baseline_sha256": {
            path.name: sha(path)
            for path in [BASELINE / "country_metrics.csv", *[BASELINE / f"{code.upper()}_heldout_predictions.csv.gz" for code in OLD_COUNTRIES]]
        },
    }
    (output / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print("Six-country macro, all six targets:")
    print(macro.to_string(index=False))
    print("Original five targets, paired with five-source baseline:")
    print(paired.groupby("method")[[f"delta_{metric}" for metric in metrics]].mean().to_string())


if __name__ == "__main__":
    main()
