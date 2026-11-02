"""Frozen strict-CN5 Poland external confirmation for the fuel-share response head."""
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
import run_cn5_fuel_resolved_fr_confirmation as fr  # noqa: E402
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_FUEL_RESOLVED_PL_CONFIRMATION_PROTOCOL_20261101.md"
PREFLIGHT_SCRIPT = ROOT / "scripts/preflight_cn5_fuel_resolved_pl.py"
PREFLIGHT = ROOT / "results/cn5-fuel-resolved-pl-preflight-r2/preflight.json"
DATA = ROOT / "data/energy-charts-v2-panel-r1"
TARGET = "pl"
TARGET_MONTHS = (9, 10, 11, 12)
SOURCES = list(nine.COUNTRIES)
METHODS = candidate.METHODS
MAPPING = {
    "coal": ("fossil_hard_coal", "fossil_brown_coal_lignite"),
    "gas": ("fossil_gas",), "oil": ("fossil_oil",),
    "other": ("fossil_coal_derived_gas", "biomass", "others"),
    "nuclear": (), "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
    "wind": ("wind_onshore",), "solar": ("solar",),
    "other_renewable": ("other_renewables",),
}
nine.six_group.base.TYPE_MAP[TARGET] = MAPPING


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_pl() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = DATA / "acquisition_manifest.json"
    raw_path = DATA / "pl_2022_2024.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = next(row["sha256"] for row in manifest["countries"]
                        if row["country"] == TARGET)
    if sha(raw_path) != expected_sha:
        raise ValueError("PL raw Energy-Charts SHA mismatch")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    records = [row for row in payload["data"] if row["timestamp"][:4] in {"2023", "2024"}]
    values = pd.DataFrame([row["values"] for row in records])
    present = set(values.columns)
    required = {key for items in MAPPING.values() for key in items}
    if required - present:
        raise ValueError(f"mapped PL series absent: {sorted(required - present)}")
    fine = pd.DataFrame(index=values.index)
    nulls = {}
    for fuel in model.FINE:
        keys = MAPPING[fuel]
        if not keys:
            fine[fuel] = 0.
        else:
            block = values[list(keys)].apply(pd.to_numeric, errors="coerce")
            nulls[fuel] = int(block.isna().sum().sum())
            fine[fuel] = block.sum(axis=1, min_count=len(keys))
    timestamps = pd.DatetimeIndex([pd.Timestamp(row["timestamp"]).tz_convert("UTC")
                                   for row in records])
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("PL timestamps must be unique and ordered")
    if len(timestamps) < 2 or not np.all(np.diff(timestamps) == pd.Timedelta(hours=1)):
        raise ValueError("PL target response requires continuous UTC-hour records")
    complete = fine[list(model.FINE)].notna().all(axis=1).to_numpy()
    block = fine[list(model.FINE)].to_numpy(float)
    valid = complete & np.isfinite(block).all(axis=1) & (block >= 0).all(axis=1)
    local = timestamps.tz_convert(payload["timezone"]).tz_localize(None)
    raw = fine.loc[valid].copy()
    raw.index = timestamps[valid]
    raw["time"] = timestamps[valid]
    raw["clock"] = local[valid]
    raw["country"], raw["region"] = "PL", "PL"
    periods = raw.clock.dt.to_period("M")
    raw["expected_hours"] = periods.map({
        p: nine.six_group.base.expected_local_hours(p, payload["timezone"])
        for p in periods.unique()
    }).to_numpy()
    raw = raw.loc[raw[list(model.FINE)].sum(axis=1).gt(0)].reset_index(drop=True)

    months = model.public_months(raw)
    features = model.public_features(months, raw.clock)
    features["country"], features["region"] = "PL", "PL"
    features["year"] = raw.clock.dt.year.to_numpy()
    features["raw_row"] = np.arange(len(raw))
    usable = features.dropna(subset=list(model.HISTORY)).copy()
    query = usable.loc[usable.year.eq(2024) & usable.clock.dt.month.isin(TARGET_MONTHS)]
    actual_months = sorted(int(m) for m in query.clock.dt.month.unique())
    if actual_months != list(TARGET_MONTHS):
        raise ValueError(f"PL target features do not meet frozen month contract: {actual_months}")
    audit = {
        "raw_sha256": expected_sha, "timezone": payload["timezone"],
        "unit": payload.get("unit"), "resolution": payload.get("resolution"),
        "mapped_production_types": MAPPING, "null_cells_by_mapped_fuel": nulls,
        "unresolved_source_components": ["waste not separately reported; omitted unless within `others`"],
        "valid_complete_nonnegative_rows": int(valid.sum()),
        "target_months": [f"2024-{m:02d}" for m in TARGET_MONTHS],
        "target_hours": int(len(query)),
        "target_hours_by_month": {str(p): int(n) for p, n in query.groupby(
            query.clock.dt.to_period("M")).size().items()},
        "nuclear_zero_basis": "Eurostat 2024; no national nuclear electricity production",
        "unused_API_series": sorted(present - required),
    }
    return raw, usable, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if preflight.get("status") != "PREFLIGHT_ELIGIBLE_RESTRICTED_WINDOW":
        raise ValueError("prediction-free PL preflight has not passed the frozen restricted window")

    input_paths = [PROTOCOL, fr.PROTOCOL, PREFLIGHT_SCRIPT, PREFLIGHT,
                   DATA / "acquisition_manifest.json", DATA / "pl_2022_2024.json"]
    for code in SOURCES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths = sorted(set(input_paths))
    code_paths = [Path(__file__).resolve(), Path(candidate.__file__).resolve(),
                  Path(fr.__file__).resolve(),
                  Path(nine.__file__).resolve(), Path(nine.six_group.__file__).resolve(),
                  Path(nine.six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(path.resolve()): sha(path) for path in input_paths}
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}

    prepared = {}
    audits = {}
    for code in SOURCES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = raw, features, audits[code]
    prepared[TARGET] = prepare_pl()
    (out / "input_audit.json").write_text(json.dumps(audits | {"PL": prepared[TARGET][2]},
                                                    ensure_ascii=False, indent=2) + "\n")

    source_models, fuelshare = fr.fit_source_models(prepared, SOURCES)
    radii = fr.source_crossfit_radii(prepared, SOURCES)
    target_raw, target_features, target_audit = prepared[TARGET]
    query = target_features.loc[
        target_features.year.eq(2024) & target_features.clock.dt.month.isin(TARGET_MONTHS)
    ].copy()
    point = fr.point_predictions(source_models, fuelshare, query)
    prediction_path = out / "PL_heldout_predictions.csv.gz"
    pd.DataFrame({"time": target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  **point}).to_csv(prediction_path, index=False, compression="gzip")
    radius_path = out / "source_crossfit_radii.json"
    radius_path.write_text(json.dumps(radii, indent=2) + "\n")
    seal = {"target": "PL", "months": [f"2024-{m:02d}" for m in TARGET_MONTHS],
            "hours": len(query), "prediction_sha256": sha(prediction_path),
            "radii_sha256": sha(radius_path), "source_countries": [c.upper() for c in SOURCES]}
    (out / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")
    if any(sha(Path(path)) != value for path, value in input_hashes.items()):
        raise RuntimeError("input changed during confirmation")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("code or frozen protocol changed during confirmation")
    if sha(prediction_path) != seal["prediction_sha256"] or sha(radius_path) != seal["radii_sha256"]:
        raise RuntimeError("prediction or calibration radius seal mismatch")

    fuel = target_raw.iloc[query.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)
    truth = fuel @ model.FACTORS / fuel.sum(axis=1)
    rows = []
    for method in METHODS:
        prediction = np.asarray(point[method], dtype=float)
        radius = float(radii[method])
        for month in TARGET_MONTHS:
            mask = query.clock.dt.month.eq(month).to_numpy()
            error = truth[mask] - prediction[mask]
            local = query.loc[mask, "clock"]
            daily = pd.DataFrame({"truth": truth[mask], "prediction": prediction[mask],
                                  "day": local.dt.strftime("%Y-%m-%d").to_numpy()})
            centered = daily[["truth", "prediction"]] - daily.groupby("day")[
                ["truth", "prediction"]].transform("mean")
            rows.append({"month": month, "method": method, "hours": int(mask.sum()),
                         "mae": float(np.abs(error).mean()),
                         "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
                         "coverage90": float((np.abs(error) <= radius).mean()),
                         "width90": float(2 * radius),
                         "interval_score90": float(np.mean(2 * radius +
                             20 * np.maximum(np.abs(error) - radius, 0)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    equal_month = monthly.groupby("method")[metrics].mean().reset_index()
    equal_month.to_csv(out / "equal_month_metrics.csv", index=False)
    pooled = []
    for method in METHODS:
        pred = np.asarray(point[method], dtype=float)
        error = truth - pred
        radius = float(radii[method])
        day = query.clock.dt.strftime("%Y-%m-%d")
        centered = pd.DataFrame({"truth": truth, "prediction": pred, "day": day})
        centered = centered[["truth", "prediction"]] - centered.groupby("day")[
            ["truth", "prediction"]].transform("mean")
        pooled.append({"method": method, "hours": len(query), "mae": float(np.abs(error).mean()),
                       "shape_mae": float(np.abs(centered.truth - centered.prediction).mean()),
                       "coverage90": float((np.abs(error) <= radius).mean()),
                       "width90": float(2 * radius),
                       "interval_score90": float(np.mean(2 * radius +
                           20 * np.maximum(np.abs(error) - radius, 0)))})
    pd.DataFrame(pooled).to_csv(out / "hourly_pooled_metrics.csv", index=False)
    status = {"status": "FROZEN_PL_EXTERNAL_CONFIRMATION_SCORED",
              "target": "PL", "target_months": [f"2024-{m:02d}" for m in TARGET_MONTHS],
              "target_hours": len(query), "target_labels_used_before_prediction_seal": False,
              "source_countries": [c.upper() for c in SOURCES], "training_year": 2022,
              "source_calibration": "2023H2 cross-fitted pseudo-target country-equal residual q90",
              "target_budget": "strict CN5 lagged t-2..t-7 shares and calendar",
              "label": "Energy-Charts mapped fixed-factor generation proxy; waste separately unresolved",
              "preflight_sha256": sha(PREFLIGHT), "protocol_sha256": sha(PROTOCOL),
              "inputs_sha256": input_hashes, "code_sha256": code_hashes,
              "target_mapping_audit": target_audit}
    (out / "status.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    print(equal_month.to_string(index=False))
    print(pd.DataFrame(pooled).to_string(index=False))


if __name__ == "__main__":
    main()
