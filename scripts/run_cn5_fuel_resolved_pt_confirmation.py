"""Frozen Portugal hourly-domain confirmation of the source-trained CN5 fuel head."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_cn5_fuel_resolved_response as candidate  # noqa: E402
import run_cn5_fuel_resolved_fr_confirmation as fr  # noqa: E402
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_FUEL_RESOLVED_PT_CONFIRMATION_PROTOCOL_20261102.md"
DATA = ROOT / "data/energy-charts-pt-cn5-r1"
TARGET = "pt"
MAPPING = {
    "coal": (), "gas": ("fossil_gas",), "oil": (),
    "other": ("biomass", "others"),
    "nuclear": (), "hydro": ("hydro_run_of_river", "hydro_water_reservoir"),
    "wind": ("wind_onshore", "wind_offshore"), "solar": ("solar",),
    "other_renewable": (),
}
METHODS = candidate.METHODS


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_pt() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path, raw_path = DATA / "acquisition_manifest.json", DATA / "pt_2022_2024.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_sha = manifest["sha256"]
    if sha(raw_path) != expected_sha:
        raise ValueError("Energy-Charts PT source hash differs from acquisition manifest")
    payload = json.loads(raw_path.read_text(encoding="utf-8"))
    if payload.get("timezone") != "Europe/Lisbon" or payload.get("resolution") != "PT1H":
        raise ValueError("PT target timezone/resolution differs from frozen protocol")
    records = [row for row in payload["data"] if row["timestamp"][:4] in {"2022", "2023", "2024"}]
    if not records:
        raise ValueError("no frozen 2022–2024 PT records")
    values = pd.DataFrame([row["values"] for row in records])
    present = set(values.columns)
    required = {series for names in MAPPING.values() for series in names}
    if required - present:
        raise ValueError(f"frozen mapped PT production series absent: {sorted(required - present)}")
    fine = pd.DataFrame(index=values.index)
    nulls = {}
    for fuel in model.FINE:
        keys = MAPPING[fuel]
        if not keys:
            fine[fuel] = 0.
            continue
        block = values[list(keys)].apply(pd.to_numeric, errors="coerce")
        nulls[fuel] = int(block.isna().sum().sum())
        fine[fuel] = block.sum(axis=1, min_count=len(keys))
    complete = fine[list(model.FINE)].notna().all(axis=1).to_numpy()
    stamps = pd.DatetimeIndex([pd.Timestamp(row["timestamp"]).tz_convert("UTC") for row in records])
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError("PT timestamps must be unique and ordered")
    if len(stamps) < 2 or not np.all(np.diff(stamps) == pd.Timedelta(hours=1)):
        raise ValueError("PT hourly label source must be continuous")
    mapped = fine[list(model.FINE)].to_numpy(float)
    valid = complete & np.isfinite(mapped).all(axis=1) & (mapped >= 0.).all(axis=1)
    local = stamps.tz_convert(payload["timezone"]).tz_localize(None)
    raw = fine.loc[valid].copy()
    raw.index = stamps[valid]
    raw["time"] = stamps[valid]
    raw["clock"] = local[valid]
    raw["country"], raw["region"] = "PT", "PT"
    periods = raw.clock.dt.to_period("M")
    raw["expected_hours"] = periods.map({
        p: nine.six_group.base.expected_local_hours(p, payload["timezone"])
        for p in periods.unique()
    }).to_numpy()
    raw = raw.loc[raw[list(model.FINE)].sum(axis=1).gt(0.)].reset_index(drop=True)

    months = model.public_months(raw)
    features = model.public_features(months, raw.clock)
    features["country"], features["region"] = "PT", "PT"
    features["year"] = raw.clock.dt.year.to_numpy()
    features["expected_hours"] = raw.expected_hours.to_numpy(float)
    features["raw_row"] = np.arange(len(raw))
    usable = features.dropna(subset=list(model.HISTORY)).copy()
    target = usable.loc[usable.year.eq(2024)].copy()
    observed = target.groupby(target.clock.dt.to_period("M")).size()
    expected = target.groupby(target.clock.dt.to_period("M")).expected_hours.first()
    coverage = observed.div(expected)
    target_months = {pd.Period(f"2024-{m:02d}") for m in range(1, 13)}
    if set(coverage.index) != target_months or coverage.min() < .99:
        raise ValueError("PT 2024 target labels/features fail the frozen 12-month 99% gate")
    audit = {
        "raw_path": str(raw_path.relative_to(ROOT)), "raw_sha256": expected_sha,
        "timezone": payload["timezone"], "resolution": payload["resolution"],
        "unit": payload.get("unit"), "mapping": MAPPING,
        "null_cells_by_fuel": nulls,
        "structural_zero_basis": {"coal": "ERSE reports coal closed by end of 2021",
                                  "nuclear": "ERSE reports no nuclear power stations"},
        "excluded_series": sorted(present - required),
        "label_coverage_by_local_month_2024": {str(p): float(v) for p, v in coverage.items()},
        "target_hours": int(len(target)),
        "complete_public_months_by_year": {
            str(int(y)): int(n) for y, n in months.groupby(months.index.year).size().items()},
        "target_input_contract": "lagged CN5 shares t-2..t-7 plus calendar",
        "labels_scored_before_prediction_seal": False,
    }
    return raw, usable, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    sources = list(nine.COUNTRIES)
    source_prepared = {code: nine.prepare_country(code) for code in sources}
    target_prepared = prepare_pt()
    prepared = {code: (raw, features, audit)
                for code, (raw, features, audit) in source_prepared.items()}
    prepared[TARGET] = target_prepared
    target_raw, target_features, target_audit = target_prepared
    query = target_features.loc[target_features.year.eq(2024)].copy()

    input_paths = [PROTOCOL, candidate.PROTOCOL, fr.PROTOCOL,
                   DATA / "acquisition_manifest.json", DATA / "pt_2022_2024.json"]
    for code in sources:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths = sorted(set(input_paths))
    code_paths = [Path(__file__).resolve(), Path(candidate.__file__).resolve(),
                  Path(fr.__file__).resolve(), Path(nine.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py",
                  ROOT / "scripts/acquire_energy_charts_pt_cn5_replication.py"]
    input_hashes = {str(path.resolve()): sha(path) for path in input_paths}
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps({"inputs": input_hashes,
        "code": code_hashes}, indent=2) + "\n", encoding="utf-8")
    (out / "input_audit.json").write_text(json.dumps(
        {code.upper(): item[2] for code, item in source_prepared.items()} | {"PT": target_audit},
        ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    radii = fr.source_crossfit_radii(prepared, sources)
    base_models, fuelshare = fr.fit_source_models(prepared, sources)
    predictions = fr.point_predictions(base_models, fuelshare, query)
    prediction_path = out / "PT_heldout_predictions.csv.gz"
    pd.DataFrame({"time": target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  "input_thermal": query.thermal.to_numpy(float), **predictions}).to_csv(
        prediction_path, index=False, compression="gzip")
    radius_path = out / "source_crossfit_radii.json"
    radius_path.write_text(json.dumps(radii, indent=2) + "\n", encoding="utf-8")
    seal = {"target": "PT", "target_year": 2024, "hours": len(query),
            "prediction_path": prediction_path.name, "prediction_sha256": sha(prediction_path),
            "radii_path": radius_path.name, "radii_sha256": sha(radius_path),
            "source_countries": [code.upper() for code in sources]}
    (out / "prediction_seal.json").write_text(json.dumps(seal, indent=2) + "\n")

    if any(sha(Path(path)) != value for path, value in input_hashes.items()):
        raise RuntimeError("input changed before target score")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("code/protocol changed before target score")
    if sha(prediction_path) != seal["prediction_sha256"] or sha(radius_path) != seal["radii_sha256"]:
        raise RuntimeError("sealed predictions or source radii failed hash verification")

    # Open target hourly fuel labels only after the prediction and calibration seals verify.
    fuel = target_raw.iloc[query.raw_row.to_numpy()][list(model.FINE)].to_numpy(float)
    denominator = fuel.sum(axis=1)
    if np.any(denominator <= 0.) or not np.isfinite(fuel).all():
        raise ValueError("PT target labels are not complete positive finite generation")
    truth = fuel @ model.FACTORS / denominator
    stored = pd.read_csv(prediction_path)
    if not np.array_equal(stored.time.astype(str).to_numpy(),
                          target_raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy()):
        raise RuntimeError("sealed PT prediction timestamps do not align with scored rows")
    seasons = {1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
               6: "summer", 7: "summer", 8: "summer", 9: "autumn", 10: "autumn",
               11: "autumn", 12: "winter"}
    rows = []
    for method in METHODS:
        prediction = stored[method].to_numpy(float)
        radius = float(radii[method])
        for month in range(1, 13):
            mask = query.clock.dt.month.eq(month).to_numpy()
            y, p = truth[mask], prediction[mask]
            centered = pd.DataFrame({"truth": y, "pred": p})
            centered["day"] = query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy()
            means = centered.groupby("day")[["truth", "pred"]].transform("mean")
            rows.append({"month": month, "season": seasons[month], "method": method,
                         "hours": int(mask.sum()), "mae": float(np.abs(y-p).mean()),
                         "shape_mae": float(np.abs((centered.truth-means.truth)-
                                                    (centered.pred-means.pred)).mean()),
                         "bias_truth_minus_pred": float((y-p).mean()),
                         "coverage90": float((np.abs(y-p) <= radius).mean()),
                         "width90": 2*radius,
                         "interval_score90": float(np.mean(2*radius +
                             20*np.maximum(np.abs(y-p)-radius, 0.)))})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "bias_truth_minus_pred", "coverage90", "width90",
               "interval_score90"]
    monthly.groupby("method")[metrics].mean().reset_index().to_csv(
        out / "month_macro_metrics.csv", index=False)
    pooled_rows = []
    for method, group in monthly.groupby("method", sort=True):
        pooled_rows.append({"method": method, **{key: float(np.average(group[key], weights=group.hours))
                            for key in metrics}, "hours": int(group.hours.sum())})
    pd.DataFrame(pooled_rows).to_csv(out / "hourly_pooled_metrics.csv", index=False)
    monthly.groupby(["season", "method"])[metrics].mean().reset_index().to_csv(
        out / "season_metrics.csv", index=False)
    (out / "status.json").write_text(json.dumps({
        "status": "FROZEN_PT_EXTERNAL_HOURLY_DOMAIN_CONFIRMATION",
        "target": "PT", "target_year": 2024, "target_hours": len(query),
        "target_labels_used_before_prediction_seal": False,
        "hourly_response_domain_previously_unscored": True,
        "response": "fixed-factor generation-side proxy, not official CIF",
        "source_training": "nine strict-CN5 countries, 2022 hourly labels",
        "risk_calibration": "source cross-fitted leave-country 2023H2 absolute residual q90",
        "candidate_parameters_changed_after_PL_result": False,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(pd.read_csv(out / "month_macro_metrics.csv").to_string(index=False), flush=True)
    print(f"scored {len(query)} sealed PT hours", flush=True)


if __name__ == "__main__":
    main()
