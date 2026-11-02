"""2026 temporal strict-CN5 evaluation of the frozen volume-trend head."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_energy_charts_nine_country_cn5_loco as loader  # noqa: E402
import run_cn5_total_volume_trajectory as candidate  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402

COUNTRIES = list(loader.COUNTRIES)
TARGETS = [country for country in COUNTRIES if country != "cz"]
DATA_2025 = ROOT / "data/energy-charts-cn5-2025-holdout-r1"
DATA_2026 = ROOT / "data/energy-charts-cn5-2026-holdout-r1"
PREFLIGHT_2026 = ROOT / "data/energy-charts-cn5-2026-preflight-r1/metadata_preflight.json"
PROTOCOL = ROOT / "docs/CN5_TOTAL_VOLUME_TRAJECTORY_2026_TEMPORAL_PROTOCOL_20261102.md"
METRICS = candidate.METRICS


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def parse_year(code: str, data_dir: Path, filename: str, expected_hash: str,
               year: int) -> tuple[pd.DataFrame, dict]:
    path = data_dir / filename
    if sha(path) != expected_hash:
        raise ValueError(f"{year} raw hash mismatch for {code}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("country") != code or payload.get("unit") != "MW":
        raise ValueError(f"unexpected country/unit in {year} response for {code}")
    rows = [r for r in payload["data"] if r["timestamp"][:4] == str(year)]
    if not rows:
        raise ValueError(f"no {year} records for {code}")
    stamps = pd.DatetimeIndex([pd.Timestamp(r["timestamp"]).tz_convert("UTC") for r in rows])
    if stamps.has_duplicates or not stamps.is_monotonic_increasing:
        raise ValueError(f"duplicate/unordered {year} timestamps: {code}")
    diffs = np.diff(stamps.as_unit("ns").asi8) / 1e9
    interval_s, count = np.unique(diffs, return_counts=True)
    interval = float(interval_s[np.argmax(count)])
    if interval not in (900., 3600.):
        raise ValueError(f"unsupported {year} interval for {code}: {interval}")

    mapping = deepcopy(loader.six_group.base.TYPE_MAP[code])
    series = {entry.get("id") for entry in payload.get("series", [])}
    # The preflight froze this known public-schema omission as structural zero.
    absent_zero = []
    if code == "sk" and "fossil_hard_coal" not in series:
        mapping["coal"] = tuple(k for k in mapping["coal"] if k != "fossil_hard_coal")
        absent_zero.append("fossil_hard_coal")
    required = {k for values in mapping.values() for k in values}
    missing = sorted(required - series)
    if missing:
        raise ValueError(f"unmapped expected types absent for {code}: {missing}")

    values = pd.DataFrame([r["values"] for r in rows])
    fine = pd.DataFrame(index=values.index)
    nulls = {}
    for fuel in model.FINE:
        names = mapping[fuel]
        if not names:
            fine[fuel] = 0.
        else:
            block = values[list(names)].apply(pd.to_numeric, errors="coerce")
            nulls[fuel] = int(block.isna().sum().sum())
            fine[fuel] = block.sum(axis=1, min_count=len(names))
    fine["__complete"] = fine[list(model.FINE)].notna().all(axis=1)
    hour_utc = stamps.floor("h")
    steps_per_hour = int(3600 / interval)
    count_per_hour = pd.Series(hour_utc).value_counts()
    valid_hours = count_per_hour.index[count_per_hour.eq(steps_per_hour)]
    block = fine.copy()
    block["hour_utc"] = hour_utc
    complete = block["__complete"] & block.hour_utc.isin(valid_hours)
    grouped = block.loc[complete].groupby("hour_utc")[list(model.FINE)].mean()
    if ((grouped < 0).any().any() or not np.isfinite(grouped.to_numpy(float)).all()):
        raise ValueError(f"negative/nonfinite {year} values for {code}")
    grouped = grouped.loc[grouped.sum(axis=1).gt(0)].copy()
    grouped["time"] = grouped.index
    grouped["clock"] = grouped.index.tz_convert(payload["timezone"]).tz_localize(None)
    grouped["country"], grouped["region"] = code.upper(), code.upper()
    periods = grouped.clock.dt.to_period("M")
    grouped["expected_hours"] = [loader.six_group.base.expected_local_hours(p, payload["timezone"])
                                 for p in periods]
    audit = {"raw_sha256": expected_hash, "timezone": payload["timezone"],
        "unit": payload["unit"], "reported_resolution": payload["resolution"],
        "target_year": year, "target_year_source_rows": len(rows), "observed_interval_seconds": int(interval),
        "dominant_interval_fraction": float(count.max() / len(diffs)),
        "complete_positive_hourly_rows": len(grouped), "null_cells_by_fuel": nulls,
        "schema_absent_categories_assumed_zero": absent_zero,
        "excluded_series": sorted(series - required)}
    return grouped.reset_index(drop=True), audit


def prepare_all() -> tuple[dict, dict]:
    manifest_2025 = json.loads((DATA_2025 / "metadata_preflight.json").read_text())
    hashes_2025 = {x["country"]: x["sha256"] for x in manifest_2025["countries"]}
    manifest_2026 = json.loads((DATA_2026 / "acquisition_manifest.json").read_text())
    hashes_2026 = {x["country"]: x["sha256"] for x in manifest_2026["countries"]}
    prepared, audit = {}, {}
    for code in COUNTRIES:
        raw_old, _, old_audit = loader.prepare_country(code)
        if code in TARGETS:
            file_2025 = f"{code}_2025.json"
            raw_2025, audit_2025 = parse_year(
                code, DATA_2025, file_2025, hashes_2025[code], 2025)
            raw_2026, audit_2026 = parse_year(
                code, DATA_2026, f"{code}_2026.json", hashes_2026[code], 2026)
            raw = pd.concat([raw_old, raw_2025, raw_2026], ignore_index=True)
        else:
            raw = raw_old.copy()
            audit_2025 = {"status": "not_needed_for_source_fit_or_calibration"}
            audit_2026 = {"status": "target_rejected_before_label_read",
                           "reason": "mapped fossil_coal_derived_gas series absent in metadata"}
        raw = raw.sort_values("time").reset_index(drop=True)
        months = model.public_months(raw)
        features = model.public_features(months, raw.clock)
        features["country"], features["region"] = code.upper(), code.upper()
        features["year"] = raw.clock.dt.year.to_numpy()
        features["raw_row"] = np.arange(len(raw))
        features = features.dropna(subset=list(model.HISTORY)).copy()
        features = candidate.augment(raw, features)
        prepared[code] = raw, features
        audit[code] = {"old_data_audit": old_audit, "2025_data_audit": audit_2025,
            "2026_data_audit": audit_2026,
            "eligible_hours_by_year": {str(int(y)): int(n)
                for y, n in features.groupby("year").size().items()},
            "complete_public_months_by_year": {str(int(y)): int(n)
                for y, n in months.groupby(months.index.year).size().items()}}
    return prepared, audit


def run_fold(target: str, prepared: dict, out: Path) -> dict:
    sources = [c for c in COUNTRIES if c != target]
    train, fine, coarse = candidate.collect(prepared, sources, 2022)
    base = model.fit(train, fine, coarse)
    base_shares = candidate.fuel_model.fit_fuelshare(train, fine)
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    y = np.divide(fine @ model.FACTORS, thermal, out=np.full(len(fine), 400.), where=active)
    x = train.loc[active, [*model.HISTORY, *candidate.FEATURES]]
    factor = candidate.HistGradientBoostingRegressor(**candidate.OPTIONS)
    factor.fit(x, y[active], sample_weight=model.weights(train.loc[active]) * len(train))
    share_volume = candidate.fit_volume_heads(train, fine)

    cal, cal_fine, _ = candidate.collect(prepared, sources, 2023, range(7, 13))
    truth = cal_fine @ model.FACTORS / cal_fine.sum(axis=1)
    weights = model.weights(cal)
    points = model.predict(base, cal)
    q = candidate.thermal_share(cal, base["profile"])
    points["fuelshare_hgb"] = candidate.fuel_model.predict_fuelshare(base_shares, base, cal)
    points["factor_volume_trajectory"] = np.clip(
        factor.predict(cal[[*model.HISTORY, *candidate.FEATURES]]), 400., 1000.) * q
    points["fuelshare_volume_trajectory"] = (
        candidate.predict_volume_shares(share_volume, cal) @ candidate.FACTORS) * q
    radii = {name: float(candidate.weighted_quantile(np.abs(truth - value), weights, .9))
             for name, value in points.items() if name in candidate.METHODS}

    raw, features = prepared[target]
    query = features.loc[features.year.eq(2026) & features.clock.dt.month.le(9) &
                         features[list(candidate.FEATURES)].notna().all(axis=1)].copy()
    point = model.predict(base, query)
    q = candidate.thermal_share(query, base["profile"])
    point["fuelshare_hgb"] = candidate.fuel_model.predict_fuelshare(base_shares, base, query)
    point["factor_volume_trajectory"] = np.clip(
        factor.predict(query[[*model.HISTORY, *candidate.FEATURES]]), 400., 1000.) * q
    point["fuelshare_volume_trajectory"] = (
        candidate.predict_volume_shares(share_volume, query) @ candidate.FACTORS) * q
    bshares = np.column_stack([np.maximum(h.predict(query[list(model.HISTORY)]), 0.)
                               for h in base_shares["heads"]])
    bshares = np.divide(bshares, bshares.sum(axis=1)[:, None],
        out=np.tile(base_shares["prior"], (len(query), 1)),
        where=bshares.sum(axis=1)[:, None] > 0.)
    vshares = candidate.predict_volume_shares(share_volume, query)
    for i, fuel in enumerate(("coal", "gas", "oil", "other")):
        point[f"base_share_{fuel}"] = bshares[:, i]
        point[f"volume_share_{fuel}"] = vshares[:, i]
    prediction = out / f"{target}_2026_predictions.csv.gz"
    pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                  **point}).to_csv(prediction, index=False, compression="gzip")
    radius = out / f"{target}_source_radii.json"
    radius.write_text(json.dumps(radii, indent=2) + "\n")
    n_available = int((features.year.eq(2026) & features.clock.dt.month.le(9)).sum())
    return {"target": target, "prediction_file": prediction.name, "radius_file": radius.name,
        "scored_hours": len(query), "available_usable_hours": n_available,
        "scored_fraction": len(query) / n_available, "sources": sources}


def score_fold(target: str, prepared: dict, out: Path, selection: dict):
    raw, features = prepared[target]
    query = features.loc[features.year.eq(2026) & features.clock.dt.month.le(9) &
                         features[list(candidate.FEATURES)].notna().all(axis=1)].copy()
    prediction = pd.read_csv(out / selection["prediction_file"])
    radii = json.loads((out / selection["radius_file"]).read_text())
    fine = candidate.fuel_model.fuel_for(raw, query)
    truth = candidate.fuel_model.truth_for(raw, query)
    thermal = fine[:, :4].sum(axis=1)
    active = thermal > 0
    shares = np.divide(fine[:, :4], thermal[:, None], out=np.zeros((len(fine), 4)),
                       where=thermal[:, None] > 0.)
    rows, mechanisms = [], []
    for method, radius in radii.items():
        yhat = prediction[method].to_numpy(float)
        error = np.abs(truth - yhat)
        daily = pd.DataFrame({"truth": truth, "pred": yhat,
            "date": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
        centered = daily[["truth", "pred"]] - daily.groupby("date")[["truth", "pred"]].transform("mean")
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            e = error[mask]
            rows.append({"target_country": target.upper(), "method": method,
                "month": int(month), "hours": int(mask.sum()), "mae": float(e.mean()),
                "shape_mae": float(np.abs(centered.truth.to_numpy()[mask] -
                                           centered.pred.to_numpy()[mask]).mean()),
                "coverage90": float((e <= radius).mean()), "width90": 2 * radius,
                "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(e-radius, 0)))})
    for method, column in (("fuelshare_hgb", "base_share"),
                           ("fuelshare_volume_trajectory", "volume_share")):
        predicted = prediction[[f"{column}_{f}" for f in ("coal", "gas", "oil", "other")]].to_numpy()
        for i, name in enumerate(("coal", "gas", "oil", "other")):
            mechanisms.append({"target_country": target.upper(), "model": method,
                "target": name, "mae": float(np.abs(shares[active, i]-predicted[active, i]).mean())})
        mechanisms.append({"target_country": target.upper(), "model": method,
            "target": "thermal_effective_ef", "mae": float(np.abs(
                shares[active] @ candidate.FACTORS - predicted[active] @ candidate.FACTORS).mean())})
    return rows, mechanisms


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    out = parser.parse_args().output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    meta_2025_path = DATA_2025 / "metadata_preflight.json"
    meta_2025 = json.loads(meta_2025_path.read_text())
    meta_2026_path = DATA_2026 / "acquisition_manifest.json"
    meta_2026 = json.loads(meta_2026_path.read_text())
    inputs = [meta_2025_path, meta_2026_path, PREFLIGHT_2026, PROTOCOL]
    inputs.extend(DATA_2025 / row["file"].split("/")[-1] for row in meta_2025["countries"])
    inputs.extend(DATA_2026 / row["file"] for row in meta_2026["countries"])
    for code in COUNTRIES:
        folder, manifest, _ = loader.six_group.data_locations(code)
        inputs += [manifest, folder / f"{code}_2022_2024.json"]
        if code in loader.six_group.OLD_COUNTRIES:
            inputs.append(folder / "audit.json")
    code_files = [Path(__file__).resolve(), Path(candidate.__file__).resolve(), PROTOCOL,
        Path(loader.__file__).resolve(), Path(loader.six_group.__file__).resolve(),
        Path(loader.six_group.base.__file__).resolve(), Path(model.__file__).resolve(),
        Path(candidate.fuel_model.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
        ROOT / "src/transcif/research/monthly_adaptation.py"]
    hashes = {"inputs": {str(p.resolve()): sha(p) for p in sorted(set(inputs))},
              "code": {str(p.resolve()): sha(p) for p in sorted(set(code_files))}}
    (out / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    prepared, audit = prepare_all()
    (out / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    selections = []
    for target in TARGETS:
        selections.append(run_fold(target, prepared, out))
        print(f"sealed 2026 {target.upper()}", flush=True)
    seals = []
    for selection in selections:
        for key in ("prediction_file", "radius_file"):
            path = out / selection[key]
            seals.append({"target": selection["target"], "file": path.name,
                          "sha256": sha(path), "kind": key})
    (out / "selection.json").write_text(json.dumps(selections, indent=2) + "\n")
    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    for group in ("inputs", "code"):
        for path, expected in hashes[group].items():
            if sha(Path(path)) != expected:
                raise RuntimeError(f"{group} changed during run: {path}")
    for seal in seals:
        if sha(out / seal["file"]) != seal["sha256"]:
            raise RuntimeError(f"sealed artifact changed: {seal['file']}")
    rows, mechanisms = [], []
    for target, selection in zip(TARGETS, selections):
        scored, mechanism = score_fold(target, prepared, out, selection)
        rows.extend(scored)
        mechanisms.extend(mechanism)
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target_country", "method"])[list(METRICS)].mean().reset_index()
    country.to_csv(out / "country_metrics.csv", index=False)
    macro = country.groupby("method")[list(METRICS)].mean().reset_index()
    macro.to_csv(out / "macro_metrics.csv", index=False)
    pd.DataFrame(mechanisms).to_csv(out / "mechanism_metrics.csv", index=False)
    summary = {"status": "TEMPORAL_2026_HOLDOUT_NOT_GEOGRAPHIC_CONFIRMATION",
        "target_year": 2026, "scored_months": "January-September",
        "training_year": 2022, "source_calibration": "2023H2",
        "features": list(candidate.FEATURES), "source_countries": [c.upper() for c in COUNTRIES],
        "target_countries": [c.upper() for c in TARGETS],
        "macro_metrics": macro.set_index("method").to_dict(orient="index"),
        "limitations": ["Target countries were previously seen; this is not geographic confirmation.",
            "2025 lagged target inputs and 2026 target responses are current API snapshots, not historical as-issued monthly vintages.",
            "Fixed-factor labels are generation-side proxies, not measured CIF or consumption CIF.",
            "CZ was rejected as an outer target before label-value reading because fossil_coal_derived_gas is missing from the 2026 API schema; it remains in 2022 source training and calibration.",
            "SK fossil_hard_coal is absent from the 2026 API schema and was treated as structural zero per the prior frozen CN5 data rule."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(macro.to_string(index=False), flush=True)
    print(f"completed 2026 temporal holdout: {out}", flush=True)


if __name__ == "__main__":
    main()
