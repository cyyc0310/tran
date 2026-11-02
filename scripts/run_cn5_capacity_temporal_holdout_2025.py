"""Frozen strict-CN5 M vs M+C temporal holdout on 2025 hourly proxies."""
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
import run_cn5_total_volume_trajectory_2025_holdout as holdout  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_CAPACITY_2025_TEMPORAL_PROTOCOL_20261102.md"
CAPACITY = ROOT / "data/eurostat-capacity-energy-charts-r2/capacity-nine-country.csv"
CAPACITY_MANIFEST = ROOT / "data/eurostat-capacity-energy-charts-r2/capacity-nine-country.manifest.json"
METHODS = ("anchor700", "ratio_recomposition", "M_direct_history", "M_ridge_history",
           "M_factor_history", "MC_direct", "MC_ridge", "MC_factor")
OPTIONS = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
               l2_regularization=10.0, random_state=416, early_stopping=False)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def add_capacity(prepared: dict, capacity: pd.DataFrame) -> dict:
    lookup = capacity.loc[capacity.stat_year.eq(2022)].set_index("country").thermal_capacity_share
    expected = {c.upper() for c in holdout.COUNTRIES}
    if set(lookup.index) != expected:
        raise ValueError("2022 capacity rows do not exactly cover the nine target/source countries")
    result = {}
    for country, (raw, features) in prepared.items():
        frame = features.copy()
        frame["capacity_thermal_share"] = frame.country.str.upper().map(lookup)
        if frame.capacity_thermal_share.isna().any():
            raise ValueError(f"missing static 2022 capacity for {country}")
        result[country] = (raw, frame)
    return result


def collect(prepared: dict, countries: list[str], year: int,
            months: range | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    pieces, fuels = [], []
    for country in countries:
        raw, features = prepared[country]
        part = features.loc[features.year.eq(year)].copy()
        part = part.dropna(subset=list(model.HISTORY))
        if months is not None:
            part = part.loc[part.clock.dt.month.isin(months)]
        if not part.empty:
            pieces.append(part)
            fuels.append(holdout.candidate.fuel_model.fuel_for(raw, part))
    if not pieces:
        raise ValueError(f"no source rows for {year=}, {countries=}")
    return pd.concat(pieces, ignore_index=True), np.concatenate(fuels)


def label_from_fuel(fuel: np.ndarray) -> np.ndarray:
    total = fuel.sum(axis=1)
    return np.divide(fuel @ model.FACTORS, total, out=np.full(len(fuel), np.nan), where=total > 0)


def fit_m_heads(train: pd.DataFrame, fuel: np.ndarray) -> dict:
    thermal = fuel[:, :4].sum(axis=1)
    coarse = np.column_stack([thermal, fuel[:, 5], fuel[:, 4], fuel[:, 6], fuel[:, 7]])
    return model.fit(train, fuel, coarse)


def predict_all(reference: dict, capacity_heads: dict,
                frame: pd.DataFrame) -> dict[str, np.ndarray]:
    base = model.predict(reference, frame)
    mc = model.predict_capacity_heads(capacity_heads, frame, reference)
    return {"anchor700": base["anchor700"],
            "ratio_recomposition": base["ratio_recomposition"],
            "M_direct_history": base["direct_history"],
            "M_ridge_history": base["ridge_history"],
            "M_factor_history": base["factor_history"],
            "MC_direct": mc["capacity_direct"],
            "MC_ridge": mc["capacity_ridge"],
            "MC_factor": mc["capacity_factor"]}


def score_block(clock: pd.Series, truth: np.ndarray, point: np.ndarray, radius: float) -> dict:
    error = truth - point
    daily = pd.DataFrame({"y": truth, "p": point,
                          "day": clock.dt.strftime("%Y-%m-%d").to_numpy()})
    centered = daily[["y", "p"]] - daily.groupby("day")[["y", "p"]].transform("mean")
    return {"hours": int(len(point)), "mae": float(np.mean(np.abs(error))),
            "shape_mae": float(np.mean(np.abs(centered.y - centered.p))),
            "coverage90": float(np.mean(np.abs(error) <= radius)),
            "width90": float(2 * radius),
            "interval_score90": float(np.mean(2 * radius + 20 * np.maximum(np.abs(error) - radius, 0)))}


def forecast(out: Path) -> tuple[dict, dict]:
    if out.exists():
        raise FileExistsError(f"refusing to overwrite {out}")
    out.mkdir(parents=True)
    capacity = pd.read_csv(CAPACITY)
    prepared, audit = holdout.prepare_all()
    prepared = add_capacity(prepared, capacity)
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(holdout.__file__).resolve(),
                  Path(holdout.candidate.__file__).resolve(),
                  Path(holdout.loader.__file__).resolve(),
                  Path(holdout.loader.six_group.__file__).resolve(),
                  Path(holdout.loader.six_group.base.__file__).resolve(),
                  Path(holdout.candidate.fuel_model.__file__).resolve(),
                  Path(model.__file__).resolve(),
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_paths = [CAPACITY, CAPACITY_MANIFEST, holdout.DATA / "metadata_preflight.json"]
    capacity_provenance = json.loads(CAPACITY_MANIFEST.read_text())
    for source in capacity_provenance["sources"]:
        input_paths.append(ROOT / source["path"])
        if "manifest" in source:
            input_paths.append(ROOT / source["manifest"])
    for country in holdout.COUNTRIES:
        folder, manifest, _ = holdout.loader.six_group.data_locations(country)
        input_paths += [manifest, folder / f"{country}_2022_2024.json"]
        if country in holdout.loader.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
        input_paths.append(holdout.DATA / f"{country}_2025.json")
    hashes = {"code": {str(p): sha(p) for p in dict.fromkeys(code_paths)},
              "inputs": {str(p): sha(p) for p in dict.fromkeys(input_paths)}}
    (out / "startup_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
    (out / "input_audit.json").write_text(json.dumps(audit, indent=2) + "\n")

    forecasts, selections = [], []
    for target in holdout.COUNTRIES:
        sources = [c for c in holdout.COUNTRIES if c != target]
        train, train_fuel = collect(prepared, sources, 2022)
        reference = fit_m_heads(train, train_fuel)
        mc_heads = model.fit_capacity_heads(train, train_fuel)
        calibration, calibration_fuel = collect(prepared, sources, 2023, range(7, 13))
        cal_truth = label_from_fuel(calibration_fuel)
        cal_predictions = predict_all(reference, mc_heads, calibration)
        cal_weights = model.weights(calibration)
        radii = {name: float(weighted_quantile(
            np.abs(cal_truth - pred), cal_weights, 0.9)) for name, pred in cal_predictions.items()}
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2025)].dropna(subset=list(model.HISTORY)).copy()
        if query.empty:
            raise ValueError(f"no target features for {target} 2025")
        points = predict_all(reference, mc_heads, query)
        label_rows = raw.iloc[query.raw_row.to_numpy()]
        prediction_path = out / f"{target}_2025_predictions.csv.gz"
        prediction_frame = pd.DataFrame({"time": label_rows.time.astype(str).to_numpy(),
                                          "clock": query.clock.astype(str).to_numpy(),
                                          "period": query.clock.dt.to_period("M").astype(str).to_numpy(),
                                          **points})
        prediction_frame.to_csv(prediction_path, index=False, compression="gzip")
        radii_path = out / f"{target}_source_radii.json"
        radii_path.write_text(json.dumps(radii, indent=2) + "\n")
        selections.append({"target": target, "prediction_file": prediction_path.name,
                           "prediction_sha256": sha(prediction_path),
                           "radii_file": radii_path.name, "radii_sha256": sha(radii_path),
                           "source_countries": sources, "source_train_rows": len(train),
                           "source_calibration_rows": len(calibration), "target_hours": len(query)})
        forecasts.append((target, prepared[target], query, radii, prediction_path))
        for group in ("code", "inputs"):
            for path, expected in hashes[group].items():
                if sha(Path(path)) != expected:
                    raise RuntimeError(f"{group} changed during fitting: {path}")
        print(f"sealed {target.upper()} 2025: {len(query)} hours", flush=True)
    (out / "prediction_seal.json").write_text(json.dumps(selections, indent=2) + "\n")
    for row in selections:
        if sha(out / row["prediction_file"]) != row["prediction_sha256"]:
            raise RuntimeError("prediction seal verification failed")
        if sha(out / row["radii_file"]) != row["radii_sha256"]:
            raise RuntimeError("radius seal verification failed")
    seal_meta = {"rows": sum(x["target_hours"] for x in selections),
                 "countries": len(selections), "prediction_sha256": sha(out / "prediction_seal.json"),
                 "target_labels_read": False, "target_year": 2025, "static_capacity_year": 2022}
    return seal_meta, {x["target"]: (p, query, radii, path)
                       for x, (_, p, query, radii, path) in zip(selections, forecasts)}


def score(out: Path, sealed: dict) -> None:
    selection = json.loads((out / "prediction_seal.json").read_text())
    if sha(out / "prediction_seal.json") != sealed["prediction_sha256"]:
        raise ValueError("prediction index seal mismatch")
    rows = []
    for item in selection:
        prediction_path = out / item["prediction_file"]
        radius_path = out / item["radii_file"]
        if sha(prediction_path) != item["prediction_sha256"] or sha(radius_path) != item["radii_sha256"]:
            raise ValueError(f"sealed forecast changed for {item['target']}")
        target = item["target"]
        raw, features = sealed["states"][target]
        query = features.loc[features.year.eq(2025)].dropna(subset=list(model.HISTORY)).copy()
        prediction = pd.read_csv(prediction_path, parse_dates=["clock"])
        truth = holdout.candidate.fuel_model.truth_for(raw, query)
        if len(prediction) != len(truth):
            raise ValueError(f"target score row mismatch for {target}")
        radii = json.loads(radius_path.read_text())
        for method in METHODS:
            point = prediction[method].to_numpy(float)
            if not np.isfinite(point).all():
                raise ValueError(f"nonfinite point prediction {target}/{method}")
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                rows.append({"target_country": target.upper(), "method": method,
                             "period": str(query.loc[mask, "clock"].dt.to_period("M").iloc[0]),
                             **score_block(query.loc[mask, "clock"].reset_index(drop=True),
                                           truth[mask], point[mask], radii[method])})
    monthly = pd.DataFrame(rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target_country", "method"])[
        ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]].mean().reset_index()
    country.to_csv(out / "country_metrics.csv", index=False)
    macro = country.groupby("method")[
        ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]].mean().reset_index()
    macro.to_csv(out / "macro_metrics.csv", index=False)
    pivot = country.pivot(index="target_country", columns="method", values="mae")
    pairs = []
    for m_method, mc_method in (("M_direct_history", "MC_direct"),
                                ("M_ridge_history", "MC_ridge"),
                                ("M_factor_history", "MC_factor")):
        for target in pivot.index:
            pairs.append({"target_country": target, "M_method": m_method,
                          "MC_method": mc_method, "MAE_M": pivot.loc[target, m_method],
                          "MAE_MC": pivot.loc[target, mc_method],
                          "delta_MC_minus_M": pivot.loc[target, mc_method] - pivot.loc[target, m_method]})
    pd.DataFrame(pairs).to_csv(out / "capacity_paired_deltas.csv", index=False)
    summary = {"status": "TEMPORAL_DEVELOPMENT_NOT_GEOGRAPHIC_CONFIRMATION",
               "track": "generation-side fixed-factor hourly proxy only",
               "target_year": 2025, "training_year": 2022, "calibration": "2023H2",
               "static_capacity_stat_year": 2022, "countries": len(selection),
               "target_hours": int(monthly.drop_duplicates(["target_country", "period"]).hours.sum()),
               "macro_metrics": macro.set_index("method").to_dict(orient="index"),
               "limitations": ["All nine target countries were previously viewed in the project.",
                   "Energy-Charts target monthly values are revision snapshots, not as-issued release vintages.",
                   "Eurostat 2022 capacity values are a current revised snapshot; country-specific historical release vintages were not reconstructed.",
                   "The target label is a fixed-factor generation proxy, not measured CIF or consumption-side CIF.",
                   "2022 capacity is held static as an intentionally conservative, lagged M+C budget; no claim is made about contemporaneous 2025 capacity."]}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "scoring_manifest.json").write_text(json.dumps({
        "forecast_seal_verified_before_target_labels": True,
        "target_labels_read_after_forecast": True,
        "target_label_source": "frozen Energy-Charts 2025 hourly generation proxy inputs",
        "prediction_index_sha256": sha(out / "prediction_seal.json"),
        "scored_hours": summary["target_hours"],
        "target_countries": sorted(monthly.target_country.unique().tolist()),
    }, indent=2) + "\n")
    print(macro.to_string(index=False), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    if args.score:
        seal = json.loads((out / "forecast_stage.json").read_text())
        expected = json.loads((out / "startup_hashes.json").read_text())
        for group in ("code", "inputs"):
            for path, digest in expected[group].items():
                if sha(Path(path)) != digest:
                    raise ValueError(f"{group} hash changed since prediction: {path}")
        seal_index = out / "prediction_seal.json"
        if sha(seal_index) != seal["prediction_seal_sha256"]:
            raise ValueError("prediction index changed since forecast stage")
        for item in json.loads(seal_index.read_text()):
            for key, hash_key in (("prediction_file", "prediction_sha256"),
                                  ("radii_file", "radii_sha256")):
                if sha(out / item[key]) != item[hash_key]:
                    raise ValueError(f"sealed forecast artifact changed: {item[key]}")
        # Only after all point forecasts, radii, protocol, code, and input files
        # pass their hashes do we parse target labels for scoring.
        prepared, _ = holdout.prepare_all()
        prepared = add_capacity(prepared, pd.read_csv(CAPACITY))
        score(out, {"states": prepared, "prediction_sha256": seal["prediction_seal_sha256"]})
    else:
        seal, states = forecast(out)
        (out / "forecast_stage.json").write_text(json.dumps({
            "prediction_seal_sha256": seal["prediction_sha256"], **{k: v for k, v in seal.items()
            if k != "prediction_sha256"}}, indent=2) + "\n")
        print(json.dumps(seal, indent=2))


if __name__ == "__main__":
    main()
