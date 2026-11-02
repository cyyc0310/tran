#!/usr/bin/env python3
"""Value of lagged local annual thermal fuel composition under strict CN5 M."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_cn5_factor_expert_transfer as base


ROOT = base.ROOT
MODEL = base.MODEL
PROTOCOL = ROOT / "docs/CN5_LAGGED_LOCAL_FUEL_MIX_INFORMATION_PROTOCOL_20261101.md"
METHODS = ("ratio_recomposition", "local_lagged_fuel_ef")
ALPHA = 0.10


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_recomposition_reference(prepared: dict, sources: list[str]) -> dict:
    """Match the point-recomposition part of MODEL.fit without fitting other heads."""
    frames, fuels, profiles, profile_countries = [], [], [], []
    for country in sources:
        raw, features = prepared[country]
        train = features.loc[features.year.eq(2022)].copy()
        fuel = base.DATA_MODULE.fuel_for(raw, train)
        frames.append(train)
        fuels.append(fuel)
        coarse = MODEL.coarse_values(raw.iloc[train.raw_row.to_numpy()])
        hourly = pd.DataFrame(coarse, columns=MODEL.GROUPS)
        hourly["hour"] = train.clock.dt.hour.to_numpy()
        profile = hourly.groupby("hour")[list(MODEL.GROUPS)].mean().reindex(range(24)).fillna(0.)
        profile = profile.div(profile.mean().replace(0., np.nan)).fillna(1.).to_numpy(float)
        profiles.append(profile)
        profile_countries.append(country)

    frame = pd.concat(frames, ignore_index=True)
    fuel = np.concatenate(fuels)
    weights = MODEL.weights(frame)
    thermal = fuel[:, :4].sum(axis=1)
    ratio_ef = float(np.sum(weights * (fuel @ MODEL.FACTORS)) / np.sum(weights * thermal))
    country_profiles = [
        np.mean([profile for profile, code in zip(profiles, profile_countries) if code == country], axis=0)
        for country in sorted(set(profile_countries))
    ]
    profile = np.mean(country_profiles, axis=0)
    return {"ratio_ef": ratio_ef, "profile": profile}


def target_thermal_ef(raw: pd.DataFrame, year: int) -> float:
    rows = raw.clock.dt.year.eq(year).to_numpy()
    if not rows.any():
        raise ValueError(f"no annual target fuel records for {year}")
    fuel = raw.loc[rows, list(MODEL.FINE)].to_numpy(float)
    thermal = fuel[:, :4].sum(axis=1)
    if thermal.sum() <= 0:
        raise ValueError(f"no thermal generation for {year}")
    return float(np.sum(fuel[:, :4] @ MODEL.FACTORS[:4]) / thermal.sum())


def predict(reference: dict, query: pd.DataFrame, thermal_ef: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    q = query[list(MODEL.GROUPS)].to_numpy(float) * reference["profile"][query.clock.dt.hour.to_numpy()]
    hourly_thermal_share = np.divide(q[:, 0], q.sum(axis=1), out=np.zeros(len(q)), where=q.sum(axis=1) > 0)
    return reference["ratio_ef"] * hourly_thermal_share, thermal_ef * hourly_thermal_share, hourly_thermal_share


def equal_country_radius(frames: list[pd.DataFrame], residuals: list[np.ndarray]) -> float:
    calibration = pd.concat(frames, ignore_index=True)
    weights = MODEL.weights(calibration)
    values = np.concatenate(residuals)
    return float(base.cn5.weighted_quantile(values, weights, 0.90))


def shape_mae(query: pd.DataFrame, truth: np.ndarray, point: np.ndarray) -> float:
    values = pd.DataFrame({
        "truth": truth,
        "point": point,
        "date": query.clock.dt.strftime("%Y-%m-%d").to_numpy(),
    })
    centered = values[["truth", "point"]] - values.groupby("date")[["truth", "point"]].transform("mean")
    return float(np.mean(np.abs(centered.truth - centered.point)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    countries = list(base.COUNTRIES)
    input_paths = [PROTOCOL]
    for code in countries:
        folder, manifest_path, _ = base.DATA_MODULE.six_group.data_locations(code)
        input_paths += [manifest_path, folder / f"{code}_2022_2024.json"]
        if code in base.DATA_MODULE.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(base.__file__).resolve(),
                  Path(base.cn5.__file__).resolve(), Path(base.DATA_MODULE.__file__).resolve(),
                  Path(base.DATA_MODULE.six_group.__file__).resolve(),
                  Path(base.DATA_MODULE.six_group.base.__file__).resolve(),
                  Path(MODEL.__file__).resolve(), ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(path.resolve()): sha(path) for path in sorted(set(input_paths))}
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps({
        "input_sha256": input_hashes, "code_sha256": code_hashes,
    }, indent=2) + "\n")

    prepared, audits = {}, {}
    for country in countries:
        raw, features, audits[country] = base.DATA_MODULE.prepare_country(country)
        prepared[country] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    calibration_rows, prediction_seals = [], []
    for target in countries:
        sources = [country for country in countries if country != target]
        crossfit_frames: list[pd.DataFrame] = []
        crossfit_residuals = {method: [] for method in METHODS}
        for pseudo in sources:
            train_sources = [country for country in sources if country != pseudo]
            reference = make_recomposition_reference(prepared, train_sources)
            raw, features = prepared[pseudo]
            query = features.loc[features.year.eq(2023) & features.clock.dt.month.gt(6)].copy()
            local_ef = target_thermal_ef(raw, 2022)
            pred_m, pred_local, _ = predict(reference, query, local_ef)
            truth = base.truth_for(raw, query)
            crossfit_frames.append(query)
            crossfit_residuals["ratio_recomposition"].append(np.abs(truth - pred_m))
            crossfit_residuals["local_lagged_fuel_ef"].append(np.abs(truth - pred_local))
        radii = {method: equal_country_radius(crossfit_frames, values)
                 for method, values in crossfit_residuals.items()}
        calibration_rows.append({
            "outer_target": target,
            "calibration_pseudo_countries": [country for country in sources],
            "radius90_ratio": radii["ratio_recomposition"],
            "radius90_local_lagged_fuel_ef": radii["local_lagged_fuel_ef"],
        })

        reference = make_recomposition_reference(prepared, sources)
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        prior_annual_ef = target_thermal_ef(raw, 2023)
        pred_m, pred_local, thermal_share = predict(reference, query, prior_annual_ef)
        frame = pd.DataFrame({
            "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
            "month": query.clock.dt.month.to_numpy(int),
            "ratio_recomposition": pred_m,
            "local_lagged_fuel_ef": pred_local,
            "predicted_thermal_share": thermal_share,
            "source_thermal_ef": reference["ratio_ef"],
            "local_prior_annual_thermal_ef": prior_annual_ef,
            "ratio_radius90": radii["ratio_recomposition"],
            "local_radius90": radii["local_lagged_fuel_ef"],
        })
        frame["ratio_lo"] = frame.ratio_recomposition - frame.ratio_radius90
        frame["ratio_hi"] = frame.ratio_recomposition + frame.ratio_radius90
        frame["local_lo"] = frame.local_lagged_fuel_ef - frame.local_radius90
        frame["local_hi"] = frame.local_lagged_fuel_ef + frame.local_radius90
        path = out / f"{target.upper()}_heldout_predictions.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        prediction_seals.append({"target": target.upper(), "path": path.name,
                                 "sha256": sha(path), "rows": len(frame)})
        print(f"sealed M vs M+Fprev predictions for {target.upper()} ({len(frame)} hours)", flush=True)

    (out / "source_calibration.json").write_text(json.dumps(calibration_rows, indent=2) + "\n")
    (out / "prediction_seal.json").write_text(json.dumps(prediction_seals, indent=2) + "\n")
    for row in prediction_seals:
        if sha(out / row["path"]) != row["sha256"]:
            raise RuntimeError(f"prediction seal mismatch before scoring: {row['target']}")

    # Score target labels only after all outer predictions have been sealed.
    monthly_rows = []
    for target in countries:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        truth = base.truth_for(raw, query)
        saved = pd.read_csv(out / f"{target.upper()}_heldout_predictions.csv.gz")
        if len(saved) != len(query):
            raise RuntimeError(f"target alignment mismatch for {target}")
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            for method, point_column, lo_column, hi_column in [
                ("ratio_recomposition", "ratio_recomposition", "ratio_lo", "ratio_hi"),
                ("local_lagged_fuel_ef", "local_lagged_fuel_ef", "local_lo", "local_hi"),
            ]:
                point = saved[point_column].to_numpy(float)[mask]
                lo, hi = saved[lo_column].to_numpy(float)[mask], saved[hi_column].to_numpy(float)[mask]
                y = truth[mask]
                score = hi - lo + 20.0 * (np.maximum(lo - y, 0.) + np.maximum(y - hi, 0.))
                monthly_rows.append({
                    "target_country": target.upper(), "month": int(month), "method": method,
                    "hours": int(mask.sum()), "mae": float(np.mean(np.abs(y - point))),
                    "bias": float(np.mean(y - point)), "shape_mae": shape_mae(query.loc[mask], y, point),
                    "coverage90": float(np.mean((y >= lo) & (y <= hi))),
                    "width90": float(np.mean(hi - lo)), "winkler90": float(np.mean(score)),
                    "source_thermal_ef": float(saved.source_thermal_ef.iloc[0]),
                    "local_prior_annual_thermal_ef": float(saved.local_prior_annual_thermal_ef.iloc[0]),
                })
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    country_metrics = monthly.groupby(["target_country", "method"], as_index=False).agg(
        mae=("mae", "mean"), bias=("bias", "mean"), shape_mae=("shape_mae", "mean"),
        coverage90=("coverage90", "mean"), width90=("width90", "mean"), winkler90=("winkler90", "mean"),
        source_thermal_ef=("source_thermal_ef", "first"),
        local_prior_annual_thermal_ef=("local_prior_annual_thermal_ef", "first"),
    )
    country_metrics.to_csv(out / "country_metrics.csv", index=False)
    country_metrics.groupby("method", as_index=False).agg(
        mae=("mae", "mean"), bias=("bias", "mean"), shape_mae=("shape_mae", "mean"),
        coverage90=("coverage90", "mean"), width90=("width90", "mean"), winkler90=("winkler90", "mean"),
        worst_country_coverage=("coverage90", "min"),
    ).to_csv(out / "macro_metrics.csv", index=False)

    for path, expected in input_hashes.items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f"input changed during run: {path}")
    for path, expected in code_hashes.items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f"code changed during run: {path}")
    if any(sha(out / row["path"]) != row["sha256"] for row in prediction_seals):
        raise RuntimeError("prediction seal mismatch after scoring")
    print(f"completed local annual fuel-mix information-value experiment: {len(monthly)} country-month rows")


if __name__ == "__main__":
    main()
