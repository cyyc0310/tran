#!/usr/bin/env python3
"""Compare CN5 with lagged local annual and monthly thermal-fuel detail."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_cn5_lagged_local_fuel_mix_information as annual


ROOT = annual.ROOT
BASE = annual.base
MODEL = annual.MODEL
PROTOCOL = ROOT / "docs/CN5_LAGGED_MONTHLY_FUEL_MIX_INFORMATION_PROTOCOL_20261101.md"
METHODS = ("ratio_recomposition", "local_lagged_fuel_ef", "local_t_minus_2_fuel_ef")
ALPHA = 0.10


def monthly_thermal_efs(raw: pd.DataFrame) -> pd.Series:
    fuel = raw[list(MODEL.FINE)].to_numpy(float)
    thermal = fuel[:, :4].sum(axis=1)
    thermal_emissions = fuel[:, :4] @ MODEL.FACTORS[:4]
    rows = pd.DataFrame({
        "period": raw.clock.dt.to_period("M"),
        "thermal": thermal,
        "emissions": thermal_emissions,
    })
    summary = rows.groupby("period").sum(numeric_only=True)
    if summary.thermal.le(0).any():
        raise ValueError("a source month has no thermal generation")
    return summary.emissions / summary.thermal


def local_t_minus_2_ef(raw: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    efs = monthly_thermal_efs(raw)
    periods = query.clock.dt.to_period("M") - 2
    values = efs.reindex(periods).to_numpy(float)
    if not np.isfinite(values).all():
        missing = sorted(str(p) for p, value in zip(periods, values) if not np.isfinite(value))
        raise ValueError(f"missing t-2 local fuel EF months: {missing[:8]}")
    return values


def predict_all(reference: dict, query: pd.DataFrame, annual_ef: float,
                monthly_ef: np.ndarray) -> tuple[dict[str, np.ndarray], np.ndarray]:
    q = query[list(MODEL.GROUPS)].to_numpy(float) * reference["profile"][query.clock.dt.hour.to_numpy()]
    thermal_share = np.divide(q[:, 0], q.sum(axis=1), out=np.zeros(len(q)), where=q.sum(axis=1) > 0)
    return {
        "ratio_recomposition": reference["ratio_ef"] * thermal_share,
        "local_lagged_fuel_ef": annual_ef * thermal_share,
        "local_t_minus_2_fuel_ef": monthly_ef * thermal_share,
    }, thermal_share


def radius(frames: list[pd.DataFrame], errors: list[np.ndarray]) -> float:
    calibration = pd.concat(frames, ignore_index=True)
    weights = MODEL.weights(calibration)
    return float(BASE.cn5.weighted_quantile(np.concatenate(errors), weights, 0.90))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    countries = list(BASE.COUNTRIES)
    input_paths = [PROTOCOL]
    for code in countries:
        folder, manifest_path, _ = BASE.DATA_MODULE.six_group.data_locations(code)
        input_paths += [manifest_path, folder / f"{code}_2022_2024.json"]
        if code in BASE.DATA_MODULE.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(annual.__file__).resolve(),
                  Path(BASE.__file__).resolve(), Path(BASE.cn5.__file__).resolve(),
                  Path(BASE.DATA_MODULE.__file__).resolve(), Path(BASE.DATA_MODULE.six_group.__file__).resolve(),
                  Path(BASE.DATA_MODULE.six_group.base.__file__).resolve(), Path(MODEL.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(path.resolve()): annual.sha(path) for path in sorted(set(input_paths))}
    code_hashes = {str(path.resolve()): annual.sha(path) for path in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps({
        "input_sha256": input_hashes, "code_sha256": code_hashes,
    }, indent=2) + "\n")

    prepared, audits = {}, {}
    for country in countries:
        raw, features, audits[country] = BASE.DATA_MODULE.prepare_country(country)
        prepared[country] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    calibration_records, seals = [], []
    for target in countries:
        sources = [country for country in countries if country != target]
        cal_frames: list[pd.DataFrame] = []
        cal_errors = {method: [] for method in METHODS}
        for pseudo in sources:
            train_sources = [country for country in sources if country != pseudo]
            reference = annual.make_recomposition_reference(prepared, train_sources)
            raw, features = prepared[pseudo]
            query = features.loc[features.year.eq(2023) & features.clock.dt.month.gt(6)].copy()
            annual_ef = annual.target_thermal_ef(raw, 2022)
            lag_ef = local_t_minus_2_ef(raw, query)
            point, _ = predict_all(reference, query, annual_ef, lag_ef)
            truth = BASE.truth_for(raw, query)
            cal_frames.append(query)
            for method in METHODS:
                cal_errors[method].append(np.abs(truth - point[method]))
        radii = {method: radius(cal_frames, values) for method, values in cal_errors.items()}
        calibration_records.append({
            "outer_target": target,
            "pseudo_countries": sources,
            "radii90": radii,
        })

        reference = annual.make_recomposition_reference(prepared, sources)
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        annual_ef = annual.target_thermal_ef(raw, 2023)
        lag_ef = local_t_minus_2_ef(raw, query)
        point, thermal_share = predict_all(reference, query, annual_ef, lag_ef)
        frame = pd.DataFrame({
            "time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
            "month": query.clock.dt.month.to_numpy(int),
            "predicted_thermal_share": thermal_share,
            "source_thermal_ef": reference["ratio_ef"],
            "prior_annual_thermal_ef": annual_ef,
            "local_t_minus_2_thermal_ef": lag_ef,
        })
        for method in METHODS:
            frame[method] = point[method]
            frame[f"{method}_radius90"] = radii[method]
            frame[f"{method}_lo"] = point[method] - radii[method]
            frame[f"{method}_hi"] = point[method] + radii[method]
        path = out / f"{target.upper()}_heldout_predictions.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        seals.append({"target": target.upper(), "path": path.name,
                      "sha256": annual.sha(path), "rows": len(frame)})
        print(f"sealed three information arms for {target.upper()} ({len(frame)} hours)", flush=True)

    (out / "source_calibration.json").write_text(json.dumps(calibration_records, indent=2) + "\n")
    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    if any(annual.sha(out / row["path"]) != row["sha256"] for row in seals):
        raise RuntimeError("prediction seal mismatch before target scoring")

    score_rows = []
    for target in countries:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        truth = BASE.truth_for(raw, query)
        saved = pd.read_csv(out / f"{target.upper()}_heldout_predictions.csv.gz")
        if len(saved) != len(query):
            raise RuntimeError(f"target alignment mismatch for {target}")
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            for method in METHODS:
                point = saved[method].to_numpy(float)[mask]
                lo = saved[f"{method}_lo"].to_numpy(float)[mask]
                hi = saved[f"{method}_hi"].to_numpy(float)[mask]
                y = truth[mask]
                score = hi - lo + (2.0 / ALPHA) * (np.maximum(lo-y, 0.) + np.maximum(y-hi, 0.))
                day = query.loc[mask].clock.dt.strftime("%Y-%m-%d").to_numpy()
                shape = pd.DataFrame({"y": y, "p": point, "day": day})
                centered = shape[["y", "p"]] - shape.groupby("day")[["y", "p"]].transform("mean")
                score_rows.append({
                    "target_country": target.upper(), "month": int(month), "method": method,
                    "hours": int(mask.sum()), "mae": float(np.mean(np.abs(y-point))),
                    "bias": float(np.mean(y-point)), "shape_mae": float(np.mean(np.abs(centered.y-centered.p))),
                    "coverage90": float(np.mean((y>=lo)&(y<=hi))),
                    "width90": float(np.mean(hi-lo)), "winkler90": float(np.mean(score)),
                })
    monthly = pd.DataFrame(score_rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    country_metrics = monthly.groupby(["target_country", "method"], as_index=False).agg(
        mae=("mae", "mean"), bias=("bias", "mean"), shape_mae=("shape_mae", "mean"),
        coverage90=("coverage90", "mean"), width90=("width90", "mean"), winkler90=("winkler90", "mean"),
    )
    country_metrics.to_csv(out / "country_metrics.csv", index=False)
    country_metrics.groupby("method", as_index=False).agg(
        mae=("mae", "mean"), bias=("bias", "mean"), shape_mae=("shape_mae", "mean"),
        coverage90=("coverage90", "mean"), width90=("width90", "mean"), winkler90=("winkler90", "mean"),
        worst_country_coverage=("coverage90", "min"),
    ).to_csv(out / "macro_metrics.csv", index=False)

    for path, expected in input_hashes.items():
        if annual.sha(Path(path)) != expected:
            raise RuntimeError(f"input changed during run: {path}")
    for path, expected in code_hashes.items():
        if annual.sha(Path(path)) != expected:
            raise RuntimeError(f"code changed during run: {path}")
    if any(annual.sha(out / row["path"]) != row["sha256"] for row in seals):
        raise RuntimeError("prediction seal mismatch after scoring")
    print(f"completed lagged monthly fuel-mix information experiment: {len(monthly)} country-month rows")


if __name__ == "__main__":
    main()
