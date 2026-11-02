#!/usr/bin/env python3
"""Nested CN5 source-pseudotarget monthly level-bias transfer experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

PROTOCOL = ROOT / "docs/CN5_SOURCE_MONTHLY_BIAS_META_PROTOCOL_20261102.md"
BASELINE_DIR = ROOT / "results/cn5-fuel-factor-head-blend-r1"
FEATURES = ["thermal", "hydro", "nuclear", "wind", "solar",
            "month_sin", "month_cos"]
BASELINES = ("factor_history", "fuelshare_history", "ratio_recomposition")
METHODS = (*BASELINES, "source_h1_global_bias", "source_monthly_ridge")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def truth_for(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    fuel = response.fuel_for(raw, part)
    return fuel @ model.FACTORS / fuel.sum(axis=1)


def fit_meta(X: np.ndarray, y: np.ndarray, weights: np.ndarray):
    # Six rows per pseudo-country receive weight 1/6, so ordinary row-wise
    # scaling equals equal-country scaling and avoids weighted-variance drift.
    scaler = StandardScaler().fit(X)
    reg = Ridge(alpha=10.0, fit_intercept=True).fit(
        scaler.transform(X), y, sample_weight=weights)
    return scaler, reg


def add_month_features(features: pd.DataFrame, prediction: pd.DataFrame,
                       values: np.ndarray, months: list[int] | None = None):
    if len(features) != len(prediction) or len(values) != len(prediction):
        raise ValueError("hourly prediction, label and feature rows must have equal length")
    frame = pd.DataFrame({
        "time": pd.to_datetime(prediction.time, utc=True),
        "base": prediction.fuelshare_history.to_numpy(float),
        "truth": values,
        "clock": features.clock.to_numpy(),
    })
    for column in ("thermal", "hydro", "nuclear", "wind", "solar"):
        frame[column] = features[column].to_numpy(float)
    frame["month"] = frame.clock.dt.month
    angle = 2 * np.pi * (frame.month.to_numpy() - 1) / 12
    frame["month_sin"], frame["month_cos"] = np.sin(angle), np.cos(angle)
    if months is not None:
        frame = frame[frame.month.isin(months)].copy()
    return frame


def macro_metrics(country_rows: pd.DataFrame, monthly_rows: pd.DataFrame) -> dict:
    cols = ["mae", "shape_mae", "coverage90", "width90", "winkler90"]
    country = country_rows.groupby("method")[cols].mean()
    monthly = monthly_rows.groupby("method")[cols].mean()
    return {"country_equal": country.to_dict(orient="index"),
            "country_month_equal": monthly.to_dict(orient="index")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in nine.COUNTRIES:
        folder, manifest_path, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest_path, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths.extend(BASELINE_DIR.glob("*_outer_2024_predictions.csv.gz"))
    input_paths.extend(BASELINE_DIR.glob("*_selection_calibration.json"))
    input_paths.extend((BASELINE_DIR / "source_pseudotarget_predictions").glob("*.csv.gz"))
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), Path(response.__file__).resolve(),
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(p.resolve()): sha(p) for p in sorted(set(input_paths))}
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps(
        {"inputs": input_hashes, "code": code_hashes}, indent=2) + "\n")

    prepared = {}
    for code in nine.COUNTRIES:
        raw, features, audit = nine.prepare_country(code)
        prepared[code] = (raw, features, audit)

    all_country_rows, all_month_rows, seals = [], [], []
    pred_dir = out / "outer_predictions"
    pred_dir.mkdir()
    for outer in nine.COUNTRIES:
        X_fit, y_fit, w_fit = [], [], []
        h1_country_rows = []
        for pseudo in nine.COUNTRIES:
            if pseudo == outer:
                continue
            first, second = sorted((outer, pseudo))
            pseudo_path = (BASELINE_DIR / "source_pseudotarget_predictions" /
                           f"exclude_{first}_{second}_{pseudo}_h1.csv.gz")
            pseudo_pred = pd.read_csv(pseudo_path)
            raw, features, _ = prepared[pseudo]
            query = features.loc[features.year.eq(2023) & features.clock.dt.month.le(6)].copy()
            truth = truth_for(raw, query)
            joined = add_month_features(query, pseudo_pred, truth, months=list(range(1, 7)))
            monthly = joined.groupby("month", as_index=False).agg(
                residual=("truth", "mean"), base=("base", "mean"),
                **{col: (col, "first") for col in FEATURES})
            error_by_month = joined.assign(error=joined.truth - joined.base).groupby(
                "month").error.mean()
            monthly["residual"] = monthly.month.map(error_by_month)
            monthly["pseudo_country"] = pseudo
            h1_country_rows.append(monthly)
            X_fit.append(monthly[FEATURES].to_numpy(float))
            y_fit.append(monthly.residual.to_numpy(float))
            w_fit.append(np.full(len(monthly), 1 / len(monthly)))

        train_X = np.concatenate(X_fit)
        train_y = np.concatenate(y_fit)
        train_w = np.concatenate(w_fit)
        scaler, reg = fit_meta(train_X, train_y, train_w)
        global_bias = float(np.average(train_y, weights=train_w))

        # H2 is an independent pseudo-target block used only for interval calibration.
        calibration_residuals = {method: [] for method in METHODS}
        calibration_weights = {method: [] for method in METHODS}
        for pseudo in nine.COUNTRIES:
            if pseudo == outer:
                continue
            first, second = sorted((outer, pseudo))
            pseudo_path = (BASELINE_DIR / "source_pseudotarget_predictions" /
                           f"exclude_{first}_{second}_{pseudo}_h2.csv.gz")
            pseudo_pred = pd.read_csv(pseudo_path)
            raw, features, _ = prepared[pseudo]
            query = features.loc[features.year.eq(2023) & features.clock.dt.month.ge(7)].copy()
            truth = truth_for(raw, query)
            joined = add_month_features(query, pseudo_pred, truth)
            month_frame = joined[["month", *FEATURES]].drop_duplicates("month").copy()
            month_frame["correction_ridge"] = reg.predict(scaler.transform(
                month_frame[FEATURES].to_numpy(float)))
            month_frame["correction_global"] = global_bias
            joined = joined.merge(month_frame[["month", "correction_ridge", "correction_global"]],
                                  on="month", validate="many_to_one")
            source_prediction = pd.read_csv(pseudo_path)
            for method in BASELINES:
                p = source_prediction[method].to_numpy(float)
                calibration_residuals[method].extend(np.abs(truth - p).tolist())
                calibration_weights[method].extend(
                    np.full(len(p), 1 / (len(nine.COUNTRIES) - 1) / len(p)).tolist())
            for method, correction in (("source_h1_global_bias", joined.correction_global.to_numpy(float)),
                                       ("source_monthly_ridge", joined.correction_ridge.to_numpy(float))):
                p = joined.base.to_numpy(float) + correction
                calibration_residuals[method].extend(np.abs(truth - p).tolist())
                calibration_weights[method].extend(
                    np.full(len(p), 1 / (len(nine.COUNTRIES) - 1) / len(p)).tolist())
        radii = {m: float(weighted_quantile(np.asarray(calibration_residuals[m]),
                                             np.asarray(calibration_weights[m]), .9))
                 for m in METHODS}

        # Score the already sealed outer 2024 prediction; no target labels enter fitting.
        raw, features, _ = prepared[outer]
        query = features.loc[features.year.eq(2024)].copy()
        outer_path = BASELINE_DIR / f"{outer.upper()}_outer_2024_predictions.csv.gz"
        base_predictions = pd.read_csv(outer_path)
        truth = truth_for(raw, query)
        joined = add_month_features(query, base_predictions, truth)
        month_features = joined[["month", *FEATURES]].drop_duplicates("month").copy()
        month_features["correction_ridge"] = reg.predict(scaler.transform(
            month_features[FEATURES].to_numpy(float)))
        month_features["correction_global"] = global_bias
        joined = joined.merge(month_features[["month", "correction_ridge", "correction_global"]],
                              on="month", validate="many_to_one")
        point = pd.DataFrame({"time": base_predictions.time})
        for method in BASELINES:
            point[method] = base_predictions[method].to_numpy(float)
        point["source_h1_global_bias"] = (joined.base.to_numpy(float) +
                                            joined.correction_global.to_numpy(float))
        point["source_monthly_ridge"] = (joined.base.to_numpy(float) +
                                           joined.correction_ridge.to_numpy(float))
        path = pred_dir / f"{outer.upper()}_outer_2024.csv.gz"
        point.to_csv(path, index=False, compression="gzip")
        seals.append({"outer_target": outer.upper(), "path": str(path.relative_to(out)),
                      "sha256": sha(path), "source_pseudotargets": [
                          c.upper() for c in nine.COUNTRIES if c != outer],
                      "radii": radii, "global_bias_h1": global_bias,
                      "monthly_ridge_coefficients": reg.coef_.tolist(),
                      "monthly_ridge_intercept": float(reg.intercept_)})

        errors = {}
        for method in METHODS:
            y = truth
            p = point[method].to_numpy(float)
            err = y - p
            radius = radii[method]
            low, high = p - radius, p + radius
            winkler = 2 * radius + 20 * np.maximum(np.abs(err) - radius, 0)
            day = pd.DataFrame({"truth": y, "pred": p,
                                "day": joined.clock.dt.strftime("%Y-%m-%d").to_numpy()})
            centered = day[["truth", "pred"]] - day.groupby("day")[
                ["truth", "pred"]].transform("mean")
            all_country_rows.append({
                "country": outer.upper(), "method": method,
                "mae": float(np.abs(err).mean()),
                "shape_mae": float(np.abs(centered.truth - centered.pred).mean()),
                "coverage90": float(((y >= low) & (y <= high)).mean()),
                "width90": 2 * radius, "winkler90": float(winkler.mean()),
                "radius": radius,
            })
            for month in sorted(joined.month.unique()):
                mask = joined.month.to_numpy() == month
                em, pm = err[mask], p[mask]
                ym = y[mask]
                w = winkler[mask]
                day_m = pd.DataFrame({"truth": ym, "pred": pm,
                                      "day": joined.loc[mask, "clock"].dt.strftime(
                                          "%Y-%m-%d").to_numpy()})
                centered_m = day_m[["truth", "pred"]] - day_m.groupby("day")[
                    ["truth", "pred"]].transform("mean")
                all_month_rows.append({
                    "country": outer.upper(), "method": method, "month": int(month),
                    "mae": float(np.abs(em).mean()),
                    "shape_mae": float(np.abs(centered_m.truth - centered_m.pred).mean()),
                    "coverage90": float(((ym >= pm - radius) & (ym <= pm + radius)).mean()),
                    "width90": 2 * radius, "winkler90": float(w.mean()), "radius": radius,
                })
        (out / "outer_seals.json").write_text(json.dumps(seals, indent=2) + "\n")
        print(f"sealed and scored {outer.upper()} with {len(nine.COUNTRIES)-1} pseudo-targets",
              flush=True)

    country = pd.DataFrame(all_country_rows)
    monthly = pd.DataFrame(all_month_rows)
    country.to_csv(out / "country_metrics.csv", index=False)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    summary = macro_metrics(country, monthly)
    summary["status"] = "DEVELOPMENT_STRICT_CN5_LOCO_TARGETS_PREVIOUSLY_VIEWED"
    summary["design"] = {
        "base_head": "fuelshare_history",
        "meta_fit": "2023H1 equal-country-weighted monthly mean residuals",
        "meta_model": "five lagged shares + month sine/cosine; StandardScaler + Ridge(alpha=10)",
        "interval_calibration": "2023H2 nested pseudo-target hourly absolute residual q90",
        "target_score": "sealed 2024 outer-country hourly labels, fixed-factor production proxy",
        "consumption_track": "not evaluated by this generation-proxy experiment",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if any(sha(Path(path)) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("an input artifact changed during analysis")
    if any(sha(Path(path)) != digest for path, digest in code_hashes.items()):
        raise RuntimeError("code or protocol changed during run")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
