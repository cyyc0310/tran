#!/usr/bin/env python3
"""Strict-CN5 LOCO: transfer a monthly fuel mix and close hourly shares to it."""
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
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = nine.COUNTRIES
PROTOCOL = ROOT / "docs/CN5_MONTHLY_COMPOSITION_CLOSURE_PROTOCOL_20261102.md"
MONTH_COLUMNS = ("cal4", "cal5", *model.GROUPS, "thermal_mean", "thermal_std",
                 "thermal_change", "total_cv", "observations")
METHODS = ("ratio_recomposition", "factor_history", "fuelshare_history",
           "monthly_composition_closure")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def monthly_rows(raw: pd.DataFrame, features: pd.DataFrame, year: int,
                 with_labels: bool) -> tuple[pd.DataFrame, np.ndarray | None]:
    part = features.loc[features.year.eq(year)].copy()
    part["period"] = part.clock.dt.to_period("M")
    part = part.groupby("period", sort=True).first().reset_index()
    if part.empty:
        raise ValueError(f"no usable monthly features for {year}")
    if not with_labels:
        return part, None
    labels = []
    raw_period = raw.clock.dt.to_period("M")
    for period in part.period:
        rows = raw.loc[raw_period.eq(period), list(model.FINE)].sum().to_numpy(float)
        thermal = rows[:4].sum()
        if thermal <= 0:
            raise ValueError(f"non-positive thermal output in {period}")
        labels.append(rows[:4] / thermal)
    return part, np.asarray(labels, dtype=float)


def fit_monthly_composition(frames: list[pd.DataFrame], labels: list[np.ndarray]) -> dict:
    x = pd.concat(frames, ignore_index=True)[list(MONTH_COLUMNS)]
    y = np.concatenate(labels)
    country = pd.concat([f[["country"]] for f in frames], ignore_index=True).country.to_numpy()
    counts = pd.Series(country).value_counts()
    weights = np.asarray([1. / counts[c] for c in country], dtype=float)
    weights /= weights.mean()
    scaler = StandardScaler().fit(x, sample_weight=weights)
    heads = []
    for index in range(y.shape[1]):
        head = Ridge(alpha=10.).fit(scaler.transform(x), y[:, index], sample_weight=weights)
        heads.append(head)
    prior = np.average(y, axis=0, weights=weights)
    prior = np.maximum(prior, 0.)
    prior /= prior.sum()
    return {"scaler": scaler, "heads": heads, "prior": prior}


def predict_monthly_composition(head: dict, frame: pd.DataFrame) -> np.ndarray:
    raw = np.column_stack([m.predict(head["scaler"].transform(frame[list(MONTH_COLUMNS)]))
                           for m in head["heads"]])
    raw = np.maximum(raw, 0.)
    totals = raw.sum(axis=1)
    return np.divide(raw, totals[:, None], out=np.tile(head["prior"], (len(raw), 1)),
                     where=totals[:, None] > 0.)


def fit_hourly_fuelshare(frame: pd.DataFrame, fuel: np.ndarray) -> dict:
    thermal = fuel[:, :4].sum(axis=1)
    active = thermal > 0
    shares = fuel[active, :4] / thermal[active, None]
    active_frame = frame.loc[active]
    weights = model.weights(active_frame)
    weights = weights / weights.mean()
    options = dict(max_iter=80, max_leaf_nodes=15, min_samples_leaf=100,
                   l2_regularization=10., random_state=416, early_stopping=False)
    heads = []
    for index in range(4):
        learner = response.HistGradientBoostingRegressor(**options)
        learner.fit(active_frame[list(model.HISTORY)], shares[:, index], sample_weight=weights)
        heads.append(learner)
    prior = np.average(shares, axis=0, weights=weights)
    prior = np.maximum(prior, 0.)
    prior /= prior.sum()
    return {"heads": heads, "prior": prior}


def predict_hourly_shares(head: dict, public: pd.DataFrame) -> np.ndarray:
    raw = np.column_stack([m.predict(public[list(model.HISTORY)]) for m in head["heads"]])
    raw = np.maximum(raw, 0.)
    totals = raw.sum(axis=1)
    return np.divide(raw, totals[:, None], out=np.tile(head["prior"], (len(raw), 1)),
                     where=totals[:, None] > 0.)


def rake_month(shares: np.ndarray, weights: np.ndarray, target: np.ndarray,
               tolerance: float = 1e-10, max_iter: int = 2000) -> np.ndarray:
    """IPF with per-hour row sums 1 and generation-weighted column margins."""
    if len(shares) == 0 or weights.sum() <= 0:
        raise ValueError("invalid monthly raking support")
    target = np.maximum(np.asarray(target, dtype=float), 0.)
    target /= target.sum()
    p = np.maximum(np.asarray(shares, dtype=float), 1e-12)
    p /= p.sum(axis=1, keepdims=True)
    w = np.asarray(weights, dtype=float)
    w /= w.sum()
    for _ in range(max_iter):
        p /= p.sum(axis=1, keepdims=True)
        current = (w[:, None] * p).sum(axis=0)
        scale = np.divide(target, current, out=np.zeros_like(target), where=current > 0.)
        p *= scale[None, :]
        current = (w[:, None] * p).sum(axis=0)
        err = max(float(np.max(np.abs(p.sum(axis=1) - 1.))),
                  float(np.max(np.abs(current - target))))
        if err < tolerance:
            return p
    raise RuntimeError(f"monthly raking failed to converge; max margin error={err}")


def predict_points(base: dict, monthly_head: dict, hourly_head: dict,
                   public: pd.DataFrame) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    public = public.reset_index(drop=True)
    base_points = model.predict(base, public)
    out = {key: base_points[key] for key in ("ratio_recomposition", "factor_history")}
    q = public[list(model.GROUPS)].to_numpy(float) * base["profile"][
        public.clock.dt.hour.to_numpy()]
    total = q.sum(axis=1)
    thermal_share = np.divide(q[:, 0], total, out=np.zeros(len(q)), where=total > 0.)
    raw_shares = predict_hourly_shares(hourly_head, public)
    months, labels = monthly_rows_from_public(public)
    month_mix = predict_monthly_composition(monthly_head, months)
    mix_map = {p: mix for p, mix in zip(months.period, month_mix)}
    coherent = raw_shares.copy()
    expected_cols = np.empty_like(raw_shares)
    for period, idx in public.assign(period=public.clock.dt.to_period("M")).groupby("period").groups.items():
        positions = np.asarray(list(idx), dtype=int)
        target = mix_map[period]
        closed = rake_month(raw_shares[positions], q[positions, 0], target)
        coherent[positions] = closed
        expected_cols[positions] = target
    raw_ef = raw_shares @ response.FUEL_FACTORS
    coherent_ef = coherent @ response.FUEL_FACTORS
    out["fuelshare_history"] = np.clip(raw_ef * thermal_share, 0., 1000.)
    out["monthly_composition_closure"] = np.clip(coherent_ef * thermal_share, 0., 1000.)
    prediction = pd.DataFrame({"predicted_thermal_share": thermal_share})
    prediction["raking_thermal_weight"] = q[:, 0]
    for k, fuel in enumerate(model.FINE[:4]):
        prediction[f"hourly_share_{fuel}"] = raw_shares[:, k]
        prediction[f"closed_share_{fuel}"] = coherent[:, k]
        prediction[f"monthly_share_{fuel}"] = expected_cols[:, k]
    return out, prediction


def monthly_rows_from_public(public: pd.DataFrame) -> tuple[pd.DataFrame, None]:
    temp = public.copy()
    temp["period"] = temp.clock.dt.to_period("M")
    return temp.groupby("period", sort=True).first().reset_index(), None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest, _ = nine.six_group.data_locations(code)
        input_paths.extend([manifest, folder / f"{code}_2022_2024.json"])
        if code in nine.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    input_paths = sorted(set(input_paths))
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(nine.__file__).resolve(),
                  Path(nine.six_group.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), Path(response.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    input_hashes = {str(p.resolve()): sha(p) for p in input_paths}
    code_hashes = {str(p.resolve()): sha(p) for p in code_paths}
    (out / "startup_hashes.json").write_text(json.dumps(
        {"input_sha256": input_hashes, "code_sha256": code_hashes}, indent=2) + "\n")

    prepared, audits = {}, {}
    for code in COUNTRIES:
        raw, features, audits[code] = nine.prepare_country(code)
        prepared[code] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    predictions, radii_by_target, seals = {}, {}, []
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        train_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        train = pd.concat(train_parts, ignore_index=True)
        train_fuel = np.concatenate([response.fuel_for(prepared[c][0], part)
                                     for c, part in zip(sources, train_parts)])
        coarse = np.concatenate([model.coarse_values(prepared[c][0].iloc[
            part.raw_row.to_numpy()]) for c, part in zip(sources, train_parts)])
        base = model.fit(train, train_fuel, coarse)
        hourly_head = fit_hourly_fuelshare(train, train_fuel)
        monthly_train, monthly_labels = [], []
        for c in sources:
            mframe, labels = monthly_rows(prepared[c][0], prepared[c][1], 2022, True)
            mframe["country"] = c
            monthly_train.append(mframe)
            monthly_labels.append(labels)
        monthly_head = fit_monthly_composition(monthly_train, monthly_labels)

        cal_parts = [prepared[c][1].loc[prepared[c][1].year.eq(2023) &
                                        prepared[c][1].clock.dt.month.le(6)] for c in sources]
        calibration = pd.concat(cal_parts, ignore_index=True)
        cal_fuel = np.concatenate([response.fuel_for(prepared[c][0], part)
                                   for c, part in zip(sources, cal_parts)])
        cal_truth = np.concatenate([response.truth_for(prepared[c][0], part)
                                    for c, part in zip(sources, cal_parts)])
        cal_points, _ = predict_points(base, monthly_head, hourly_head, calibration)
        cal_weights = model.weights(calibration)
        radii = {method: float(weighted_quantile(np.abs(cal_truth - point), cal_weights, .90))
                 for method, point in cal_points.items()}

        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        point, mix = predict_points(base, monthly_head, hourly_head, query)
        artifact = pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                                 "period": query.clock.dt.to_period("M").astype(str).to_numpy(),
                                 **point})
        artifact = pd.concat([artifact, mix.reset_index(drop=True)], axis=1)
        path = out / f"{target.upper()}_heldout_predictions.csv.gz"
        artifact.to_csv(path, index=False, compression="gzip")
        radius_path = out / f"{target.upper()}_source_radii.json"
        radius_path.write_text(json.dumps(radii, indent=2) + "\n")
        seals.append({"target": target.upper(), "path": path.name, "sha256": sha(path),
                      "radii_path": radius_path.name, "radii_sha256": sha(radius_path),
                      "hours": len(query), "source_countries": [c.upper() for c in sources]})
        predictions[target] = (query, point, mix)
        radii_by_target[target] = radii
        print(f"sealed {target.upper()}: {len(query)} hours", flush=True)

    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    if any(sha(Path(p)) != h for p, h in input_hashes.items()):
        raise RuntimeError("input changed during run")
    if any(sha(Path(p)) != h for p, h in code_hashes.items()):
        raise RuntimeError("code changed during run")
    if any(sha(out / row[key]) != row[digest] for row in seals
           for key, digest in (("path", "sha256"), ("radii_path", "radii_sha256"))):
        raise RuntimeError("prediction seal mismatch")

    raking_audit = {}
    for row in seals:
        saved = pd.read_csv(out / row["path"])
        frame = pd.DataFrame({"period": saved.period,
                              "weight": saved.raking_thermal_weight})
        row_error = 0.
        margin_error = 0.
        for fuel in model.FINE[:4]:
            frame[fuel] = saved[f"closed_share_{fuel}"]
            frame[f"target_{fuel}"] = saved[f"monthly_share_{fuel}"]
        row_error = float(np.max(np.abs(saved[[f"closed_share_{f}" for f in model.FINE[:4]]]
                                     .sum(axis=1).to_numpy() - 1.)))
        for _, group in frame.groupby("period"):
            weights = group.weight.to_numpy(float)
            weighted = group[list(model.FINE[:4])].to_numpy(float).T @ weights / weights.sum()
            target = group[[f"target_{f}" for f in model.FINE[:4]]].iloc[0].to_numpy(float)
            margin_error = max(margin_error, float(np.max(np.abs(weighted-target))))
        if row_error > 1e-8 or margin_error > 1e-8:
            raise RuntimeError(f"raking constraints failed for {row['target']}: {row_error}, {margin_error}")
        raking_audit[row["target"]] = {"max_hourly_simplex_error": row_error,
                                       "max_weighted_month_margin_error": margin_error}
    (out / "raking_audit.json").write_text(json.dumps(raking_audit, indent=2) + "\n")

    scores = []
    mix_scores = []
    for target in COUNTRIES:
        raw, _ = prepared[target]
        query, point, _ = predictions[target]
        truth = response.truth_for(raw, query)
        saved = pd.read_csv(out / f"{target.upper()}_heldout_predictions.csv.gz")
        if len(saved) != len(query):
            raise RuntimeError(f"target alignment failure for {target}")
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            day = query.loc[mask].clock.dt.strftime("%Y-%m-%d").to_numpy()
            y = truth[mask]
            for method in METHODS:
                pred = saved[method].to_numpy(float)[mask]
                radius = radii_by_target[target][method]
                lo, hi = pred-radius, pred+radius
                score = hi-lo + 20.*(np.maximum(lo-y, 0.) + np.maximum(y-hi, 0.))
                centered = pd.DataFrame({"y": y, "p": pred, "day": day})
                centered[["y", "p"]] -= centered.groupby("day")[["y", "p"]].transform("mean")
                scores.append({"target_country": target.upper(), "month": int(month),
                    "method": method, "hours": int(mask.sum()),
                    "mae": float(np.mean(np.abs(y-pred))),
                    "shape_mae": float(np.mean(np.abs(centered.y-centered.p))),
                    "coverage90": float(np.mean((y >= lo) & (y <= hi))),
                    "width90": float(2*radius), "winkler90": float(np.mean(score))})
            raw_month = raw.loc[raw.clock.dt.to_period("M").eq(query.loc[mask].clock.dt.to_period("M").iloc[0]),
                                list(model.FINE[:4])].sum().to_numpy(float)
            actual_mix = raw_month / raw_month.sum()
            for k, fuel in enumerate(model.FINE[:4]):
                predicted = saved[f"monthly_share_{fuel}"].to_numpy(float)[mask]
                mix_scores.append({"target_country": target.upper(), "month": int(month),
                                   "fuel": fuel, "actual_share": float(actual_mix[k]),
                                   "predicted_share": float(predicted[0]),
                                   "share_abs_error": float(abs(actual_mix[k]-predicted[0]))})
    monthly = pd.DataFrame(scores)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    country = monthly.groupby(["target_country", "method"], as_index=False).agg(
        mae=("mae", "mean"), shape_mae=("shape_mae", "mean"), coverage90=("coverage90", "mean"),
        width90=("width90", "mean"), winkler90=("winkler90", "mean"))
    country.to_csv(out / "country_metrics.csv", index=False)
    country.groupby("method", as_index=False).agg(
        mae=("mae", "mean"), shape_mae=("shape_mae", "mean"), coverage90=("coverage90", "mean"),
        width90=("width90", "mean"), winkler90=("winkler90", "mean"),
        worst_country_coverage=("coverage90", "min")).to_csv(out / "macro_metrics.csv", index=False)
    pd.DataFrame(mix_scores).to_csv(out / "monthly_mix_diagnostics.csv", index=False)
    print(f"completed monthly-composition closure experiment: {len(monthly)} country-month rows")


if __name__ == "__main__":
    main()
