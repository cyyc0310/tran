#!/usr/bin/env python3
"""Strict-CN5 LOCO evaluation of coarse-composition-gated factor experts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import run_pseudotarget_coarse_conditional_stack_cn5 as cn5


ROOT = cn5.ROOT
DATA_MODULE = cn5.nine
MODEL = cn5.model
COUNTRIES = cn5.COUNTRIES
PROTOCOL = ROOT / "docs/CN5_FACTOR_EXPERT_TRANSFER_PROTOCOL_20261101.md"
TAUS = (0.05, 0.10, 0.20, 0.40, 0.80, float("inf"))
METHODS = ("ratio_recomposition", "macro_recomposition", "factor_history",
           "uniform_factor_experts", "gated_factor_experts")


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def source_fit(prepared: dict, sources: list[str]) -> dict:
    parts = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
    frame = pd.concat(parts, ignore_index=True)
    fuel = np.concatenate([DATA_MODULE.fuel_for(prepared[c][0], part)
                           for c, part in zip(sources, parts)])
    coarse = np.concatenate([
        MODEL.coarse_values(prepared[c][0].iloc[part.raw_row.to_numpy()])
        for c, part in zip(sources, parts)
    ])
    return MODEL.fit(frame, fuel, coarse)


def expert_fit(prepared: dict, country: str) -> tuple[dict, np.ndarray]:
    raw, features = prepared[country]
    train = features.loc[features.year.eq(2022)]
    fuel = DATA_MODULE.fuel_for(raw, train)
    coarse = MODEL.coarse_values(raw.iloc[train.raw_row.to_numpy()])
    model = MODEL.fit(train, fuel, coarse)
    return model, train[list(MODEL.GROUPS)].mean().to_numpy(float)


def gate_weights(public: pd.DataFrame, available: list[str], centroids: dict,
                 scales: np.ndarray, tau: float) -> np.ndarray:
    values = public[list(MODEL.GROUPS)].to_numpy(float)
    prototypes = np.stack([centroids[c] for c in available])
    distance = np.mean(((values[:, None, :] - prototypes[None, :, :]) /
                        scales[None, None, :]) ** 2, axis=2)
    if np.isinf(tau):
        return np.full(distance.shape, 1.0 / len(available))
    logits = -distance / (2 * tau * tau)
    logits -= logits.max(axis=1, keepdims=True)
    weights = np.exp(logits)
    return weights / weights.sum(axis=1, keepdims=True)


def expert_predict(public: pd.DataFrame, available: list[str], experts: dict,
                   centroids: dict, scales: np.ndarray, tau: float) -> np.ndarray:
    weights = gate_weights(public, available, centroids, scales, tau)
    points = np.column_stack([MODEL.predict(experts[c], public)["factor_history"]
                              for c in available])
    return np.sum(weights * points, axis=1)


def truth_for(raw: pd.DataFrame, query: pd.DataFrame) -> np.ndarray:
    fuel = DATA_MODULE.fuel_for(raw, query)
    return fuel @ MODEL.FACTORS / fuel.sum(axis=1)


def shape_mae(query: pd.DataFrame, y: np.ndarray, p: np.ndarray) -> float:
    daily = pd.DataFrame({"y": y, "p": p,
                          "date": query.clock.dt.strftime("%Y-%m-%d").to_numpy()})
    centered = daily[["y", "p"]] - daily.groupby("date")[["y", "p"]].transform("mean")
    return float(np.mean(np.abs(centered.y - centered.p)))


def equal_country_radius(frames: list[pd.DataFrame], residuals: list[np.ndarray]) -> float:
    calibration = pd.concat(frames, ignore_index=True)
    weights = MODEL.weights(calibration)
    values = np.concatenate(residuals)
    return float(cn5.weighted_quantile(values, weights, 0.90))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    input_paths = [PROTOCOL]
    for code in COUNTRIES:
        folder, manifest_path, _ = DATA_MODULE.six_group.data_locations(code)
        input_paths += [manifest_path, folder / f"{code}_2022_2024.json"]
        if code in DATA_MODULE.six_group.OLD_COUNTRIES:
            input_paths.append(folder / "audit.json")
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(cn5.__file__).resolve(),
                  Path(DATA_MODULE.__file__).resolve(), Path(DATA_MODULE.six_group.__file__).resolve(),
                  Path(DATA_MODULE.six_group.base.__file__).resolve(), Path(MODEL.__file__).resolve(),
                  ROOT / "src/transcif/research/cross_country.py",
                  ROOT / "src/transcif/research/monthly_adaptation.py"]
    (out / "startup_hashes.json").write_text(json.dumps({
        "input_sha256": {str(p.resolve()): sha(p) for p in sorted(set(input_paths))},
        "code_sha256": {str(p.resolve()): sha(p) for p in code_paths},
    }, indent=2) + "\n")

    prepared, audits = {}, {}
    for country in COUNTRIES:
        raw, features, audits[country] = DATA_MODULE.prepare_country(country)
        prepared[country] = raw, features
    (out / "input_audit.json").write_text(json.dumps(audits, ensure_ascii=False, indent=2) + "\n")

    seals, tuning_records, calibration_records = [], [], []
    for target in COUNTRIES:
        sources = [c for c in COUNTRIES if c != target]
        experts, centroids = {}, {}
        for source in sources:
            experts[source], centroids[source] = expert_fit(prepared, source)
        source_train_frames = [prepared[c][1].loc[prepared[c][1].year.eq(2022)] for c in sources]
        source_train = pd.concat(source_train_frames, ignore_index=True)
        scales = np.maximum(source_train[list(MODEL.GROUPS)].std(ddof=0).to_numpy(float), 0.02)

        # Tune temperature only on cross-fitted country pseudo-targets in 2023H1.
        tune_losses = {tau: [] for tau in TAUS}
        for pseudo in sources:
            raw, features = prepared[pseudo]
            query = features.loc[features.year.eq(2023) & features.clock.dt.month.le(6)]
            y = truth_for(raw, query)
            available = [c for c in sources if c != pseudo]
            for tau in TAUS:
                p = expert_predict(query, available, experts, centroids, scales, tau)
                loss = float(np.abs(y - p).mean())
                tune_losses[tau].append(loss)
                tuning_records.append({"outer_target": target.upper(), "pseudo_target": pseudo.upper(),
                                       "tau": "inf" if np.isinf(tau) else tau, "mae": loss,
                                       "hours": len(y)})
        selected_tau = min(TAUS, key=lambda tau: (float(np.mean(tune_losses[tau])), tau))

        # Cross-fitted H2 risk calibration: no pseudo-target sees its own expert.
        cal_frames: list[pd.DataFrame] = []
        cal_residuals = {method: [] for method in METHODS}
        for pseudo in sources:
            raw, features = prepared[pseudo]
            query = features.loc[features.year.eq(2023) & features.clock.dt.month.gt(6)].copy()
            y = truth_for(raw, query)
            available = [c for c in sources if c != pseudo]
            pseudo_model = source_fit(prepared, available)
            pooled = MODEL.predict(pseudo_model, query)
            points = {
                "ratio_recomposition": pooled["ratio_recomposition"],
                "macro_recomposition": pooled["macro_recomposition"],
                "factor_history": pooled["factor_history"],
                "uniform_factor_experts": expert_predict(query, available, experts, centroids, scales,
                                                           float("inf")),
                "gated_factor_experts": expert_predict(query, available, experts, centroids, scales,
                                                         selected_tau),
            }
            cal_frames.append(query)
            for method, point in points.items():
                cal_residuals[method].append(np.abs(y - point))
        radii = {method: equal_country_radius(cal_frames, values)
                 for method, values in cal_residuals.items()}
        calibration_records.append({"target": target.upper(),
                                    "selected_tau": "inf" if np.isinf(selected_tau) else selected_tau,
                                    "source_countries": [c.upper() for c in sources],
                                    "radii90": radii})

        # Full source models for the sealed outer-target predictions.
        pooled_model = source_fit(prepared, sources)
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        base = MODEL.predict(pooled_model, query)
        expert_gated = expert_predict(query, sources, experts, centroids, scales, selected_tau)
        expert_uniform = expert_predict(query, sources, experts, centroids, scales, float("inf"))
        points = {"ratio_recomposition": base["ratio_recomposition"],
                  "macro_recomposition": base["macro_recomposition"],
                  "factor_history": base["factor_history"],
                  "uniform_factor_experts": expert_uniform,
                  "gated_factor_experts": expert_gated}
        frame = pd.DataFrame({"time": raw.iloc[query.raw_row.to_numpy()].time.astype(str).to_numpy(),
                              **points})
        for method, radius in radii.items():
            frame[f"{method}_lo"] = frame[method] - radius
            frame[f"{method}_hi"] = frame[method] + radius
        path = out / f"{target.upper()}_heldout_predictions.csv.gz"
        frame.to_csv(path, index=False, compression="gzip")
        seals.append({"target": target.upper(), "path": path.name, "sha256": sha(path),
                      "rows": len(frame), "source_countries": [c.upper() for c in sources],
                      "selected_tau": "inf" if np.isinf(selected_tau) else selected_tau})
        print(f"sealed strict-CN5 target {target.upper()} ({len(frame)} hours)", flush=True)

    (out / "prediction_seal.json").write_text(json.dumps(seals, indent=2) + "\n")
    (out / "inner_tuning.csv").write_text(pd.DataFrame(tuning_records).to_csv(index=False))
    (out / "source_calibration.json").write_text(json.dumps(calibration_records, indent=2) + "\n")

    # Only after all target predictions are sealed are outer-target labels scored.
    monthly_rows = []
    for target in COUNTRIES:
        raw, features = prepared[target]
        query = features.loc[features.year.eq(2024)].copy()
        truth = truth_for(raw, query)
        saved = pd.read_csv(out / f"{target.upper()}_heldout_predictions.csv.gz")
        for method in METHODS:
            pred = saved[method].to_numpy(float)
            lo, hi = saved[f"{method}_lo"].to_numpy(float), saved[f"{method}_hi"].to_numpy(float)
            for month in sorted(query.clock.dt.month.unique()):
                mask = query.clock.dt.month.eq(month).to_numpy()
                error = truth[mask] - pred[mask]
                radius_lo, radius_hi = pred[mask] - lo[mask], hi[mask] - pred[mask]
                width = hi[mask] - lo[mask]
                interval_score = width + 20 * np.maximum(lo[mask] - truth[mask], 0) + 20 * np.maximum(truth[mask] - hi[mask], 0)
                monthly_rows.append({"target_country": target.upper(), "method": method,
                                     "month": int(month), "hours": int(mask.sum()),
                                     "mae": float(np.mean(np.abs(error))),
                                     "shape_mae": shape_mae(query.loc[mask], truth[mask], pred[mask]),
                                     "coverage90": float(np.mean((truth[mask] >= lo[mask]) & (truth[mask] <= hi[mask]))),
                                     "width90": float(np.mean(width)),
                                     "interval_score90": float(np.mean(interval_score))})
    monthly = pd.DataFrame(monthly_rows)
    monthly.to_csv(out / "monthly_metrics.csv", index=False)
    metrics = ["mae", "shape_mae", "coverage90", "width90", "interval_score90"]
    country_metrics = monthly.groupby(["target_country", "method"])[metrics].mean().reset_index()
    country_metrics.to_csv(out / "country_metrics.csv", index=False)
    country_metrics.groupby("method")[metrics].mean().reset_index().to_csv(out / "macro_metrics.csv", index=False)

    hashes = json.loads((out / "startup_hashes.json").read_text())
    for path, expected in hashes["input_sha256"].items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f"input changed during run: {path}")
    for path, expected in hashes["code_sha256"].items():
        if sha(Path(path)) != expected:
            raise RuntimeError(f"code changed during run: {path}")
    if any(sha(out / row["path"]) != row["sha256"] for row in seals):
        raise RuntimeError("prediction seal mismatch")
    print(f"completed strict-CN5 factor-expert LOCO: {len(monthly)} monthly rows", flush=True)


if __name__ == "__main__":
    main()
