#!/usr/bin/env python3
"""Source-only nearest-composition routing diagnostic for the CN5 head blend."""
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
import run_energy_charts_nine_country_cn5_loco as nine  # noqa: E402
import run_cn5_fuel_resolved_response as response  # noqa: E402
from transcif.research import multicountry_cn5 as model  # noqa: E402
from transcif.research.monthly_adaptation import weighted_quantile  # noqa: E402

COUNTRIES = nine.COUNTRIES
ALPHAS = (0., .25, .5, .75, 1.)
ROUND1 = ROOT / "results/cn5-fuel-factor-head-blend-r1"
PROTOCOL = ROOT / "docs/CN5_FUEL_FACTOR_HEAD_PROFILE_GATE_PROTOCOL_20261102.md"


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pair_path(a: str, b: str, heldout: str, period: str) -> Path:
    left, right = sorted((a, b))
    return ROUND1 / "source_pseudotarget_predictions" / f"exclude_{left}_{right}_{heldout}_{period}.csv.gz"


def profile(raw: pd.DataFrame, part: pd.DataFrame) -> np.ndarray:
    """Mean six-month lagged CN5 shares for a target block; no hourly target features."""
    months = model.public_months(raw)
    denom = months.sum(axis=1).replace(0., np.nan)
    shares = months.div(denom, axis=0)
    period_values = part.clock.dt.to_period("M").drop_duplicates().sort_values()
    monthly_profiles = []
    for period in period_values:
        lag = [period-i for i in range(2, 8)]
        available = shares.reindex(lag).dropna(axis=0, how="any")
        if period-2 not in available.index:
            continue
        monthly_profiles.append(available.mean(axis=0).to_numpy(float))
    if not monthly_profiles:
        raise ValueError("no complete lagged six-month CN5 profiles in block")
    return np.mean(monthly_profiles, axis=0)


def hellinger(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.square(np.sqrt(a)-np.sqrt(b)).sum()) / np.sqrt(2.))


def best_alpha(pred: pd.DataFrame, truth: np.ndarray) -> tuple[float, dict[float, float]]:
    scores = {}
    factor = pred.factor_history.to_numpy(float)
    fuel = pred.fuelshare_history.to_numpy(float)
    for alpha in ALPHAS:
        scores[alpha] = float(np.abs(truth-((1-alpha)*factor+alpha*fuel)).mean())
    return min(ALPHAS, key=lambda a: (scores[a], a)), scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)

    seal_path = ROUND1 / "source_pseudotarget_seal.json"
    outer_seal_path = ROUND1 / "outer_prediction_seal.json"
    input_paths = [PROTOCOL, seal_path, outer_seal_path]
    prepared = {}
    input_hashes = {str(path.resolve()): sha(path) for path in input_paths}
    code_paths = [Path(__file__).resolve(), PROTOCOL, Path(__file__).resolve().parent /
                  "run_cn5_fuel_factor_head_blend.py", Path(response.__file__).resolve(),
                  Path(nine.__file__).resolve(), Path(nine.six_group.base.__file__).resolve(),
                  Path(model.__file__).resolve(), ROOT/"src/transcif/research/monthly_adaptation.py"]
    code_hashes = {str(path.resolve()): sha(path) for path in code_paths}
    source_seal = json.loads(seal_path.read_text(encoding="utf-8"))
    outer_seal = json.loads(outer_seal_path.read_text(encoding="utf-8"))
    for entry in source_seal:
        input_paths.append(ROUND1 / entry["path"])
    for entry in outer_seal:
        input_paths.extend([ROUND1/entry["prediction_file"], ROUND1/entry["calibration_file"]])
    input_hashes = {str(path.resolve()): sha(path) for path in sorted(set(input_paths))}
    if any(sha(Path(path)) != entry["sha256"] for entry in source_seal
           for path, field in [(ROUND1/entry["path"], "sha256")]):
        raise RuntimeError("source pseudo-target prediction seal mismatch")
    if any(sha(ROUND1/entry["prediction_file"]) != entry["prediction_sha256"] or
           sha(ROUND1/entry["calibration_file"]) != entry["calibration_sha256"] for entry in outer_seal):
        raise RuntimeError("outer prediction/calibration seal mismatch")
    (out/"startup_hashes.json").write_text(json.dumps({"inputs": input_hashes,
        "code": code_hashes}, indent=2)+"\n", encoding="utf-8")

    for code in COUNTRIES:
        raw, features, _ = nine.prepare_country(code)
        prepared[code] = raw, features

    # For every outer exclusion, derive each pseudo-target's H1 alpha using only its
    # own source-only crossfit prediction. Store signatures and candidate scores.
    contexts = {}
    alpha_rows = []
    for outer in COUNTRIES:
        pseudo = [c for c in COUNTRIES if c != outer]
        contexts[outer] = {}
        for country in pseudo:
            raw, features = prepared[country]
            part = features.loc[features.year.eq(2023) & features.clock.dt.month.le(6)].copy()
            pred = pd.read_csv(pair_path(outer, country, country, "h1"))
            truth = response.truth_for(raw, part)
            selected, scores = best_alpha(pred, truth)
            contexts[outer][country] = {
                "alpha": selected, "scores": scores,
                "h1_profile": profile(raw, part),
            }
            alpha_rows.append({"excluded_outer": outer.upper(), "pseudo_target": country.upper(),
                               "alpha": selected, **{f"mae_alpha_{a:g}": scores[a] for a in ALPHAS}})
    pd.DataFrame(alpha_rows).to_csv(out/"source_pseudotarget_alpha.csv", index=False)

    validation_rows = []
    outer_meta = []
    outer_monthly = []
    for outer in COUNTRIES:
        pseudo = [c for c in COUNTRIES if c != outer]
        context = contexts[outer]

        # Double-excluded H1 pseudo-target validation for nearest-prototype routing
        # versus an LOPO global alpha chosen on the other pseudo-target countries.
        for target in pseudo:
            candidates = [c for c in pseudo if c != target]
            raw, features = prepared[target]
            part = features.loc[features.year.eq(2023) & features.clock.dt.month.le(6)].copy()
            target_profile = profile(raw, part)
            nearest = min(candidates, key=lambda c: (hellinger(target_profile, context[c]["h1_profile"]), c))
            route_alpha = context[nearest]["alpha"]
            global_scores = {a: float(np.mean([context[c]["scores"][a] for c in candidates])) for a in ALPHAS}
            global_alpha = min(ALPHAS, key=lambda a: (global_scores[a], a))
            pred = pd.read_csv(pair_path(outer, target, target, "h1"))
            truth = response.truth_for(raw, part)
            pair = pred[["factor_history", "fuelshare_history"]]
            route_prediction = ((1-route_alpha)*pair.factor_history.to_numpy()+
                                route_alpha*pair.fuelshare_history.to_numpy())
            global_prediction = ((1-global_alpha)*pair.factor_history.to_numpy()+
                                 global_alpha*pair.fuelshare_history.to_numpy())
            validation_rows.append({"excluded_outer": outer.upper(), "pseudo_target": target.upper(),
                "nearest_source_prototype": nearest.upper(), "route_alpha": route_alpha,
                "global_lopo_alpha": global_alpha,
                "route_distance": hellinger(target_profile, context[nearest]["h1_profile"]),
                "route_mae": float(np.abs(truth-route_prediction).mean()),
                "global_lopo_mae": float(np.abs(truth-global_prediction).mean()),
                "factor_mae": float(np.abs(truth-pair.factor_history.to_numpy()).mean()),
                "fuelshare_mae": float(np.abs(truth-pair.fuelshare_history.to_numpy()).mean())})

        # H2 source-only calibration: route each calibration pseudo-target using
        # nearest H1 prototype, with both the outer and pseudo-target excluded.
        cal_abs, cal_weights = [], []
        for target in pseudo:
            candidates = [c for c in pseudo if c != target]
            raw, features = prepared[target]
            part = features.loc[features.year.eq(2023) & features.clock.dt.month.ge(7)].copy()
            target_profile = profile(raw, part)
            nearest = min(candidates, key=lambda c: (hellinger(target_profile, context[c]["h1_profile"]), c))
            alpha = context[nearest]["alpha"]
            pred = pd.read_csv(pair_path(outer, target, target, "h2"))
            truth = response.truth_for(raw, part)
            center = ((1-alpha)*pred.factor_history.to_numpy()+alpha*pred.fuelshare_history.to_numpy())
            cal_abs.extend(np.abs(truth-center).tolist())
            cal_weights.extend(np.full(len(part), 1/len(pseudo)/len(part)).tolist())
        radius = float(weighted_quantile(np.asarray(cal_abs), np.asarray(cal_weights), .9))

        # Select a target-specific alpha using only its 2024 forecast-visible coarse profile.
        raw, features = prepared[outer]
        query = features.loc[features.year.eq(2024)].copy()
        target_profile = profile(raw, query)
        nearest = min(pseudo, key=lambda c: (hellinger(target_profile, context[c]["h1_profile"]), c))
        alpha = context[nearest]["alpha"]
        pred_path = ROUND1/next(entry["prediction_file"] for entry in outer_seal
                                if entry["target"].lower() == outer)
        pred = pd.read_csv(pred_path)
        factor = pred.factor_history.to_numpy(float)
        fuel = pred.fuelshare_history.to_numpy(float)
        center = (1-alpha)*factor + alpha*fuel
        truth = response.truth_for(raw, query)
        outer_meta.append({"target": outer.upper(), "nearest_source_prototype": nearest.upper(),
                           "alpha": alpha, "distance": hellinger(target_profile, context[nearest]["h1_profile"]),
                           "h2_crossfit_radius90": radius, "hours": len(query)})
        for month in sorted(query.clock.dt.month.unique()):
            mask = query.clock.dt.month.eq(month).to_numpy()
            errors = truth[mask]-center[mask]
            daily = pd.DataFrame({"prediction": center[mask], "truth": truth[mask],
                "day": query.loc[mask, "clock"].dt.strftime("%Y-%m-%d").to_numpy()})
            centered = daily[["prediction", "truth"]]-daily.groupby("day")[["prediction", "truth"]].transform("mean")
            outer_monthly.append({"target_country": outer.upper(), "month": int(month),
                "nearest_source_prototype": nearest.upper(), "alpha": alpha, "hours": int(mask.sum()),
                "mae": float(np.abs(errors).mean()),
                "shape_mae": float(np.abs(centered.prediction-centered.truth).mean()),
                "coverage90": float((np.abs(errors) <= radius).mean()), "width90": 2*radius,
                "winkler90": float(np.mean(2*radius+20*np.maximum(np.abs(errors)-radius, 0)))})

    source_validation = pd.DataFrame(validation_rows)
    source_validation.to_csv(out/"source_nested_gate_validation.csv", index=False)
    detail = source_validation.groupby("excluded_outer", as_index=False).agg(
        route_mae=("route_mae", "mean"), global_lopo_mae=("global_lopo_mae", "mean"),
        factor_mae=("factor_mae", "mean"), fuelshare_mae=("fuelshare_mae", "mean"))
    detail.to_csv(out/"source_nested_gate_by_outer.csv", index=False)
    pd.DataFrame(outer_meta).to_csv(out/"outer_gate_routing.csv", index=False)
    monthly = pd.DataFrame(outer_monthly)
    monthly.to_csv(out/"outer_gate_monthly_metrics.csv", index=False)
    country = monthly.groupby("target_country", as_index=False).agg(
        mae=("mae", "mean"), shape_mae=("shape_mae", "mean"), coverage90=("coverage90", "mean"),
        width90=("width90", "mean"), winkler90=("winkler90", "mean"))
    country.to_csv(out/"outer_gate_country_metrics.csv", index=False)
    summary = {
        "source_nested_validation": {k: float(v) for k, v in source_validation[
            ["route_mae", "global_lopo_mae", "factor_mae", "fuelshare_mae"]].mean().items()},
        "outer_2024_macro": {k: float(v) for k, v in country[
            ["mae", "shape_mae", "coverage90", "width90", "winkler90"]].mean().items()},
        "outer_2024_worst_country_mae": float(country.mae.max()),
        "outer_2024_min_country_coverage": float(country.coverage90.min()),
        "targets_previously_viewed": True,
        "source_label": "fixed-factor generation proxy",
        "prediction_input": "strict CN5 lagged monthly shares and calendar",
    }
    (out/"summary.json").write_text(json.dumps(summary, indent=2)+"\n", encoding="utf-8")
    if any(sha(Path(path)) != value for path, value in input_hashes.items()):
        raise RuntimeError("input changed during profile-gate analysis")
    if any(sha(Path(path)) != value for path, value in code_hashes.items()):
        raise RuntimeError("code/protocol changed during profile-gate analysis")
    (out/"status.json").write_text(json.dumps({"status": "COMPLETED_EXPLORATORY_NESTED_PROFILE_GATE",
        "source_nested_rows": len(source_validation), "outer_targets": len(outer_meta),
        **summary}, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
