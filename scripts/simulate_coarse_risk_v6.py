#!/usr/bin/env python3
"""Artificial mechanism diagnostics only; never read observed hourly labels."""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd

from transcif.research.conditional_transfer_v6 import (
    ConditionalTransport, FEATURE_NAMES, MODES, information_feature_audit, information_features,
)
from transcif.research.eia_proxy import FUELS


SEEDS = (17023, 17024, 17025)
FACTORS = (.8, 1., 1.2)
SCENARIOS = ("repeatable_season", "directional_drift", "unidentifiable_shape")
SOURCES = ("S1", "S2")
TARGETS = ("T1", "T2")
HOUR_PATTERN = np.sin(2 * np.pi * np.arange(24) / 24)


def monthly_world(scenario, seed):
    """Generate gas/wind simplex shares; period-end + 30 day publication."""
    starts = pd.date_range("2022-01-01", "2026-12-01", freq="MS", tz="UTC")
    rng = np.random.default_rng(seed)
    shared_innovation = rng.normal(0., 1., len(starts))
    frames, latent = [], {}
    for index, region in enumerate(SOURCES + TARGETS):
        if scenario == "repeatable_season":
            amplitude, slope = (60., 100., 130., 160.)[index], 0.
            base = 200.
        elif scenario == "directional_drift":
            amplitude, slope = 20., (1., 2., 2.5, 3.)[index]
            base = 100.
        else:
            amplitude, slope, base = 30., 0., 200.
        level = base + amplitude * np.sin(2 * np.pi * starts.month.to_numpy() / 12)
        level = level + slope * np.arange(len(starts)) + shared_innovation
        if np.any((level <= 0) | (level >= 400)):
            raise AssertionError("Artificial gas shares must remain inside the simplex.")
        frame = pd.DataFrame({"region": region, "period_start": starts,
                              "period_end": starts + pd.offsets.MonthBegin(1),
                              "available_at": starts + pd.offsets.MonthBegin(1) + pd.Timedelta(days=30)})
        for fuel in FUELS:
            frame[f"fuelshare_{fuel}"] = 0.
        frame["fuelshare_gas"] = level / 400
        frame["fuelshare_wind"] = 1 - frame.fuelshare_gas
        frames.append(frame)
        latent[region] = dict(zip(starts.strftime("%Y-%m"), level))
    return pd.concat(frames, ignore_index=True), latent


def origins_for(year, dense=False):
    days = range(1, 19) if dense else (3, 13, 23)
    return pd.DatetimeIndex([f"{year}-{month:02d}-{day:02d}" for month in range(1, 13)
                             for day in days], tz="UTC")


def make_cohort(scenario, seed, monthly, latent, regions, origins):
    rows = {"point": [], "truth": [], "features": [], "regions": [], "days": []}
    for index, region in enumerate(regions):
        x, _ = information_features(origins, region, monthly)
        history = monthly.loc[monthly.region.eq(region)]
        # Twin target domains share all random draws. Their only difference is
        # the unobserved zero-mean fine shape amplitude in the third scenario.
        domain_seed = 0 if scenario == "unidentifiable_shape" and region in TARGETS else index
        rng = np.random.default_rng(seed + 1000 * origins[0].year + domain_seed)
        level_noise = rng.normal(0., 2., len(origins))
        shape_noise = rng.normal(0., 1., (len(origins), 24))
        shape_noise -= shape_noise.mean(axis=1, keepdims=True)
        for i, origin in enumerate(origins):
            public = history.loc[(history.available_at <= origin) & (history.period_end <= origin)]
            same_month = public.loc[public.period_start.dt.month.eq(origin.month)]
            last = (same_month if len(same_month) else public).iloc[-1]
            coarse_point = 400 * last.fuelshare_gas
            point = coarse_point + 6 * HOUR_PATTERN
            level = latent[region][origin.strftime("%Y-%m")] + level_noise[i]
            amplitude = (4., 8.)[index] if region in SOURCES else 8.
            fine_shape = amplitude * HOUR_PATTERN + shape_noise[i]
            if scenario == "unidentifiable_shape" and region == "T2":
                fine_shape *= 4
            rows["point"].append(point)
            rows["truth"].append(level + fine_shape)
            rows["features"].append(x[i])
            rows["regions"].append(region)
            rows["days"].append(origin)
    return {key: pd.DatetimeIndex(value) if key == "days" else np.asarray(value)
            for key, value in rows.items()}


def metrics(truth, prediction, factor):
    point = prediction["corrected_point"]
    lower = point + factor * (prediction["lower"] - point)
    upper = point + factor * (prediction["upper"] - point)
    width = upper - lower
    score = width + 20 * np.maximum(lower - truth, 0) + 20 * np.maximum(truth - upper, 0)
    return {"coverage90": float(np.mean((lower <= truth) & (truth <= upper))),
            "width90": float(width.mean()), "interval_score90": float(score.mean()),
            "point_mae": float(np.abs(point - truth).mean()),
            "level_mae": float(np.abs(point.mean(axis=1) - truth.mean(axis=1)).mean()),
            "shape_mae": float(np.abs((point - point.mean(axis=1, keepdims=True))
                                      - (truth - truth.mean(axis=1, keepdims=True))).mean())}


def subset_prediction(prediction, mask):
    return {key: value[mask] for key, value in prediction.items()}


def level_relation(cohort):
    result = {}
    for region in np.unique(cohort["regions"]):
        mask = cohort["regions"] == region
        residual = (cohort["truth"][mask] - cohort["point"][mask]).mean(axis=1)
        coarse_bias = cohort["features"][mask, FEATURE_NAMES.index("seasonal_hindcast_bias")]
        result[region] = {"true_daily_level_residual_mean": float(residual.mean()),
                          "seasonal_hindcast_bias_mean": float(coarse_bias.mean()),
                          "mean_absolute_residual_minus_coarse_bias": float(np.abs(residual - coarse_bias).mean())}
    return result


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    previous = pd.read_csv(output / "metrics.csv") if (output / "metrics.csv").exists() else None
    results, selections, identities, diagnostics = [], [], [], []
    for seed in SEEDS:
        for scenario in SCENARIOS:
            monthly, latent = monthly_world(scenario, seed)
            fit = make_cohort(scenario, seed, monthly, latent, SOURCES, origins_for(2024, dense=True))
            selection = make_cohort(scenario, seed, monthly, latent, SOURCES, origins_for(2025))
            target = make_cohort(scenario, seed, monthly, latent, TARGETS,
                                 origins_for(2026)[origins_for(2026).month <= 6])
            diagnostics.append({"seed": seed, "scenario": scenario,
                "fit": information_feature_audit(fit["features"]),
                "selection": information_feature_audit(selection["features"]),
                "target": information_feature_audit(target["features"]),
                "level_relation": {"fit": level_relation(fit), "selection": level_relation(selection),
                                   "target": level_relation(target)}})
            for mode in MODES:
                model = ConditionalTransport(SOURCES, mode=mode, fit_year=2024).fit(
                    fit["point"], fit["truth"], fit["features"], fit["regions"], days=fit["days"],
                    level_scales=10., shape_scales=3.)
                valid = model.predict(selection["point"], selection["features"],
                                      level_scales=10., shape_scales=3., n_samples=32, seed=seed)
                scores = {factor: metrics(selection["truth"], valid, factor)["interval_score90"]
                          for factor in FACTORS}
                factor = min(FACTORS, key=lambda candidate: (scores[candidate], candidate))
                selections.append({"seed": seed, "scenario": scenario, "method": mode,
                                   "selected_factor": factor, "source_2025_scores": scores,
                                   "source_2025_bias_mean": float(valid["bias"].mean()),
                                   "source_2025_level_scale_mean": float(valid["level_scale"].mean())})
                # Seed each target region identically, so paired twin ensembles
                # also coincide; no score or target truth affects prediction.
                twin_predictions = []
                for region in TARGETS:
                    mask = target["regions"] == region
                    prediction = model.predict(target["point"][mask], target["features"][mask],
                                               level_scales=10., shape_scales=3., n_samples=32, seed=seed)
                    twin_predictions.append(prediction)
                    results.append({"seed": seed, "scenario": scenario, "method": mode, "target": region,
                                    "factor": factor, **metrics(target["truth"][mask], prediction, factor),
                                    "predicted_bias_mean": float(prediction["bias"].mean()),
                                    "predicted_level_scale_mean": float(prediction["level_scale"].mean()),
                                    "true_level_residual_mean": float((target["truth"][mask]
                                                                       - target["point"][mask]).mean())})
                if scenario == "unidentifiable_shape":
                    a, b = (target["regions"] == region for region in TARGETS)
                    input_equal = all(np.array_equal(target[key][a], target[key][b])
                                      for key in ("point", "features"))
                    prediction_equal = all(np.array_equal(twin_predictions[0][key], twin_predictions[1][key])
                                           for key in twin_predictions[0])
                    level_equal = np.allclose(target["truth"][a].mean(axis=1),
                                              target["truth"][b].mean(axis=1), atol=1e-12)
                    coarse_equal = monthly.loc[monthly.region.eq("T1")].drop(columns="region").reset_index(drop=True).equals(
                        monthly.loc[monthly.region.eq("T2")].drop(columns="region").reset_index(drop=True))
                    if not (coarse_equal and input_equal and prediction_equal and level_equal):
                        raise AssertionError("Same-coarse shape twins must have identical inputs and predictions.")
                    identities.append({"seed": seed, "method": mode, "inputs_strictly_equal": input_equal,
                                       "all_prediction_arrays_strictly_equal": prediction_equal,
                                       "monthly_rows_strictly_equal": coarse_equal,
                                       "truth_daily_levels_equal": bool(level_equal), "fine_shape_multiplier": 4.})
            print(f"completed artificial scenario={scenario} seed={seed}", flush=True)
    frame = pd.DataFrame(results)
    reproduction = None
    if previous is not None:
        shared = list(previous.columns)
        pd.testing.assert_frame_equal(previous, frame[shared], check_exact=False, rtol=1e-12, atol=1e-12)
        reproduction = "prior metrics reproduced to 1e-12; diagnostics-only extension"
    frame.to_csv(output / "metrics.csv", index=False)
    summary = frame.groupby(["scenario", "method", "target"], sort=False)[
        ["coverage90", "width90", "interval_score90", "level_mae", "shape_mae"]].agg(["mean", "min", "max"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(output / "summary.csv", index=False)
    payload = {"data_kind": "ARTIFICIAL ONLY; no observed hourly target labels were read",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "conditional_module_sha256": hashlib.sha256(Path(inspect.getfile(ConditionalTransport)).read_bytes()).hexdigest(),
        "prior_metrics_reproduction": reproduction,
        "purpose": "controlled mechanism diagnosis, not evidence of external generalization",
        "seeds": SEEDS, "scenarios": SCENARIOS, "methods": MODES, "factors": FACTORS,
        "feature_names": FEATURE_NAMES, "proper_year_half": "2024H1", "bank_year_half": "2024H2",
        "selection": "source 2025 mean 90% interval score; factor only",
        "unseen_evaluation": "artificial T1/T2 2026H1; frozen source selections",
        "point_forecast": "latest published same-calendar-month level + fixed amplitude-6 daily sine",
        "publication": "period_end + 30 days, no extra delay; native simplex gas/wind shares",
        "counts_per_seed_scenario": {"proper_rows": 216, "bank_rows": 216,
                                     "selection_rows": 72, "unseen_rows": 36},
        "selections": selections, "same_coarse_shape_identity": identities,
        "feature_diagnostics": diagnostics}
    (output / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(summary.reset_index().to_string(index=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/coarse-risk-v6-controlled"))
    run(parser.parse_args().output)
