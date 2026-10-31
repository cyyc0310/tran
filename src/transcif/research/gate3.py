"""Frozen seven-area controlled-information feasibility experiment.

Prediction artifacts are persisted and hashed before each outer target's
hourly evaluation table is loaded. Retrospective monthly input construction
is a distinct preprocessing step; it is not an issued-public-data claim.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .eia_proxy import FUELS, REGIONS, latest_observation
from .inference import infer
from .mechanisms import fit, local_hours, make_evaluator
from .references import calibrate_reference_errors, fit_references, proxy_values
from .weighted_metrics import PreparedDistribution


METRICS = ("mae", "rmse", "crps", "coverage90", "width90", "shape_mae", "level_mae")
POINT_METRICS = ("basepoint_mae", "basepoint_rmse", "basepoint_shape_mae", "basepoint_level_mae")
LOW_AMBIGUITY = ("BPAT", "ERCO", "ISNE", "MISO")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def _load_source_frames(data: Path, target: str):
    frames = {}
    for source in REGIONS:
        if source == target:
            continue
        frame = pd.read_csv(data / f"{source}_hourly.csv", parse_dates=["time"])
        # Do not let even source 2024 outcomes enter fitting or residual banks.
        frames[source] = frame.loc[pd.to_datetime(frame.time, utc=True) < pd.Timestamp("2024-01-01", tz="UTC")].copy()
    return frames


def predict_fold(data: Path, output: Path, target: str, other_ef: float,
                 source_frames, monthly, model):
    """No target hourly frame is accepted or opened by this function."""
    if target in source_frames or target in model.metadata["provided_source_ids"]:
        raise ValueError("Outer target must be excluded from every fitted source")
    start = time.perf_counter()
    ef = np.array([1000, 400, 800, 0, 0, 0, 0, other_ef], dtype=float)
    references = fit_references(source_frames, monthly, ef, max_year=2023)
    residuals, calibration_counts = calibrate_reference_errors(source_frames, monthly, ef)
    arrays = {f"residual_{name}": values for name, values in residuals.items()}
    index, budgets, contexts = [], [], {}
    counter = 0
    origins = pd.date_range("2024-01-01", "2024-12-31", freq="D", tz="UTC")
    for origin in origins:
        observation = latest_observation(monthly, target, origin)
        if observation is None:
            continue
        hours = pd.date_range(origin, periods=24, freq="h")
        local = np.asarray(local_hours(hours, target))
        observed_mix = observation[[f"fuelshare_{fuel}" for fuel in FUELS]].to_numpy(float)
        key = (str(observation.period_start), tuple(local))
        if key not in contexts:
            context_name = f"c{len(contexts):03d}"
            evaluate = make_evaluator(model, observed_mix, local, ef, observation_sigma=0.06)
            context_entries = []
            for method, budget in (("prior", 128), ("gaussian", 128), ("diverse", 128), ("prior", 2048)):
                name = method if budget == 128 else "prior_2048"
                for seed in (0, 1, 2):
                    result = infer(evaluate, model.dimension, budget, seed, method)
                    artifact_key = f"{context_name}_{name}_{seed}"
                    arrays[artifact_key + "_pred"] = result.predictions
                    arrays[artifact_key + "_weights"] = result.weights
                    entry = {"method": name, "seed": seed, "key": artifact_key, "ess": result.ess}
                    context_entries.append(entry)
                    budgets.append({**entry, "context": context_name, "evaluations": result.evaluations,
                                    "elapsed_seconds": result.elapsed_seconds})
            contexts[key] = context_entries
        for entry in contexts[key]:
            index.append({**entry, "origin": origin.isoformat(), "period_end": str(observation.period_end),
                          "available_at": str(observation.available_at), "kind": "inference"})
        points = references.predict(hours, target, monthly)
        for name, values in points.items():
            artifact_key = f"p{counter:04d}_{name}"
            arrays[artifact_key] = values
            index.append({"method": name, "seed": -1, "key": artifact_key, "origin": origin.isoformat(),
                          "period_end": str(observation.period_end), "available_at": str(observation.available_at),
                          "ess": 128.0, "kind": "reference"})
        counter += 1
    output.mkdir(parents=True, exist_ok=True)
    prediction_path = output / "predictions.npz"
    np.savez_compressed(prediction_path, **arrays)
    pd.DataFrame(index).to_csv(output / "prediction_index.csv", index=False)
    pd.DataFrame(budgets).to_csv(output / "decoder_budgets.csv", index=False)
    seal = {"target": target, "sources": sorted(source_frames), "other_ef": other_ef,
            "prediction_sha256": sha256(prediction_path),
            "index_sha256": sha256(output / "prediction_index.csv"),
            "predictions_completed_before_target_hourly_load": True,
            "forecast_days": counter, "unique_inference_contexts": len(contexts),
            "calibration": "source 2022-fit / 2023 temporal residuals; no target calibration",
            "source_residual_days": calibration_counts, "model": model.metadata,
            "wall_seconds": time.perf_counter() - start}
    write_json(output / "prediction_seal.json", seal)
    return seal


def score_fold(data: Path, output: Path, target: str, other_ef: float):
    seal = json.loads((output / "prediction_seal.json").read_text())
    if seal["target"] != target or seal["other_ef"] != other_ef:
        raise ValueError("Prediction seal does not match the requested region/EF scenario")
    if sha256(output / "predictions.npz") != seal["prediction_sha256"]:
        raise ValueError("Prediction artifact changed after its seal")
    if sha256(output / "prediction_index.csv") != seal["index_sha256"]:
        raise ValueError("Prediction index changed after its seal")
    # This is the first outer-target hourly-label read in this fold.
    labels = pd.read_csv(data / f"{target}_hourly.csv", parse_dates=["time"])
    labels["day"] = pd.to_datetime(labels.time, utc=True).dt.floor("D")
    ef = np.array([1000, 400, 800, 0, 0, 0, 0, other_ef], dtype=float)
    truth_by_day = {}
    for day, frame in labels.groupby("day"):
        if day.year == 2024 and len(frame) == 24 and frame.valid.all():
            truth = proxy_values(frame.sort_values("time"), ef)
            if np.isfinite(truth).all():
                truth_by_day[day.isoformat()] = truth
    index = pd.read_csv(output / "prediction_index.csv")
    results, prepared = [], {}
    with np.load(output / "predictions.npz") as arrays:
        for record in index.to_dict("records"):
            if record["origin"] not in truth_by_day:
                continue
            key = record["key"]
            if record["kind"] == "inference":
                if key not in prepared:
                    prepared[key] = PreparedDistribution(arrays[key + "_pred"], arrays[key + "_weights"])
                distribution = prepared[key]
            else:
                scenarios = np.clip(arrays[key][None] + arrays[f"residual_{record['method']}"], 0, ef.max())
                distribution = PreparedDistribution(scenarios, np.ones(len(scenarios)))
            truth = truth_by_day[record["origin"]]
            scores = distribution.score(truth)
            point = arrays[key] if record["kind"] == "reference" else distribution.mean
            error = point - truth
            scores.update(basepoint_mae=float(np.abs(error).mean()),
                          basepoint_rmse=float(np.sqrt(np.square(error).mean())),
                          basepoint_shape_mae=float(np.abs(error - error.mean()).mean()),
                          basepoint_level_mae=float(abs(error.mean())))
            results.append({"region": target, "other_ef": other_ef, "origin": record["origin"],
                            "method": record["method"], "seed": record["seed"], "ess": record["ess"], **scores})
    frame = pd.DataFrame(results)
    frame.to_csv(output / "window_scores.csv", index=False)
    return frame


def audit_prediction_fold(output: Path):
    """Inspect an already persisted forecast without opening any hourly label."""
    seal = json.loads((output / "prediction_seal.json").read_text())
    index = pd.read_csv(output / "prediction_index.csv")
    origins = pd.to_datetime(index.origin, utc=True)
    available = pd.to_datetime(index.available_at, utc=True)
    period_end = pd.to_datetime(index.period_end, utc=True)
    if not ((available <= origins) & (period_end <= origins)).all():
        raise ValueError("Forecast uses a statistic from after its origin")
    if seal["target"] in seal["sources"] or seal["target"] in seal["model"]["provided_source_ids"]:
        raise ValueError("Target appears in fitted source provenance")
    budgets = pd.read_csv(output / "decoder_budgets.csv")
    expected = np.where(budgets.method == "prior_2048", 2048, 128)
    if not np.array_equal(budgets.evaluations.to_numpy(), expected):
        raise ValueError("Inference exceeded or omitted its charged decoder budget")
    with np.load(output / "predictions.npz") as arrays:
        for entry in budgets.itertuples():
            weights = arrays[entry.key + "_weights"]
            predictions = arrays[entry.key + "_pred"]
            if not np.isfinite(predictions).all() or (weights < 0).any() or not np.isclose(weights.sum(), 1):
                raise ValueError("Invalid archived predictive distribution")
    audit = {"passed": True, "target_hourly_labels_opened": False,
             "forecast_days": seal["forecast_days"], "decoder_calls": int(budgets.evaluations.sum()),
             "unique_inference_contexts": seal["unique_inference_contexts"]}
    write_json(output / "pre_score_audit.json", audit)
    return audit


def summarize(scores: pd.DataFrame, output: Path):
    # Average repeated Monte Carlo estimates on the same region/day first.
    scores = scores.copy()
    scores["rmse"] = scores.rmse ** 2
    scores["basepoint_rmse"] = scores.basepoint_rmse ** 2
    metric_columns = list(METRICS + POINT_METRICS)
    days = scores.groupby(["other_ef", "region", "origin", "method"], as_index=False)[metric_columns].mean()
    regions = days.groupby(["other_ef", "region", "method"], as_index=False)[metric_columns].mean()
    regions["rmse"] = np.sqrt(regions.rmse)
    regions["basepoint_rmse"] = np.sqrt(regions.basepoint_rmse)
    regions.to_csv(output / "region_metrics.csv", index=False)
    all_tables = []
    for name, selected in (("all_seven", regions), ("lower_accounting_ambiguity", regions[regions.region.isin(LOW_AMBIGUITY)])):
        table = selected.groupby(["other_ef", "method"], as_index=False)[metric_columns].mean()
        table.insert(0, "subset", name)
        all_tables.append(table)
    macro = pd.concat(all_tables, ignore_index=True)
    macro.to_csv(output / "macro_metrics.csv", index=False)
    # Synchronized month resampling preserves cross-region common-month shocks;
    # intervals describe these seven areas, not a population of independent BAs.
    main = days[days.other_ef == 400].copy()
    main["month"] = pd.to_datetime(main.origin, utc=True).dt.month
    comparisons = []
    for baseline in ("prior", "gaussian", "prior_2048", "level_template", "extra_trees", "ridge"):
        left = main[main.method == "diverse"].set_index(["region", "origin"])
        right = main[main.method == baseline].set_index(["region", "origin"])
        common = left.index.intersection(right.index)
        difference = left.loc[common, "crps"] - right.loc[common, "crps"]
        blocks = difference.rename("delta").reset_index()
        blocks["month"] = pd.to_datetime(blocks.origin, utc=True).dt.month
        grouped = blocks.groupby(["region", "month"]).delta
        sums = grouped.sum().unstack("month").reindex(columns=range(1, 13)).fillna(0).to_numpy()
        counts = grouped.count().unstack("month").reindex(columns=range(1, 13)).fillna(0).to_numpy()
        rng = np.random.default_rng(801)
        samples = []
        for _ in range(2000):
            chosen = rng.integers(0, 12, 12)
            denominator = counts[:, chosen].sum(axis=1)
            region_means = np.divide(sums[:, chosen].sum(axis=1), denominator,
                                     out=np.full(len(denominator), np.nan), where=denominator > 0)
            samples.append(float(np.nanmean(region_means)))
        region_delta = blocks.groupby("region").delta.mean()
        comparisons.append({"baseline": baseline, "diverse_minus_baseline_crps": float(region_delta.mean()),
                            "synchronized_month_bootstrap_ci95": np.quantile(samples, [.025, .975]).tolist(),
                            "regions_improved": int((region_delta < 0).sum()), "regions": len(region_delta)})
    main_table = macro[(macro.other_ef == 400) & (macro.subset == "all_seven")].set_index("method")
    diverse = main_table.loc["diverse"]
    candidate_supported = all(c["synchronized_month_bootstrap_ci95"][1] < 0 for c in comparisons
                              if c["baseline"] in ("prior", "gaussian", "level_template", "extra_trees", "ridge"))
    candidate_supported &= diverse.coverage90 >= .85
    summary = {"completed": True, "candidate_supported_in_this_pilot": bool(candidate_supported),
               "comparison_note": "Descriptive 12-month block intervals conditional on seven observed regions; seeds are not independent regions.",
               "point_metric_note": "mae/rmse/shape_mae/level_mae use predictive distribution means including source residual calibration; basepoint_* use uncalibrated direct point predictions (identical for inference methods).",
               "comparisons": comparisons,
               "main_macro": main_table.reset_index().to_dict("records"),
               "region_count": len(REGIONS), "scored_days_by_region": days[days.other_ef == 400].groupby("region").origin.nunique().to_dict(),
               "limitations": ["controlled synthetic release dates", "fixed-EF nonnegative-reported-generation proxy",
                               "static monthly persistence and low-rank Gaussian misspecification", "one test year and seven related US balancing areas",
                               "source-only temporal residual calibration for direct references", "no real China hourly validation or issued-public deployment"]}
    write_json(output / "summary.json", summary)
    return summary


def run(data: Path, output: Path):
    output.mkdir(parents=True, exist_ok=True)
    monthly = pd.read_csv(data / "monthly.csv")
    all_scores, folds = [], []
    for target in REGIONS:
        print(f"gate3: fitting sources excluding {target}", flush=True)
        frames = _load_source_frames(data, target)
        model = fit(frames, max_year=2023)
        np.savez_compressed(output / f"{target}_source_model.npz", center=model.center,
                            slow_loadings=model.slow_loadings, daily_center=model.daily_center,
                            daily_loadings=model.daily_loadings)
        for other_ef in (400, 0, 900):
            fold = output / f"{target}_other{other_ef}"
            seal = predict_fold(data, fold, target, other_ef, frames, monthly, model)
            folds.append(seal)
            audit_prediction_fold(fold)
            print(f"gate3: {target} other={other_ef} saved {seal['forecast_days']} days; now scoring", flush=True)
            all_scores.append(score_fold(data, fold, target, other_ef))
    scores = pd.concat(all_scores, ignore_index=True)
    scores.to_csv(output / "window_scores.csv", index=False)
    write_json(output / "folds.json", folds)
    return summarize(scores, output)
