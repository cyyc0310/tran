"""Reproducible controlled ambiguity and equal-budget inference experiments."""

import csv
import json
from pathlib import Path

import numpy as np

from .inference import infer
from .toy import ToyMechanism
from .weighted_metrics import distribution_metrics, weighted_quantile

BUDGETS = (32, 64, 128, 256)
REPEATS = 30
TRUTH_SEED = 73021
METHODS = (("prior", "prior", 1), ("gaussian", "gaussian", 1), ("diverse", "diverse", 1), ("prior_2B", "prior", 2), ("prior_4B", "prior", 4))


def _cdf_at(predictions: np.ndarray, weights: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    result = np.empty_like(thresholds)
    for hour in range(predictions.shape[1]):
        order = np.argsort(predictions[:, hour])
        cumulative = np.r_[0.0, np.cumsum(weights[order])]
        indexes = np.searchsorted(predictions[order, hour], thresholds[:, hour], side="right")
        result[:, hour] = cumulative[indexes]
    return result


def _reference_details(model: ToyMechanism) -> dict:
    reference = model.reference()
    predictions, weights = reference["predictions"], reference["weights"]
    thresholds = weighted_quantile(predictions, weights, np.linspace(0.025, 0.975, 39))
    truths = model.test_truths(seed=TRUTH_SEED)
    truth_metrics = [distribution_metrics(predictions, weights, truth) for truth in truths]
    reference_metrics = {
        key: float(np.mean([row[key] for row in truth_metrics]))
        for key in ("mae", "rmse", "crps", "coverage90", "width90", "shape_mae", "level_mae")
    }
    return {
        **reference, "thresholds": thresholds,
        "cdf": _cdf_at(predictions, weights, thresholds),
        "mean": weights @ predictions, "truths": truths,
        "reference_metrics": reference_metrics,
    }


def _score(result, reference: dict) -> dict[str, float]:
    truth_metrics = [distribution_metrics(result.predictions, result.weights, truth) for truth in reference["truths"]]
    metrics = {key: float(np.mean([row[key] for row in truth_metrics])) for key in truth_metrics[0]}
    positive_mass = float(result.weights[result.z[:, 0] > 0].sum())
    reference_positive_mass = float(reference["weights"][reference["z"][:, 0] > 0].sum())
    two_modes = 0.1 < reference_positive_mass < 0.9
    return {
        **metrics,
        "mean_error_to_reference": float(np.abs(result.weights @ result.predictions - reference["mean"]).mean()),
        "cdf_distance": float(np.abs(_cdf_at(result.predictions, result.weights, reference["thresholds"]) - reference["cdf"]).mean()),
        "positive_mode_mass": positive_mass,
        "mode_omission": float(two_modes and (positive_mass < 0.1 or positive_mass > 0.9)),
        "ess": result.ess,
        "evaluations": result.evaluations,
        "particles_retained": len(result.weights),
        "pilot_evaluations": result.pilot_evaluations,
        "elapsed_seconds": result.elapsed_seconds,
    }


def _paired_mean_interval(differences: np.ndarray) -> list[float]:
    rng = np.random.default_rng(65019)
    bootstrap_means = differences[rng.integers(len(differences), size=(10000, len(differences)))].mean(axis=1)
    return [float(value) for value in np.quantile(bootstrap_means, [0.025, 0.975])]


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run(output: Path) -> dict:
    """Run all prespecified conditions; failure of the candidate is a valid result.

    The primary gate-2 comparison is harmful ambiguity at B=128. A candidate
    must reduce CDF approximation error by at least 10% against *both* same-
    budget baselines, with paired-seed 95% bootstrap intervals below zero, and
    may not increase the mode-omission rate. Higher-budget prior methods have
    separate cost columns and cannot be described as equal-budget competitors.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "cases": ["harmful", "harmless", "resolved"], "budgets": list(BUDGETS),
        "repeats": REPEATS, "inference_seeds": list(range(REPEATS)), "truth_seed": TRUTH_SEED,
        "test_truth_count": 16, "reference_points": 20001,
        "primary_case": "harmful", "primary_budget": 128,
        "candidate_success": "At least 10% lower CDF error against both prior and gaussian at B=128; both paired-seed 95% bootstrap difference intervals below zero; no higher mode-omission rate.",
        "high_budget_check": {"budget": 8192, "repeats": 5},
        "numpy_version": np.__version__,
        "cdf_metric": "Mean absolute CDF difference at 39 reference quantiles across 24 hours; an approximation diagnostic, not a probability score.",
        "truth_note": "Sixteen common test curves sampled from each reference posterior using a seed independent of all inference draws; outcomes are used for scoring only.",
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    references = {case: _reference_details(ToyMechanism(case)) for case in protocol["cases"]}
    for case, reference in references.items():
        np.savez_compressed(output / f"reference_{case}.npz", **{key: value for key, value in reference.items() if isinstance(value, np.ndarray)})

    nuisance_z = np.column_stack((np.linspace(-3, 3, 25), np.zeros(25)))
    displaced_z = nuisance_z.copy()
    displaced_z[:, 1] = np.linspace(-100.0, 100.0, 25)
    nuisance_error = max(float(np.max(np.abs(ToyMechanism(case).evaluate(nuisance_z)[1] - ToyMechanism(case).evaluate(displaced_z)[1]))) for case in references)
    widths = {case: reference["reference_metrics"]["width90"] for case, reference in references.items()}
    # Verify that the quadrature, rather than a particular sampler, resolves
    # both harmful/harmless ambiguity and the extra sign-sensitive observation.
    reference_convergence = {}
    for case, reference in references.items():
        finer = ToyMechanism(case).reference(points=40001)
        reference_convergence[case] = float(np.max(np.abs(finer["weights"] @ finer["predictions"] - reference["mean"])))
    gate1 = {
        "passed": bool(widths["harmful"] > 3 * widths["harmless"] and widths["resolved"] < 0.5 * widths["harmful"] and nuisance_error < 1e-9 and max(reference_convergence.values()) < 1e-5),
        "reference_width90": widths, "common_logit_shift_max_error": nuisance_error,
        "reference_mean_grid_refinement_error": reference_convergence,
        "reference_metrics": {case: reference["reference_metrics"] for case, reference in references.items()},
    }

    rows = []
    for case, reference in references.items():
        model = ToyMechanism(case)
        for base_budget in BUDGETS:
            for seed in range(REPEATS):
                for label, method, multiplier in METHODS:
                    result = infer(model.evaluate, model.dimension, base_budget * multiplier, seed, method)
                    rows.append({"case": case, "base_budget": base_budget, "method": label, "seed": seed, **_score(result, reference)})
    _write_csv(output / "repeats.csv", rows)
    aggregate = []
    for case in references:
        for budget in BUDGETS:
            for label, _, _ in METHODS:
                subset = [row for row in rows if row["case"] == case and row["base_budget"] == budget and row["method"] == label]
                aggregate.append({"case": case, "base_budget": budget, "method": label, **{key: float(np.mean([row[key] for row in subset])) for key in subset[0] if key not in {"case", "base_budget", "method", "seed"}}})
    _write_csv(output / "aggregate.csv", aggregate)

    primary = {method: [row for row in rows if row["case"] == "harmful" and row["base_budget"] == 128 and row["method"] == method] for method in ("prior", "gaussian", "diverse")}
    candidate = np.array([row["cdf_distance"] for row in primary["diverse"]])
    comparisons = {}
    for baseline in ("prior", "gaussian"):
        errors = np.array([row["cdf_distance"] for row in primary[baseline]])
        interval = _paired_mean_interval(candidate - errors)
        reduction = float(1.0 - candidate.mean() / errors.mean())
        no_more_omissions = sum(row["mode_omission"] for row in primary["diverse"]) <= sum(row["mode_omission"] for row in primary[baseline])
        comparisons[baseline] = {"cdf_relative_reduction": reduction, "paired_difference_ci95": interval, "no_more_mode_omissions": no_more_omissions, "passed": bool(reduction >= 0.1 and interval[1] < 0 and no_more_omissions)}
    high_budget_rows = []
    for case, reference in references.items():
        for seed in range(5):
            for method in ("prior", "gaussian", "diverse"):
                result = infer(ToyMechanism(case).evaluate, 2, 8192, 10000 + seed, method)
                high_budget_rows.append({"case": case, "method": method, "seed": seed, **_score(result, reference)})
    _write_csv(output / "high_budget.csv", high_budget_rows)
    gate2 = {
        "candidate_passed": all(comparison["passed"] for comparison in comparisons.values()),
        "primary_comparisons": comparisons,
        "high_budget_mean_cdf_distance": {case: {method: float(np.mean([row["cdf_distance"] for row in high_budget_rows if row["case"] == case and row["method"] == method])) for method in ("prior", "gaussian", "diverse")} for case in references},
        "interpretation": "This gate tests sampling efficiency of a defensive proposal, not novelty of Bayesian conditioning or a new coverage guarantee.",
    }
    summary = {"gate1": gate1, "gate2": gate2, "protocol": protocol, "repeats": len(rows), "high_budget_repeats": len(high_budget_rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    return summary
