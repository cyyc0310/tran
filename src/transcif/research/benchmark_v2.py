"""Auditable exploration of budgeted, prediction-oriented posterior inference."""

import csv
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import scipy

from .gates12 import _cdf_at, _paired_mean_interval
from .inference_v2 import METHODS, infer_v2
from .stress_cases import DEVELOPMENT_CASES, STRESS_CASES
from .weighted_metrics import PreparedDistribution, weighted_quantile


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _trim(reference):
    keep = reference["weights"] >= 1e-15
    removed = float(reference["weights"][~keep].sum())
    result = {key: value[keep] if isinstance(value, np.ndarray) else value
              for key, value in reference.items()}
    result["weights"] = result["weights"] / result["weights"].sum()
    result["removed_mass"] = removed
    return result


def reference_details(case):
    started = perf_counter()
    coarse = _trim(case.reference(1201, 161))
    finer = _trim(case.reference(2401, 321))
    thresholds = weighted_quantile(finer["predictions"], finer["weights"], np.linspace(.025, .975, 39))
    cdf = _cdf_at(finer["predictions"], finer["weights"], thresholds)
    cdf_difference = np.abs(_cdf_at(coarse["predictions"], coarse["weights"], thresholds) - cdf)
    endpoints = weighted_quantile(finer["predictions"], finer["weights"], np.array([.05, .95]))
    coarse_endpoints = weighted_quantile(coarse["predictions"], coarse["weights"], np.array([.05, .95]))
    mean = finer["weights"] @ finer["predictions"]
    rng = np.random.default_rng(96311)
    truths = finer["predictions"][rng.choice(len(finer["weights"]), 16, p=finer["weights"])]
    prepared = PreparedDistribution(finer["predictions"], finer["weights"])
    scores = {key: float(np.mean([prepared.score(truth)[key] for truth in truths]))
              for key in ("crps", "coverage90", "width90", "mae", "shape_mae")}
    return {"thresholds": thresholds, "cdf": cdf, "mean": mean, "truths": truths,
            "endpoints": endpoints, "reference_scores": scores,
            "audit": {"coarse_evaluations": coarse["quadrature_evaluations"],
                      "fine_evaluations": finer["quadrature_evaluations"],
                      "fine_retained": len(finer["weights"]),
                      "fine_removed_mass": finer["removed_mass"],
                      "cdf_refinement_mean": float(cdf_difference.mean()),
                      "cdf_refinement_max": float(cdf_difference.max()),
                      "endpoint_refinement_mean": float(np.abs(endpoints - coarse_endpoints).mean()),
                      "endpoint_refinement_max": float(np.abs(endpoints - coarse_endpoints).max()),
                      "reference_seconds": perf_counter() - started}}


def joint_scores(predictions, weights, truths):
    """Exact forecast-observation terms, MC independent forecast pairs for ES.

    Fixed 512 iid pairs estimate E||X-X'||. This is not a learned score or
    comparison of signal power. Variogram score uses exact weighted moments.
    """
    rng = np.random.default_rng(401991)
    pair = rng.choice(len(weights), size=(2, 512), p=weights)
    pair_distance = np.linalg.norm(predictions[pair[0]] - predictions[pair[1]], axis=1) / np.sqrt(24)
    first = np.linalg.norm(predictions[:, None, :] - truths[None, :, :], axis=2) / np.sqrt(24)
    energy = float(np.mean(weights @ first) - .5 * pair_distance.mean())
    left, right = np.triu_indices(24, 1)
    predicted = weights @ np.sqrt(np.abs(predictions[:, left] - predictions[:, right]))
    observed = np.sqrt(np.abs(truths[:, left] - truths[:, right]))
    variogram = float(np.mean(np.square(predicted[None, :] - observed)))
    return {"energy_score_mc": energy, "variogram_score": variogram}


def score_result(result, reference):
    prepared = PreparedDistribution(result.predictions, result.weights)
    scores = [prepared.score(truth) for truth in reference["truths"]]
    averaged = {key: float(np.mean([row[key] for row in scores])) for key in scores[0]}
    return {**averaged,
            "crps_excess_to_reference": averaged["crps"] - reference["reference_scores"]["crps"],
            "cdf_distance": float(np.abs(_cdf_at(result.predictions, result.weights,
                                                  reference["thresholds"]) - reference["cdf"]).mean()),
            "mean_error_to_reference": float(np.abs(prepared.mean - reference["mean"]).mean()),
            "endpoint_error_to_reference": float(np.abs(np.stack([prepared.lower, prepared.upper])
                                                         - reference["endpoints"]).mean()),
            **joint_scores(result.predictions, result.weights, reference["truths"]),
            "ess": result.ess, "evaluations": result.evaluations,
            "retained": len(result.weights), "setup_evaluations": result.pilot_evaluations,
            "inference_seconds": result.elapsed_seconds}


def _hashes(root):
    paths = [root / "src/transcif/research" / name for name in
             ("inference.py", "toy.py", "gates12.py", "weighted_metrics.py", "inference_v2.py",
              "representations.py", "stress_cases.py", "benchmark_v2.py")]
    paths += [root / "scripts/run_gate2_v2.py", root / "docs/GATE2_V2_PROTOCOL.md"]
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def run(output, root, suite="development"):
    output, root = Path(output), Path(root)
    if output.exists() and any(output.iterdir()):
        raise ValueError("Use a fresh output directory; existing experiments are immutable")
    output.mkdir(parents=True, exist_ok=True)
    if suite not in {"development", "stress"}:
        raise ValueError("Unknown suite")
    cases = DEVELOPMENT_CASES if suite == "development" else STRESS_CASES
    seeds = list(range(2000, 2010)) if suite == "development" else list(range(4000, 4020))
    protocol = {"suite": suite, "status": "exploratory; no model promotion or novelty established",
                "cases": [case.metadata() for case in cases], "budgets": [128, 512],
                "seeds": seeds, "methods": list(METHODS), "reference_truth_seed": 96311,
                "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                "sha256": _hashes(root)}
    write_json(output / "protocol.json", protocol)
    (output / "frozen_protocol.md").write_bytes((root / "docs/GATE2_V2_PROTOCOL.md").read_bytes())
    rows, diagnostic_rows, reference_audits = [], [], {}
    for case in cases:
        reference = reference_details(case)
        reference_audits[case.name] = {**reference["audit"], **reference["reference_scores"]}
        np.savez_compressed(output / f"reference_{case.name}.npz", **{
            key: reference[key] for key in ("thresholds", "cdf", "mean", "truths", "endpoints")})
        for budget in protocol["budgets"]:
            for seed in seeds:
                for method in METHODS:
                    counted = [0]

                    def evaluate(z):
                        counted[0] += len(z)
                        if counted[0] > budget:
                            raise AssertionError("Sampler exceeded forward-row budget")
                        return case.evaluate(z)

                    result = infer_v2(evaluate, case.dimension, budget, seed, method)
                    if counted[0] != result.evaluations or result.evaluations > budget:
                        raise AssertionError("Sampler cost report does not match evaluator")
                    rows.append({"case": case.name, "dimension": case.dimension, "budget": budget,
                                 "seed": seed, "method": method, **score_result(result, reference)})
                    diagnostic_rows.append({"case": case.name, "budget": budget, "seed": seed,
                                            "method": method, **result.diagnostics})
        write_csv(output / "repeats.csv", rows)
        write_json(output / "diagnostics.json", diagnostic_rows)
        write_json(output / "reference_audit.json", reference_audits)
        print(f"Completed {case.name}: {len(rows)} trials", flush=True)
    metric_keys = [key for key in rows[0] if key not in {"case", "dimension", "budget", "seed", "method"}]
    aggregate = []
    comparisons = []
    for case in cases:
        for budget in protocol["budgets"]:
            for method in METHODS:
                selected = [r for r in rows if r["case"] == case.name and r["budget"] == budget and r["method"] == method]
                aggregate.append({"case": case.name, "budget": budget, "method": method,
                                  **{key: float(np.mean([r[key] for r in selected])) for key in metric_keys}})
            for candidate in ("goal_time", "goal_dct", "goal_haar", "goal_cdf_time", "goal_cdf_dct", "goal_cdf_haar"):
                for baseline in ("sobol", "local_mixture_qmc", "smc"):
                    values = {m: np.array([r["cdf_distance"] for r in rows if r["case"] == case.name
                                          and r["budget"] == budget and r["method"] == m])
                              for m in (candidate, baseline)}
                    difference = values[candidate] - values[baseline]
                    low, high = _paired_mean_interval(difference)
                    comparisons.append({"case": case.name, "budget": budget, "candidate": candidate,
                                        "baseline": baseline, "cdf_difference": float(difference.mean()),
                                        "relative_reduction": float(1 - values[candidate].mean() / values[baseline].mean()),
                                        "descriptive_ci_low": low, "descriptive_ci_high": high})
    write_csv(output / "aggregate.csv", aggregate)
    write_csv(output / "comparisons.csv", comparisons)
    macro = []
    for budget in protocol["budgets"]:
        for method in METHODS:
            selected = [r for r in aggregate if r["budget"] == budget and r["method"] == method]
            macro.append({"budget": budget, "method": method,
                          **{key: float(np.mean([r[key] for r in selected])) for key in metric_keys}})
    write_csv(output / "macro.csv", macro)
    current = _hashes(root)
    mismatches = [name for name in current if current[name] != protocol["sha256"][name]]
    summary = {"trials": len(rows), "cases": len(cases), "methods": len(METHODS),
               "all_actual_budgets_verified": True, "frozen_hash_mismatches": mismatches,
               "status": protocol["status"], "macro": macro, "reference_audits": reference_audits}
    write_json(output / "summary.json", summary)
    if mismatches:
        raise RuntimeError("Code changed during experiment; output is not a frozen run")
    return summary
