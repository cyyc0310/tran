"""Fixed-protocol outer target/seed sweep; descriptive macro aggregation."""

import json
from pathlib import Path

import numpy as np

from transcif.data.dataset import load_bundle
from transcif.experiment import run

METRICS = ("mae", "rmse", "crps", "energy_score", "pointwise_90_coverage", "mean_90_width")


def benchmark(manifest, output, targets=None, seeds=(0, 1, 2), **options):
    doc, grids = load_bundle(manifest)
    targets = list(targets) if targets is not None else list(grids)
    seeds = list(seeds)
    if (not targets or len(set(targets)) != len(targets) or not seeds or
            len(set(seeds)) != len(seeds) or any(s < 0 for s in seeds)):
        raise ValueError("nonempty unique targets and nonnegative unique seeds required")
    if any(name not in grids or grids[name].labels_path is None for name in targets):
        raise ValueError("every benchmark target must exist and declare scoring labels")
    for i, name in enumerate(targets):
        for other in targets[i + 1:]:
            if grids[name].coverage & grids[other].coverage:
                raise ValueError("benchmark targets must have disjoint coverage; select targets explicitly")
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError("benchmark output must be empty")
    output.mkdir(parents=True, exist_ok=True)
    records, by_target = [], {}
    for i, target in enumerate(targets):
        results = []
        for seed in seeds:
            folder = f"target-{i:03d}-seed-{seed}"
            result = run(manifest, output / folder, target=target, seed=seed, **options)
            if not result["target_scored"]:
                raise ValueError("benchmark requires target scoring")
            records.append({"target": target, "seed": seed, "run": folder,
                            "metrics": result["metrics"], "transfer": result["transfer"]})
            results.append(result["metrics"])
        by_target[target] = {method: {metric: float(np.mean([r[method][metric] for r in results]))
            for metric in METRICS} for method in results[0]}
    methods = next(iter(by_target.values()))
    macro = {method: {metric: float(np.mean([by_target[t][method][metric] for t in targets]))
        for metric in METRICS} for method in methods}
    jurisdictions = sorted({grids[t].jurisdiction for t in targets})
    by_country = {country: {method: {metric: float(np.mean([
        by_target[t][method][metric] for t in targets if grids[t].jurisdiction == country]))
        for metric in METRICS} for method in methods} for country in jurisdictions}
    country_macro = {method: {metric: float(np.mean([
        by_country[c][method][metric] for c in jurisdictions])) for metric in METRICS} for method in methods}
    summary = {"track": doc["track"], "targets": targets, "seeds": seeds,
        "protocol": options, "per_target_seed_average": by_target,
        "target_macro": macro, "per_jurisdiction": by_country, "jurisdiction_macro": country_macro,
        "runs": records,
        "limitations": ["descriptive seed-averaged summaries, not confidence intervals",
                        "outer scores must not be used to retune this benchmark"]}
    (output / "benchmark.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary
