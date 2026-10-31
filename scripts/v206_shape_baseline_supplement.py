"""Exploratory, frozen-rule point-shape baselines for the opened V206 target."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/v206-cross-simulator-transfer-r1"
RUNNER = ROOT / "scripts/v206_cross_simulator_transfer.py"
BOOTSTRAPS = 100_000
SEED = 206_207


def load_runner():
    spec = importlib.util.spec_from_file_location("v206_runner", RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import frozen V206 runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    runner = load_runner()
    source, _ = runner.read_source()
    candidates = runner.source_month_candidates(source)
    target, _, _ = runner.read_target()
    names = (
        "constant_anchor",
        "nearest_one_conditioned",
        "all30_region_balanced_conditioned",
        "k15_conditioned",
        "region_month_climatology_unconditioned",
    )
    rows = []
    manifest = OUT / "prediction_manifest.jsonl"
    for line in manifest.read_text().splitlines():
        query = json.loads(line)
        region, month = query["province"], int(query["month"])
        year, scenario = int(query["year"]), query["scenario"]
        target_mean = float(query["target_mean"])
        donors = runner.choose_donors(candidates, region, month, target_mean)
        means = np.asarray([donor[3] for donor in donors], dtype=np.float64)
        shapes = np.stack([donor[4] for donor in donors])
        order = np.argsort(np.abs(means - target_mean), kind="stable")

        region_shapes = []
        for source_region, by_month in candidates.items():
            if source_region == region:
                continue
            source_profiles = by_month[month]
            region_shapes.append(np.mean(np.stack([item[3] for item in source_profiles]), axis=0))
        estimates = {
            "constant_anchor": np.zeros(shapes.shape[1]),
            "nearest_one_conditioned": shapes[order[0]],
            "all30_region_balanced_conditioned": shapes.mean(axis=0),
            "k15_conditioned": shapes[order[: runner.K]].mean(axis=0),
            "region_month_climatology_unconditioned": np.mean(np.stack(region_shapes), axis=0),
        }
        lo, hi = runner.month_bounds(year)[month - 1]
        truth = target[(scenario, year, region)][lo:hi] - target_mean
        for name in names:
            error = estimates[name] - truth
            rows.append({
                "scenario": scenario,
                "year": year,
                "province": region,
                "month": month,
                "method": name,
                "shape_mae": float(np.mean(np.abs(error))),
                "shape_rmse": float(np.sqrt(np.mean(error**2))),
            })

    monthly = pd.DataFrame(rows)
    province = monthly.groupby(["province", "method"], as_index=False)[
        ["shape_mae", "shape_rmse"]
    ].mean()
    macro = province.groupby("method", as_index=False)[["shape_mae", "shape_rmse"]].mean()
    monthly.to_csv(OUT / "shape_baseline_supplement_monthly.csv", index=False)
    province.to_csv(OUT / "shape_baseline_supplement_province.csv", index=False)
    macro.to_csv(OUT / "shape_baseline_supplement_macro.csv", index=False)

    wide = province.pivot(index="province", columns="method", values="shape_mae")
    rng = np.random.default_rng(SEED)
    comparisons = []
    for baseline in names:
        if baseline == "k15_conditioned":
            continue
        differences = (wide["k15_conditioned"] - wide[baseline]).to_numpy()
        indices = rng.integers(0, len(differences), size=(BOOTSTRAPS, len(differences)))
        bootstrap = differences[indices].mean(axis=1)
        comparisons.append({
            "comparison": f"k15_conditioned - {baseline}",
            "difference_shape_mae": float(differences.mean()),
            "ci95_low": float(np.quantile(bootstrap, 0.025)),
            "ci95_high": float(np.quantile(bootstrap, 0.975)),
            "n_province_clusters": len(differences),
            "bootstrap_draws": BOOTSTRAPS,
            "seed": SEED,
        })
    pd.DataFrame(comparisons).to_csv(OUT / "shape_baseline_supplement_bootstrap.csv", index=False)
    print(macro.to_string(index=False))
    print(pd.DataFrame(comparisons).to_string(index=False))


if __name__ == "__main__":
    main()
