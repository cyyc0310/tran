"""Paired region-cluster bootstrap for V380 hourly interval-score differences."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/v380-mix-similarity-calibration-r1/region_month_scores.csv"
OUT = ROOT / "results/v380-mix-similarity-calibration-r1/paired_region_bootstrap.csv"
METHODS = ("global_full_mix_blend_0.25", "global_full_mix_blend_0.50", "full_mix_tv_kernel_bw_0.40")
N_BOOT = 100_000
SEED = 20261031


def main() -> None:
    data = pd.read_csv(INPUT)
    rows = []
    rng = np.random.default_rng(SEED)
    for alpha in (0.05, 0.10):
        sample = data.loc[data["alpha"].eq(alpha) & data["threshold_g_kwh"].eq(300.0)]
        baseline = sample.loc[sample["method"].eq("global_local_asymmetric")].groupby("outer_region")["hourly_interval_score"].mean()
        for method in METHODS:
            candidate = sample.loc[sample["method"].eq(method)].groupby("outer_region")["hourly_interval_score"].mean()
            difference = (candidate - baseline).reindex(baseline.index).to_numpy(float)
            draws = rng.choice(difference, size=(N_BOOT, len(difference)), replace=True).mean(axis=1)
            rows.append({
                "alpha": alpha,
                "nominal_coverage": 1.0 - alpha,
                "threshold_g_kwh": 300.0,
                "method": method,
                "region_mean_score_difference_vs_global": float(difference.mean()),
                "regions_better": int(np.sum(difference < 0.0)),
                "regions_total": int(len(difference)),
                "bootstrap_95_low": float(np.quantile(draws, 0.025)),
                "bootstrap_95_high": float(np.quantile(draws, 0.975)),
                "bootstrap_probability_improvement": float(np.mean(draws < 0.0)),
                "seed": SEED,
                "bootstrap_draws": N_BOOT,
            })
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
