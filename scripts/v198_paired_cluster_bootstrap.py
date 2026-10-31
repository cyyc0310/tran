"""Paired province-cluster bootstrap for the frozen V198 time holdout."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/china-projected-cif-v198-time-holdout-r1/monthly_scores.csv"
OUTPUT = ROOT / "results/china-projected-cif-v198-time-holdout-r1/paired_cluster_bootstrap.csv"
METHOD = "v197_scaled_k15"
BASELINES = ("loo_pooled_k15", "all_available_law", "k15_law")
METRICS = ("coverage90", "width90", "winkler90")
TRACKS = ("same_scenario", "leave_scenario_out")
N_BOOT = 5_000
SEED = 198


def main() -> None:
    scores = pd.read_csv(INPUT)
    scores = scores[scores.method.isin((METHOD, *BASELINES))]
    rng = np.random.default_rng(SEED)
    rows: list[dict[str, object]] = []

    for track in TRACKS:
        track_scores = scores[
            (scores.track == track) & scores.coverage90.notna()
        ]
        method_scores = track_scores[track_scores.method == METHOD].set_index(
            ["scenario", "year", "province", "month"]
        )
        provinces = np.sort(method_scores.index.get_level_values("province").unique())

        for baseline in BASELINES:
            base_scores = track_scores[track_scores.method == baseline].set_index(
                ["scenario", "year", "province", "month"]
            )
            paired = method_scores[list(METRICS)].join(
                base_scores[list(METRICS)],
                lsuffix="_method",
                rsuffix="_baseline",
                how="inner",
            )
            province_ids = paired.index.get_level_values("province").to_numpy()
            draw_indices = rng.integers(0, len(provinces), size=(N_BOOT, len(provinces)))

            for metric in METRICS:
                delta = (
                    paired[f"{metric}_method"] - paired[f"{metric}_baseline"]
                ).to_numpy()
                province_delta = np.array(
                    [delta[province_ids == province].mean() for province in provinces]
                )
                bootstrap = province_delta[draw_indices].mean(axis=1)
                lo, hi = np.quantile(bootstrap, [0.025, 0.975])
                rows.append(
                    {
                        "track": track,
                        "method": METHOD,
                        "baseline": baseline,
                        "metric": metric,
                        "n_queries": len(paired),
                        "n_provinces": len(provinces),
                        "paired_query_mean_delta": float(delta.mean()),
                        "province_macro_mean_delta": float(province_delta.mean()),
                        "cluster_bootstrap_ci_low": float(lo),
                        "cluster_bootstrap_ci_high": float(hi),
                        "bootstrap_replicates": N_BOOT,
                        "seed": SEED,
                    }
                )

    pd.DataFrame(rows).to_csv(OUTPUT, index=False, float_format="%.10g")
    print(f"wrote {len(rows)} paired contrasts to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
