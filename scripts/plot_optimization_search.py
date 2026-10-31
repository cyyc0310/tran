"""Summarize frozen V3 results; this script never fits or selects a model."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from transcif.research.search_analysis import paired_month_interval


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    structural = args.results / "structural-v3-2025-confirmation"
    joint = args.results / "joint-v3-2026h1-confirmation"
    s25 = pd.read_csv(structural / "macro_metrics.csv")
    r25 = pd.read_csv(structural / "region_metrics.csv")
    s26 = pd.read_csv(joint / "macro_metrics.csv")
    days = pd.read_csv(joint / "day_metrics.csv")
    # Quality subset was frozen before the first 2026 label was scored.
    counts = days[days.method.eq("joint_0") & days.delay.eq(0)].groupby("region").origin.nunique()
    quality_regions = counts[counts >= 91].index
    intervals = []
    for subset, frame in [("all7", days), ("quality6", days[days.region.isin(quality_regions)])]:
        for delay, group in frame.groupby("delay"):
            for candidate, baseline in [("joint_0.5", "joint_0"), ("joint_source_alpha", "joint_0"),
                                        ("joint_0", "latest"), ("joint_0.5", "latest")]:
                intervals.append({"subset": subset, "total_lag_days": int(delay) + 30,
                                  "candidate": candidate, "baseline": baseline,
                                  **paired_month_interval(group, candidate, baseline)})
    pd.DataFrame(intervals).to_csv(args.output / "2026_month_intervals.csv", index=False)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "#f7f9fc", "axes.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.3), constrained_layout=True)
    colors = {"latest": "#798695", "seasonal": "#087e8b", "analog_shape_half": "#da8b20",
              "joint_0": "#087e8b", "joint_0.5": "#6851a4"}
    for method, label in [("latest", "Latest coarse level"), ("seasonal", "Same-month history"),
                          ("analog_shape_half", "Latest + analog shape")]:
        data = s25[s25.method.eq(method)].sort_values("delay")
        axes[0, 0].plot(data.delay + 30, data.crps, "o-", color=colors[method], label=label, lw=2)
    axes[0, 0].set(title="A  |  2025: coarse level has the larger benefit", ylabel="Equal-region CRPS (lower is better)",
                   xlabel="Assumed total release lag (days)", xticks=[30, 60, 120])
    axes[0, 0].legend(frameon=False, fontsize=9)
    reg = r25[r25.delay.eq(0)].pivot(index="region", columns="method", values="crps")
    gain = (reg.latest - reg.seasonal).sort_values()
    axes[0, 1].barh(gain.index, gain, color=["#b95e57" if x < 0 else "#087e8b" for x in gain])
    axes[0, 1].axvline(0, color="#798695", lw=1)
    axes[0, 1].set(title="B  |  2025: regional gains are heterogeneous", xlabel="Latest CRPS − seasonal CRPS, 30-day lag")
    sampler = pd.read_csv(args.results / "sampler-search-v3-20261023-audited/round3/macro.csv")
    x = np.arange(2)
    for offset, method, color in [(-.18, "sobol", "#798695"), (.18, "antithetic", "#6851a4")]:
        data = sampler[sampler.method.eq(method)].sort_values("budget")
        axes[1, 0].bar(x + offset, data.cdf_distance, width=.34, label=method.title(), color=color)
    axes[1, 0].set(title="C  |  New compound cases: sampler gains uncertain", xticks=x,
                   xticklabels=["128 calls", "512 calls"], ylabel="CDF distance (lower is better)")
    axes[1, 0].legend(frameon=False)
    axes[1, 0].text(.03, .97, "All 12 paired case-budget intervals cross zero.\nAntithetic CRPS excess is worse at 512 calls.",
                    transform=axes[1, 0].transAxes, va="top", fontsize=8.5)
    axes[1, 0].set_ylim(0, .18)
    for method, label in [("latest", "Latest coarse level"), ("joint_0", "Same-month history"),
                          ("joint_0.5", "Fixed level + shape, α = 0.5")]:
        data = s26[s26.method.eq(method)].sort_values("delay")
        axes[1, 1].plot(data.delay + 30, data.crps, "o-", color=colors[method], label=label, lw=2)
    axes[1, 1].set(title="D  |  2026 H1: small fixed shape increment", xlabel="Assumed total release lag (days)",
                   ylabel="Equal-region CRPS (lower is better)", xticks=[30, 60, 120])
    axes[1, 1].legend(frameon=False, fontsize=9)
    fig.suptitle("TransCIF optimization search: information, level, shape, and inference", fontsize=15, weight="bold")
    fig.savefig(args.output / "optimization_search.png", dpi=180)
    fig.savefig(args.output / "optimization_search.pdf")
    manifest = {"purpose": "Post-score descriptive visualization; no model selection",
                "2026_quality_regions": list(quality_regions), "2026_unique_region_days": int(counts.sum()),
                "2026_bootstrap_note": "Six synchronized months; observed regions only, no global-grid guarantee"}
    (args.output / "summary.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
