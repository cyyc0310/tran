"""Plot V2 inference, dynamic validation and delayed-information sensitivity."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/v2-figures"))
    args = parser.parse_args()
    root = args.results
    args.output.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), layout="constrained")
    methods = ["prior", "sobol", "local_mixture_qmc", "goal_haar", "goal_cdf_haar", "smc"]
    labels = ["Prior IS", "Sobol IS", "Local QMC", "Mean-risk Haar", "CDF-risk Haar", "SMC"]
    for axis, suite, title in zip(axes[0], ("development", "stress"),
                                 ("Five development cases", "Eight fixed stress cases")):
        frame = pd.read_csv(root / "gate2-v2-20261023" / suite / "macro.csv").set_index(["method", "budget"])
        positions = np.arange(len(methods))
        for offset, budget, color in ((-.18, 128, "#496c91"), (.18, 512, "#c48a35")):
            values = [frame.loc[(method, budget), "cdf_distance"] for method in methods]
            bars = axis.bar(positions + offset, values, width=.34, label=f"Budget {budget}", color=color)
            axis.bar_label(bars, fmt="%.3f", fontsize=8)
        axis.set_xticks(positions, labels, rotation=22, ha="right")
        axis.set_ylabel("Mean marginal CDF discrepancy (lower is better)")
        axis.set_title(title, weight="bold")
        axis.legend(frameon=False)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
    base = pd.read_csv(root / "dynamic-v2-20261023-r2" / "macro_metrics.csv").set_index("method")
    dynamic = ["calendar", "level_template", "ridge", "static", "age_global", "age_time", "age_dct", "age_haar", "quarterly_haar"]
    dynamic_labels = ["Calendar", "Level + template", "Ridge", "Static correction", "Global age", "Time age", "DCT age", "Haar age", "Quarterly Haar"]
    values = [base.loc[method, "crps"] for method in dynamic]
    bars = axes[1, 0].barh(dynamic_labels, values,
                           color=["#496c91" if method == "level_template" else "#9ba9b6" for method in dynamic])
    axes[1, 0].bar_label(bars, fmt="%.1f", padding=3)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("CRPS: fixed-EF generation proxy gCO2/kWh (lower is better)")
    axes[1, 0].set_xlim(0, max(values) * 1.14)
    axes[1, 0].set_title("2023 leave-region-out source validation: seven US regions", weight="bold")
    delayed = [base,
               pd.read_csv(root / "dynamic-v2-20261023-r3-extra30" / "macro_metrics.csv").set_index("method"),
               pd.read_csv(root / "dynamic-v2-20261023-r3-extra90" / "macro_metrics.csv").set_index("method")]
    for method, label, color in (("level_template", "Level + template", "#496c91"),
                                  ("static", "Static correction", "#9ba9b6"),
                                  ("age_dct", "DCT age", "#c48a35"),
                                  ("age_haar", "Haar age", "#a43e59")):
        axes[1, 1].plot([30, 60, 120], [frame.loc[method, "crps"] for frame in delayed],
                        "o-", label=label, color=color)
    axes[1, 1].set_xticks([30, 60, 120])
    axes[1, 1].set_xlabel("Assumed monthly release lag (days after month end)")
    axes[1, 1].set_ylabel("CRPS: fixed-EF proxy gCO2/kWh")
    axes[1, 1].set_title("Target information delayed; fitted model unchanged", weight="bold")
    axes[1, 1].legend(frameon=False)
    axes[1, 1].grid(alpha=.2)
    fig.suptitle("TransCIF V2: CDF-risk allocation and age-aware multiscale prototypes", fontsize=16, weight="bold")
    fig.supxlabel("Exploratory evidence; no candidate superiority or China validation established. "
                  "Synthetic references have finite grid resolution; real data use fixed emission factors and assumed releases.", fontsize=9)
    fig.savefig(args.output / "v2_experiments.png", dpi=180)
    fig.savefig(args.output / "v2_experiments.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
