"""Plot completed gate evidence without rerunning or altering experiments."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/transcif-mpl-cache")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transcif.research.weighted_metrics import weighted_quantile


COLORS = {"prior": "#476B91", "gaussian": "#CB8B33", "diverse": "#A33D54"}
LABELS = {
    "source_template": "Source template",
    "level_template": "Level-adjusted template",
    "lagged_level": "Lagged level",
    "ridge": "Ridge",
    "extra_trees": "Extra Trees",
    "prior": "Prior IS (128)",
    "gaussian": "Gaussian proposal (128)",
    "diverse": "Diversity proposal (128)",
    "prior_2048": "Prior IS (2048)",
}
UNIT = "fixed-EF proxy gCO2/kWh"


def plot(research: Path, output: Path) -> tuple[Path, Path]:
    """Require completed gate 3 and plot the frozen main accounting scenario."""
    if (research / "gate3/SUPERSEDED.md").exists():
        raise ValueError("This run was superseded by a data timing correction; select the definitive rerun")
    summary_path = research / "gate3/summary.json"
    if not summary_path.is_file():
        raise ValueError(f"Gate 3 is incomplete: missing {summary_path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("completed") is not True:
        raise ValueError("Gate 3 summary does not report a completed experiment")
    if summary.get("region_count") != 7:
        raise ValueError("This figure requires the seven-region gate-3 experiment")

    aggregate = pd.read_csv(research / "gates12/aggregate.csv")
    macro = pd.read_csv(research / "gate3/macro_metrics.csv")
    main = macro.loc[(macro["subset"] == "all_seven") & (macro["other_ef"] == 400)].set_index("method")
    if main.index.has_duplicates or set(LABELS) - set(main.index):
        raise ValueError("Expected one main-scenario macro row for every plotted method")
    required_metrics = ["crps", "basepoint_mae", "basepoint_shape_mae"]
    if not np.isfinite(main.loc[list(LABELS), required_metrics].to_numpy()).all():
        raise ValueError("Main-scenario metrics must be finite")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titleweight": "bold", "axes.titlesize": 12,
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), layout="constrained")
    ax = axes[0, 0]
    for case, color in (("harmful", "#A33D54"), ("harmless", "#476B91"), ("resolved", "#388566")):
        with np.load(research / "gates12" / f"reference_{case}.npz") as reference:
            predictions, weights = reference["predictions"], reference["weights"]
            lower, upper = weighted_quantile(predictions, weights, np.array([0.05, 0.95]))
            mean = weights @ predictions
        hours = np.arange(len(mean))
        width = float(np.mean(upper - lower))
        ax.fill_between(hours, lower, upper, color=color, alpha=0.16)
        ax.plot(hours, mean, color=color, linewidth=1.8,
                label=f"{case.capitalize()}: mean width {width:.1f}")
    ax.set(title="1 | Controlled ambiguity: reference posterior", xlabel="Forecast hour",
           ylabel=UNIT, xticks=[0, 6, 12, 18, 23])
    ax.legend(title="Shading: central 90% interval", fontsize=9, title_fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.2)

    ax = axes[0, 1]
    for method, label in (("prior", "Prior IS"), ("gaussian", "Gaussian proposal"), ("diverse", "Diversity proposal")):
        selected = aggregate.loc[(aggregate["case"] == "harmful") & (aggregate["method"] == method)].sort_values("evaluations")
        if selected.empty or selected["evaluations"].duplicated().any():
            raise ValueError(f"Missing or duplicate gate-2 budget rows for {method}")
        ax.plot(selected["evaluations"], selected["cdf_distance"], marker="o",
                label=label, color=COLORS[method], linewidth=1.8)
    ax.set(title="2 | Harmful ambiguity: inference accuracy", xlabel="Actual decoder evaluations (including pilot)",
           ylabel="Mean CDF distance to reference (lower is better)", xscale="log", xticks=[32, 64, 128, 256])
    ax.set_xticklabels(["32", "64", "128", "256"])
    ax.minorticks_off()
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)
    ax.text(0.02, 0.03, "Mean over 30 fixed seeds; approximation diagnostic", transform=ax.transAxes, fontsize=8)

    ax = axes[1, 0]
    names = list(LABELS)
    colors = [COLORS.get(name, "#9BA9B8") for name in names]
    bars = ax.barh(np.arange(len(names)), main.loc[names, "crps"], color=colors)
    ax.set_yticks(np.arange(len(names)), [LABELS[name] for name in names], fontsize=9)
    ax.invert_yaxis()
    ax.set(title="3 | Held-region test: distribution error", xlabel=f"CRPS ({UNIT}); lower is better")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xlim(0, float(main.loc[names, "crps"].max()) * 1.17)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)

    ax = axes[1, 1]
    core = ["source_template", "level_template", "extra_trees", "diverse"]
    positions = np.arange(len(core))
    width = 0.36
    for offset, field, label, color in (
        (-width / 2, "basepoint_mae", "Point MAE", "#476B91"),
        (width / 2, "basepoint_shape_mae", "Daily-centered shape MAE", "#CB8B33"),
    ):
        bars = ax.bar(positions + offset, main.loc[core, field], width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    ax.set_xticks(positions, ["Source\ntemplate", "Level-adjusted\ntemplate", "Extra Trees", "Diversity\nproposal"], fontsize=9)
    ax.set(title="3 | Point error and within-day shape error", ylabel=f"MAE ({UNIT})")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_ylim(0, float(main.loc[core, ["basepoint_mae", "basepoint_shape_mae"]].to_numpy().max()) * 1.27)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)

    fig.suptitle(
        "TransCIF research pilot: three sequential evidence gates\n"
        "Gate 3: seven US balancing areas, 2024; equal-region macro averages; other-fuel EF = 400",
        fontsize=14, fontweight="bold",
    )
    fig.supxlabel(
        "Controlled information budget with assumed release dates; fixed-EF generation proxy. "
        "Not an issued-public deployment or measured China CO2 validation.\n"
        "Direct references use source residual calibration for CRPS; bottom-right uses uncalibrated point predictions. "
        "Point and shape errors are not additive.",
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=True)
    png, pdf = output / "three_gates.png", output / "three_gates.pdf"
    fig.savefig(png, dpi=200, metadata={"Title": "TransCIF three-gate research evidence"})
    fig.savefig(pdf, metadata={"Title": "TransCIF three-gate research evidence", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research", type=Path, default=ROOT / "results/research")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    try:
        paths = plot(args.research, args.output or args.research / "figures")
    except (ValueError, FileNotFoundError, KeyError) as error:
        parser.exit(2, f"Cannot plot completed research gates: {error}\n")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
