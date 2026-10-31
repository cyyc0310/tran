"""Post-score presentation correction using only immutable fixed-role tables.

This supplementary plot fixes semantic role colors across panels. Its five
displayed roles are chosen by their predeclared comparison meaning, never by
target performance. Frozen analysis code, decisions and score values are intact.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


DISPLAY_ROLES = ("primary", "baseline", "source2_primary", "source2_simple_anchor", "source2_noanchor")
REGIONS = ("AZPS", "DUK", "LGEE", "PACE", "PSCO", "SRP")
DELAYS = (0, 30, 90)
METRICS = ("empirical_coverage90", "empirical_width90")
STYLES = {
    "primary": {"color": "#0072B2", "marker": "o", "linestyle": "-", "label": "Joint primary (R1)"},
    "baseline": {"color": "#6B6B6B", "marker": "s", "linestyle": "-", "label": "Source-selected global reference"},
    "source2_primary": {"color": "#D55E00", "marker": "^", "linestyle": "--", "label": "R2 structured + anchor"},
    "source2_simple_anchor": {"color": "#009E73", "marker": "D", "linestyle": "--", "label": "R2 simple global + anchor"},
    "source2_noanchor": {"color": "#CC79A7", "marker": "+", "linestyle": ":", "label": "R2 structured, no anchor"},
}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_fixed_tables(directory):
    """Validate existing fixed rows and averaging; do not rank any candidate."""
    directory = Path(directory)
    macro = pd.read_csv(directory / "fixed_roles_macro_metrics.csv")
    regional = pd.read_csv(directory / "fixed_roles_region_delay_metrics.csv")
    if macro.duplicated("role").any() or regional.duplicated(["role", "region", "delay"]).any():
        raise ValueError("Fixed-role tables contain duplicate rows.")
    if not set(DISPLAY_ROLES).issubset(macro.role) or not set(DISPLAY_ROLES).issubset(regional.role):
        raise ValueError("Both fixed tables must retain all five predeclared display roles.")
    selected = macro.set_index("role").loc[list(DISPLAY_ROLES)]
    detailed = regional.loc[regional.role.isin(DISPLAY_ROLES)].copy()
    for frame in (selected, detailed):
        if not np.isfinite(frame[list(METRICS)]).all().all():
            raise ValueError("Coverage and width must be finite.")
        if not frame.empirical_coverage90.between(0, 1).all() or frame.empirical_width90.lt(0).any():
            raise ValueError("Invalid coverage or interval width.")
    for role in DISPLAY_ROLES:
        part = detailed.loc[detailed.role.eq(role)]
        if set(zip(part.region, part.delay)) != {(region, delay) for region in REGIONS for delay in DELAYS}:
            raise ValueError("Each fixed display role must retain all six regions and all three delays.")
        for key in ("variant", "method", "calibration"):
            if not part[key].eq(selected.loc[role, key]).all():
                raise ValueError("Macro and regional fixed configurations differ.")
        if not np.allclose(part[list(METRICS)].mean(), selected.loc[role, list(METRICS)].to_numpy(dtype=float),
                           rtol=1e-10, atol=1e-10):
            raise ValueError("Fixed macro differs from equal-region/equal-delay means.")
    return selected, detailed


def plot(directory, output=None):
    directory = Path(directory).resolve()
    output = Path(output).resolve() if output is not None else directory
    paths = [output / f"confirmation_summary.{extension}" for extension in ("png", "pdf", "json")]
    if any(path.exists() for path in paths):
        raise FileExistsError("Supplementary confirmation outputs already exist; refusing to overwrite.")
    inputs = [directory / name for name in ("fixed_roles_macro_metrics.csv", "fixed_roles_region_delay_metrics.csv")]
    input_hashes = {str(path): sha(path) for path in inputs}
    macro, regional = read_fixed_tables(directory)
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output / ".presentation-matplotlib-cache"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    fig.subplots_adjust(left=.075, right=.985, top=.82, bottom=.32, wspace=.28)
    for order, role in enumerate(DISPLAY_ROLES):
        style = STYLES[role]
        row = macro.loc[role]
        # Semantic mapping is reused explicitly; no panel-local color cycle.
        axes[0].scatter(row.empirical_width90, row.empirical_coverage90, color=style["color"],
            marker=style["marker"], s=110 if role != "source2_noanchor" else 170,
            linewidths=1.5, label=style["label"], zorder=3 + order)
        values = regional.loc[regional.role.eq(role)].groupby("region")[list(METRICS)].mean().reindex(REGIONS)
        axes[1].plot(REGIONS, values.empirical_coverage90, color=style["color"],
            marker=style["marker"], linestyle=style["linestyle"], linewidth=1.5,
            markersize=7 if role != "source2_noanchor" else 10, label=style["label"], zorder=3 + order)
    for ax in axes:
        ax.axhline(.9, color="#111111", linestyle=(0, (4, 3)), linewidth=1, zorder=1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(axis="y", color="#EEEEEE", linewidth=.7, zorder=0)
    axes[0].set(xlabel="90% interval width (fixed-EF proxy gCO2/kWh)", ylabel="Hourly coverage",
                title="Macro coverage and width", ylim=(.6, 1.015), xlim=(185, 270))
    axes[1].set(xlabel="New confirmation region", ylabel="Hourly coverage",
                title="Coverage by region (equal mean over delays)", ylim=(.2, 1.015))
    axes[0].text(.98, .9, "Nominal 90%", ha="right", va="bottom", transform=axes[0].get_yaxis_transform(), fontsize=8)
    fig.suptitle("2026 H1 confirmation: fixed comparisons across six new regions", fontsize=14, y=.96)
    fig.text(.5, .875, "Source choices stay fixed; higher coverage can come with wider intervals.",
             ha="center", fontsize=10, color="#444444")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .145), ncol=3,
               frameon=False, fontsize=9, columnspacing=1.5, handletextpad=.6)
    fig.text(.075, .11, "Display subset: five pre-fixed mechanism roles; CQR, unconstrained and historical references remain in the fixed tables.",
             fontsize=8.5, color="#444444")
    fig.text(.075, .075, "R1 primary and R2 noanchor overlap; R2 structured + anchor is nearly identical at this scale. Dashed horizontal line: nominal 90%.",
             fontsize=8.5, color="#444444")
    fig.text(.075, .04, "Post-score presentation correction only: consistent role colors and short labels; no change to frozen values, selections or decisions.",
             fontsize=8.5, color="#444444")
    for extension in ("png", "pdf"):
        fig.savefig(output / f"confirmation_summary.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    if any(sha(path) != digest for path, digest in input_hashes.items()):
        raise RuntimeError("Fixed input tables changed while rendering.")
    metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), "presentation_only": True,
        "post_score_presentation_correction": True, "selection_performed": False,
        "frozen_values_or_decisions_changed": False, "display_roles": list(DISPLAY_ROLES), "styles": STYLES,
        "display_rule": "Five pre-fixed mechanism-comparison roles by semantic identity; no target-score ranking",
        "input_hashes": input_hashes, "script_sha256": sha(__file__),
        "output_hashes": {path.name: sha(path) for path in paths[:2]},
        "omitted_from_display_only": ["cqr", "unconstrained_baseline", "historical_v5"]}
    paths[2].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True, help="Existing joint analysis directory containing fixed-role CSVs")
    parser.add_argument("--output", type=Path, help="Optional supplemental output directory; defaults to --analysis")
    args = parser.parse_args()
    print(json.dumps({"created": [str(path) for path in plot(args.analysis, args.output)], "presentation_only": True}))


if __name__ == "__main__":
    main()
