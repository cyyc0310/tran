"""Generate publication data figures for docs/paper/2026-07-22-transcif-full-paper.md.

All numerical values are copied verbatim from docs/experiments/
2026-07-17-all-experiments-summary.md. Each block cites the section.

Run:  python3 scripts/make_paper_figures.py
Outputs 300-dpi PNGs into docs/paper/figures/.

Style -- strict IEEE Transactions minimalism:
  * Two-colour palette: dark navy + neutral grey. Light grey for reference fills.
  * White background, thin black axes, faint gridlines.
  * No callout chips, no stars, no arrows pointing at features, no headline boxes.
  * Hatching only where strictly needed for B/W print distinction.
  * Times New Roman 8 pt body, 9 pt panel titles.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================================
# Restrained palette -- only what an IEEE typesetter would permit
# =====================================================================================
NAVY    = "#1F3A5F"   # primary accent (winners, +D+E, Term 1)
STEEL   = "#5A7AA0"   # secondary accent (baseline, Term 2)
GREY_D  = "#5A5A5A"   # dark grey (reference floor, persistence)
GREY_M  = "#8E8E8E"   # mid grey (neutral / auxiliary)
GREY_L  = "#C8C8C8"   # light grey (reference fill)
GREY_XL = "#ECECEC"   # extra light (background tints)
INK     = "#1A1A1A"   # near-black: text, axes

# semantic aliases
C_PERS = GREY_D
C_BASE = STEEL
C_DE   = NAVY
C_T1   = NAVY
C_T2   = STEEL
C_CLIM = GREY_M

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "font.family": "Times New Roman",
    "font.size": 8,
    "axes.titlesize": 9, "axes.titleweight": "bold",
    "axes.labelsize": 8.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.7, "axes.edgecolor": INK,
    "axes.grid": True, "axes.axisbelow": True,
    "grid.color": GREY_L, "grid.alpha": 0.55, "grid.linewidth": 0.4,
    "legend.frameon": False, "legend.fontsize": 7.5,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.direction": "in", "ytick.direction": "in",
    "mathtext.fontset": "stix",
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "docs", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

PERSISTENCE_SA1 = 67.568  # docs summary S0


def save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


# =====================================================================================
# Figure 2 -- Theorem 1 numerical decomposition (S3.1 + S4.4)
# =====================================================================================
def fig2_theorem1() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.0, 2.9))

    # ---- Panel (a): stacked bars ----------------------------------------------------
    labels = ["baseline", "+D+E"]
    term1 = [84.413, 73.966]
    term2 = [22.123, 20.288]
    x = np.arange(len(labels))
    w = 0.5
    axA.bar(x, term1, w, color=C_T1, edgecolor=INK, linewidth=0.5,
            label="Term (1), transfer amplification")
    axA.bar(x, term2, w, bottom=term1, color=C_T2, edgecolor=INK,
            linewidth=0.5, label="Term (2), residual estimation")

    for i, (t1, t2) in enumerate(zip(term1, term2)):
        total = t1 + t2
        # total label above stack
        axA.text(i, total + 2.5, f"{total:.1f}", ha="center", va="bottom",
                 fontsize=7.5, color=INK)
        # segment values, small and inside
        axA.text(i, t1 / 2, f"{t1:.1f}", ha="center", va="center",
                 fontsize=7, color="white")
        axA.text(i, t1 + t2 / 2, f"{t2:.1f}", ha="center", va="center",
                 fontsize=7, color="white")
        # Term (1) share as a quiet right-side annotation
        share = 100 * t1 / total
        axA.text(i + w / 2 + 0.05, total / 2,
                 f"({share:.1f}%)\n",
                 ha="left", va="center", fontsize=7, color=GREY_D,
                 style="italic")

    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_ylim(0, 120)
    axA.set_ylabel(r"mean $|$component$|$  (gCO$_2$/kWh)")
    axA.set_title(r"(a)  Error decomposition, SA1 calib.")
    axA.legend(loc="upper right", fontsize=6.8)
    axA.grid(axis="x", visible=False)

    # ---- Panel (b): Term 1 share vs L_T ---------------------------------------------
    regions = ["SA1", "QLD1", "NSW1", "VIC1"]
    lt = [490.43, 841.59, 875.14, 1160.12]
    base_share = [79.66, 83.71, 89.28, 91.84]
    de_share   = [78.84, 82.51, 88.98, 90.50]

    axB.plot(lt, base_share, "o-", color=C_BASE,
             markersize=5, markeredgecolor=INK, markeredgewidth=0.4,
             linewidth=1.1, label="baseline")
    axB.plot(lt, de_share, "s--", color=C_DE,
             markersize=5, markeredgecolor=INK, markeredgewidth=0.4,
             linewidth=1.1, label="+D+E")

    for xi, yi, r in zip(lt, base_share, regions):
        axB.annotate(r, (xi, yi), textcoords="offset points", xytext=(4, 5),
                     fontsize=7, color=INK)

    axB.axhline(50, color=GREY_D, linestyle=":", linewidth=0.8)
    axB.text(lt[0], 51, "50% threshold", fontsize=6.8, color=GREY_D,
             style="italic")

    axB.set_ylim(45, 100)
    axB.set_xlabel(r"$L_T = |C_{ren}-C_{non}|$  (gCO$_2$/kWh)")
    axB.set_ylabel("Term (1) share  (%)")
    axB.set_title(r"(b)  Term (1) dominance, 4 regions")
    axB.legend(loc="lower right")

    fig.tight_layout()
    save(fig, "fig2_theorem1_decomposition.png")


# =====================================================================================
# Figure 3 -- per-region persistence comparison (S4.5)
# =====================================================================================
def fig3_region_persistence() -> None:
    regions = ["QLD1", "NSW1", "VIC1", "SA1"]
    pers = [103.181, 133.492, 104.764, 67.568]
    base = [62.397, 82.997, 116.534, 76.239]
    de   = [58.396, 75.172, 103.405, 65.561]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(regions))
    w = 0.26

    # Persistence: light grey reference floor (no hatch; distinguished by lightness)
    ax.bar(x - w, pers, w, color=GREY_L, edgecolor=INK, linewidth=0.5,
           label="persistence")
    ax.bar(x,     base, w, color=C_BASE, edgecolor=INK, linewidth=0.5,
           label="baseline")
    ax.bar(x + w, de,   w, color=C_DE,   edgecolor=INK, linewidth=0.5,
           label="+D+E")

    for i in range(len(regions)):
        ax.text(x[i] - w, pers[i] + 1.8, f"{pers[i]:.1f}", ha="center",
                va="bottom", fontsize=6.6, color=GREY_D)
        ax.text(x[i],     base[i] + 1.8, f"{base[i]:.1f}", ha="center",
                va="bottom", fontsize=6.6, color=INK)
        ax.text(x[i] + w, de[i]   + 1.8, f"{de[i]:.1f}",   ha="center",
                va="bottom", fontsize=6.6, color=INK)

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("corrected MAE  (gCO$_2$/kWh)")
    ax.set_title("Per-region generalisation vs persistence floor")
    ax.legend(loc="upper right", ncol=3)
    ax.set_ylim(0, 152)
    ax.grid(axis="x", visible=False)
    save(fig, "fig3_region_persistence.png")


# =====================================================================================
# Figure 4 -- ablation ladder (SA1)  (S1.1 / S2.1 / S4.2)
# =====================================================================================
def fig4_ablation() -> None:
    rows = [
        ("pure ERM",                    77.629),
        ("+REG / NEG loss",             81.948),
        ("+E (CORAL only)",             75.788),
        ("baseline (no adaptation)",    76.725),
        ("+D (gradual unfreeze)",       67.240),
        ("+D+E (full pipeline)",        66.004),
    ]
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows_sorted]
    vals   = [r[1] for r in rows_sorted]
    colors = [C_DE if v < PERSISTENCE_SA1 else GREY_M for v in vals]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    ax.barh(y, vals, color=colors, height=0.6,
            edgecolor=INK, linewidth=0.5)

    ax.axvline(PERSISTENCE_SA1, color=INK, linestyle="--", linewidth=0.8)
    ax.text(PERSISTENCE_SA1, len(labels) - 0.4,
            f" persistence = {PERSISTENCE_SA1:.2f}",
            fontsize=7.2, color=INK, va="center", ha="left")

    for yi, v in zip(y, vals):
        ax.text(v + 0.25, yi, f"{v:.2f}", va="center", fontsize=7.2,
                color=INK)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("corrected MAE  (gCO$_2$/kWh)")
    ax.set_xlim(60, 88)
    ax.set_title("SA1 ablation ladder")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_ablation_ladder.png")


# =====================================================================================
# Figure 5 -- multi-seed robustness (S5)
# =====================================================================================
def fig5_multiseed() -> None:
    seeds = [0, 1, 2, 3, 4]
    base = [76.531, 75.663, 80.626, 79.729, 77.122]
    de   = [67.161, 66.380, 67.422, 65.850, 65.648]

    fig, ax = plt.subplots(figsize=(7.0, 3.0))
    x = np.arange(len(seeds))

    for i in x:
        ax.plot([i, i], [base[i], de[i]], color=GREY_M,
                linewidth=0.8, zorder=1)

    ax.scatter(x, base, s=36, color=C_BASE, marker="o",
               edgecolor=INK, linewidth=0.4, label="baseline", zorder=3)
    ax.scatter(x, de,   s=36, color=C_DE,   marker="s",
               edgecolor=INK, linewidth=0.4, label="+D+E", zorder=3)

    ax.axhline(PERSISTENCE_SA1, color=INK, linestyle="--", linewidth=0.8,
               label=f"persistence = {PERSISTENCE_SA1:.2f}", zorder=2)

    mb, sb = np.mean(base), np.std(base)
    md, sd = np.mean(de),   np.std(de)
    agg_x = len(seeds) + 0.45
    ax.errorbar(agg_x - 0.18, mb, yerr=sb, fmt="o", color=C_BASE,
                capsize=3, markersize=5, markeredgecolor=INK,
                markeredgewidth=0.4, zorder=4)
    ax.errorbar(agg_x + 0.18, md, yerr=sd, fmt="s", color=C_DE,
                capsize=3, markersize=5, markeredgecolor=INK,
                markeredgewidth=0.4, zorder=4)
    ax.text(agg_x - 0.18, mb + sb + 1.4, f"{mb:.2f}\n($\\pm${sb:.2f})",
            fontsize=6.8, ha="center", color=INK)
    ax.text(agg_x + 0.18, md - sd - 1.4, f"{md:.2f}\n($\\pm${sd:.2f})",
            fontsize=6.8, ha="center", va="top", color=INK)

    ax.set_xticks(list(x) + [agg_x])
    ax.set_xticklabels([f"seed {s}" for s in seeds] + ["mean"])
    ax.set_ylabel("corrected MAE  (gCO$_2$/kWh)")
    ax.set_ylim(60, 88)
    ax.set_title("Multi-seed robustness (5 seeds)")
    ax.legend(loc="upper center", ncol=4)
    save(fig, "fig5_multiseed.png")


# =====================================================================================
# Figure 6 -- Bates-Granger fusion (S4.1)
# =====================================================================================
def fig6_fusion() -> None:
    labels = ["climatology", "network\n(+D+E)", "persistence\n(floor)",
              "TransCIF\nfused"]
    vals = [86.359, 71.226, 67.568, 61.196]
    colors = [GREY_M, C_BASE, GREY_L, C_DE]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    bars = ax.bar(x, vals, 0.6, color=colors,
                  edgecolor=INK, linewidth=0.5)

    ax.axhline(PERSISTENCE_SA1, color=INK, linestyle="--",
               linewidth=0.7)

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.0, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.4, color=INK)
        pct = 100 * (v - PERSISTENCE_SA1) / PERSISTENCE_SA1
        if abs(pct) > 0.1:
            ax.text(b.get_x() + b.get_width() / 2, v + 5.5,
                    f"({pct:+.1f}%)", ha="center", va="bottom",
                    fontsize=6.8, color=GREY_D)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.8)
    ax.set_ylabel("MAE  (gCO$_2$/kWh)")
    ax.set_ylim(0, 100)
    ax.set_title("Bates-Granger Bayes-optimal fusion")
    ax.grid(axis="x", visible=False)
    save(fig, "fig6_fusion.png")


if __name__ == "__main__":
    fig2_theorem1()
    fig3_region_persistence()
    fig4_ablation()
    fig5_multiseed()
    fig6_fusion()
    print("done: 5 data figures ->", os.path.relpath(OUT, os.path.join(HERE, "..")))
