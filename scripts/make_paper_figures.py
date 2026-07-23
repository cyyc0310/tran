"""Generate the publication figures for docs/paper/2026-07-22-transcif-full-paper.md.

Every numerical value below is copied verbatim from the repository's real experiment
logs -- there is NO synthetic or invented data. Sources (single authority = docs/):
  - docs/experiments/2026-07-17-all-experiments-summary.md
  - docs/experiments/2026-07-14-sa1-domain-adaptation.md
Each data block cites the exact section it comes from. Run:
  python3 scripts/make_paper_figures.py
Outputs 300-dpi PNGs into docs/paper/figures/.

Design system (applied uniformly to all figures):
  * one cohesive blue-purple palette, used with fixed semantics everywhere
    (baseline = purple, +D+E = indigo, win = indigo, lose = orchid, persistence =
    slate reference, Term 1 = indigo, Term 2 = purple, auxiliary = periwinkle);
  * consistent typography, spines, grid, value labels, and 300-dpi vector-quality PNGs.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# =====================================================================================
# Unified design system
# =====================================================================================
# --- Unified blue-purple scientific palette (single source of truth) -----------------
# One cohesive blue -> violet -> purple family: roles are separated by hue position and
# lightness (not by clashing hues), giving a calm, publication-grade look while staying
# mutually distinguishable.  The pipeline itself reads as a blue -> violet -> purple flow.
INDIGO = "#2E3C8F"   # deepest: primary model / +D+E / Term 1 / win / fused forecaster
BLUE = "#3D6FB8"     # medium blue: encoder (Stage 1) accent / secondary series
VIOLET = "#6A57C0"   # blue-violet: physics (Stage 2) accent
PURPLE = "#8C5DB0"   # purple: baseline / Term 2 (residual)
SKY = "#7B9FD4"      # light periwinkle: auxiliary series (e.g. climatology)
ORCHID = "#B173BE"   # light orchid: lose / degradation vs floor
SLATE = "#8B93AD"    # neutral blue-grey: persistence / reference
INK = "#212842"      # deep indigo-slate: text and reference lines

# --- fixed semantic aliases (never reassign per-figure) ------------------------------
C_PERS = SLATE
C_BASE = PURPLE
C_DE = INDIGO
C_WIN = INDIGO
C_LOSE = ORCHID
C_T1 = INDIGO
C_T2 = PURPLE

# --- schematic box tints (fill, edge): all in the blue-purple family -----------------
# the pipeline flows Stage 1 (blue) -> Stage 2 (violet) -> output (purple); the spanning
# config/theorem bands stay neutral lavender/slate so they frame without competing.
TINT_INPUT = ("#ECEEF3", "#8890A6")
TINT_STAGE1 = ("#D6E0F4", BLUE)
TINT_STAGE2 = ("#DFDBF3", VIOLET)
TINT_OUT = ("#ECE1F3", PURPLE)
TINT_CONFIG = ("#E7E4F5", "#6E60AE")
TINT_THM = ("#E1E4F1", "#565F86")

plt.rcParams.update(
    {
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10.5,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#444444",
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 8.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.color": "#333333",
        "ytick.color": "#333333",
    }
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "docs", "paper", "figures")
os.makedirs(OUT, exist_ok=True)

PERSISTENCE_SA1 = 67.568  # docs summary S0: persistence_mae = 67.56788...


def save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


# =====================================================================================
# Figure 1 -- system architecture schematic (structural diagram; reflects paper S4).
# Shows: config-driven deployment layer -> two-stage pipeline (multi-source MLDG
# pretraining, D/E target adaptation, physics reconstruction with emission factors,
# residual + gated skip, split-conformal band) -> Theorem 1 exact error decomposition.
# =====================================================================================
def fig1_architecture() -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.65))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 59)
    ax.set_aspect("equal")
    ax.axis("off")

    def box(cx, cy, w, h, tint, lw=1.1):
        fc, ec = tint
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.5,rounding_size=1.3",
            linewidth=lw, edgecolor=ec, facecolor=fc, zorder=2,
        ))

    def arrow(x1, y1, x2, y2, color=INK, ls="-", lw=1.2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=12,
            linewidth=lw, color=color, linestyle=ls, zorder=3,
            shrinkA=0, shrinkB=0,
        ))

    def txt(x, y, s, size=8, color=INK, weight="normal", style="normal", ha="center"):
        ax.text(x, y, s, ha=ha, va="center", fontsize=size, color=color,
                fontweight=weight, fontstyle=style, zorder=4)

    # ---- config-driven deployment band (top) ----------------------------------------
    box(50, 54, 96, 7.2, TINT_CONFIG, lw=1.2)
    txt(50, 55.4, "Config-driven deployment  \u00b7  RegionConfig / DeploymentConfig",
        size=8.5, weight="bold", color="#443A88")
    txt(50, 52.5, "onboard a new region = edit one JSON  \u00b7  inline emission factors \u21d2 no code change",
        size=7.2, color="#564C96")

    # ---- input cluster (left) --------------------------------------------------------
    box(10, 37, 16, 8.5, TINT_INPUT)
    txt(10, 38.4, "Source regions", size=7.6, weight="bold")
    txt(10, 35.6, "QLD1 \u00b7 NSW1 \u00b7 VIC1", size=6.8)
    box(10, 22, 16, 8.5, TINT_INPUT)
    txt(10, 23.4, "Target region", size=7.6, weight="bold")
    txt(10, 20.6, "SA1 calib. split", size=6.8)

    # ---- Stage 1: domain-invariant encoder -------------------------------------------
    box(43, 30, 29, 23, TINT_STAGE1)
    txt(43, 39.2, "Stage 1 \u00b7 Encoder $h$", size=8.6, weight="bold", color="#25397E")
    txt(43, 36.0, "(domain-invariant)", size=6.8, style="italic", color="#45589C")
    txt(43, 32.4, "LT-MWKC \u2014 multi-freq temporal", size=7.1)
    txt(43, 29.2, "CV-DWCC \u2014 cross-variable", size=7.1)
    ax.plot([29.5, 56.5], [26.6, 26.6], color="#AEBEE6", lw=0.8, zorder=3)
    txt(43, 23.5, r"$\to\ \hat{s}\in[0,1]^{H}$  (renewable share)", size=7.3, weight="bold",
        color="#25397E")

    # ---- Stage 2: physics + residual -------------------------------------------------
    box(75, 30, 29, 23, TINT_STAGE2)
    txt(75, 39.2, "Stage 2 \u00b7 Physics + residual", size=8.6, weight="bold", color="#463A8A")
    txt(75, 35.4, r"CIF $= s\,C_{ren}+(1{-}s)\,C_{non}$", size=7.4)
    txt(75, 31.9, r"$+\ \hat{\Delta}$  residual head", size=7.1)
    txt(75, 28.6, "$+$  volatility-gated persistence skip", size=6.8)
    ax.plot([61.5, 88.5], [25.8, 25.8], color="#C3BCE8", lw=0.8, zorder=3)
    txt(75, 23.0, "$+$  split-conformal band", size=6.9, style="italic", color="#564C96")

    # ---- output ----------------------------------------------------------------------
    box(95, 30, 8, 11, TINT_OUT)
    txt(95, 31.2, r"$CI_{pred}$", size=8.6, weight="bold", color="#6A3E96")
    txt(95, 28.2, "gCO$_2$/kWh", size=6.0, color="#6A3E96")

    # ---- main-flow arrows ------------------------------------------------------------
    arrow(18, 37, 28.5, 33)          # sources -> encoder
    arrow(18, 22, 28.5, 27)          # target  -> encoder
    arrow(57.5, 30, 60.5, 30)        # encoder -> stage 2
    arrow(89.5, 30, 91, 30)          # stage 2 -> output

    # arrow annotations placed in the arrow-free corridor (arrows live at y<=27 and
    # y>=33; the band y=28..32 is clear), kept narrow to clear the encoder edge x=28.5
    txt(21, 31, "MLDG multi-source pretrain", size=6.2, color="#5A5F80", style="italic")
    txt(21, 28.4, "D: fine-tune  \u00b7  E: CORAL", size=6.2, color="#5A5F80", style="italic")

    # ---- config feed arrows (dashed) -------------------------------------------------
    arrow(24, 50.4, 12, 41.5, color=VIOLET, ls="--", lw=1.0)
    txt(8.5, 46.6, "region data\n+ channels", size=6.2, color="#564C96", ha="center")
    arrow(75, 50.4, 75, 41.6, color=VIOLET, ls="--", lw=1.0)
    txt(87.5, 46.6, r"$C_{ren},C_{non}\Rightarrow L_T$", size=6.6, color="#564C96")

    # ---- Theorem 1 band (bottom) -----------------------------------------------------
    box(50, 7.2, 96, 9.4, TINT_THM, lw=1.2)
    txt(50, 9.2,
        r"Theorem 1 (exact):  $CI_{pred}-CI_{true} = "
        r"(\hat{s}-s)(C_{ren}-C_{non})\ +\ (\hat{\Delta}-\varepsilon)$",
        size=9, weight="bold", color="#2A2F5C")
    txt(50, 5.1,
        "Term \u2460  transfer amplification  (scaled by $L_T=|C_{ren}-C_{non}|$)"
        "          Term \u2461  residual estimation",
        size=7.1, color="#5A5F80")
    arrow(95, 24.4, 82, 12.1, color="#A6ACCC", ls="--", lw=0.9)

    save(fig, "fig1_architecture.png")


# =====================================================================================
# Figure 2 -- Theorem 1 decomposition.  docs summary S3.1 (panel a) + S4.4 (panel b)
# =====================================================================================
def fig2_theorem1() -> None:
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.4, 3.3))

    # Panel a: mean|Term1| / mean|Term2| stacked, SA1 calib=0.7 (summary S3.1)
    labels = ["baseline", "+D+E"]
    term1 = [84.413, 73.966]
    term2 = [22.123, 20.288]
    x = np.arange(len(labels))
    axA.bar(x, term1, 0.56, label="Term \u2460 (transfer amp.)", color=C_T1)
    axA.bar(x, term2, 0.56, bottom=term1, label="Term \u2461 (residual)", color=C_T2)
    for i, (t1, t2) in enumerate(zip(term1, term2)):
        share = 100 * t1 / (t1 + t2)
        axA.text(i, t1 / 2, f"{share:.1f}%", ha="center", va="center",
                 color="white", fontsize=8.5, fontweight="bold")
        axA.text(i, t1 + t2 + 1.8, f"{t1 + t2:.1f}", ha="center", va="bottom",
                 color="#333333", fontsize=7.5)
    axA.set_xticks(x)
    axA.set_xticklabels(labels)
    axA.set_ylim(0, 120)
    axA.set_ylabel("mean |error component|  (gCO$_2$/kWh)")
    axA.set_title("(a) Error decomposition (SA1, split 0.7)")
    axA.legend(loc="upper right")

    # Panel b: Term1-share vs L_T across 4 rotated regions (summary S4.4)
    regions = ["SA1", "QLD1", "NSW1", "VIC1"]
    lt = [490.43, 841.59, 875.14, 1160.12]
    base_share = [79.66, 83.71, 89.28, 91.84]
    de_share = [78.84, 82.51, 88.98, 90.50]
    axB.plot(lt, base_share, "o-", color=C_BASE, label="baseline", markersize=6,
             markeredgecolor="white", markeredgewidth=0.6, linewidth=1.6)
    axB.plot(lt, de_share, "s--", color=C_DE, label="+D+E", markersize=6,
             markeredgecolor="white", markeredgewidth=0.6, linewidth=1.6)
    for xi, yi, r in zip(lt, base_share, regions):
        axB.annotate(r, (xi, yi), textcoords="offset points", xytext=(4, 7),
                     fontsize=7.5, fontweight="bold")
    axB.axhline(50, color=C_LOSE, linestyle=":", linewidth=1.2)
    axB.text(505, 52.5, "50% dominance threshold", color=C_LOSE, fontsize=7)
    axB.set_ylim(45, 100)
    axB.set_xlabel("$L_T = |C_{ren}-C_{non}|$  (gCO$_2$/kWh)")
    axB.set_ylabel("Term \u2460 share (%)")
    axB.set_title("(b) 8/8 Term \u2460 dominance across regions")
    axB.legend(loc="lower right")

    fig.tight_layout()
    save(fig, "fig2_theorem1_decomposition.png")


# =====================================================================================
# Figure 3 -- per-region persistence comparison (headline).  docs summary S4.5
# =====================================================================================
def fig3_region_persistence() -> None:
    regions = ["QLD1", "NSW1", "VIC1", "SA1"]
    pers = [103.181, 133.492, 104.764, 67.568]
    base = [62.397, 82.997, 116.534, 76.239]
    de = [58.396, 75.172, 103.405, 65.561]

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(len(regions))
    w = 0.26
    ax.bar(x - w, pers, w, label="persistence", color=C_PERS)
    b2 = ax.bar(x, base, w, label="baseline", color=C_BASE)
    b3 = ax.bar(x + w, de, w, label="+D+E", color=C_DE)

    for i in range(len(regions)):
        for bar, val in ((b2[i], base[i]), (b3[i], de[i])):
            pct = 100 * (val - pers[i]) / pers[i]
            win = val < pers[i]
            ax.text(bar.get_x() + bar.get_width() / 2, val + 2,
                    f"{pct:+.1f}%", ha="center", va="bottom", fontsize=7,
                    color=C_WIN if win else C_LOSE, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(regions)
    ax.set_ylabel("corrected MAE  (gCO$_2$/kWh)")
    ax.set_title("Per-region vs persistence: +D+E wins 4/4, baseline only 2/4")
    ax.legend(loc="upper left", ncol=3)
    ax.set_ylim(0, 152)
    save(fig, "fig3_region_persistence.png")


# =====================================================================================
# Figure 4 -- ablation ladder (SA1).  docs summary S1.1 / S2.1 / S4.2
# =====================================================================================
def fig4_ablation() -> None:
    rows = [
        ("pure ERM (no MLDG)", 77.629),
        ("+REG/NEG", 81.948),
        ("+MLDG weighting B", 78.317),
        ("+gating A", 76.820),
        ("baseline (no mitigation)", 76.725),
        ("+temperature C1", 76.599),
        ("+E (CORAL)", 75.788),
        ("all-combined (stage1)", 75.508),
        ("all-combined (stage2)", 74.712),
        ("+D (fine-tuning)", 67.240),
        ("+D+E", 66.004),
    ]
    rows_sorted = sorted(rows, key=lambda r: r[1], reverse=True)
    labels = [r[0] for r in rows_sorted]
    vals = [r[1] for r in rows_sorted]
    # winners (beat the floor) highlighted indigo; the rest neutral blue-grey -> the
    # dashed floor line carries the "beats persistence?" signal without alarm colours.
    colors = [C_WIN if v < PERSISTENCE_SA1 else "#C8CADB" for v in vals]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    ax.barh(y, vals, color=colors, height=0.66)
    ax.axvline(PERSISTENCE_SA1, color=INK, linestyle="--", linewidth=1.3)
    ax.text(PERSISTENCE_SA1 - 0.3, len(labels) - 0.35,
            f"persistence floor = {PERSISTENCE_SA1:.2f}", fontsize=7.5, color=INK,
            ha="right", va="center")
    for yi, v in zip(y, vals):
        beats = v < PERSISTENCE_SA1
        ax.text(v + 0.25, yi, f"{v:.2f}", va="center", fontsize=7.2,
                color=C_WIN if beats else "#5A5F80",
                fontweight="bold" if beats else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("corrected MAE  (gCO$_2$/kWh)   \u2014   indigo = beats persistence floor")
    ax.set_xlim(60, 85)
    ax.set_title("SA1 ablation ladder: only +D and +D+E beat the persistence floor")
    ax.grid(axis="y", visible=False)
    save(fig, "fig4_ablation_ladder.png")


# =====================================================================================
# Figure 5 -- multi-seed robustness (5 seeds).  docs summary S5
# =====================================================================================
def fig5_multiseed() -> None:
    seeds = [0, 1, 2, 3, 4]
    base = [76.531, 75.663, 80.626, 79.729, 77.122]
    de = [67.161, 66.380, 67.422, 65.850, 65.648]

    fig, ax = plt.subplots(figsize=(7.4, 3.5))
    x = np.arange(len(seeds))
    for i in x:
        ax.plot([i, i], [base[i], de[i]], color="#CBCEDD", linewidth=1.2, zorder=1)
    ax.scatter(x, base, color=C_BASE, s=52, label="baseline", zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.scatter(x, de, color=C_DE, s=52, marker="s", label="+D+E", zorder=3,
               edgecolor="white", linewidth=0.6)
    ax.axhline(PERSISTENCE_SA1, color=INK, linestyle="--", linewidth=1.3,
               label=f"persistence = {PERSISTENCE_SA1:.2f}")

    mb, sb = np.mean(base), np.std(base)
    md, sd = np.mean(de), np.std(de)
    ax.errorbar(len(seeds) + 0.35, mb, yerr=sb, fmt="o", color=C_BASE, capsize=4,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.errorbar(len(seeds) + 0.7, md, yerr=sd, fmt="s", color=C_DE, capsize=4,
                markeredgecolor="white", markeredgewidth=0.6)
    ax.text(len(seeds) + 0.35, mb + sb + 1.6, f"{mb:.2f}\n\u00b1{sb:.2f}", fontsize=6.6,
            ha="center", color=C_BASE)
    ax.text(len(seeds) + 0.7, md - sd - 1.6, f"{md:.2f}\n\u00b1{sd:.2f}", fontsize=6.6,
            ha="center", va="top", color=C_DE)

    ax.set_xticks(list(x) + [len(seeds) + 0.52])
    ax.set_xticklabels([f"seed {s}" for s in seeds] + ["agg."])
    ax.set_ylabel("corrected MAE  (gCO$_2$/kWh)")
    ax.set_ylim(60, 86)
    ax.set_title("+D+E beats baseline in 5/5 seeds; direction robust, margin seed-sensitive")
    ax.legend(loc="upper center", ncol=3)
    save(fig, "fig5_multiseed.png")


# =====================================================================================
# Figure 6 -- Bates-Granger fusion.  docs summary S4.1
# =====================================================================================
def fig6_fusion() -> None:
    labels = ["persistence\n(floor)", "network-only\n(+D+E)", "climatology\n-only",
              "fused\n(Bates-Granger)"]
    vals = [67.568, 71.226, 86.359, 61.196]
    # persistence=slate reference, network=baseline-purple, climatology=periwinkle
    # auxiliary, fused=indigo (the winning combined forecaster).
    colors = [C_PERS, C_BASE, SKY, C_DE]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(6.6, 3.5))
    bars = ax.bar(x, vals, 0.62, color=colors)
    ax.axhline(PERSISTENCE_SA1, color=INK, linestyle="--", linewidth=1.3)
    for b, v in zip(bars, vals):
        pct = 100 * (v - PERSISTENCE_SA1) / PERSISTENCE_SA1
        if abs(v - PERSISTENCE_SA1) < 1e-6:
            tag, col = f"{v:.2f}", INK
        else:
            tag = f"{v:.2f}\n({pct:+.1f}%)"
            col = C_WIN if v < PERSISTENCE_SA1 else C_LOSE
        ax.text(b.get_x() + b.get_width() / 2, v + 0.7, tag, ha="center",
                va="bottom", fontsize=7.5, color=col, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("MAE  (gCO$_2$/kWh)")
    ax.set_ylim(0, 98)
    ax.set_title("Bates-Granger fusion: \u22129.4% vs persistence (project best)")
    save(fig, "fig6_fusion.png")


if __name__ == "__main__":
    fig1_architecture()
    fig2_theorem1()
    fig3_region_persistence()
    fig4_ablation()
    fig5_multiseed()
    fig6_fusion()
    print("done: 6 figures ->", os.path.relpath(OUT, os.path.join(HERE, "..")))
