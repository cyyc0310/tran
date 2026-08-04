"""Generate three IEEE-style architecture figures for
docs/paper/2026-07-22-transcif-full-paper.md.

Outputs (300-dpi PNG, docs/paper/figures/):
  fig1_architecture.png       (referenced as Figure 1 in the paper)
  fig_stage1_encoder.png      (Stage 1 detail)
  fig_stage2_physics.png      (Stage 2 detail)

Style -- strict IEEE Transactions minimalism. Three-colour palette only:
  * Navy   -- primary accent (Stage 1, white-box physics, +D+E, Term 1)
  * Steel  -- secondary (Stage 2, residual, baseline, Term 2)
  * Greys  -- reference floors, persistence, neutral boxes
Source vs target domain distinction is made by warm vs cool tone (pale sand vs
pale blue), not by hue saturation or hatching.
"""

from __future__ import annotations

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Circle

# =====================================================================================
# Three-tone IEEE palette
# =====================================================================================
NAVY    = "#1F3A5F"   # primary accent
STEEL   = "#4A6E8A"   # secondary accent
GREY_D  = "#3A3A3A"   # darkest grey (borders, text)
GREY_M  = "#6E6E6E"   # mid grey (reference)
GREY_L  = "#B5B5B5"   # light grey (auxiliary)
GREY_XL = "#E5E5E5"   # extra light (panel fills)
GREY_XX = "#F4F4F4"   # subtlest tint (background bands)
INK     = "#1A1A1A"   # near-black: text and arrows

# Box tints -- source vs target distinguished by warm vs cool tone (no hatching)
TINT_SRC  = ("#E6EEF5", "#5A7AA0")   # source = pale cool blue
TINT_TGT  = ("#F5EAE0", "#A07A5A")   # target = pale warm sand
TINT_S1   = ("#E5EAF0", NAVY)        # stage 1 = pale navy
TINT_S2   = ("#E8E8E8", STEEL)       # stage 2 = pale steel
TINT_OUT  = ("#EDEDEDP", NAVY)       # output (placeholder, fixed below)
TINT_OUT  = ("#EDEDED", NAVY)
TINT_CFG  = ("#F1F1F1", GREY_M)      # config band = neutral grey
TINT_THM  = ("#F1F1F1", GREY_D)      # theorem band = neutral grey
TINT_INNER = ("#FAFAFA", GREY_L)     # inner blocks = near-white

plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
    "font.family": "Times New Roman",
    "font.size": 8,
    "mathtext.fontset": "stix",
})

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "..", "docs", "paper", "figures")
os.makedirs(OUT, exist_ok=True)


def save(fig, name: str) -> None:
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


# =====================================================================================
# Shared schematic helpers
# =====================================================================================
def box(ax, cx, cy, w, h, tint, lw=0.8, rounding=0.0, hatch=None, z=2):
    fc, ec = tint
    if rounding > 0:
        ax.add_patch(FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=f"round,pad=0.02,rounding_size={rounding}",
            linewidth=lw, edgecolor=ec, facecolor=fc,
            hatch=hatch, zorder=z,
        ))
    else:
        ax.add_patch(Rectangle(
            (cx - w / 2, cy - h / 2), w, h,
            linewidth=lw, edgecolor=ec, facecolor=fc,
            hatch=hatch, zorder=z,
        ))


def rect(ax, cx, cy, w, h, tint, lw=0.8, hatch=None, z=2):
    """Sharp rectangle (white-box / interpretable semantics)."""
    box(ax, cx, cy, w, h, tint, lw=lw, rounding=0.0, hatch=hatch, z=z)


def rbox(ax, cx, cy, w, h, tint, lw=0.8, hatch=None, z=2):
    """Rounded rectangle (black-box / learned semantics)."""
    box(ax, cx, cy, w, h, tint, lw=lw, rounding=0.18, hatch=hatch, z=z)


def arrow(ax, x1, y1, x2, y2, color=INK, ls="-", lw=0.9, z=3):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10,
        linewidth=lw, color=color, linestyle=ls, zorder=z,
        shrinkA=0, shrinkB=0,
    ))


def txt(ax, x, y, s, size=8, color=INK, weight="normal", style="normal",
        ha="center", va="center", z=4):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color,
            fontweight=weight, fontstyle=style, zorder=z)


# =====================================================================================
# Figure 1 -- Overall TransCIF Architecture
# =====================================================================================
def fig_overall_architecture() -> None:
    fig, ax = plt.subplots(figsize=(10.4, 6.4))
    ax.set_xlim(0, 104); ax.set_ylim(0, 64)
    ax.set_aspect("equal"); ax.axis("off")

    # ---- (0) Title -------------------------------------------------------------------
    txt(ax, 52, 62.5,
        "TransCIF:  Physics-Informed, Config-Driven Cross-Region CI Forecasting",
        size=10.5, weight="bold", color=INK)

    # ---- (1) Config-driven deployment band (top) ------------------------------------
    # Taller band so every text element sits comfortably inside.
    rect(ax, 52, 56.5, 100, 7.0, TINT_CFG, lw=0.8)
    txt(ax, 14, 58.4, "Config-Driven Deployment",
        size=9, weight="bold", color=INK)
    txt(ax, 14, 55.6, "RegionConfig  ·  DeploymentConfig",
        size=7.2, color=GREY_D)
    txt(ax, 48, 58.0,
        "onboard a new grid  =  edit one JSON",
        size=8.4, color=INK)
    txt(ax, 48, 55.6,
        r"inline  $C_{ren},\,C_{non}$   $\Rightarrow$   zero code change",
        size=7.4, color=GREY_D, style="italic")
    # JSON snippet in its own well-sized bordered cell
    rect(ax, 88, 56.5, 26, 5.4, ("#FAFAFA", GREY_L), lw=0.5)
    txt(ax, 88, 57.6, '{"region": "SA1",',
        size=7.0, color=GREY_D)
    txt(ax, 88, 55.4, r'  $C_{ren}$: 0,   $C_{non}$: 490.43 }',
        size=7.0, color=GREY_D)

    # ---- (2) Inputs (left column) ----------------------------------------------------
    rect(ax, 11, 42, 18, 9, TINT_SRC, lw=0.8)
    txt(ax, 11, 44.8, "Source Regions", size=8.4, weight="bold", color=INK)
    txt(ax, 11, 41.4, "QLD1 · NSW1 · VIC1", size=7.2, color=GREY_D)

    rect(ax, 11, 28, 18, 9, TINT_TGT, lw=0.8)
    txt(ax, 11, 30.8, "Target Region", size=8.4, weight="bold", color=INK)
    txt(ax, 11, 27.4, "SA1  (calib. split 0.7)", size=7.2, color=GREY_D)

    rect(ax, 11, 14.5, 18, 8, TINT_INNER, lw=0.7)
    txt(ax, 11, 17.3, "Scale-Invariant Reparam",
        size=7.6, weight="bold", color=INK)
    txt(ax, 11, 14.0, "RenewShare · LoadNorm\nTempAnomaly  (Open-Meteo)",
        size=6.8, color=GREY_D)

    # ---- (3) Stage 1 -- title INSIDE the container, header strip --------------------
    # Outer container: tall enough for header + 2 inner blocks + output label.
    rect(ax, 42, 32, 30, 26, TINT_S1, lw=1.0)        # y in [19, 45]
    # Header strip line
    ax.plot([27.5, 56.5], [41.2, 41.2], color=NAVY, linewidth=0.5)
    txt(ax, 42, 43.4, "Stage 1  ·  Encoder $h$",
        size=9.2, weight="bold", color=NAVY)
    txt(ax, 42, 40.0, "(domain-invariant)",
        size=7.2, style="italic", color=GREY_D)

    rect(ax, 42, 36.0, 26, 4.8, TINT_INNER, lw=0.6)
    txt(ax, 42, 36.9, "DLinear Decomp", size=8.0, weight="bold", color=NAVY)
    txt(ax, 42, 35.0, "trend/seasonal decomposition + config bias",
        size=6.6, color=GREY_D)

    rect(ax, 42, 28.2, 26, 4.8, TINT_INNER, lw=0.6)
    txt(ax, 42, 29.1, "Adaptive Persistence Gate", size=8.0, weight="bold", color=NAVY)
    txt(ax, 42, 27.2, "config-conditioned + volatility-aware",
        size=6.6, color=GREY_D)

    txt(ax, 42, 22.5, r"$\hat{s}\in[0,1]^{H}$   (renewable share)",
        size=8.2, weight="bold", color=NAVY)

    # Side-callout below Stage 1 -- taller box for 2 lines
    rect(ax, 42, 13.0, 30, 7.2, TINT_INNER, lw=0.7)
    txt(ax, 42, 15.5, "Config-Weighted Source Sampling",
        size=7.6, weight="bold", color=INK)
    txt(ax, 42, 12.2,
        "cosine warmup    ·    5-seed LORO eval",
        size=7.0, color=GREY_D)

    # ---- (4) Stage 2 -- mirror layout of Stage 1 ------------------------------------
    rect(ax, 75, 32, 30, 26, TINT_S2, lw=1.0)        # y in [19, 45]
    ax.plot([60.5, 89.5], [41.2, 41.2], color=STEEL, linewidth=0.5)
    txt(ax, 75, 43.4, "Stage 2  ·  Physics + Residual",
        size=9.2, weight="bold", color=STEEL)
    txt(ax, 75, 40.0, "(white-box + black-box)",
        size=7.2, style="italic", color=GREY_D)

    # White-box physics: sharp rectangle, sized to hold the formula + caption
    rect(ax, 75, 36.0, 26, 5.6, TINT_INNER, lw=0.7)
    txt(ax, 75, 37.5,
        r"CIF$(s)=s\,C_{ren}{+}(1{-}s)\,C_{non}$",
        size=8.2, weight="bold", color=STEEL)
    txt(ax, 75, 34.5, "fixed linear emission map",
        size=6.6, style="italic", color=GREY_D)

    # Black-box residual: rounded, sized for header + caption
    rbox(ax, 75, 28.2, 26, 5.4, TINT_INNER, lw=0.7)
    txt(ax, 75, 29.6, r"ZS+ calibration",
        size=7.8, weight="bold", color=STEEL)
    txt(ax, 75, 26.9, "level anchoring  ·  residual correction",
        size=6.4, color=GREY_D)

    txt(ax, 75, 22.5, r"$CI_{pred}$   (gCO$_2$/kWh)",
        size=8.2, weight="bold", color=STEEL)

    rect(ax, 75, 13.0, 30, 7.2, TINT_INNER, lw=0.7)
    txt(ax, 75, 15.5, r"Physical constant   $L_T = |C_{ren}{-}C_{non}|$",
        size=7.6, weight="bold", color=INK)
    txt(ax, 75, 12.2,
        "SA1 490  ·  QLD1 842  ·  NSW1 875  ·  VIC1 1160",
        size=6.8, color=GREY_D)

    # ---- (5) Output ------------------------------------------------------------------
    rect(ax, 99, 32, 9, 16, TINT_OUT, lw=1.0)
    txt(ax, 99, 36.0, r"$\hat{CI}$", size=12, weight="bold", color=NAVY)
    txt(ax, 99, 32.0, "gCO$_2$/kWh", size=7.0, color=GREY_D)
    txt(ax, 99, 28.8, "+ conf. band", size=6.8, style="italic", color=GREY_D)

    # ---- (6) Main flow arrows --------------------------------------------------------
    arrow(ax, 20.0, 42, 27.0, 37.0, lw=1.0)
    arrow(ax, 20.0, 28, 27.0, 31.0, lw=1.0)
    arrow(ax, 57.0, 32, 60.0, 32, lw=1.2)
    arrow(ax, 90.0, 32, 94.5, 32, lw=1.2)

    arrow(ax, 11, 18.5, 11, 23.4, color=GREY_M, lw=0.8)

    # config feed arrows (dashed grey) -- drop from band to each consumer
    arrow(ax, 11, 53.0, 11, 46.6, color=GREY_M, ls="--", lw=0.7)
    arrow(ax, 42, 53.0, 42, 45.2, color=GREY_M, ls="--", lw=0.7)
    arrow(ax, 75, 53.0, 75, 45.2, color=GREY_M, ls="--", lw=0.7)
    txt(ax, 11, 49.7, "region cfg", size=6.0, color=GREY_D, style="italic")
    txt(ax, 42, 49.0, "channels", size=6.0, color=GREY_D, style="italic")
    txt(ax, 75, 49.0, r"$C_{ren},C_{non}$", size=6.6, color=GREY_D)

    # ---- (7) Theorem 1 band (bottom) -------------------------------------------------
    # Taller band so the equation line and caption line both sit comfortably inside.
    rect(ax, 52, 4.0, 100, 9.0, TINT_THM, lw=0.8)   # y in [-0.5, 8.5]
    txt(ax, 52, 7.0,
        r"Theorem 1 (exact):   "
        r"$CI_{pred}-CI_{true} "
        r"=\;(\hat{s}-s)(C_{ren}-C_{non})\;+\;(\hat{\Delta}-\varepsilon)$",
        size=9.2, weight="bold", color=INK)
    txt(ax, 52, 2.6,
        "Term (1)  transfer amplification (scaled by $L_T$)"
        "               Term (2)  residual estimation",
        size=7.6, color=GREY_D)

    arrow(ax, 99, 24.0, 92, 8.5, color=GREY_L, ls="--", lw=0.7)

    save(fig, "fig1_architecture.png")


# =====================================================================================
# Figure 2 -- Stage 1: Encoder + Domain Adaptation
# =====================================================================================
def fig_stage1_encoder() -> None:
    """Stage 1 in two vertically-stacked panels.

    Geometry: each panel is ~10 in wide × ~3 in tall (xlim 0-100, ylim 0-30).
    With aspect=equal, 1 data unit ≈ 0.1 in ≈ 7.2 pt. A 20-unit-wide box is
    ~1.4 in wide -- enough for any reasonable label at fontsize 8-9.
    """
    fig = plt.figure(figsize=(10.4, 7.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.12)
    axA = fig.add_subplot(gs[0])
    axB = fig.add_subplot(gs[1])

    # ======================== Panel (a): encoder =================================
    axA.set_xlim(0, 100); axA.set_ylim(0, 30)
    axA.set_aspect("equal"); axA.axis("off")
    txt(axA, 50, 28.5,
        "(a)  Stage 1 encoder -- DLinear Decomp + Adaptive Persistence Gate",
        size=9.5, weight="bold", color=INK)

    # Inputs (left column)
    rect(axA, 8, 14, 12, 17, TINT_SRC, lw=0.8)
    txt(axA, 8, 21.5, "Inputs", size=9, weight="bold", color=INK)
    txt(axA, 8, 17.5, "RenewShare", size=7.5, color=GREY_D)
    txt(axA, 8, 14.5, "LoadNorm", size=7.5, color=GREY_D)
    txt(axA, 8, 11.5, "TempAnom", size=7.5, color=GREY_D)

    # DLinear trend block
    rect(axA, 31, 21, 24, 9, TINT_S1, lw=0.9)
    txt(axA, 31, 23.2, "DLinear Trend", size=9, weight="bold", color=NAVY)
    txt(axA, 31, 19.0, "AvgPool decomposition", size=7, color=GREY_D)

    # DLinear seasonal block
    rect(axA, 31, 8, 24, 9, TINT_S1, lw=0.9)
    txt(axA, 31, 10.2, "DLinear Seasonal", size=9, weight="bold", color=NAVY)
    txt(axA, 31, 6.0, "config-conditioned sigmoid", size=7, color=GREY_D)

    # Feature fusion
    rect(axA, 57, 14.5, 14, 16, TINT_INNER, lw=0.8)
    txt(axA, 57, 21, "Feature", size=8.5, weight="bold", color=INK)
    txt(axA, 57, 18.5, "fusion", size=8.5, weight="bold", color=INK)
    txt(axA, 57, 14, "pool + head", size=6.8, color=GREY_D)
    txt(axA, 57, 10, r"$h(x)$", size=9, color=INK, weight="bold")

    # Gate (mixer)
    rect(axA, 79, 21, 14, 9, TINT_INNER, lw=0.8)
    txt(axA, 79, 23.2, r"gate $\sigma$", size=8.5, weight="bold", color=INK)
    txt(axA, 79, 19, "adaptive gate", size=7, color=GREY_D)

    # Persistence
    rect(axA, 79, 8, 14, 9, TINT_INNER, lw=0.8)
    txt(axA, 79, 10.5, "persistence", size=8, weight="bold", color=INK)
    txt(axA, 79, 6.5, r"$s_t^{\,last}$", size=8, color=INK)

    # Output share (right)
    rect(axA, 95.5, 14.5, 8, 16, TINT_OUT, lw=0.9)
    txt(axA, 95.5, 19, r"$\hat{s}$", size=11, weight="bold", color=NAVY)
    txt(axA, 95.5, 14.5, r"$[0,1]^H$", size=7.5, color=GREY_D)
    txt(axA, 95.5, 10.5, "share", size=7, color=GREY_D)

    # Arrows
    arrow(axA, 14, 17, 19, 20, color=INK, lw=0.9)
    arrow(axA, 14, 11, 19, 9, color=INK, lw=0.9)
    arrow(axA, 43, 21, 50, 18, color=INK, lw=0.9)
    arrow(axA, 43, 8, 50, 11, color=INK, lw=0.9)
    arrow(axA, 64, 18, 72, 21, color=INK, lw=0.9)
    arrow(axA, 64, 11, 72, 9, color=INK, lw=0.9)
    arrow(axA, 86, 21, 92, 18, color=INK, lw=1.0)
    arrow(axA, 86, 8, 92, 11, color=INK, lw=1.0, ls="--")

    # ======================== Panel (b): adaptation ==============================
    axB.set_xlim(0, 100); axB.set_ylim(0, 30)
    axB.set_aspect("equal"); axB.axis("off")
    txt(axB, 50, 28.5,
        "(b)  Config-weighted source training  +  zero-shot target inference",
        size=9.5, weight="bold", color=INK)

    # Row 1: 3 source domains (cool tone)
    src_x = [18, 50, 82]
    src_names = ["QLD1", "NSW1", "VIC1"]
    for x, n in zip(src_x, src_names):
        rect(axB, x, 22, 22, 6, TINT_SRC, lw=0.8)
        txt(axB, x, 23.5, f"source  {n}", size=8.5, weight="bold", color=INK)
        txt(axB, x, 20.8, "AEMO 2023", size=6.6, color=GREY_D)
        arrow(axB, x, 18.5, x, 16.5, color=GREY_M, lw=0.8)

    # Row 2: source region sampling bar
    rect(axB, 50, 13.0, 86, 5.5, TINT_INNER, lw=0.9)
    txt(axB, 50, 14.5, "Config-Weighted Source Sampling",
        size=9, weight="bold", color=INK)
    txt(axB, 50, 11.5, "weight = 1 / (config_distance + epsilon)",
        size=7, color=GREY_D, style="italic")

    # Branch arrows down to D / E
    arrow(axB, 32, 10.0, 24, 7.2, color=GREY_M, lw=0.8)
    arrow(axB, 68, 10.0, 76, 7.2, color=GREY_M, lw=0.8)

    # Row 3: ZS and ZS+ branches
    rect(axB, 24, 3.0, 30, 7, TINT_TGT, lw=0.9)
    txt(axB, 24, 5.2, "ZS  Zero-Shot Inference",
        size=8.5, weight="bold", color=INK)
    txt(axB, 24, 1.8, "config-only prediction",
        size=6.8, color=GREY_D)

    rect(axB, 76, 3.0, 30, 7, ("#FAF2EA", "#A07A5A"), lw=0.9)
    txt(axB, 76, 5.2, "ZS+  Test-Time Calibration",
        size=8.5, weight="bold", color=INK)
    txt(axB, 76, 1.8, "level anchoring + residual corr",
        size=6.8, color=GREY_D)

    save(fig, "fig_stage1_encoder.png")


# =====================================================================================
# Figure 3 -- Stage 2: Physics + Residual + Theorem 1 numerical validation
# =====================================================================================
def fig_stage2_physics() -> None:
    fig = plt.figure(figsize=(10.4, 3.8))
    axA = fig.add_subplot(111)

    # ======================== Stage 2 pipeline ===================================
    axA.set_xlim(0, 100); axA.set_ylim(0, 30)
    axA.set_aspect("equal"); axA.axis("off")
    txt(axA, 50, 28.5,
        "Stage 2  ·  physics reconstruction + residual correction",
        size=9.5, weight="bold", color=INK)

    # Input share (from Stage 1)
    rect(axA, 7, 17, 10, 13, TINT_SRC, lw=0.8)
    txt(axA, 7, 23, "Stage 1", size=8, weight="bold", color=INK)
    txt(axA, 7, 18.5, r"$\hat{s}$", size=12, color=INK, weight="bold")
    txt(axA, 7, 13.5, "share", size=7, color=GREY_D)

    # White-box physics
    rect(axA, 28, 17, 24, 13, TINT_S2, lw=1.0)
    txt(axA, 28, 25, "White-Box Physics", size=8.5,
        weight="bold", color=STEEL)
    txt(axA, 28, 20.5,
        r"$\mathrm{CIF}(\hat{s})$",
        size=10, color=INK, weight="bold")
    txt(axA, 28, 15.5, r"$= \hat{s}\,C_{ren} + (1{-}\hat{s})\,C_{non}$",
        size=7.5, color=GREY_D)

    # Constants feed (below white-box)
    rect(axA, 28, 5, 24, 6, TINT_INNER, lw=0.7)
    txt(axA, 28, 8.0, r"$C_{ren},\,C_{non}$  region table",
        size=7.5, weight="bold", color=INK)
    arrow(axA, 28, 11.0, 28, 10.3, color=GREY_M, lw=0.7, ls="--")

    # Black-box residual (rounded)
    rbox(axA, 60, 17, 20, 13, TINT_INNER, lw=0.9)
    txt(axA, 60, 24, "Black-Box", size=8.5, weight="bold", color=INK)
    txt(axA, 60, 20.5, "Residual", size=8.5, weight="bold", color=INK)
    txt(axA, 60, 16.0, r"$\hat{\Delta}$ MLP head",
        size=7.5, color=INK, weight="bold")

    # Volatility-gated skip (below residual)
    rbox(axA, 60, 5, 20, 6, TINT_INNER, lw=0.7)
    txt(axA, 60, 8.0, "adaptive persistence gate",
        size=7.5, weight="bold", color=INK)
    arrow(axA, 60, 11.0, 60, 10.3, color=GREY_M, lw=0.7, ls="--")

    # Combine circle
    axA.add_patch(Circle((80, 17), 2.6, facecolor="white",
                         edgecolor=INK, linewidth=0.9, zorder=3))
    txt(axA, 80, 17, "+", size=12, weight="bold", color=INK)

    # Split-conformal band
    rbox(axA, 80, 6, 14, 6, TINT_INNER, lw=0.7)
    txt(axA, 80, 8.5, "conformal band",
        size=7.0, weight="bold", color=INK)
    txt(axA, 80, 5.7, "90% interval", size=6.2, color=GREY_D)

    # Output
    rect(axA, 95, 17, 8, 13, TINT_OUT, lw=1.0)
    txt(axA, 95, 23, r"$\hat{CI}$", size=12, weight="bold", color=NAVY)
    txt(axA, 95, 17, "+ band", size=7, color=GREY_D)

    # Arrows
    arrow(axA, 12, 17, 16, 17, color=INK, lw=1.0)
    arrow(axA, 40, 17, 50, 17, color=INK, lw=1.0)
    arrow(axA, 70, 17, 77.5, 17, color=INK, lw=1.0)
    arrow(axA, 82.6, 17, 91, 17, color=INK, lw=1.0)

    save(fig, "fig_stage2_physics.png")


if __name__ == "__main__":
    fig_overall_architecture()
    fig_stage1_encoder()
    fig_stage2_physics()
    print("done: 3 architecture figures ->", os.path.relpath(
        OUT, os.path.join(HERE, "..")))
