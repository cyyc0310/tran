#!/usr/bin/env python
"""TransCIF-FD overall architecture figure (data/physics layer, FD-16..FD-33 state).

Vertical pipeline (bottom -> top), style matched to the legacy
fig_overall_architecture_v2: rounded boxes, purple palette, colour-coded
arrows (blue = main signal, dashed purple = config conditioning,
orange = deterministic routing), right-hand annotation column.

Usage:
    .venv/bin/python scripts/figures/make_fd_architecture.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent.parent / "figures"

# palette
C_DATA_FC, C_DATA_EC = "#eef1f9", "#9aa7cc"
C_FEAT_FC, C_FEAT_EC = "#e3ecf9", "#7f9fd4"
C_HEAD_FC, C_HEAD_EC = "#f3edf9", "#b090d6"
C_AGG_FC, C_AGG_EC = "#ece2f4", "#9d7cc0"
C_ROUTE_FC, C_ROUTE_EC = "#fdf3e0", "#e0a83f"
C_PHYS_FC, C_PHYS_EC = "#e9f7ee", "#79bb8f"
C_TIER_FC = ["#ece7f7", "#d9cff0", "#bda9e3"]
C_TIER_EC = "#8a5fc0"
C_NOTE_FC, C_NOTE_EC = "#f7f5fb", "#b9aed6"
C_MAIN, C_CFG, C_ROUTE = "#4a6fb5", "#8a5fc0", "#e09b2d"
C_TXT = "#3a3a4a"


def box(ax, x, y, w, h, lines, fc, ec, fs=8.0, lw=1.1, title_fs=None,
        title_c="#4a3f6b"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.9",
        fc=fc, ec=ec, lw=lw, mutation_aspect=1.0))
    if isinstance(lines, str):
        lines = [lines]
    tfs = title_fs or fs + 0.6
    ax.text(x + w / 2, y + h - 1.55, lines[0], ha="center", va="top",
            fontsize=tfs, fontweight="bold", color=title_c)
    for i, ln in enumerate(lines[1:]):
        ax.text(x + w / 2, y + h - 1.55 - (i + 1) * 2.6, ln, ha="center",
                va="top", fontsize=fs, color=C_TXT)


def arrow(ax, x0, y0, x1, y1, color=C_MAIN, lw=2.0, style="-"):
    ax.annotate(
        "", xy=(x1, y1), xytext=(x0, y0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw,
                        linestyle=style, mutation_scale=16,
                        shrinkA=0, shrinkB=0))


def container_label(ax, x, y, text):
    ax.text(x, y, text, ha="left", va="center", fontsize=10,
            fontweight="bold", color="#5a4d85")


fig, ax = plt.subplots(figsize=(15.5, 19.5), dpi=300)
ax.set_xlim(0, 100)
ax.set_ylim(0, 142)
ax.axis("off")

# ---------------------------------------------------------------- title
ax.text(50, 139.5,
        "TransCIF-FD: Fuel-Decomposition Architecture for Zero-Shot Cross-Region CIF Forecasting",
        ha="center", va="center", fontsize=14.5, fontweight="bold", color="#2e2845")
ax.text(50, 136.4,
        "data / physics layer as of FD-16..FD-33  ·  model frozen (~21k params)  ·  "
        "145-pair LORO:  $I_{cfg}$ 40.94 · $I_0$ 37.17 · $I_+$ 38.2",
        ha="center", va="center", fontsize=10, color="#6a6284")

MX, MW = 4.0, 63.0          # main column
RX, RW = 70.5, 28.5         # right annotation column
CX = MX + MW / 2            # centre line of the main column

# ---------------------------------------------------------------- output (top)
box(ax, MX + 6, 124.0, MW - 12, 6.5,
    ["OUTPUT — 24 h day-ahead CIF trajectory",
     "hourly CIF curve  ·  per-fuel decomposition  ·  ranking signal (Spearman)"],
    C_PHYS_FC, C_PHYS_EC, fs=8.2)

# ---------------------------------------------------------------- information tiers
container_label(ax, MX, 122.3, "3 · INFORMATION TIERS (strictly nested legal inputs)")
tiers = [
    ("$I_+$   + observable CIF history",
     "ZS+ test-time calibration · 6 branches, per-lead inverse-power weights",
     "38.2  (multi-year)", 2),
    ("$I_0$   + 336 h renewable-share stream",
     "level anchoring:  $\\widehat{CIF} += g\\,((1-\\bar{r}_{48h})\\,ef_{nr} - \\overline{\\widehat{CIF}})$",
     "37.17  (beats 43.50)", 1),
    ("$I_{cfg}$   ZERO TELEMETRY (config + weather + calendar)",
     "the deployable tier for telemetry-scarce grids (e.g. Chinese provinces)",
     "40.94  (−35%)", 0),
]
ty = 97.0
for title, sub, num, k in tiers:
    box(ax, MX + 4, ty, MW - 8, 6.6, [title, sub],
        C_TIER_FC[k], C_TIER_EC, fs=8.0, title_fs=9.6)
    ax.text(MX + MW - 5.5, ty + 3.3, num, ha="right", va="center",
            fontsize=9.6, fontweight="bold", color="#5a4d85")
    ty += 8.8
# ladder arrows between tiers, then into the output box
arrow(ax, CX, 103.8, CX, 105.6)
arrow(ax, CX, 112.6, CX, 114.4)
arrow(ax, CX, 121.4, CX, 123.8)
ax.text(MX + MW - 5.5, 95.4,
        "$I_S$ supervised upper bound (PatchTST, 80% local labels): 43.50",
        ha="right", va="center", fontsize=8.2, style="italic", color="#6a6284")

# ---------------------------------------------------------------- physics layer
container_label(ax, MX, 91.2, "2 · PHYSICS SYNTHESIS (closed form)")
box(ax, MX, 81.0, MW, 9.4,
    ["$\\widehat{CIF}(h) \\;=\\; \\sum_f \\hat{s}_f(h)\\;\\mathrm{ef}_f\\;(1 \\pm 0.35\\,\\mathrm{tanh}(\\cdot))$"
     "        Theorem 1:  $|\\widehat{CIF}-CIF| \\approx \\kappa_{region}\\,|\\hat{r}-r| + \\epsilon_{phys}$",
     "bounded EF correction (calibrated EFs, FD-23)  ·  level anchored separately from shape"],
    C_PHYS_FC, C_PHYS_EC, fs=8.4, title_fs=9.8)
arrow(ax, CX, 90.6, CX, 96.8, lw=2.4)

# ---------------------------------------------------------------- model
container_label(ax, MX, 77.8, "1 · FuelDecompNet  (~21k params — deep conditioning proven harmful ×3)")
heads = [
    ("Solar", "astro envelope ×\nbounded wx modulation"),
    ("Wind", "IEC wcf / drought-\nanchored reference"),
    ("Baseload ×5", "level + support mask\n(no phantom fuel)"),
    ("Thermal ×3", "residual + config-\nanchored softmax split"),
    ("Aggregate head", "DLinear(rs) + logit\nanchor + persist gate"),
]
hw_, gap = 11.6, 1.5
hx = MX + 1.0
for i, (t, s) in enumerate(heads):
    fc, ec = (C_AGG_FC, C_AGG_EC) if i == 4 else (C_HEAD_FC, C_HEAD_EC)
    ax.add_patch(FancyBboxPatch((hx, 68.0), hw_, 8.2,
                                boxstyle="round,pad=0.25,rounding_size=0.9",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(hx + hw_ / 2, 74.2, t, ha="center", va="center",
            fontsize=8.8, fontweight="bold", color="#4a3f6b")
    for j, ln in enumerate(s.split("\n")):
        ax.text(hx + hw_ / 2, 71.7 - j * 2.3, ln, ha="center", va="center",
                fontsize=6.9, color=C_TXT)
    hx += hw_ + gap
arrow(ax, CX, 76.4, CX, 80.8, lw=2.4)          # heads -> physics

# routing bar
box(ax, MX, 57.0, MW, 8.6,
    ["DETERMINISTIC STRUCTURE ROUTING  (zero parameters)",
     "$c_{wind} \\geq \\tau$ or no fuel telemetry → aggregate path   ·   "
     "hydro gate $\\sigma(20\\,(0.5-c_{hydro}))$ → load-following (BPAT 46.7→16.4)"],
    C_ROUTE_FC, C_ROUTE_EC, fs=8.0, title_fs=9.4, title_c="#8a6210")
arrow(ax, CX, 65.8, CX, 67.8, color=C_ROUTE, lw=2.2)   # routing -> heads

# ---------------------------------------------------------------- feature stack
box(ax, MX, 48.0, MW, 6.4,
    ["FEATURE STACK   $x_{fuel}\\,(L{\\times}10)$ · $x_{wx}\\,(L{\\times}10)$ · "
     "$fut_{exog}\\,(H{\\times}17)$ · $config\\,(16)$ · $ef\\,(10)$",
     "L = 336 h history · H = 24 h horizon · cold-mode dropout p = 0.3 on history "
     "→ one weight set serves $I_{cfg}$ and $I_0$"],
    C_FEAT_FC, C_FEAT_EC, fs=8.0, title_fs=9.0)
arrow(ax, CX, 54.6, CX, 56.8, lw=2.4)          # features -> routing

# ---------------------------------------------------------------- data layer
container_label(ax, MX, 45.6, "0 · DATA LAYER — public tracks only, no keys")
data_boxes = [
    ("Fuel telemetry — 28 sources", "EIA-930 · UK CI API · AU NEMED\nDUID (train side only)"),
    ("Weather — ERA5 + farmblend", "capacity-weighted farm weather\n(wind R² up to 0.71)"),
    ("Astro & calendar", "solar geometry / clear-sky\nUTC-normalised (deterministic)"),
    ("Day-ahead load forecast", "EIA-930, z-scored,\nwindow de-meaned"),
    ("Fuel prices", "WB pink sheets · FRED\n(1-month publication lag)"),
    ("Region config — 16-dim", "mean_rs · ef_nr\nfuel shares · climate"),
    ("Curated unit registry", "FD-28 — fixes fuel-bucket\nlabels (hydro in wind)"),
    ("Calibrated EFs", "FD-23 — ridge λ=15\nvs reported CIF, clip [0,1400]"),
]
bw, bh = 15.1, 7.6
for i, (t, s) in enumerate(data_boxes):
    row, col = divmod(i, 4)
    bx = MX + col * (bw + 1.2)
    by = 35.6 - row * (bh + 1.6)
    box(ax, bx, by, bw, bh, [t, s], C_DATA_FC, C_DATA_EC, fs=6.8, title_fs=7.6)
arrow(ax, CX, 45.0, CX, 47.8, lw=2.4)          # data -> features

# FD-17 callout, one line, clear of the boxes and the LORO band
ax.text(CX, 23.9,
        "FD-17 wind-speed unit fix (km/h fed to m/s IEC curve; 39–42% of hours misread as cut-out) — largest single gain",
        ha="center", va="center", fontsize=7.6, style="italic", color="#8a6210",
        bbox=dict(boxstyle="round,pad=0.35", fc="#fdf3e0", ec="#e0a83f", lw=0.9))
ax.text(CX, 20.4,
        "source regions (unrestricted) ──── LORO: 28 sources → 1 unseen target · 29 regions × 5 seeds ────",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color="#5a4d85")

# config conditioning side-arrow (dashed purple, along the left of the model column)
arrow(ax, MX - 1.3, 41.0, MX - 1.3, 76.0, color=C_CFG, lw=1.8, style="--")
ax.text(MX - 2.1, 58.5, "config conditioning (FiLM / prior anchor)", rotation=90,
        ha="center", va="center", fontsize=7.4, color=C_CFG)

# ---------------------------------------------------------------- right column
notes = [
    ("145-PAIR LORO RESULT",
     ["median MAE, gCO₂/kWh",
      "$I_{cfg}$   62.75 → 40.94  (−35%)",
      "$I_0$     53.39 → 37.17",
      "$I_+$     46.9 → 38.2 (multi-year)",
      "beats supervised PatchTST",
      "reference 43.50 by 6.3",
      "win 75% · p = 7.1e−11"]),
    ("TRAINING",
     ["$\\mathcal{L} = MAE + 1.0\\,L_{shareEF}$",
      "     $+\\ 0.3\\,L_{rs} + 0.5\\,L_{shape}$",
      "config-distance sampling",
      "$w \\propto 1/(|\\Delta mean_{rs}| + 0.05)$",
      "600 epochs · Adam 1e−3",
      "all dynamic corrections",
      "zero-initialised (physics warm start)"]),
    ("ALL GAINS = DATA / PHYSICS",
     ["model architecture frozen",
      "unit fix · farmblend ×4 rounds",
      "calibrated EFs · registry",
      "deterministic routing",
      "negative probes kept honest:",
      "regime feats, seasonal head,",
      "source bias, shrink λ, demand"]),
    ("ZERO-SHOT DEPLOY (CN)",
     ["monthly provincial config →",
      "hourly CIF + fuel decomposition",
      "hydro provinces  ~15–19",
      "coal-flat provinces  18–22",
      "(demo: demo_cn_province.py)"]),
]
ny = 122.0
for title, rows in notes:
    bh_ = 4.4 + 2.35 * len(rows)
    box(ax, RX, ny - bh_, RW, bh_, [title] + rows, C_NOTE_FC, C_NOTE_EC,
        fs=7.3, title_fs=8.4)
    ny -= bh_ + 3.4

# ---------------------------------------------------------------- footer
ax.text(50, 3.2,
        "Reproducibility: results/fuel_decomp_eval_full_fd28.json (2023 protocol) · "
        "fd29_multiyear.json · verdicts in results/fd16..fd33_*_verdict.md · 170 tests green",
        ha="center", va="center", fontsize=7.8, color="#8a86a0")

fig.savefig(OUT / "fd_architecture.png", bbox_inches="tight", facecolor="white")
fig.savefig(OUT / "fd_architecture.pdf", bbox_inches="tight", facecolor="white")
print(f"saved {OUT/'fd_architecture.png'} and .pdf")
