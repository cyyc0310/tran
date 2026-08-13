"""Information-set tier and difficulty figures for the zero-shot config-CIF paper.

Generates 5 publication figures directly from results/*.json:
  1. information_set_tiers   - headline ladder (ZS -> ZS+ -> Joint vs PatchTST)
  2. equalizer_effect        - 4 architectures converge to the same ZS+ median
  3. calibration_curve       - calibration-hours curve (beautified, replaces existing)
  4. difficulty_stratification - persistence vs joint MAE scatter by difficulty tier
  5. fuel_regime_clusters    - 26-region fuel-share stacked bars, cluster-ordered

All figures: matplotlib Agg backend, PNG + PDF at 150 dpi, English labels,
Okabe-Ito colorblind-safe palette. Run from repo root:
    python scripts/figures/make_information_set_figures.py
"""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "results"
FIGDIR = "figures"

# Okabe-Ito colorblind-safe palette
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000", "grey": "#999999",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.4,
    "savefig.dpi": 150,
    "pdf.fonttype": 42,
})


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"),
                    bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  wrote figures/{name}.png + .pdf")


def short(region):
    if region.startswith("UK_"):
        return "_".join(region.split("_")[:2])
    return region


# ---------------------------------------------------------------------------
# Figure 1: information-set tier ladder (headline)
# ---------------------------------------------------------------------------
def fig_information_set_tiers():
    """Three information-set tiers vs supervised reference.

    Data: Table 1 headline numbers, confirmed against:
      - results/unified_eval_full.json  (ZS 52.1, ZS+ 46.88, PatchTST 41.47)
      - results/joint_train_native_full.json (Joint 39.53, torch-native Phase 9)
    """
    # Headline numbers from docs/paper Table 1 (line 185-188), cross-checked
    # against the JSON aggregations.
    tiers = [
        {"x": 0, "mae": 52.1,  "label": "ZS\n($\\mathcal{I}_0$)",
         "budget": "0 h", "ratio": "$\\rho$=1.24"},
        {"x": 1, "mae": 46.88, "label": "ZS+\n($\\mathcal{I}_+$)",
         "budget": "0 h\n(observable)", "ratio": "$\\rho^+$=1.08"},
        {"x": 2, "mae": 39.53, "label": "Joint\n($\\mathcal{I}_J$)",
         "budget": "288 h\n(12 days)", "ratio": "$\\rho^J$=0.95"},
    ]
    patch_mae = 41.47

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    xs = [t["x"] for t in tiers]
    ys = [t["mae"] for t in tiers]

    # Dashed trend line ZS -> ZS+ -> Joint (decreasing).
    ax.plot(xs, ys, "o--", color=C["blue"], lw=1.8, ms=11,
            markerfacecolor="white", markeredgewidth=2.0,
            markeredgecolor=C["blue"], zorder=4, label="Information-set tiers")

    # Color each tier marker distinctly.
    tier_colors = [C["orange"], C["green"], C["purple"]]
    for t, col in zip(tiers, tier_colors):
        ax.plot(t["x"], t["mae"], "o", ms=11, color=col,
                markeredgecolor="black", markeredgewidth=0.6, zorder=6)

    # PatchTST supervised reference as a horizontal band.
    ax.axhline(patch_mae, color=C["red"], ls="-.", lw=1.6, zorder=3,
               label=f"PatchTST-supervised ({patch_mae})")
    ax.axhspan(patch_mae - 0.5, patch_mae + 0.5, color=C["red"],
               alpha=0.10, zorder=1)

    # Annotate each tier with budget + ratio + MAE.
    for t, col in zip(tiers, tier_colors):
        ax.annotate(
            f"MAE {t['mae']:.2f}",
            xy=(t["x"], t["mae"]), xytext=(t["x"], t["mae"] - 4.6),
            ha="center", fontsize=9.5, fontweight="bold", color=col, zorder=7,
        )
        ax.annotate(
            t["ratio"],
            xy=(t["x"], t["mae"]), xytext=(t["x"], t["mae"] + 2.4),
            ha="center", fontsize=8.5, color=C["black"], zorder=7,
        )

    # X axis: tier budget labels.
    ax.set_xticks(xs)
    ax.set_xticklabels([f'{t["label"]}\n{t["budget"]}' for t in tiers],
                       fontsize=9)

    ax.set_xlim(-0.6, 2.6)
    ax.set_ylim(34, 58)
    ax.set_ylabel("Median MAE (gCO$_2$/kWh)")
    ax.set_xlabel("Information set (target-domain label budget)")
    ax.set_title("Performance ladder by information-set tier (29-region LORO)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(frameon=False, loc="upper right")

    # Caption-style note on the right.
    ax.text(2.55, 34.6,
            "Lower is better\nJoint < PatchTST\nin 61% of pairs",
            fontsize=7.5, color=C["grey"], ha="right", va="bottom", style="italic")

    fig.tight_layout()
    save(fig, "information_set_tiers")


# ---------------------------------------------------------------------------
# Figure 2: equalizer effect
# ---------------------------------------------------------------------------
def fig_equalizer_effect():
    """4 architectures diverge wildly in ZS but converge under ZS+ calibration."""
    pm = load("probe_models.json")
    pw = load("probe_weather.json")

    variants = [
        ("Flagship", pm["results"]["flagship"]),
        ("MoE",      pm["results"]["moe3"]),
        ("RevIN",    pm["results"]["revin"]),
        ("Weather",  pw["results"]["weather"]),
    ]
    names = [v[0] for v in variants]
    zs_med  = [float(np.median([d["transcif_zs"]      for d in v[1].values()])) for v in variants]
    zsp_med = [float(np.median([d["transcif_zs_plus"] for d in v[1].values()])) for v in variants]

    conv_center = float(np.median(zsp_med))
    conv_half = 0.05  # specified convergence band +/-0.05

    x = np.arange(len(names))
    w = 0.36

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    b1 = ax.bar(x - w/2, zs_med, w, color=C["orange"], edgecolor="black",
                linewidth=0.5, label="ZS (0 labels)", zorder=3)
    b2 = ax.bar(x + w/2, zsp_med, w, color=C["green"], edgecolor="black",
                linewidth=0.5, label="ZS+ (0 labels, calibrated)", zorder=3)

    # Value labels.
    for rect, val in zip(b1, zs_med):
        ax.text(rect.get_x() + rect.get_width()/2, val + 1.0, f"{val:.1f}",
                ha="center", fontsize=8.5, color=C["orange"], fontweight="bold")
    for rect, val in zip(b2, zsp_med):
        ax.text(rect.get_x() + rect.get_width()/2, val + 1.0, f"{val:.2f}",
                ha="center", fontsize=8.5, color=C["green"], fontweight="bold")

    # Convergence band for ZS+.
    ax.axhspan(conv_center - conv_half, conv_center + conv_half,
               color=C["sky"], alpha=0.30, zorder=1,
               label=f"ZS+ convergence band ({conv_center:.2f}$\\pm${conv_half})")
    ax.axhline(conv_center, color=C["blue"], ls="--", lw=1.2, zorder=2)

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylabel("Median MAE (gCO$_2$/kWh)")
    ax.set_ylim(0, max(zs_med) * 1.18)
    ax.set_title("Equalizer effect: 4 architectures converge to the same ZS+ median")
    ax.legend(frameon=False, loc="upper center", ncol=1)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    save(fig, "equalizer_effect")


# ---------------------------------------------------------------------------
# Figure 3: calibration curve (beautified)
# ---------------------------------------------------------------------------
def fig_calibration_curve():
    """Median MAE vs calibration hours, with sweet spot + overfit annotation."""
    doc = load("probe_calibration_curve.json")
    hours = doc["calibration_hours"]      # [0, 72, 144, 288, 576]
    results = doc["results"]              # {n_train: {target: mae}}
    sweep = doc["n_train_sweep"]          # [0, 3, 6, 12, 24]

    medians, p25s, p75s = [], [], []
    for n_train in sweep:
        vals = list(results[str(n_train)].values())
        medians.append(float(np.nanmedian(vals)))
        p25s.append(float(np.nanpercentile(vals, 25)))
        p75s.append(float(np.nanpercentile(vals, 75)))

    fig, ax = plt.subplots(figsize=(7.0, 4.2))

    ax.fill_between(hours, p25s, p75s, alpha=0.18, color=C["blue"],
                    label="25–75 percentile", zorder=2)
    ax.plot(hours, medians, "o-", color=C["blue"], lw=2.0, ms=9,
            markeredgecolor="black", markeredgewidth=0.5, zorder=5,
            label="Median MAE")

    # Value labels.
    for h, m in zip(hours, medians):
        ax.annotate(f"{m:.1f}", xy=(h, m), xytext=(0, 8),
                    textcoords="offset points", ha="center",
                    fontsize=8.5, color=C["blue"], fontweight="bold")

    # Sweet spot at 144 h.
    sweet_idx = hours.index(144)
    ax.axvline(144, color=C["green"], ls=":", lw=1.4, alpha=0.8, zorder=3)
    ax.annotate(
        "Sweet spot\n144 h (6 days)\n$\\rightarrow$ first clear drop",
        xy=(144, medians[sweet_idx]),
        xytext=(150, medians[sweet_idx] + 18),
        fontsize=8.5, color=C["green"],
        arrowprops=dict(arrowstyle="->", color=C["green"], lw=1.0), zorder=6,
    )

    # Overfit region 144 -> 576.
    ax.axvspan(180, 600, color=C["red"], alpha=0.07, zorder=1)
    ax.text(430, ax.get_ylim()[1] * 0.0 + 8, "no further gain\n(diminishing / overfit)",
            fontsize=8, color=C["red"], ha="center", style="italic", zorder=6)

    # ZS+ regime annotation at 0 h.
    ax.annotate(
        "ZS+\n0 h labels\n(pure test-time)",
        xy=(0, medians[0]),
        xytext=(12, medians[0] - 14),
        fontsize=8.5, color=C["orange"], ha="left",
        arrowprops=dict(arrowstyle="->", color=C["orange"], lw=1.0), zorder=6,
    )

    ax.set_xlabel("Target-domain calibration labels (hours)")
    ax.set_ylabel("Median MAE (gCO$_2$/kWh)")
    ax.set_title("Calibration-data curve: information-set stratification")
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h} h" for h in hours])
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    save(fig, "calibration_curve")


# ---------------------------------------------------------------------------
# Figure 4: difficulty stratification
# ---------------------------------------------------------------------------
def fig_difficulty_stratification():
    """Persistence vs joint MAE scatter, colored by difficulty tier."""
    doc = load("regime_split_report.json")
    rows = [r for r in doc["per_region"]
            if r["joint_mae"] is not None and not (isinstance(r["joint_mae"], float) and np.isnan(r["joint_mae"]))]

    def diff_color(d):
        if d.startswith("easy"):
            return C["green"]
        if d.startswith("medium"):
            return C["orange"]
        return C["red"]

    def diff_label(d):
        if d.startswith("easy"):
            return "easy (<30)"
        if d.startswith("medium"):
            return "medium (30–60)"
        return "pathological (>60)"

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    # Draw per-tier point sets for legend.
    drawn = set()
    for r in rows:
        col = diff_color(r["difficulty"])
        lab = diff_label(r["difficulty"])
        ax.scatter(r["persist_mae"], r["joint_mae"], s=70, color=col,
                   edgecolor="black", linewidth=0.5, alpha=0.9, zorder=4,
                   label=lab if lab not in drawn else None)
        drawn.add(lab)

    # y = x diagonal: if joint == persistence, calibration is ineffective.
    lim_lo = 0
    lim_hi = max(max(r["persist_mae"] for r in rows),
                 max(r["joint_mae"] for r in rows)) * 1.05
    ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "--", color=C["grey"],
            lw=1.3, zorder=2, label="y = x (calibration ineffective)")
    # Better-than-diagonal region shading.
    ax.fill_between([lim_lo, lim_hi], [lim_lo, lim_hi],
                    [lim_lo, lim_lo], color=C["green"], alpha=0.05, zorder=1)

    # Annotate key pathological / hard regions.
    annotate_regions = {
        "VIC1": "VIC1", "SA1": "SA1", "UK_09_East_Midlands": "UK_09",
        "UK_08_West_Midlands": "UK_08", "UK_07_South_Wales": "UK_07",
        "UK_17_Wales": "UK_17", "US_ERCO": "ERCO",
    }
    for r in rows:
        key = r["region"]
        if key in annotate_regions:
            ax.annotate(annotate_regions[key],
                        xy=(r["persist_mae"], r["joint_mae"]),
                        xytext=(6, 4), textcoords="offset points",
                        fontsize=8, color="black", fontweight="bold", zorder=7)

    ax.set_xlim(lim_lo, lim_hi)
    ax.set_ylim(lim_lo, lim_hi)
    ax.set_xlabel("Persistence MAE (gCO$_2$/kWh) — zero-shot difficulty proxy")
    ax.set_ylabel("Joint-trained MAE (gCO$_2$/kWh)")
    ax.set_title("Difficulty stratification: where joint calibration pays off")
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    save(fig, "difficulty_stratification")


# ---------------------------------------------------------------------------
# Figure 5: fuel-regime clusters
# ---------------------------------------------------------------------------
def fig_fuel_regime_clusters():
    """Stacked fuel-share bars for 26 regions, cluster-ordered, with pseudo-nbr pairs."""
    with open(os.path.join("data_2023", "fuel", "fuel_shares_us.json")) as f:
        us = json.load(f)
    with open(os.path.join("data_2023", "fuel", "fuel_shares_uk.json")) as f:
        uk = json.load(f)

    # Canonical fuel columns for plotting (merge biomass into "other_renew").
    plot_fuels = ["coal", "gas", "nuclear", "hydro", "solar", "wind"]
    fuel_colors = {
        "coal": C["black"], "gas": C["orange"], "nuclear": C["purple"],
        "hydro": C["blue"], "solar": C["yellow"], "wind": C["green"],
    }

    # Build per-region normalized vectors over plot_fuels (fold biomass + imports/other into gas bucket
    # as "thermal/other" is not the point; keep renewables clean).
    regions = {}
    for src in (us, uk):
        for name, shares in src["regions"].items():
            vec = {}
            total = 0.0
            for f in plot_fuels:
                val = shares.get(f, 0.0) or 0.0
                vec[f] = val
                total += val
            # normalize to plot_fuels sum so the stack is comparable
            if total > 0:
                for f in plot_fuels:
                    vec[f] /= total
            regions[name] = vec

    # Cluster labels from regime_split_report.json.
    try:
        regime = load("regime_split_report.json")
        cluster_of = {r["region"]: r["cluster"] for r in regime["per_region"]}
    except Exception:
        cluster_of = {}

    # Derive a simple cluster for regions missing one using a quick heuristic:
    # group by dominant renewable vs fossil. Fall back to NaN -> a sentinel.
    def cluster_key(name):
        c = cluster_of.get(name)
        if c is not None and not (isinstance(c, float) and np.isnan(c)):
            return int(c)
        v = regions[name]
        renewable = v["hydro"] + v["solar"] + v["wind"]
        if renewable > 0.5:
            return 3
        if v["nuclear"] > 0.25:
            return 4
        return 2

    names = list(regions.keys())
    # Sort: cluster asc, then renewable share desc within cluster.
    def renewable_share(name):
        v = regions[name]
        return v["hydro"] + v["solar"] + v["wind"]
    names.sort(key=lambda n: (cluster_key(n), -renewable_share(n)))

    fig, ax = plt.subplots(figsize=(8.0, 5.0))

    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    for f in plot_fuels:
        vals = np.array([regions[n][f] for n in names])
        ax.bar(x, vals, bottom=bottom, color=fuel_colors[f], edgecolor="white",
               linewidth=0.3, width=0.82, label=f.capitalize(), zorder=3)
        bottom += vals

    # Cluster separators + top labels.
    prev_c = None
    boundaries = []
    for i, n in enumerate(names):
        c = cluster_key(n)
        if prev_c is not None and c != prev_c:
            boundaries.append(i - 0.5)
        prev_c = c
    for b in boundaries:
        ax.axvline(b, color=C["grey"], lw=0.8, ls="-", alpha=0.6, zorder=2)

    # Cluster band labels at top.
    seg_starts = [0] + [int(round(b + 0.5)) for b in boundaries]
    seg_ends = [int(round(b + 0.5)) for b in boundaries] + [len(names)]
    seen_clusters = set()
    for s, e in zip(seg_starts, seg_ends):
        if s >= e:
            continue
        c = cluster_key(names[s])
        if c in seen_clusters:
            lab = ""
        else:
            seen_clusters.add(c)
            lab = f"cluster {c}"
        if lab:
            ax.text((s + e - 1) / 2, 1.06, lab, ha="center", va="bottom",
                    fontsize=8, color=C["grey"], fontweight="bold")

    # Pseudo-neighbor pair brackets: ERCO/MISO and FPL/PJM.
    def idx(name):
        return names.index(name)

    def bracket(i1, i2, label):
        y = 1.12
        x1, x2 = i1, i2
        ax.plot([x1, x1, x2, x2], [y - 0.015, y, y, y - 0.015],
                color=C["red"], lw=1.0, zorder=8)
        ax.text((x1 + x2) / 2, y + 0.005, label, ha="center", va="bottom",
                fontsize=7.5, color=C["red"], fontweight="bold")

    try:
        bracket(idx("US_ERCO"), idx("US_MISO"), "ERCO / MISO")
    except ValueError:
        pass
    try:
        bracket(idx("US_FPL"), idx("US_PJM"), "FPL / PJM")
    except ValueError:
        pass

    ax.set_xticks(x)
    ax.set_xticklabels([short(n) for n in names], rotation=75,
                       fontsize=7.0, ha="right")
    ax.set_ylim(0, 1.20)
    ax.set_ylabel("Fuel share (normalized)")
    ax.set_title("Fuel-mix composition of 26 regions, cluster-ordered")
    ax.legend(frameon=False, loc="lower center", ncol=6,
              bbox_to_anchor=(0.5, -0.30), fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    ax.set_xlim(-0.6, len(names) - 0.4)

    fig.tight_layout()
    save(fig, "fuel_regime_clusters")


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    print("Figure 1: information_set_tiers")
    fig_information_set_tiers()
    print("Figure 2: equalizer_effect")
    fig_equalizer_effect()
    print("Figure 3: calibration_curve")
    fig_calibration_curve()
    print("Figure 4: difficulty_stratification")
    fig_difficulty_stratification()
    print("Figure 5: fuel_regime_clusters")
    fig_fuel_regime_clusters()
    print("\nDone.")


if __name__ == "__main__":
    main()
