"""Figures for the MAE->10 analysis (Phase 5 follow-up).

Reads results/mae_floor_analysis.json (+ results/residuals/*.npz for the
error-decomposition panel) and writes:

  figures/mae_vs_persistence_floor.png   scatter: joint MAE vs persistence floor
  figures/mae_floor_by_region.png        per-region bars: floor + joint MAE
  figures/error_decomposition.png        per-hour error profile + bias/variance

Style matches scripts/figures/make_mae_overview.py (Okabe-Ito palette).
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

C_JOINT = "#0072B2"     # blue — joint model
C_PERSIST = "#999999"   # grey — persistence floor
C_NOISE = "#E69F00"     # orange — noise floor
C_TARGET = "#D55E00"    # vermillion — MAE 10 target line
TIER_COLOR = {"easy": "#009E73", "medium": "#E69F00", "hard": "#D55E00"}

HERE = Path(__file__).resolve().parent.parent.parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
FIGS.mkdir(exist_ok=True)


def short(name):
    if name.startswith("UK"):
        return "_".join(name.split("_")[:2])
    return name


def load():
    return json.loads((RESULTS / "mae_floor_analysis.json").read_text())


def fig_scatter(data):
    regions = [r for r in data["regions"] if r["joint_median_mae"] is not None]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for r in regions:
        ax.scatter(r["persistence_floor_cif_mae"], r["joint_median_mae"],
                   c=TIER_COLOR[r["tier"]], s=42, alpha=0.85, edgecolor="k",
                   linewidth=0.4, zorder=3)
    # y = x reference (model = persistence floor)
    lim = max(r["persistence_floor_cif_mae"] for r in regions) * 1.05
    ax.plot([0, lim], [0, lim], "--", c="0.6", lw=1, label="joint = persistence floor", zorder=1)
    ax.axhline(10, c=C_TARGET, lw=1.2, ls=":", label="MAE 10 target")
    # label the easy-region cluster
    for r in regions:
        if r["tier"] == "easy" or r["joint_median_mae"] > 80 or r["region"] in (
                "US_BPAT", "UK_08_West_Midlands", "UK_09_East_Midlands"):
            ax.annotate(short(r["region"]),
                        (r["persistence_floor_cif_mae"], r["joint_median_mae"]),
                        fontsize=6.5, xytext=(3, 3), textcoords="offset points")
    from scipy.stats import spearmanr
    sp = spearmanr([r["persistence_floor_cif_mae"] for r in regions],
                   [r["joint_median_mae"] for r in regions])
    ax.set_xlabel("Persistence floor (24h-ahead lag-24 MAE, gCO$_2$/kWh)")
    ax.set_ylabel("Joint-trained median MAE (gCO$_2$/kWh)")
    ax.set_title(f"What bounds MAE: persistence floor drives error "
                 f"(Spearman ρ = {sp.correlation:.2f})")
    # tier legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                      markersize=8, label=f"{t} tier")
               for t, c in TIER_COLOR.items()]
    handles += [Line2D([0], [0], color="0.6", ls="--", label="joint = floor"),
                Line2D([0], [0], color=C_TARGET, ls=":", label="MAE 10")]
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", framealpha=0.9)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = FIGS / "mae_vs_persistence_floor.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out}")


def fig_bars(data):
    regions = sorted(
        [r for r in data["regions"] if r["joint_median_mae"] is not None],
        key=lambda r: r["joint_median_mae"],
    )
    names = [short(r["region"]) for r in regions]
    joint = [r["joint_median_mae"] for r in regions]
    pfloor = [r["persistence_floor_cif_mae"] for r in regions]
    nfloor = [r["noise_floor_mae"] for r in regions]

    x = np.arange(len(regions))
    w = 0.28
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - w, nfloor, w, color=C_NOISE, label="Noise floor (0.8·σ detrended)", alpha=0.9)
    ax.bar(x, pfloor, w, color=C_PERSIST, label="Persistence floor (lag-24)", alpha=0.9)
    ax.bar(x + w, joint, w, color=C_JOINT, label="Joint-trained MAE", alpha=0.9)
    ax.axhline(10, c=C_TARGET, lw=1.2, ls=":", label="MAE 10 target")
    ax.set_ylabel("MAE (gCO$_2$/kWh)")
    ax.set_title("Per-region MAE vs theoretical and practical floors (sorted by joint MAE)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=55, ha="right", fontsize=7)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_ylim(top=max(joint) * 1.12)
    fig.tight_layout()
    out = FIGS / "mae_floor_by_region.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out}")


def fig_error_decomposition():
    """Per-hour-of-day error profile + bias/variance by tier from residuals."""
    res_dir = RESULTS / "residuals"
    files = sorted(res_dir.glob("*_seed*.npz")) if res_dir.exists() else []
    if not files:
        print("[SKIP] error_decomposition: no residuals yet")
        return
    # load floor data for tier lookup
    data = load()
    tier_of = {r["region"]: r["tier"] for r in data["regions"]}

    per_tier_hour_err = {"easy": [], "medium": [], "hard": []}
    per_tier_bias = {"easy": [], "medium": [], "hard": []}
    per_tier_var = {"easy": [], "medium": [], "hard": []}
    for f in files:
        region = f.stem.rsplit("_seed", 1)[0]
        tier = tier_of.get(region, "medium")
        d = np.load(f)
        if "pred_joint_trained" not in d.files:
            continue
        y = d["y_true"]
        pred = d["pred_joint_trained"]
        err = pred - y  # signed
        n_eval = y.shape[0]
        # reshape per-hour: each origin has 24 hours; hour index 0..23
        per_hour_err = np.abs(err).reshape(n_eval, 24).mean(axis=0)  # (24,)
        per_tier_hour_err[tier].append(per_hour_err)
        per_tier_bias[tier].append(err.mean())
        per_tier_var[tier].append(err.std())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5),
                                   gridspec_kw={"width_ratios": [2, 1]})
    hours = np.arange(24)
    for tier in ["easy", "medium", "hard"]:
        if not per_tier_hour_err[tier]:
            continue
        mat = np.array(per_tier_hour_err[tier])
        mean_prof = mat.mean(axis=0)
        sem = mat.std(axis=0) / np.sqrt(mat.shape[0])
        ax1.plot(hours, mean_prof, "-o", ms=3, color=TIER_COLOR[tier],
                 label=f"{tier} tier (n={mat.shape[0]})")
        ax1.fill_between(hours, mean_prof - sem, mean_prof + sem,
                         alpha=0.2, color=TIER_COLOR[tier])
    ax1.set_xlabel("Hour ahead (0 = first forecast hour)")
    ax1.set_ylabel("Mean |error| (gCO$_2$/kWh)")
    ax1.set_title("Error by forecast horizon (joint model)")
    ax1.set_xticks([0, 6, 12, 18, 23])
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # bias/variance box
    for i, tier in enumerate(["easy", "medium", "hard"]):
        if not per_tier_bias[tier]:
            continue
        biases = per_tier_bias[tier]
        vars_ = per_tier_var[tier]
        ax2.scatter(biases, vars_, c=TIER_COLOR[tier], s=50, alpha=0.8,
                    edgecolor="k", linewidth=0.4, label=f"{tier}")
    ax2.axvline(0, c="0.6", lw=1)
    ax2.set_xlabel("Bias = mean(pred − true)")
    ax2.set_ylabel("Error std (variance)")
    ax2.set_title("Bias vs variance per region")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out = FIGS / "error_decomposition.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out}")


def main():
    data = load()
    fig_scatter(data)
    fig_bars(data)
    fig_error_decomposition()


if __name__ == "__main__":
    main()
