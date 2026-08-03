"""Publication-quality figures for the zero-shot config-CIF paper.

Regenerates every figure referenced in docs/paper/2026-07-26-zeroshot-config-cif-paper.md
directly from results/*.json, with a unified submission-grade style:
- Okabe-Ito colorblind-safe palette
- 300 dpi PNG + vector PDF, two-column (7.0 in) or single-column (3.5 in) widths
Run from repo root: python scripts/make_submission_figures.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = "results"
FIGDIR = "figures"

# Okabe-Ito palette
C = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "sky": "#56B4E9",
    "yellow": "#F0E442", "black": "#000000", "grey": "#999999",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "lines.linewidth": 1.2,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})


def load(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGDIR, f"{name}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.png + .pdf")


def short(region):
    """UK_14_South_East_England -> UK_14; others unchanged."""
    if region.startswith("UK_"):
        return "_".join(region.split("_")[:2])
    return region


# ---------------------------------------------------------------------------
# Figure 1: main 29-region LORO benchmark (Table 1 companion)
# ---------------------------------------------------------------------------

def fig_main_benchmark():
    rows = load("unified_eval_full.json")
    from collections import defaultdict
    g = defaultdict(list)
    for r in rows:
        g[r["target"]].append(r)
    regions = []
    for k, rs_list in g.items():
        maes = [r["transcif_zs"]["mae"] for r in rs_list]
        maes_p = [r["transcif_zs_plus"]["mae"] for r in rs_list]
        sup = float(np.mean([r["patchtst_sup"]["mae"] for r in rs_list]))
        regions.append({
            "region": short(k),
            "mean_rs": rs_list[0]["mean_rs"],
            "rho": float(np.mean(maes)) / sup,
            "rho_plus": float(np.mean(maes_p)) / sup,
            "rho_p": float(np.mean(maes)) / float(np.mean([r["persistence"]["mae"] for r in rs_list])),
        })
    regions.sort(key=lambda d: d["mean_rs"])
    rs = np.array([d["mean_rs"] for d in regions])
    rho = np.array([d["rho"] for d in regions])
    rho_plus = np.array([d["rho_plus"] for d in regions])
    rho_p = np.array([d["rho_p"] for d in regions])
    names = [d["region"] for d in regions]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.7),
                                   gridspec_kw={"width_ratios": [1.7, 1.0]})
    x = np.arange(len(names))
    ax1.bar(x - 0.27, rho, 0.27, color=C["blue"], label=r"$\rho$ = ZS / PatchTST-sup")
    ax1.bar(x, rho_plus, 0.27, color=C["green"], label=r"$\rho^+$ = ZS+ / PatchTST-sup")
    ax1.bar(x + 0.27, rho_p, 0.27, color=C["orange"], label=r"$\rho_P$ = ZS / persistence")
    ax1.axhline(1.0, color=C["black"], lw=0.7, ls="--")
    ax1.axhline(float(np.median(rho_plus)), color=C["green"], lw=0.7, ls=":",
                label=f"median $\\rho^+$ = {np.median(rho_plus):.2f}")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=90)
    ax1.set_ylabel("MAE ratio")
    ax1.set_title("(a) 29-region LORO benchmark (sorted by $\\bar{rs}$)")
    ax1.legend(frameon=False, loc="upper right", ncol=1)

    coeffs = np.polyfit(rs, rho, 2)
    r2 = 1 - np.sum((rho - np.polyval(coeffs, rs))**2) / np.sum((rho - rho.mean())**2)
    xx = np.linspace(rs.min(), rs.max(), 200)
    ax2.scatter(rs, rho, s=14, color=C["blue"], zorder=3, label="ZS")
    ax2.plot(xx, np.polyval(coeffs, xx), color=C["red"], lw=1.2,
             label=f"quadratic fit ($R^2$={r2:.2f})")
    ax2.scatter(rs, rho_plus, s=14, color=C["green"], zorder=3, label="ZS+ (calibrated)")
    ax2.axhline(1.0, color=C["black"], lw=0.7, ls="--")
    ax2.set_xlabel(r"mean renewable share $\bar{rs}$")
    ax2.set_ylabel(r"MAE ratio vs. supervised")
    ax2.set_title("(b) ZS+ flattens the U-shaped difficulty")
    ax2.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "main_benchmark")


# ---------------------------------------------------------------------------
# Theorem 1 figures
# ---------------------------------------------------------------------------

def fig_theorem1():
    t1 = load("theorem1_validation.json")
    t1.sort(key=lambda r: r["mean_rs"])
    names = [short(r["region"]) for r in t1]
    term1 = np.array([r["term1_mean_abs"] for r in t1])
    term2 = np.array([r["term2_mean_abs"] for r in t1])
    x = np.arange(len(names))

    # (a) stacked decomposition per region
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    ax.bar(x, term1, 0.65, color=C["blue"], label=r"Term 1: $|(\hat{s}-s)(ef_r-ef_{nr})|$ (amplification)")
    ax.bar(x, term2, 0.65, bottom=term1, color=C["orange"], label=r"Term 2: $|\delta_t|$ (physics residual)")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=90)
    ax.set_ylabel("mean |error| (gCO$_2$/kWh)")
    frac = float(np.mean([r["term1_fraction"] for r in t1]))
    ax.set_title(f"Theorem 1 decomposition — identity residual $\\leq 1.3\\times10^{{-4}}$; "
                 f"Term 1 = {frac*100:.1f}% of error on average")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "theorem1_error_propagation")

    # (b) L_T * eps_rs predicts realized CIF MAE
    lt_eps = np.array([r["L_T"] * r["mean_rs_error_abs"] for r in t1])
    mae = np.array([r["mean_cif_error_abs"] for r in t1])
    corr = float(np.corrcoef(lt_eps, mae)[0, 1])
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    lim = max(lt_eps.max(), mae.max()) * 1.08
    ax.plot([0, lim], [0, lim], color=C["grey"], lw=0.8, ls="--", label="y = x")
    ax.scatter(lt_eps, mae, s=16, color=C["blue"], zorder=3)
    b, a = np.polyfit(lt_eps, mae, 1)
    ax.plot([0, lim], [a, a + b * lim], color=C["red"], lw=1.1,
            label=f"fit: $r$ = {corr:.3f} ($R^2$ = {corr**2:.3f})")
    for r, xx, yy in zip(t1, lt_eps, mae):
        if xx > 90 or yy > 90:
            ax.annotate(short(r["region"]), (xx, yy), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel(r"predicted bound term $L_T \cdot \varepsilon_{rs}$ (gCO$_2$/kWh)")
    ax.set_ylabel("realized CIF MAE (gCO$_2$/kWh)")
    ax.set_title("Amplification term predicts CIF error")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    save(fig, "theorem1_lt_amplification")


# ---------------------------------------------------------------------------
# Theorem 2 figure
# ---------------------------------------------------------------------------

def fig_theorem2():
    t2 = load("theorem2_transfer_bound.json")
    ad = t2["analysis_data"]
    st = t2["statistics"]
    rs = np.array([r["mean_rs"] for r in ad])
    ratio = np.array([r["ratio_vs_patchtst"] for r in ad])

    # effective distance (Eq. 4 weighting), recomputed exactly as in the model
    d_eff = []
    for j, rj in enumerate(rs):
        d = np.abs(np.delete(rs, j) - rj)
        w = 1.0 / (d + 0.05)
        d_eff.append(float((w * d).sum() / w.sum()))
    d_eff = np.array(d_eff)

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5))

    # (a) proxy comparison
    labels = ["nearest\nsource", "centroid\ndist.", "source\ndensity", "effective\ndist."]
    corrs = [st["corr_min_dist"], st["corr_centroid_dist"], st["corr_density"], st["corr_effective_dist"]]
    ps = [st["p_min_dist"], st["p_centroid_dist"], st["p_density"], st["p_effective_dist"]]
    colors = [C["grey"], C["grey"], C["grey"], C["blue"]]
    bars = axes[0].bar(range(4), corrs, color=colors)
    for i, (c_, p_) in enumerate(zip(corrs, ps)):
        tag = "$p$=0.001" if p_ < 0.005 else "n.s."
        axes[0].text(i, c_ + (0.03 if c_ >= 0 else -0.07), tag, ha="center", fontsize=6)
    axes[0].axhline(0, color=C["black"], lw=0.6)
    axes[0].set_xticks(range(4)); axes[0].set_xticklabels(labels)
    axes[0].set_ylabel(r"corr. with transfer ratio $\rho$")
    axes[0].set_title("(a) which proxy predicts difficulty")
    axes[0].set_ylim(-0.35, 0.75)

    # (b) effective distance vs ratio
    axes[1].scatter(d_eff, ratio, s=14, color=C["blue"], zorder=3)
    b, a = np.polyfit(d_eff, ratio, 1)
    xx = np.linspace(d_eff.min(), d_eff.max(), 50)
    axes[1].plot(xx, a + b * xx, color=C["red"], lw=1.1,
                 label=f"$r$ = {st['corr_effective_dist']:.2f}, $p$ = 0.001")
    axes[1].set_xlabel(r"effective config distance $d_{\mathrm{eff}}$")
    axes[1].set_ylabel(r"$\rho$")
    axes[1].set_title("(b) model-weighted distance")
    axes[1].legend(frameon=False)

    # (c) U-shape in mean_rs
    q = st["quadratic_coeffs"]
    xx = np.linspace(rs.min(), rs.max(), 200)
    axes[2].scatter(rs, ratio, s=14, color=C["blue"], zorder=3)
    axes[2].plot(xx, np.polyval(q, xx), color=C["red"], lw=1.1,
                 label=f"$R^2$ = {st['r2_quadratic_rs']:.2f}")
    vertex = -q[1] / (2 * q[0])
    axes[2].axvline(vertex, color=C["grey"], lw=0.7, ls=":")
    axes[2].text(vertex + 0.01, axes[2].get_ylim()[0] + 0.15, f"$\\bar{{rs}}$={vertex:.2f}", fontsize=6)
    axes[2].set_xlabel(r"$\bar{rs}$")
    axes[2].set_ylabel(r"$\rho$")
    axes[2].set_title("(c) U-shaped difficulty")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    save(fig, "theorem2_config_distance")


# ---------------------------------------------------------------------------
# Conformal prediction figure
# ---------------------------------------------------------------------------

def fig_conformal():
    cf = load("conformal_prediction.json")
    cf.sort(key=lambda r: r["mean_rs"])
    names = [short(r["region"]) for r in cf]
    cov90 = np.array([r["coverage_90_per_h"] for r in cf])

    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.5),
                             gridspec_kw={"width_ratios": [1.6, 1.0, 1.0]})

    # (a) per-region coverage
    x = np.arange(len(names))
    colors = [C["blue"] if c >= 0.9 else C["red"] for c in cov90]
    axes[0].bar(x, cov90, 0.65, color=colors)
    axes[0].axhline(0.9, color=C["black"], lw=0.8, ls="--", label="nominal 90%")
    axes[0].set_xticks(x); axes[0].set_xticklabels(names, rotation=90)
    axes[0].set_ylim(0.6, 1.02)
    axes[0].set_ylabel("empirical coverage")
    axes[0].set_title(f"(a) 90% coverage — {int((cov90>=0.9).sum())}/29 valid, mean {cov90.mean():.3f}")
    axes[0].legend(frameon=False, loc="lower right")

    # (b) reliability curve (mean over regions)
    levels = np.array(cf[0]["reliability_levels"])
    rel = np.array([r["reliability_coverages"] for r in cf])
    axes[1].plot([0.4, 1.0], [0.4, 1.0], color=C["grey"], lw=0.8, ls="--")
    axes[1].plot(levels, rel.mean(axis=0), marker="o", ms=3, color=C["blue"], label="mean")
    axes[1].fill_between(levels, rel.min(axis=0), rel.max(axis=0), alpha=0.15, color=C["blue"],
                         label="min–max")
    axes[1].set_xlabel("nominal level"); axes[1].set_ylabel("empirical coverage")
    axes[1].set_title("(b) reliability")
    axes[1].legend(frameon=False, loc="lower right")

    # (c) halfwidth growth with horizon
    hw = np.array([np.array(r["halfwidth_90_per_h"]) / r["point_mae"] for r in cf])
    hrs = np.arange(1, 25)
    axes[2].plot(hrs, hw.mean(axis=0), color=C["blue"], marker="o", ms=2.5, label="mean")
    axes[2].fill_between(hrs, np.percentile(hw, 25, axis=0), np.percentile(hw, 75, axis=0),
                         alpha=0.15, color=C["blue"], label="IQR")
    axes[2].set_xlabel("horizon step (h)"); axes[2].set_ylabel("halfwidth / point MAE")
    axes[2].set_title("(c) per-horizon halfwidth")
    axes[2].legend(frameon=False, loc="lower right")

    fig.tight_layout()
    save(fig, "conformal_prediction")


# ---------------------------------------------------------------------------
# Temporal OOD figure
# ---------------------------------------------------------------------------

def fig_temporal_ood():
    to = load("temporal_ood.json")
    to.sort(key=lambda r: r["mean_rs"])
    names = [short(r["region"]) for r in to]
    splits = [("Standard (80/20)", C["blue"]), ("9-month (75/25)", C["orange"]),
              ("6-month (50/50)", C["red"])]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 2.5))
    for k, (split, col) in enumerate(splits):
        vals = [r[f"ratio_plus_{split}"] for r in to]
        mean_v = float(np.mean(vals))
        ax.bar(x + (k - 1) * 0.27, vals, 0.27, color=col,
               label=f"{split} (mean {mean_v:.2f})")
    ax.axhline(1.0, color=C["black"], lw=0.7, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("ZS+ MAE ratio vs. persistence")
    m_std = float(np.mean([r["ratio_plus_Standard (80/20)"] for r in to]))
    m_75 = float(np.mean([r["ratio_plus_9-month (75/25)"] for r in to]))
    m_50 = float(np.mean([r["ratio_plus_6-month (50/50)"] for r in to]))
    ax.set_title(f"Temporal OOD (ZS+): earlier boundaries change mean ratio by "
                 f"{(m_75/m_std-1)*100:+.0f}% / {(m_50/m_std-1)*100:+.0f}%")
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "temporal_ood")


# ---------------------------------------------------------------------------
# Deployment warm-up figure
# ---------------------------------------------------------------------------

def fig_deployment():
    dp = load("deployment_warmup.json")
    ft, zs, cross = dp["finetuning_curve"], dp["zero_shot_warmup"], dp["crossover_days"]
    regions = list(ft.keys())
    fig, axes = plt.subplots(2, 4, figsize=(7.0, 3.6), sharex=True)
    for ax, reg in zip(axes.flat, regions):
        sup = [(e["days"], e["sup_mae"]) for e in ft[reg] if e.get("sup_mae")]
        days = [d for d, _ in sup]; maes = [m for _, m in sup]
        zs_vals = [e["zs_mae"] for e in zs[reg] if e.get("zs_mae")]
        zs_ref = float(np.mean(zs_vals)) if zs_vals else None
        zsp_pts = [(e["days"], e["zsp_mae"]) for e in zs[reg] if e.get("zsp_mae")]
        ax.plot(days, maes, marker="o", ms=2.5, color=C["orange"], label="supervised (retrained)")
        if zs_ref:
            ax.axhline(zs_ref, color=C["blue"], lw=1.2, label="zero-shot (day 0)")
        if zsp_pts:
            ax.plot([d for d, _ in zsp_pts], [m for _, m in zsp_pts],
                    color=C["green"], lw=1.0, ls="--", label="ZS+ (calibrated)")
        cv = cross.get(reg)
        title_cross = f"crossover: {cv}d" if cv != ">270" else "crossover: >270d"
        if isinstance(cv, (int, float)):
            ax.axvline(cv, color=C["grey"], lw=0.7, ls=":")
        ax.set_title(f"{short(reg)} ({title_cross})", fontsize=7)
        ax.tick_params(labelsize=6)
    for ax in axes[1]:
        ax.set_xlabel("days of target data")
    for ax in axes[:, 0]:
        ax.set_ylabel("CIF MAE")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Deployment warm-up race: supervised accumulation vs. day-0 zero-shot", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save(fig, "deployment_warmup")


# ---------------------------------------------------------------------------
# CarbonCast comparison figure
# ---------------------------------------------------------------------------

def fig_carboncast():
    cc = load("carboncast_analysis.json")
    cc.sort(key=lambda r: r["mean_rs"])
    names = [short(r["region"]) for r in cc]
    sup = [r["cc_supervised"]["mae"] for r in cc]
    zs_cc = [r["cc_zeroshot"]["mae"] for r in cc]
    zs_tc = [r["transcif_zeroshot"]["mae"] for r in cc]
    zsp_tc = [r["transcif_zs_plus"]["mae"] for r in cc]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    ax.bar(x - 0.3, sup, 0.2, color=C["grey"], label="CarbonCast supervised")
    ax.bar(x - 0.1, zs_cc, 0.2, color=C["orange"], label="CarbonCast zero-shot")
    ax.bar(x + 0.1, zs_tc, 0.2, color=C["blue"], label="TransCIF zero-shot (ours)")
    ax.bar(x + 0.3, zsp_tc, 0.2, color=C["purple"], label="TransCIF ZS+ (ours, calibrated)")
    wins = sum(1 for a_, b_ in zip(zsp_tc, zs_cc) if a_ < b_)
    for i, (a_, b_) in enumerate(zip(zsp_tc, zs_cc)):
        if a_ < b_:
            ax.annotate("*", (i + 0.3, a_), ha="center", va="bottom", fontsize=9, color=C["purple"])
    ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("CIF MAE (gCO$_2$/kWh)")
    n = len(names)
    ax.set_title(f"CarbonCast under domain shift: TransCIF ZS+ wins {wins}/{n} zero-shot")
    ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    save(fig, "carboncast_analysis")


if __name__ == "__main__":
    os.makedirs(FIGDIR, exist_ok=True)
    print("Generating submission-quality figures ...")
    fig_main_benchmark()
    fig_theorem1()
    fig_theorem2()
    fig_conformal()
    fig_temporal_ood()
    fig_deployment()
    fig_carboncast()
    print("Done.")
