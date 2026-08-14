"""Figures for Section 6.10 (joint calibration), using stable existing results.

  figures/joint_mae_progression.png   45.96 -> 46.89 -> 40.53 -> 39.53 (+ PatchTST)
  figures/joint_per_region_scatter.png frozen-proxy vs native, 29 regions, 45 deg
  figures/joint_dropone_ablation.png   meta-learner dead-weight at zero-shot
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")
plt.rcParams.update({"font.size": 10, "figure.dpi": 150})

HERE = Path(__file__).resolve().parent.parent.parent
RESULTS = HERE / "results"
FIGS = HERE / "figures"
TIER_COLOR = {"easy": "#009E73", "medium": "#E69F00", "hard": "#D55E00"}
C_FROZEN = "#999999"
C_NATIVE = "#0072B2"
C_TARGET = "#D55E00"


def fig_progression():
    stages = [
        ("best single\n(causal+ZS+)", 45.96, C_FROZEN),
        ("BasisMix+ZS+\n(5-dir fusion)", 46.89, C_FROZEN),
        ("Joint\nfrozen-proxy", 40.53, "#56B4E9"),
        ("Joint\ntorch-native", 39.53, C_NATIVE),
    ]
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    x = np.arange(len(stages))
    bars = ax.bar(x, [s[1] for s in stages], 0.6,
                  color=[s[2] for s in stages], alpha=0.92, edgecolor="k", lw=0.5)
    ax.axhline(41.47, color=C_TARGET, ls="--", lw=1.3,
               label="supervised PatchTST (41.47)")
    ax.axhline(39.53, color=C_NATIVE, ls=":", lw=1.0, alpha=0.5)
    for bar, s in zip(bars, stages):
        ax.annotate(f"{s[1]:.2f}", (bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
    # arrow showing the differentiable lift
    ax.annotate("", xy=(3, 39.53), xytext=(2, 40.53),
                arrowprops=dict(arrowstyle="->", color=C_NATIVE, lw=1.5))
    ax.text(2.5, 39.0, "torch-native\nend-to-end\n(+1.62, p=5e-14)",
            fontsize=7.5, color=C_NATIVE, ha="center")
    ax.set_xticks(x); ax.set_xticklabels([s[0] for s in stages], fontsize=8.5)
    ax.set_ylabel("Median MAE (gCO$_2$/kWh, 29 regions × 5 seeds)")
    ax.set_title("From zero-shot fusion to sub-40 joint calibration")
    ax.legend(fontsize=8, loc="upper right", framealpha=0.9)
    ax.set_ylim(35, 50)
    fig.tight_layout()
    out = FIGS / "joint_mae_progression.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out}")


def _tier_of(persist_mae):
    if persist_mae < 25:
        return "easy"
    if persist_mae < 50:
        return "medium"
    return "hard"


def fig_per_region_scatter():
    frozen = json.loads((RESULTS / "joint_train_full.json").read_text())
    native = json.loads((RESULTS / "joint_train_native_full.json").read_text())
    floor = json.loads((RESULTS / "mae_floor_analysis.json").read_text())
    tier_by_region = {r["region"]: _tier_of(r["persistence_floor_cif_mae"])
                      for r in floor["regions"]}

    def per_region_median(rows):
        by = defaultdict(list)
        for r in rows:
            if "held_out_mae" in r and r["held_out_mae"] is not None:
                by[r["target"]].append(r["held_out_mae"])
        return {k: float(np.median(v)) for k, v in by.items()}

    fm = per_region_median(frozen)
    nm = per_region_median(native)
    regions = sorted(set(fm) & set(nm))

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    for reg in regions:
        t = tier_by_region.get(reg, "medium")
        ax.scatter(fm[reg], nm[reg], c=TIER_COLOR[t], s=55, alpha=0.85,
                   edgecolor="k", linewidth=0.4, zorder=3)
    lim = max(max(fm.values()), max(nm.values())) * 1.05
    ax.plot([0, lim], [0, lim], "--", c="0.6", lw=1, label="frozen = native", zorder=1)
    # label biggest improvements + regressions
    deltas = {r: fm[r] - nm[r] for r in regions}
    for reg, d in sorted(deltas.items(), key=lambda kv: -kv[1])[:3]:
        ax.annotate(reg.replace("_", "\n", 1)[:14], (fm[reg], nm[reg]),
                    fontsize=6.5, xytext=(4, 4), textcoords="offset points",
                    color=TIER_COLOR["hard"])
    for reg, d in sorted(deltas.items(), key=lambda kv: kv[1])[:2]:
        ax.annotate(reg[:10], (fm[reg], nm[reg]),
                    fontsize=6.5, xytext=(4, -10), textcoords="offset points",
                    color=TIER_COLOR["easy"])
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c,
                      markersize=8, label=f"{t} tier") for t, c in TIER_COLOR.items()]
    handles += [Line2D([0], [0], color="0.6", ls="--", label="frozen = native")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", framealpha=0.9)
    ax.set_xlabel("Frozen-proxy joint MAE (gCO$_2$/kWh)")
    ax.set_ylabel("Torch-native joint MAE (gCO$_2$/kWh)")
    ax.set_title("Per-region improvement: torch-native wins concentrate on hard grids")
    ax.set_xlim(left=0); ax.set_ylim(bottom=0)
    fig.tight_layout()
    out = FIGS / "joint_per_region_scatter.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out}")


def fig_dropone():
    path = RESULTS / "fused_five_dropone.json"
    if not path.exists():
        print("[SKIP] drop-one (no data)")
        return
    d = json.loads(path.read_text())
    drop_dirs = ["rag", "phys", "causal", "icl", "hier"]
    by_drop = defaultdict(list)
    rows = d if isinstance(d, list) else d.get("results", d.get("rows", []))
    for r in rows:
        drops = r.get("drops", r)
        for dd in drop_dirs:
            cell = drops.get(dd, {})
            mae = (cell.get("fused_plus") or cell.get("fused") or {}).get("mae")
            if mae is not None:
                by_drop[dd].append(float(mae))
    if not by_drop:
        print("[SKIP] drop-one (no drop data found)")
        return
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    labels = [dd for dd in drop_dirs if dd in by_drop]
    vals = [by_drop[dd] for dd in labels]
    medians = [np.median(v) for v in vals]
    x = np.arange(len(labels))
    bp = ax.boxplot(vals, positions=x, widths=0.5, patch_artist=True,
                    boxprops=dict(facecolor="#56B4E9", alpha=0.6),
                    medianprops=dict(color=C_NATIVE, lw=2))
    ax.scatter(np.repeat(x, [len(v) for v in vals]),
               np.concatenate(vals), s=14, alpha=0.4, color="0.3", zorder=2)
    full = np.median(np.concatenate(vals)) if vals else 0
    ax.axhline(full, color=C_TARGET, ls="--", lw=1, label=f"overall median ({full:.1f})")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("BasisMix+ MAE when direction dropped (gCO$_2$/kWh)")
    ax.set_title("Drop-one ablation: removing any direction barely moves MAE\n"
                 "(meta-learner is dead weight at zero-shot)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = FIGS / "joint_dropone_ablation.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[WRITE] {out}")


def main():
    fig_progression()
    fig_per_region_scatter()
    fig_dropone()


if __name__ == "__main__":
    main()
