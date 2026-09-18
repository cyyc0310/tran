#!/usr/bin/env python
"""29-region curves: zero-train lag tier (best3f) vs supervised PatchTST.

Per panel (one region): a representative 96 h stretch from mid-test
(4 consecutive day-ahead origins, stride 24 h):
    actual       black solid   ground truth
    I_lag        purple        best3f = EWMA-corrected I_cfg + persistence
                               + Chronos-2 (convex, early-fitted, zero training)
    PatchTST-sup grey dashed   10-month local supervised training (gap4
                               protocol, seed 0)

I_lag series is rebuilt from the official FD-50p archives with the exact
fd50p3_seeds recipe (recipe_R + EWMA + early-fitted 3-way weights); the
per-region late-half MAE is asserted against results/fd50p3_seeds.json
seed 0.  Supervised series come from results/gap4_sup_cache (seed 0, MAE
asserted against gap4_supervised_vs_fd41.json).  No model training here.

Usage:
    .venv-nemed/bin/python scripts/figures/make_fd_lag_sup_curves.py
Outputs:
    figures/fd_lag_sup_curves.png (+pdf)
"""

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
from fd50p3_seeds import (  # official recipe, verbatim
    CACHE as LAG_CACHE, DUMP, load_region, recipe_R, ewma_hod, mae, BETA,
)
import json

SUP_CACHE = ROOT / "results" / "gap4_sup_cache"
NWP_CACHE = ROOT / "results" / "fd47_nwp"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 8.5,
    "legend.fontsize": 11,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

C = {"black": "#1a1a1a", "grey": "#8a8a8a", "purple": "#7d3c98"}
N_ORIGINS = 4


def short_label(name):
    s = name
    for pre in ("UK_", "US_"):
        if s.startswith(pre):
            s = s[len(pre):]
    s = s.replace("_", " ")
    for a, b in (("Greater London", "London"), ("North ", "N "),
                 ("South ", "S "), ("East ", "E "), ("West ", "W ")):
        s = s.replace(a, b)
    return s


def lag_series(target):
    """best3f series for seed 0, official fd50p3 recipe."""
    o, y, p, per, ev = load_region(target, 0)
    nn, h = y.shape
    e = slice(0, nn // 2)
    l = slice(nn // 2, nn)
    r = recipe_R(y, p, per, ev)
    g = ewma_hod(r, y, BETA)

    c = np.load(LAG_CACHE / f"{target}_c.npz")
    valid = np.asarray(c["valid"], dtype=bool)
    med = c["med2048"].astype(float)
    med[~valid] = per[~valid]

    best_w, best_m = (1.0, 0.0, 0.0), mae(g[e], y[e])
    for wg in np.arange(0.0, 1.001, 0.05):
        for wp in np.arange(0.0, 1.001 - wg + 1e-9, 0.05):
            wc = round(1.0 - wg - wp, 2)
            if wc < -1e-9:
                continue
            m = mae(wg * g[e] + wp * per[e] + wc * med[e], y[e])
            if m < best_m:
                best_w, best_m = (round(float(wg), 2), round(float(wp), 2), wc), m
    wg, wp, wc = best_w
    best3f = wg * g + wp * per + wc * med
    return o, y, best3f, mae(best3f[l], y[l])


def main():
    p3 = json.load(open(ROOT / "results" / "fd50p3_seeds.json"))["0"]["rows"]
    gap4 = {r["target"]: r for r in json.load(
        open(ROOT / "results" / "gap4_supervised_vs_fd41.json"))["rows"]}

    targets = sorted(p3)
    assert len(targets) == 29, f"expected 29 regions, got {len(targets)}"

    regions = []
    for t in targets:
        o, y, lag, lag_late = lag_series(t)
        # trace check vs official fd50p3 seed-0 rows
        ref = p3[t]["best3f_late"]
        assert abs(lag_late - ref) < 0.02, \
            f"{t}: lag late {lag_late:.3f} != official {ref:.3f}"

        zs = np.load(SUP_CACHE / f"{t}_s0.npz")
        sup_y, sup = zs["y_true"].astype(float), zs["pred"].astype(float)
        sup_m = float(np.abs(sup - sup_y).mean())
        ref_s = gap4[t]["patchtst_mae_s0"]
        assert abs(sup_m - ref_s) < 0.5, \
            f"{t}: sup mae {sup_m:.2f} != gap4 {ref_s:.2f}"
        # supervised windows must be the same test windows as the lag cache
        n = min(len(y), len(sup_y))
        assert np.allclose(y[:n], sup_y[:n], atol=1e-3), \
            f"{t}: y_true misaligned between lag and supervised caches"

        regions.append((t, y, lag, sup, lag_late, sup_m))

    regions.sort(key=lambda r: r[4])   # sort by late-half lag MAE

    n_rows, n_cols = 6, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True)
    hours = np.arange(N_ORIGINS * 24)

    for ax, (t, y, lag, sup, lag_late, sup_m) in zip(axes.flat, regions):
        n = y.shape[0]
        o0 = n // 2
        sl = slice(o0, o0 + N_ORIGINS)
        ax.plot(hours, y[sl].ravel(), color=C["black"], lw=1.5, label="Actual")
        ax.plot(hours, lag[sl].ravel(), color=C["purple"], lw=1.3,
                label="I$_{lag}$ best3f (zero-train)")
        ax.plot(hours, sup[sl].ravel(), color=C["grey"], lw=1.1, ls="--",
                label="PatchTST-sup (10-month)")
        for hb in range(24, N_ORIGINS * 24, 24):
            ax.axvline(hb, color=C["grey"], lw=0.5, alpha=0.45)
        ax.set_title(f"{short_label(t)} — {sup_m:.0f} / {lag_late:.0f}",
                     fontsize=8.5)
        ax.tick_params(labelsize=7.5)

    for ax in axes.flat[len(regions):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(
        "Day-ahead CIF forecasts, 29 regions — zero-train lag tier vs "
        "supervised training (seed 0)\n"
        "black: actual · purple: I$_{lag}$ best3f (EWMA+persist+Chronos, "
        "delayed labels, no training) · grey dashed: PatchTST supervised "
        "(10-month local labels)\npanel title: full-test PatchTST MAE / "
        "late-half lag MAE (online warm-up consumes the early half); "
        "panels sorted by lag MAE",
        fontsize=12)
    fig.supxlabel("hours from first shown origin")
    fig.supylabel("CIF (gCO$_2$/kWh)")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fd_lag_sup_curves.{ext}", facecolor="white")
    plt.close(fig)

    pooled_lag = float(np.mean([r[4] for r in regions]))
    pooled_sup = float(np.mean([r[5] for r in regions]))
    print("saved figures/fd_lag_sup_curves.png (+pdf); 29 panels")
    print(f"trace check -> lag late-half pooled (seed 0) {pooled_lag:.2f} "
          f"(paper 27.97 head), PatchTST full-test pooled {pooled_sup:.2f} "
          f"(paper Table 1 43.5)")


if __name__ == "__main__":
    main()
