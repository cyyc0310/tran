#!/usr/bin/env python
"""29-region day-ahead curves: ALL arms in ONE figure (seed 0).

Per panel (one region), a representative 96 h stretch from mid-test
(4 consecutive day-ahead origins, stride 24 h):
    actual     black solid   ground truth
    I_cfg      blue          zero telemetry (FD-41 official, ERA5-proxy
                             arm of the FD-47 cache)
    I_0        green         + live fuel-share telemetry
    I_lag      purple        best3f zero-train lag tier (EWMA-corrected
                             I_cfg + persistence + Chronos-2, convex
                             weights fitted on the early half only)
    PatchTST   grey dashed   supervised, 10-month local labels (gap4
                             protocol)

Panel title (2 lines): region name, then seed-0 MAE
I_cfg / I_0 / lag(late half) / PatchTST(full test).  Panels sorted by
I_cfg MAE.  Tier colors match fd_ladder.png / fd_region_curves.png /
fd_lag_sup_curves.png.

Every series is rebuilt from archives and trace-asserted:
  I_cfg / I_0  vs FD-47 cache (pooled seed-0 I_cfg 43.98, paper Tab. 1)
  I_lag        vs results/fd50p3_seeds.json best3f_late (late-half
               pooled 27.95, paper head 27.97 +- 0.10)
  PatchTST     vs results/gap4_supervised_vs_fd41.json (pooled 43.35)
Usage:
    .venv-nemed/bin/python scripts/figures/make_fd_all_curves.py
Outputs:
    figures/fd_all_curves.png (+pdf)
"""

from pathlib import Path
import sys
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "experiments"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))
from make_fd_lag_sup_curves import lag_series  # official best3f rebuild

NWP = ROOT / "results" / "fd47_nwp"
SUP = ROOT / "results" / "gap4_sup_cache"
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 9.5,
    "axes.titlesize": 8,
    "legend.fontsize": 10.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
})

C = {"black": "#1a1a1a", "grey": "#8a8a8a", "blue": "#2471a3",
     "green": "#1e8449", "purple": "#7d3c98"}
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


def main():
    p3 = json.load(open(ROOT / "results" / "fd50p3_seeds.json"))["0"]["rows"]
    gap4 = {r["target"]: r for r in json.load(
        open(ROOT / "results" / "gap4_supervised_vs_fd41.json"))["rows"]}

    files = sorted(NWP.glob("*_seed0.npz"))
    assert len(files) == 29, f"expected 29 region caches, got {len(files)}"

    regions = []
    for f in files:
        t = f.name.replace("_seed0.npz", "")
        z = np.load(f)
        y = z["y_true"].astype(float)
        p_cfg = z["pred_era5_icfg"].astype(float)
        p_i0 = z["pred_era5_i0"].astype(float)
        m_cfg = float(np.abs(p_cfg - y).mean())
        m_i0 = float(np.abs(p_i0 - y).mean())

        o, yl, lag, m_lag = lag_series(t)
        ref_lag = p3[t]["best3f_late"]
        assert abs(m_lag - ref_lag) < 0.02, \
            f"{t}: lag late {m_lag:.3f} != official {ref_lag:.3f}"

        zs = np.load(SUP / f"{t}_s0.npz")
        ys, sup = zs["y_true"].astype(float), zs["pred"].astype(float)
        m_sup = float(np.abs(sup - ys).mean())
        ref_sup = gap4[t]["patchtst_mae_s0"]
        assert abs(m_sup - ref_sup) < 0.5, \
            f"{t}: sup mae {m_sup:.2f} != gap4 {ref_sup:.2f}"

        # all three caches share the same day-ahead test windows
        n = min(len(y), len(yl), len(ys))
        assert np.allclose(y[:n], yl[:n], atol=1e-3), f"{t}: y misalign (lag)"
        assert np.allclose(y[:n], ys[:n], atol=1e-3), f"{t}: y misalign (sup)"

        regions.append(dict(t=t, y=y, cfg=p_cfg, i0=p_i0, lag=lag, sup=sup,
                            m_cfg=m_cfg, m_i0=m_i0, m_lag=m_lag, m_sup=m_sup,
                            n=n))

    regions.sort(key=lambda r: r["m_cfg"])

    n_rows, n_cols = 6, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12.5), sharex=True)
    hours = np.arange(N_ORIGINS * 24)

    for ax, r in zip(axes.flat, regions):
        o0 = r["n"] // 2
        sl = slice(o0, o0 + N_ORIGINS)
        ax.plot(hours, r["y"][sl].ravel(), color=C["black"], lw=1.6,
                label="Actual")
        ax.plot(hours, r["cfg"][sl].ravel(), color=C["blue"], lw=1.05,
                label="I$_{cfg}$ zero-telemetry")
        ax.plot(hours, r["i0"][sl].ravel(), color=C["green"], lw=1.15,
                label="I$_0$ fuel-telemetry")
        ax.plot(hours, r["lag"][sl].ravel(), color=C["purple"], lw=1.15,
                label="I$_{lag}$ best3f (zero-train)")
        ax.plot(hours, r["sup"][sl].ravel(), color=C["grey"], lw=1.0,
                ls="--", label="PatchTST-sup (10-month)")
        for hb in range(24, N_ORIGINS * 24, 24):
            ax.axvline(hb, color=C["grey"], lw=0.5, alpha=0.45)
        ax.set_title(
            f"{short_label(r['t'])}\n"
            f"{r['m_cfg']:.0f} / {r['m_i0']:.0f} / {r['m_lag']:.0f} / "
            f"{r['m_sup']:.0f}", fontsize=8, pad=2.5)
        ax.tick_params(labelsize=7.5)

    for ax in axes.flat[len(regions):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5)
    fig.suptitle(
        "Day-ahead CIF forecasts, 29 regions — all tiers and the "
        "supervised baseline in one figure (LORO, seed 0)\n"
        "black: actual · blue: I$_{cfg}$ zero-telemetry · green: I$_0$ "
        "fuel-telemetry · purple: I$_{lag}$ best3f (EWMA+persist+Chronos-2, "
        "delayed labels, no training) · grey dashed: PatchTST supervised "
        "(10-month local labels)\npanel title: seed-0 MAE — "
        "I$_{cfg}$ / I$_0$ / I$_{lag}$ (late half, online warm-up) / "
        "PatchTST (full test); panels sorted by I$_{cfg}$",
        fontsize=12)
    fig.supxlabel("hours from first shown origin")
    fig.supylabel("CIF (gCO$_2$/kWh)")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fd_all_curves.{ext}", facecolor="white")
    plt.close(fig)

    pooled_cfg = float(np.mean([r["m_cfg"] for r in regions]))
    med_i0 = float(np.median([r["m_i0"] for r in regions]))
    pooled_lag = float(np.mean([r["m_lag"] for r in regions]))
    pooled_sup = float(np.mean([r["m_sup"] for r in regions]))
    print("saved figures/fd_all_curves.png (+pdf); 29 panels, 5 series each")
    print(f"trace check -> I_cfg pooled {pooled_cfg:.2f} (paper 43.98), "
          f"I_0 median {med_i0:.2f} (paper 36.9), "
          f"lag late pooled {pooled_lag:.2f} (paper 27.97 head), "
          f"PatchTST pooled {pooled_sup:.2f} (paper Tab.1 43.5)")


if __name__ == "__main__":
    main()
