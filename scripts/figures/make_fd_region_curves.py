#!/usr/bin/env python
"""29-region day-ahead CIF prediction curves (FD-41 official, seed 0).

Reads the archived per-region prediction cache from the FD-47 NWP study
(results/fd47_nwp/<region>_seed0.npz).  Its ERA5-proxy arm is exactly the
FD-41 official pipeline: pooled I_cfg MAE 43.98 / median 38.4 (paper
Table 1 and section 8.5).  No model training happens here -- every curve
is traceable to the archived npz.

Per panel (one region): a representative 96 h stretch from mid-test
(4 consecutive day-ahead origins, stride 24 h):
    actual    black   ground truth
    I_cfg     blue    config + weather + calendar (zero telemetry)
    I_0       green   + live fuel-share stream
Panel title shows full-test seed-0 MAE (I_cfg / I_0); panels sorted by
I_cfg.  Tier colors match fd_ladder.png (blue = I_cfg, green = I_0).

Usage:
    .venv-nemed/bin/python scripts/figures/make_fd_region_curves.py
Outputs:
    figures/fd_region_curves.png (300 dpi) and .pdf
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
CACHE = ROOT / "results" / "fd47_nwp"
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

C = {"black": "#1a1a1a", "grey": "#8a8a8a", "blue": "#2471a3",
     "orange": "#e67e22", "green": "#1e8449"}

N_ORIGINS = 4   # 4 consecutive day-ahead origins -> continuous 96 h


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
    files = sorted(CACHE.glob("*_seed0.npz"))
    assert len(files) == 29, f"expected 29 region caches, got {len(files)}"

    regions = []
    for f in files:
        z = np.load(f)
        yt = z["y_true"].astype(float)
        p_cfg = z["pred_era5_icfg"].astype(float)
        p_i0 = z["pred_era5_i0"].astype(float)
        m_cfg = float(np.abs(p_cfg - yt).mean())
        m_i0 = float(np.abs(p_i0 - yt).mean())
        regions.append((f, yt, p_cfg, p_i0, m_cfg, m_i0))

    regions.sort(key=lambda r: r[4])   # sort by full-test I_cfg MAE

    n_rows, n_cols = 6, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12), sharex=True)
    hours = np.arange(N_ORIGINS * 24)

    for ax, (f, yt, p_cfg, p_i0, m_cfg, m_i0) in zip(axes.flat, regions):
        name = f.name.replace("_seed0.npz", "")
        n = yt.shape[0]
        o0 = n // 2
        sl = slice(o0, o0 + N_ORIGINS)
        ax.plot(hours, yt[sl].ravel(), color=C["black"], lw=1.5, label="Actual")
        ax.plot(hours, p_cfg[sl].ravel(), color=C["blue"], lw=1.1,
                label="I$_{cfg}$ (zero telemetry)")
        ax.plot(hours, p_i0[sl].ravel(), color=C["green"], lw=1.2,
                label="I$_0$ (+fuel telemetry)")
        for h in range(24, N_ORIGINS * 24, 24):
            ax.axvline(h, color=C["grey"], lw=0.5, alpha=0.45)
        ax.set_title(f"{short_label(name)} — {m_cfg:.0f} / {m_i0:.0f}",
                     fontsize=8.5)
        ax.tick_params(labelsize=7.5)

    for ax in axes.flat[len(regions):]:
        ax.axis("off")

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(
        "Day-ahead CIF forecasts, 29 regions (LORO, seed 0, ERA5-proxy arm "
        "of the FD-47 cache = FD-41 official)\n"
        "black: actual · blue: I$_{cfg}$ zero-telemetry · green: I$_0$ "
        "fuel-telemetry · panel title: full-test MAE (I$_{cfg}$ / I$_0$), "
        "panels sorted by I$_{cfg}$ · light bars mark day-ahead origins",
        fontsize=12)
    fig.supxlabel("hours from first shown origin")
    fig.supylabel("CIF (gCO$_2$/kWh)")
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"fd_region_curves.{ext}", facecolor="white")
    plt.close(fig)

    med_cfg = float(np.median([r[4] for r in regions]))
    med_i0 = float(np.median([r[5] for r in regions]))
    pooled_cfg = float(np.mean([r[4] for r in regions]))
    print("saved figures/fd_region_curves.png (+pdf); 29 panels")
    print(f"trace check -> seed-0 pooled I_cfg {pooled_cfg:.2f} (paper 43.98), "
          f"median I_cfg {med_cfg:.2f} (paper 38.4), median I_0 {med_i0:.2f} "
          f"(paper 36.9)")


if __name__ == "__main__":
    main()
