"""Plot the calibration-data-amount curve from probe_calibration_curve.json.

Produces a figure showing median MAE vs calibration hours, with the three
information-set regimes annotated (ZS+ / Joint-minimal / Joint-rich).

Usage:
    PYTHONPATH=src python scripts/figures/plot_calibration_curve.py
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from transcif.config import RESULTS_DIR, DATA_DIR

FIG_DIR = Path(__file__).resolve().parent.parent.parent / "figures"
FIG_DIR.mkdir(exist_ok=True)


def main():
    path = RESULTS_DIR / "probe_calibration_curve.json"
    if not path.exists():
        print(f"Missing {path}; run probe_calibration_curve.py first.")
        return
    doc = json.loads(path.read_text())
    hours = doc["calibration_hours"]  # [0, 72, 144, 288, 576]
    results = doc["results"]          # {n_train: {target: mae}}

    # Aggregate: median + per-region spread.
    medians, p25s, p75s = [], [], []
    for n_train in doc["n_train_sweep"]:
        vals = list(results[str(n_train)].values())
        medians.append(np.nanmedian(vals))
        p25s.append(np.nanpercentile(vals, 25))
        p75s.append(np.nanpercentile(vals, 75))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(hours, medians, "o-", color="#2563eb", lw=2, markersize=8, zorder=5)
    ax.fill_between(hours, p25s, p75s, alpha=0.15, color="#2563eb", label="25–75 %ile")

    # Annotate regimes.
    ax.axvline(x=0, color="gray", ls="--", alpha=0.4)
    ax.annotate("ZS+\n(0 h labels,\npure unsupervised)",
                xy=(0, medians[0]), xytext=(40, medians[0] + 8),
                fontsize=8, ha="left",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
    ax.annotate("Joint minimal\n(72–144 h)",
                xy=(120, medians[2]), xytext=(120, medians[2] + 10),
                fontsize=8, ha="center",
                arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))

    ax.set_xlabel("Target-domain calibration labels (hours)")
    ax.set_ylabel("Median MAE (gCO$_2$/kWh)")
    ax.set_title("Calibration-data-amount curve: information-set stratification")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)

    out = FIG_DIR / "calibration_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    fig.savefig(out.with_suffix(".pdf"))
    print(f"[WRITE] {out}")
    print(f"\nCurve data:")
    for h, m, lo, hi in zip(hours, medians, p25s, p75s):
        print(f"  {h:4d}h: median={m:.2f}  (IQR {lo:.1f}–{hi:.1f})")


if __name__ == "__main__":
    main()
