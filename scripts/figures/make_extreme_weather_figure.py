#!/usr/bin/env python
"""Extreme-weather event figure: CIF + wind share vs gust/precip/pressure.

Eight verified 2023 events (see analyze_extreme_weather.py).  Each panel
is a 7-day window: top = CIF (black, left) + wind share (blue, right);
bottom = gust (orange) + hourly precipitation (teal bars) + MSL pressure
(purple, right).  Shows the two transmission channels: renewable-share
swings (storms/calm) and demand-driven thermal response (freeze/heat).

Usage:
    .venv/bin/python scripts/figures/make_extreme_weather_figure.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "analysis"))

from analyze_extreme_weather import build_region_table, hourly_indices
from transcif.data.loaders import all_region_configs

FIG = Path(__file__).resolve().parent.parent.parent / "figures"

# (region, event label, center date, half-window days)
EVENTS = [
    ("UK_01_North_Scotland", "UK_01 N Scotland — Sept heatwave (calm: wind share collapse, 27× CIF vol)", "2023-09-07", 3),
    ("SA1", "SA1 — September wind drought (wind-lull, 1.6× CIF vol)", "2023-09-12", 4),
    ("UK_03_North_West_England", "UK_03 NW England — Storm Debi (gust z+4.5, rain)", "2023-11-14", 2),
    ("UK_06_North_Wales_Merseyside", "UK_06 N Wales — Storm Gerrit (gust+rain+freeze)", "2023-12-28", 2),
    ("US_FPL", "US_FPL — Hurricane Idalia (gust 5σ, CIF unmoved: wind+solar ≈ 5%)", "2023-08-30", 2),
    ("US_ERCO", "US_ERCO — Winter Storm Mara (freeze, 1.35× CIF vol)", "2023-02-01", 2),
    ("US_PJM", "US_PJM — July heat dome (1.52× CIF vol)", "2023-07-30", 2),
    ("QLD1", "QLD1 — TC Jasper (cyclone + 2.2 m rain)", "2023-12-15", 3),
]


def main():
    cfgs = all_region_configs()
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = hourly_indices(build_region_table(name, cfgs))
        return cache[name]

    fig, axes = plt.subplots(len(EVENTS), 2, figsize=(14, 2.3 * len(EVENTS)),
                             gridspec_kw={"width_ratios": [1, 1], "hspace": 0.55,
                                          "wspace": 0.30})
    for row, (name, label, center, half) in enumerate(EVENTS):
        h = get(name)
        t0 = pd.Timestamp(center) - pd.Timedelta(days=half)
        t1 = pd.Timestamp(center) + pd.Timedelta(days=half + 1)
        w = h[(h.index >= t0) & (h.index < t1)]
        tt = w.index

        axc, axw = axes[row, 0], axes[row, 1]
        # left: CIF + wind share
        axc.plot(tt, w["cif"], color="k", lw=1.2, label="CIF")
        if w["wind_share"].notna().any():
            ax2 = axc.twinx()
            ax2.plot(tt, w["wind_share"], color="tab:blue", lw=1.0, alpha=0.8,
                     label="wind share")
            ax2.set_ylim(-0.02, 1.0)
            ax2.set_ylabel("wind share", color="tab:blue", fontsize=8)
            ax2.tick_params(axis="y", labelsize=7, colors="tab:blue")
        axc.set_title(label, fontsize=8.5, loc="left")
        axc.set_ylabel("CIF gCO₂/kWh", fontsize=8)
        axc.tick_params(axis="both", labelsize=7)

        # right: gust + precip + pressure
        axw.plot(tt, w["gust"], color="tab:orange", lw=1.0, label="gust")
        pr = w["precip"].fillna(0.0)
        axw.bar(tt, pr, width=1 / 26, color="tab:cyan", alpha=0.7,
                label="precip")
        ax3 = axw.twinx()
        ax3.plot(tt, w["pres"], color="tab:purple", lw=0.9, alpha=0.7,
                 label="MSL pressure")
        ax3.set_ylabel("hPa", color="tab:purple", fontsize=8)
        ax3.tick_params(axis="y", labelsize=7, colors="tab:purple")
        ax3.set_ylim(975, 1035)
        axw.set_ylabel("m/s | mm/h", fontsize=8)
        axw.tick_params(axis="both", labelsize=7)

    fig.suptitle("Extreme weather → CIF: transmission is via weather-dependent "
                 "generation share (2023, ERA5 + telemetry)", fontsize=11)
    fig.savefig(FIG / "extreme_weather_events.png", dpi=170,
                bbox_inches="tight")
    fig.savefig(FIG / "extreme_weather_events.pdf", bbox_inches="tight")
    print(f"[extreme-fig] wrote {FIG / 'extreme_weather_events.png'}")


if __name__ == "__main__":
    main()
