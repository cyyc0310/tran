#!/usr/bin/env python
"""Per-region fuel ↔ weather/load coupling structure.

For each region with fuel telemetry, quantify how each fuel's hourly
share couples to the physical drivers:

    wcf    fleet wind capacity factor      -> wind share should track
    csi    clear-sky index (daytime)       -> solar share should track
    hour   local hour (sin/cos)            -> dispatchable load-following
    temp   temperature (HDH/CDH proxy)     -> demand-driven dispatch
    resid  1 - (wind+solar explained)      -> what the thermal residual
                                             actually rides on

Output: a coupling table per region + residual attribution for the
hard-tail regions (what explains CIF variance beyond weather).

Usage:
    .venv/bin/python scripts/analysis/analyze_fuel_weather_coupling.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from transcif.config.region_meta import REGION_META
from transcif.data.fuel import FUEL_INDEX, CANONICAL_FUELS
from transcif.data.loaders import all_region_configs, load_region_data
from transcif.data.fuel import attach_fuel_and_exog

REGIONS = ["VIC1", "SA1", "NSW1", "QLD1",
           "UK_09_East_Midlands", "UK_07_South_Wales", "UK_08_West_Midlands",
           "US_CISO", "US_ERCO", "US_PJM"]


def main():
    cfgs = all_region_configs()
    print("=== 每燃料小时份额 与 物理驱动的相关性(全年)===")
    print(f"{'region':22s} {'fuel':8s} {'~wcf':>6s} {'~csi_d':>6s} {'~sin_h':>6s} {'~temp':>6s}  {'解读'}")
    for n in REGIONS:
        d = load_region_data(n, cfgs)
        attach_fuel_and_exog(d, n, cfgs)
        if not d.get("has_fuel"):
            continue
        fs = d["fuel_shares"]
        ex = d["exog"]
        wcf = ex["wind_cf"]
        csi = ex["clearsky_index"]
        temp = ex["weather"][:, 0]
        hours = pd.DatetimeIndex(d["hours"])
        # LOCAL hour-of-day (US/UK rows are UTC in ``hours``; AU local).
        _, _, tz = REGION_META[n]
        local = hours + pd.to_timedelta(tz, unit="h")
        sinh = np.sin(2 * np.pi * (local.hour + 0.5) / 24.0)
        df = pd.DataFrame({
            "wcf": wcf, "csi_d": csi * (temp * 0 + 1) * (sinh > 0.05),  # daytime csi
            "sinh": sinh, "temp": temp,
        })
        for f in CANONICAL_FUELS:
            share = fs[:, FUEL_INDEX[f]]
            if share.mean() < 0.02:
                continue
            r = {c: df[c].corr(pd.Series(share)) for c in df.columns}
            note = ""
            if f == "wind":
                note = "weather-driven" if r["wcf"] > 0.5 else "WEAK-weather?!"
            if f == "solar":
                note = "astro-driven" if r["csi_d"] > 0.5 else "WEAK-astro?!"
            if f in ("gas", "coal") and abs(r["sinh"]) > 0.3:
                note = "load-following"
            if f in ("gas", "coal") and abs(r["wcf"]) > 0.3:
                note = "wind-residual"
            if f == "imports" and abs(r["sinh"]) > 0.2:
                note = "peak-import"
            print(f"{n:22s} {f:8s} {r['wcf']:+6.2f} {r['csi_d']:+6.2f} "
                  f"{r['sinh']:+6.2f} {r['temp']:+6.2f}  {note}")
        print()


if __name__ == "__main__":
    main()
