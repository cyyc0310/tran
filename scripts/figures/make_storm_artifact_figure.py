#!/usr/bin/env python
"""FD-45 evidence figure: Storm Ciarán ERA5 point-value artifact (UK_01).

Top: ERA5 farm-blend 100 m wind at UK_01's sampled sites during the storm
window (2023-10-28 16:00 - 11-01 15:00) with the IEC cut-out threshold —
the input the model sees.  Bottom: actual CIF (flat at zero: the real
dispersed fleet rode through) vs the model's day-ahead forecast, which
faithfully converts the ERA5 dip into false thermal-dispatch spikes.

Usage:
    .venv/bin/python scripts/figures/make_storm_artifact_figure.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transcif.config import SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import build_fd_windows
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, predict_fuel_windows, train_fuel_zero_shot,
)

OUT = Path(__file__).resolve().parent.parent.parent / "figures"
TARGET = "UK_01_North_Scotland"
A0 = 7216          # figure-window start in the region series (storm Ciaran)
CUTOUT = 25.0      # IEC cut-out wind speed (m/s)


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfgs = all_region_configs()
    fd_regions = {n: prepare_fd_region(n, cfgs) for n in cfgs}
    data = fd_regions[TARGET]
    model = train_fuel_zero_shot(fd_regions, TARGET, seed=0, epochs=900,
                                 device=device, use_monthly=True,
                                 dynamic_residual=True, wind_route_tau=1.1)
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(split - SEQ_LEN, None)
    sliced = {**data, "rs": data["rs"][sl], "cif": data["cif"][sl],
              "fuel_shares": data["fuel_shares"][sl],
              "hours": data["hours"][sl],
              "exog": {k: v[sl] for k, v in data["exog"].items()}}
    w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                         stride=TEST_STRIDE,
                         monthly_table=data.get("monthly_table"), lag_months=1)
    sel = list(range(10, 14))
    ws = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in w.items()}
    ef = data["ef_vec"].astype(np.float32)
    cfg_p, _, _ = predict_fuel_windows(model, ws, data["fd_config"], ef,
                                       cold=True, device=device)
    i0_p, _, _ = predict_fuel_windows(model, ws, data["fd_config"], ef,
                                      cold=False, device=device)
    y = ws["y_cif"].ravel()

    hours = pd.date_range("2023-10-28 16:00", periods=96, freq="h")
    wf = pd.read_csv("data_2023/weather/"
                     f"{TARGET}_farmblend_weather_2023_hourly.csv",
                     parse_dates=["hour"])
    wind = wf.set_index("hour").loc[hours, "wind_speed_100m"].values

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    h = np.arange(96)
    ax1.plot(h, wind, color="tab:blue", lw=1.4, label="ERA5 wind 100 m (sampled sites)")
    ax1.axhline(CUTOUT, color="red", ls="--", lw=1,
                label="IEC cut-out 25 m/s")
    ax1.axvspan(24, 48, color="orange", alpha=0.15)
    ax1.annotate("point-value artifact:\n40-60 m/s vs real fleet ~full output",
                 xy=(36, 44), xytext=(52, 50), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))
    ax1.set_ylabel("wind 100 m (m/s)")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.set_title(f"{TARGET} — Storm Ciarán window (2023-10-28 16:00 → 11-01), "
                  "the model converts an ERA5 point-value dip into false "
                  "thermal-dispatch spikes", fontsize=11)

    ax2.plot(h, y, color="black", lw=2, label="Actual (fleet rides through: CIF = 0)")
    ax2.plot(h, i0_p.ravel(), color="tab:orange", lw=1.4,
             label=f"I$_0$ forecast (MAE {np.abs(i0_p.ravel()-y).mean():.0f})")
    ax2.plot(h, cfg_p.ravel(), color="tab:green", lw=1.2,
             label=f"I$_{{cfg}}$ forecast (MAE {np.abs(cfg_p.ravel()-y).mean():.0f})")
    ax2.axvspan(24, 48, color="orange", alpha=0.15)
    ax2.annotate("false thermal-dispatch spike\n(ERA5 dip ≠ real fleet)",
                 xy=(36, float(i0_p.ravel()[24:48].max())),
                 xytext=(2, 110), fontsize=9,
                 arrowprops=dict(arrowstyle="->"))
    ax2.set_ylabel("CIF (gCO$_2$/kWh)")
    ax2.set_xlabel("hours from 2023-10-28 16:00 UTC")
    ax2.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"uk01_storm_ciaran_artifact.{ext}", dpi=150)
    print(f"[storm] wrote {OUT / 'uk01_storm_ciaran_artifact.png'}")


if __name__ == "__main__":
    main()
