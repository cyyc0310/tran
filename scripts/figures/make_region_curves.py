#!/usr/bin/env python
"""29-region CIF curves: the full information ladder vs actual.

For each region (seed 0): train on the other 28 regions, then forecast 4
consecutive day-ahead origins (96 h) from the middle of the test split.
Each panel shows five curves:

    actual          (black)      ground truth
    I_cfg           (green thin) config + weather + calendar, ZERO telemetry
    I_0             (orange)     + 336 h renewable-share telemetry
    I_+             (blue)       + observable CIF history (ZS+ calibration)
    I_S             (red dashed) same architecture fine-tuned on the
                                target's 80 % local labels (upper bound)

Usage:
    .venv/bin/python scripts/figures/make_region_curves.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from transcif.config import (
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION, RESULTS_DIR,
)
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import build_fd_windows
from transcif.models.zeroshot.fuel import (
    finetune_fuel_supervised, make_zs_plus_share_fn, prepare_fd_region,
    predict_fuel_windows, train_fuel_zero_shot,
)
from transcif.calibration.zs_plus import zs_plus_predict

OUT = Path(__file__).resolve().parent.parent.parent / "figures"
N_ORIGINS = 4          # consecutive day-ahead origins -> 96 h curve
ORIGIN_OFFSET = 10     # start mid-test for a representative stretch

# Deployment route table (FD-22/23 official): these regions' models are
# trained on the fuel path; every other region uses the default tau.
ROUTE_TAU = {"UK_01_North_Scotland": 1.1, "UK_02_South_Scotland": 1.1,
             "UK_16_Scotland": 1.1}


def region_curves(target, fd_regions, device):
    import copy
    model = train_fuel_zero_shot(fd_regions, target, seed=0, epochs=900,
                                 device=device, use_monthly=True,
                                 dynamic_residual=True,
                                 wind_route_tau=ROUTE_TAU.get(target, 1.1))

    data = fd_regions[target]
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(split - SEQ_LEN, None)
    sliced = {**data,
              "rs": data["rs"][sl], "cif": data["cif"][sl],
              "fuel_shares": data["fuel_shares"][sl],
              "hours": data["hours"][sl],
              "exog": {k: v[sl] for k, v in data["exog"].items()}}
    w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                         stride=TEST_STRIDE,
                         monthly_table=data.get("monthly_table"), lag_months=1)
    n = len(w["x_rs"])
    o0 = min(ORIGIN_OFFSET, max(0, n - N_ORIGINS))
    sel = list(range(o0, min(o0 + N_ORIGINS, n)))
    w_sel = {k: (v[sel] if isinstance(v, (np.ndarray, pd.DatetimeIndex)) else v)
             for k, v in w.items()}

    ef_vec = data["ef_vec"].astype(np.float32)
    cif_i0, _, _ = predict_fuel_windows(model, w_sel, data["fd_config"],
                                        ef_vec, cold=False, device=device)
    cif_cfg, _, _ = predict_fuel_windows(model, w_sel, data["fd_config"],
                                         ef_vec, cold=True, device=device)
    # I_+ via ZS+ calibration at the same origins.
    share_fn = make_zs_plus_share_fn(model, data, device=device)
    origins = np.array([split + st for st in
                        range(0, len(data["cif"][split - SEQ_LEN:])
                              - SEQ_LEN - HORIZON + 1, TEST_STRIDE)])[sel]
    cif_plus = zs_plus_predict(model, data["fd_config"], data["rs"],
                               data["cif"], data["ef_r"], data["ef_nr"],
                               origins, share_fn=share_fn)
    actual = w_sel["y_cif"]
    return actual, cif_plus, cif_cfg, cif_i0


def main():
    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu")
    cfgs = all_region_configs()
    pool = list(cfgs)
    print(f"[curves] preparing {len(pool)} regions ...")
    fd_regions = {n: prepare_fd_region(n, cfgs) for n in pool}

    # Sort panels by I_cfg MAE (the headline telemetry-free tier).
    import json
    rows = json.load(open(RESULTS_DIR / "fuel_decomp_eval_full_fd41.json"))["rows"]
    from collections import defaultdict
    mae = defaultdict(list)
    for r in rows:
        mae[r["target"]].append(r["fuel_i_cfg"]["mae"])
    order = sorted(pool, key=lambda n: np.mean(mae.get(n, [1e9])))

    n_rows, n_cols = 6, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 13),
                             sharex=True)
    hours = np.arange(N_ORIGINS * HORIZON)
    for ax, target in zip(axes.flat, order):
        actual, plus, cfg_pred, i0 = region_curves(target, fd_regions, device)
        m = lambda p: np.abs(p.ravel() - actual.ravel()).mean()
        ax.plot(hours, actual.ravel(), color="black", lw=1.5, label="Actual")
        ax.plot(hours, cfg_pred.ravel(), color="tab:green", lw=1.1,
                label="I$_{cfg}$ (no telemetry)")
        ax.plot(hours, i0.ravel(), color="tab:orange", lw=1.2,
                label="I$_0$ (+rs telemetry)")
        ax.set_title(f"{target}  ({m(cfg_pred):.0f} / {m(i0):.0f})", fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.flat[len(order):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=11)
    fig.suptitle(
        "Day-ahead CIF, 29 regions — zero-telemetry (green) vs share "
        "telemetry (orange)\npanel title: I$_{cfg}$ / I$_0$ MAE over the "
        "plotted 96 h; panels sorted by I$_{cfg}$; fd41 official (900ep, fuel path)",
        fontsize=12)
    fig.supxlabel("hours")
    fig.supylabel("CIF (gCO$_2$/kWh)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    OUT.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"region_curves_29_cfg0.{ext}", dpi=160)
    print(f"[curves] wrote {OUT / 'region_curves_29_cfg0.png'}")


if __name__ == "__main__":
    main()
