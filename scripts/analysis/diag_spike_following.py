#!/usr/bin/env python
"""Spike-following diagnostic for high-MAE regions (FD-35).

Hypothesis (user): the residual error concentrates in SPIKE hours and/or
the model over-chases transients.  For each region (seed 0, deployment
mode): predict the test split, then measure

    ramp_regime   |dCIF| per hour (actual) — top tercile = spike hours
    err_spike     mean |err| in actual-spike hours
    err_calm      mean |err| in calm hours (concentration ratio)
    disp_ratio    median per-window std(pred)/std(actual)
                  (>1.15 over-follows, <0.85 too flat)
    lag_corr      argmax hour-lag correlation ±6 h (delayed tracking)

Usage:
    .venv/bin/python scripts/analysis/diag_spike_following.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_fuel_decomp_eval import build_target_test_windows
from transcif.data.loaders import all_region_configs
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, predict_fuel_windows, train_fuel_zero_shot,
)

REGIONS = ["UK_09_East_Midlands", "UK_08_West_Midlands", "UK_07_South_Wales", "SA1"]


def main():
    cfgs = all_region_configs()
    pool = {n: prepare_fd_region(n, cfgs) for n in cfgs}
    print(f"{'region':24s} {'尖峰err':>7s} {'平稳err':>7s} {'集中比':>6s} "
          f"{'离散比':>6s} {'滞后h':>5s}  判定")
    for name in REGIONS:
        model = train_fuel_zero_shot(pool, name, seed=0, epochs=600,
                                     wind_route_tau=0.45, device="mps",
                                     dynamic_residual=True)
        d = pool[name]
        w = build_target_test_windows(d, use_monthly=True)
        ef = d["ef_vec"].astype(np.float32)
        cif, _, _ = predict_fuel_windows(model, w, d["fd_config"], ef,
                                         cold=True, device="mps")
        y = w["y_cif"]
        err = np.abs(cif - y)
        # actual ramp regime on the flattened horizon
        dy = np.abs(np.diff(y, axis=1, prepend=y[:, :1]))
        thr = np.percentile(dy, 66)
        spike = dy > thr
        e_spike = err[spike].mean()
        e_calm = err[~spike].mean()
        ratio = e_spike / max(e_calm, 1e-6)
        # dispersion
        disp = np.median(cif.std(axis=1) / y.std(axis=1).clip(1e-6))
        # lag correlation (flatten to a single series, hourly)
        def corr_at_lag(lag):
            a = cif.ravel()
            b = y.ravel()
            if lag > 0:
                a, b = a[lag:], b[:-lag]
            elif lag < 0:
                a, b = a[:lag], b[-lag:]
            return np.corrcoef(a, b)[0, 1]
        lags = range(-6, 7)
        best_lag = max(lags, key=lambda l: corr_at_lag(l))
        verdict = []
        if ratio > 2.0:
            verdict.append("尖峰集中" if disp < 0.85 else "追不上尖峰")
        if disp > 1.15:
            verdict.append("过度跟随")
        elif disp < 0.85:
            verdict.append("过平(漏尖峰)")
        if abs(best_lag) >= 2:
            verdict.append(f"滞后{best_lag:+d}h")
        if not verdict:
            verdict.append("无主导模式(水平/标签)")
        print(f"{name:24s} {e_spike:7.1f} {e_calm:7.1f} {ratio:6.2f} "
              f"{disp:6.2f} {best_lag:+5d}  {'+'.join(verdict)}")


if __name__ == "__main__":
    main()
