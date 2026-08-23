#!/usr/bin/env python
"""FD-16 mechanism diagnostic: per-window MAE by wind-regime tercile.

Trains two models per wind-heavy region (seed 0): the FD-16 path (10 wx
channels + drought-anchored wind reference) and the legacy path
(FuelDecompNet(n_weather=8) fed the first 8 channels).  Splits test
windows into lull/mid/normal terciles by the trailing-24 h wind regime
at the origin and reports MAE per tercile — the attribution study says
lull-transition days are where extreme CIF error lives.

Usage:
    .venv/bin/python scripts/analysis/diag_wind_lull.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_fuel_decomp_eval import build_target_test_windows
from transcif.data.loaders import all_region_configs
from transcif.models.fuel_decomp import FuelDecompNet
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, predict_fuel_windows, train_fuel_zero_shot,
)

REGIONS = ["VIC1", "SA1", "UK_02_South_Scotland", "UK_07_South_Wales"]


def trunc8(w):
    w = dict(w)
    w["x_weather"] = w["x_weather"][:, :, :8]
    w["fut_weather"] = w["fut_weather"][:, :, :8]
    return w


def main():
    cfgs = all_region_configs()
    names = set(REGIONS) | {"US_CISO", "US_PJM", "US_BPAT", "QLD1"}
    pool = {n: prepare_fd_region(n, cfgs) for n in names}

    def trunc_pool(d):
        d8 = dict(d)
        ex = dict(d["exog"])
        ex["weather"] = ex["weather"][:, :8]
        ex.pop("wind_regime24", None)   # legacy arm: no regime channels
        ex.pop("wind_tend6", None)
        d8["exog"] = ex
        return d8

    def trunc_fex(w):
        w = dict(w)
        w["fut_exog"] = w["fut_exog"][:, :, :15]
        return w

    pool8 = {n: trunc_pool(d) for n, d in pool.items()}
    for region in REGIONS:
        tgt = pool[region]
        w = build_target_test_windows(tgt)
        truth = w["y_cif"]
        regime = w["x_weather"][:, -1, 8]   # trailing-24 h regime at origin
        q1, q2 = np.quantile(regime, [1 / 3, 2 / 3])
        bins = [("lull", regime <= q1), ("mid", (regime > q1) & (regime <= q2)),
                ("normal", regime > q2)]

        model = train_fuel_zero_shot(pool, region, seed=0, epochs=600)
        cif_new, _, _ = predict_fuel_windows(model, w, tgt["fd_config"],
                                             tgt["ef_vec"].astype(np.float32), cold=False)
        model_old = FuelDecompNet(n_weather=8, n_exog=15)
        model_old = train_fuel_zero_shot(pool8, region, seed=0, epochs=600,
                                         model=model_old)
        cif_old, _, _ = predict_fuel_windows(model_old, trunc_fex(trunc8(w)),
                                             tgt["fd_config"], tgt["ef_vec"].astype(np.float32),
                                             cold=False)

        print(f"--- {region} (regime terciles at {q1:.2f}/{q2:.2f}) ---")
        for name, m in bins:
            if m.sum() < 5:
                continue
            e_new = np.abs(cif_new[m] - truth[m]).mean()
            e_old = np.abs(cif_old[m] - truth[m]).mean()
            print(f"  {name:7s} n={m.sum():3d}  old {e_old:6.1f} -> new "
                  f"{e_new:6.1f}  ({e_old - e_new:+.1f})")
        e_all_n = np.abs(cif_new - truth).mean()
        e_all_o = np.abs(cif_old - truth).mean()
        print(f"  ALL     n={len(truth):3d}  old {e_all_o:6.1f} -> new "
              f"{e_all_n:6.1f}  ({e_all_o - e_all_n:+.1f})")


if __name__ == "__main__":
    main()
