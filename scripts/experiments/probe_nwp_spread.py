#!/usr/bin/env python
"""FD-43: NWP inter-model spread channels — Scotland family + controls.

8 regions x seeds {0, 1} x 900ep with NWP_SPREAD=1, paired against the
fd41 official rows (same seeds).  Run with the env set by the caller.

Usage:
    NWP_SPREAD=1 .venv/bin/python scripts/experiments/probe_nwp_spread.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from transcif.config import RESULTS_DIR, SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION
from transcif.data.loaders import all_region_configs
from transcif.data.fuel import build_fd_windows
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, predict_fuel_windows, train_fuel_zero_shot,
)

REGIONS = None   # None = all 29
SEEDS = [0, 1, 2, 3, 4]


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfgs = all_region_configs()
    fd_regions = {n: prepare_fd_region(n, cfgs) for n in cfgs}

    results = []
    for target in (REGIONS or list(fd_regions)):
        data = fd_regions[target]
        split = int(len(data["rs"]) * TRAIN_FRACTION)
        sl = slice(split - SEQ_LEN, None)
        sliced = {**data, "rs": data["rs"][sl], "cif": data["cif"][sl],
                  "fuel_shares": data["fuel_shares"][sl],
                  "hours": data["hours"][sl],
                  "exog": {k: v[sl] for k, v in data["exog"].items()}}
        w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                             stride=TEST_STRIDE,
                             monthly_table=data.get("monthly_table"),
                             lag_months=1)
        y = w["y_cif"]
        ef_vec = data["ef_vec"].astype(np.float32)
        for seed in SEEDS:
            t0 = time.time()
            model = train_fuel_zero_shot(
                fd_regions, target, seed=seed, epochs=900, device=device,
                use_monthly=True, dynamic_residual=True, wind_route_tau=1.1)
            c, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                           ef_vec, cold=True, device=device)
            i0, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                            ef_vec, cold=False, device=device)
            row = {"target": target, "seed": seed,
                   "mae_cfg": float(np.abs(c - y).mean()),
                   "mae_i0": float(np.abs(i0 - y).mean()),
                   "train_s": round(time.time() - t0, 1)}
            results.append(row)
            print(f"[fd43] {target:26s} s{seed} cfg {row['mae_cfg']:.1f} "
                  f"i0 {row['mae_i0']:.1f}", flush=True)
            RESULTS_DIR.joinpath("fd44_probe.json").write_text(
                json.dumps(results, indent=1))
    print("[fd43] done")


if __name__ == "__main__":
    main()
