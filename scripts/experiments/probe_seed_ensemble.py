#!/usr/bin/env python
"""FD-40: seed-ensemble deployment mode (29 regions, seeds 0-4).

The official protocol trains one model per (target, seed) and reports the
mean MAE.  A real deployment trains once — but nothing stops it training
five cheap models (30 s each) and averaging their forecasts.  This probe
quantifies that variance reduction per region: same windows, same seeds
as the official run, predictions averaged instead of scored separately.

Usage:
    .venv/bin/python scripts/experiments/probe_seed_ensemble.py \
        --out results/fd40_seed_ensemble.json
"""

import argparse
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


def build_target_test_windows(data):
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(split - SEQ_LEN, None)
    sliced = {**data, "rs": data["rs"][sl], "cif": data["cif"][sl],
              "fuel_shares": data["fuel_shares"][sl],
              "hours": data["hours"][sl],
              "exog": {k: v[sl] for k, v in data["exog"].items()}}
    return build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                            stride=TEST_STRIDE,
                            monthly_table=data.get("monthly_table"),
                            lag_months=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RESULTS_DIR / "fd40_seed_ensemble.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--regions", nargs="+", default=None)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfgs = all_region_configs()
    fd_regions = {n: prepare_fd_region(n, cfgs) for n in cfgs}
    loop = [n for n in (args.regions or list(fd_regions))
            if n in fd_regions]

    results = []
    for target in loop:
        t0 = time.time()
        data = fd_regions[target]
        w = build_target_test_windows(data)
        y = w["y_cif"]
        ef_vec = data["ef_vec"].astype(np.float32)
        cfg_preds, i0_preds = [], []
        for seed in args.seeds:
            model = train_fuel_zero_shot(
                fd_regions, target, seed=seed, epochs=args.epochs,
                device=device, use_monthly=True, dynamic_residual=True,
                wind_route_tau=1.1)
            c, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                           ef_vec, cold=True, device=device)
            i0, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                            ef_vec, cold=False, device=device)
            cfg_preds.append(c)
            i0_preds.append(i0)
            del model
        cfg_preds = np.stack(cfg_preds)
        i0_preds = np.stack(i0_preds)
        ens_cfg = cfg_preds.mean(axis=0)
        ens_i0 = i0_preds.mean(axis=0)
        row = {"target": target,
               "mae_seed_mean_cfg": float(np.abs(cfg_preds - y[None]).mean()),
               "mae_ensemble_cfg": float(np.abs(ens_cfg - y).mean()),
               "mae_seed_mean_i0": float(np.abs(i0_preds - y[None]).mean()),
               "mae_ensemble_i0": float(np.abs(ens_i0 - y).mean()),
               "train_s": round(time.time() - t0, 1)}
        results.append(row)
        print(f"[fd40] {target:26s} cfg {row['mae_seed_mean_cfg']:.1f} -> "
              f"ens {row['mae_ensemble_cfg']:.1f} | i0 {row['mae_seed_mean_i0']:.1f} "
              f"-> {row['mae_ensemble_i0']:.1f} ({row['train_s']}s)", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"[fd40] wrote {args.out}")


if __name__ == "__main__":
    main()
