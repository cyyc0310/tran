#!/usr/bin/env python
"""FD-41: single-seed full-scale hyperparameter arms (29 regions, seed 0).

Each arm reruns the official v3 stack with one knob changed and reports
per-region MAE for paired comparison against fd39e seed 0.

Usage:
    .venv/bin/python scripts/experiments/probe_hparam_arm.py \
        --epochs 900 --tag ep900
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
    ap.add_argument("--tag", required=True)
    ap.add_argument("--epochs", type=int, default=600)
    ap.add_argument("--p-cold", type=float, default=0.3)
    ap.add_argument("--lambda-shape", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfgs = all_region_configs()
    fd_regions = {n: prepare_fd_region(n, cfgs) for n in cfgs}

    results = []
    for target in fd_regions:
        t0 = time.time()
        model = train_fuel_zero_shot(
            fd_regions, target, seed=0, epochs=args.epochs,
            p_cold=args.p_cold, lambda_shape=args.lambda_shape,
            device=device, use_monthly=True, dynamic_residual=True,
            wind_route_tau=1.1)
        data = fd_regions[target]
        w = build_target_test_windows(data)
        y = w["y_cif"]
        ef_vec = data["ef_vec"].astype(np.float32)
        c, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                       ef_vec, cold=True, device=device)
        i0, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                        ef_vec, cold=False, device=device)
        row = {"target": target, "seed": 0, "tag": args.tag,
               "epochs": args.epochs, "p_cold": args.p_cold,
               "lambda_shape": args.lambda_shape,
               "mae_cfg": float(np.abs(c - y).mean()),
               "mae_i0": float(np.abs(i0 - y).mean()),
               "train_s": round(time.time() - t0, 1)}
        results.append(row)
        print(f"[{args.tag}] {target:26s} cfg {row['mae_cfg']:.1f} "
              f"i0 {row['mae_i0']:.1f}", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"[fd41:{args.tag}] wrote {args.out}")


if __name__ == "__main__":
    main()
