#!/usr/bin/env python
"""LOJO (leave-one-jurisdiction-out) deployment-stack ensemble: 4 AU targets,
trained ONLY on US+UK sources, 5-seed prediction averaging (FD-40/42 mode).

Companion to results/lojo_au_full.json (per-seed scoring).  This script
averages the five per-seed predictions (variance reduction, deployment
mode) instead of scoring them separately.

Usage:
    PYTHONPATH=src .venv-nemed/bin/python scripts/experiments/lojo_ensemble.py
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

AU_TARGETS = ["QLD1", "NSW1", "VIC1", "SA1"]


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
    ap.add_argument("--out", default=str(RESULTS_DIR / "lojo_ensemble.json"))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--targets", nargs="+", default=AU_TARGETS)
    args = ap.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    cfgs = all_region_configs()
    fd_regions = {n: prepare_fd_region(n, cfgs) for n in cfgs}

    from transcif.calibration.zs_plus import zs_plus_predict
    from transcif.models.zeroshot.fuel import make_zs_plus_share_fn
    from transcif.config import TRAIN_FRACTION, TEST_STRIDE as _TS

    results = []
    for target in args.targets:
        t0 = time.time()
        data = fd_regions[target]
        w = build_target_test_windows(data)
        y = w["y_cif"]
        n = len(w["x_rs"])
        ef_vec = data["ef_vec"].astype(np.float32)
        cfg_preds, i0_preds, ip_preds = [], [], []
        for seed in args.seeds:
            model = train_fuel_zero_shot(
                fd_regions, target, seed=seed, epochs=args.epochs,
                device=device, use_monthly=True, dynamic_residual=True,
                wind_route_tau=1.1, cross_jurisdiction=True)
            c, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                           ef_vec, cold=True, device=device)
            i0, _, _ = predict_fuel_windows(model, w, data["fd_config"],
                                            ef_vec, cold=False, device=device)
            split = int(len(data["rs"]) * TRAIN_FRACTION)
            share_fn = make_zs_plus_share_fn(model, data, device=device)
            origins = [split + st for st in range(
                0, len(data["cif"][split - SEQ_LEN:]) - SEQ_LEN - HORIZON + 1,
                _TS)][:n]
            ip = zs_plus_predict(model, data["fd_config"], data["rs"],
                                 data["cif"], data["ef_r"], data["ef_nr"],
                                 origins, share_fn=share_fn)
            cfg_preds.append(c)
            i0_preds.append(i0)
            ip_preds.append(ip)
            del model
        cfg_preds = np.stack(cfg_preds)
        i0_preds = np.stack(i0_preds)
        ip_preds = np.stack(ip_preds)
        row = {"target": target,
               "mae_seed_mean_cfg": float(np.abs(cfg_preds - y[None]).mean()),
               "mae_ensemble_cfg": float(np.abs(cfg_preds.mean(axis=0) - y).mean()),
               "mae_seed_mean_i0": float(np.abs(i0_preds - y[None]).mean()),
               "mae_ensemble_i0": float(np.abs(i0_preds.mean(axis=0) - y).mean()),
               "mae_seed_mean_iplus": float(np.abs(ip_preds - y[None]).mean()),
               "mae_ensemble_iplus": float(np.abs(ip_preds.mean(axis=0) - y).mean()),
               "train_s": round(time.time() - t0, 1)}
        results.append(row)
        print(f"[lojo-ens] {target:8s} cfg {row['mae_seed_mean_cfg']:.1f} -> "
              f"{row['mae_ensemble_cfg']:.1f} | i0 {row['mae_seed_mean_i0']:.1f} -> "
              f"{row['mae_ensemble_i0']:.1f} | i+ {row['mae_seed_mean_iplus']:.1f} -> "
              f"{row['mae_ensemble_iplus']:.1f} ({row['train_s']}s)", flush=True)
        Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"[lojo-ens] wrote {args.out}")


if __name__ == "__main__":
    main()
