#!/usr/bin/env python
"""Route selection on the target's TRAIN period (deployment-legal).

The deterministic router cannot separate SA1 (wind .561 -> aggregate)
from UK_16 (wind .544 -> fuel) with one threshold.  For the four
aggregate-routed wind-heavy regions this script trains BOTH route modes
(tau=0.45 aggregate-favouring, tau=1.1 fuel), selects the winner by
I_cfg MAE on the target's TRAIN-period windows (never the test split),
then reports test metrics with the selected route.

Usage:
    .venv/bin/python scripts/experiments/run_route_selection_eval.py \
        --out results/fd22_route_selection.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from transcif.config import SEQ_LEN, HORIZON, TRAIN_FRACTION, TEST_STRIDE
from transcif.data.fuel import build_fd_windows
from transcif.models.zeroshot.fuel import (
    prepare_fd_region, predict_fuel_windows, shape_metrics_with_months,
    train_fuel_zero_shot,
)

REGIONS = ["UK_01_North_Scotland", "UK_02_South_Scotland",
           "UK_16_Scotland", "SA1"]
TAUS = {"aggregate": 0.45, "fuel": 1.1}


def train_val_windows(data):
    """Windows from the train split only (route-selection validation)."""
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(0, split)
    sliced = {**data,
              "rs": data["rs"][sl], "cif": data["cif"][sl],
              "fuel_shares": data["fuel_shares"][sl],
              "hours": data["hours"][sl],
              "exog": {k: v[sl] for k, v in data["exog"].items()}}
    return build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                            stride=24, max_windows=120,
                            rng=np.random.default_rng(0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    from transcif.data.loaders import all_region_configs
    cfgs = all_region_configs()
    pool = {n: prepare_fd_region(n, cfgs) for n in cfgs}

    rows = []
    for region in REGIONS:
        data = pool[region]
        w_val = train_val_windows(data)
        for seed in args.seeds:
            t0 = time.time()
            scores = {}
            models = {}
            for name, tau in TAUS.items():
                m = train_fuel_zero_shot(pool, region, seed=seed, epochs=600,
                                         wind_route_tau=tau)
                cif_val, _, _ = predict_fuel_windows(
                    m, w_val, data["fd_config"],
                    data["ef_vec"].astype(np.float32), cold=True)
                scores[name] = float(np.abs(cif_val - w_val["y_cif"]).mean())
                models[name] = m
            pick = min(scores, key=scores.get)
            # test windows (same protocol as run_fuel_decomp_eval)
            split = int(len(data["rs"]) * TRAIN_FRACTION)
            sl = slice(split - SEQ_LEN, None)
            sliced = {**data,
                      "rs": data["rs"][sl], "cif": data["cif"][sl],
                      "fuel_shares": data["fuel_shares"][sl],
                      "hours": data["hours"][sl],
                      "exog": {k: v[sl] for k, v in data["exog"].items()}}
            w_test = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                                      stride=TEST_STRIDE)
            y = w_test["y_cif"]
            ef = data["ef_vec"].astype(np.float32)
            row = {"target": region, "seed": seed, "route": pick,
                   "val_mae": scores, "n_test": len(y)}
            for tier, cold in [("fuel_i_cfg", True), ("fuel_i0", False)]:
                cif, _, _ = predict_fuel_windows(models[pick], w_test,
                                                 data["fd_config"], ef,
                                                 cold=cold)
                row[tier] = shape_metrics_with_months(cif, y,
                                                      w_test["origin_hours"])
            rows.append(row)
            print(f"[route] {region} s{seed}: val {scores} -> {pick} | "
                  f"I_cfg {row['fuel_i_cfg']['mae']:.1f} "
                  f"({time.time() - t0:.0f}s)")
            with open(args.out, "w") as f:
                json.dump({"rows": rows}, f, indent=1)
    print(f"[route] wrote {args.out}")


if __name__ == "__main__":
    main()
