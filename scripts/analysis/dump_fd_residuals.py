#!/usr/bin/env python
"""Per-hour residual dump for the FD pipeline (error-vs-actual attribution).

Trains the zero-shot model per region (seed 0, default FD-19 router),
predicts the full test split at I_0 and I_cfg, and saves per-window
origin/bias/error joined with actual-CIF context (level, volatility,
wind regime, month, hour) for downstream analysis.

Usage:
    .venv/bin/python scripts/analysis/dump_fd_residuals.py [--regions A B]
Writes results/fd_residuals/{REGION}.npz
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))

from run_fuel_decomp_eval import build_target_test_windows
from transcif.config import SEQ_LEN, HORIZON, TRAIN_FRACTION
from transcif.data.fuel import FUEL_INDEX
from transcif.data.loaders import all_region_configs
from transcif.models.zeroshot.fuel import (
    predict_fuel_windows, prepare_fd_region, train_fuel_zero_shot,
)

OUT = Path(__file__).resolve().parent.parent.parent / "results" / "fd_residuals"

REGIONS = [
    "US_CISO", "US_ERCO", "US_NYIS", "US_PJM",
    "UK_09_East_Midlands", "UK_07_South_Wales", "UK_12_South_England",
    "NSW1", "QLD1", "SA1", "VIC1",
]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=REGIONS)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    cfgs = all_region_configs()
    pool = {n: prepare_fd_region(n, cfgs) for n in cfgs}
    for name in args.regions:
        out = OUT / f"{name}.npz"
        if out.exists():
            print(f"[dump] {name}: exists, skip")
            continue
        model = train_fuel_zero_shot(pool, name, seed=0, epochs=600,
                                     wind_route_tau=0.45)
        d = pool[name]
        w = build_target_test_windows(d)
        ef = d["ef_vec"].astype(np.float32)
        c0, _, _ = predict_fuel_windows(model, w, d["fd_config"], ef, cold=False)
        cc, _, _ = predict_fuel_windows(model, w, d["fd_config"], ef, cold=True)
        y = w["y_cif"]
        oh = pd.DatetimeIndex(w["origin_hours"])
        # actual-context features at the origin (deployment-observable or
        # truth-side diagnostics): trailing regime, actual window stats
        regime = w["x_weather"][:, -1, 8]
        wind_share_t = w["x_fuel"][:, -168:, FUEL_INDEX["wind"]].mean(axis=1)
        np.savez_compressed(
            out,
            y_true=y.astype(np.float32),
            pred_i0=c0.astype(np.float32),
            pred_icfg=cc.astype(np.float32),
            origin_hours=oh.values,
            regime24=regime.astype(np.float32),
            wind_share_t=wind_share_t.astype(np.float32),
            y_std=w["y_cif"].std(axis=1).astype(np.float32),
        )
        print(f"[dump] {name}: wrote {out.name} ({len(y)} windows)")


if __name__ == "__main__":
    main()
