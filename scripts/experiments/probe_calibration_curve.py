"""Calibration-data-amount curve: how much target CIF is enough?

Stage B/D follow-up: sweep n_train_origins ∈ {0, 3, 6, 12, 24} to show how the
joint-trained MAE decreases as more target-domain calibration labels become
available.  n_train=0 is pure ZS+ (no labels); n_train=24 is the data-rich end.

This produces the headline figure for Route 1's "information-set stratification"
narrative: calibration data quantity determines the performance ceiling.

Protocol (matches run_joint_train_full.py):
  - origins = test-segment days (stride=24, ~58 total)
  - train_origins = first n_train days (target CIF labels used for calibration)
  - eval_origins  = next 12 disjoint days (held-out evaluation)
  - 12 representative targets × 3 seeds, median MAE per (target, n_train)

Output: results/probe_calibration_curve.json + console table.

Usage:
    PYTHONPATH=src python scripts/experiments/probe_calibration_curve.py
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Allow importing run_joint_train from scripts/experiments (not a package).
_REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_REPO))

from transcif.config import (
    RESULTS_DIR, AU_REGIONS, US_REGIONS, UK_REGIONS,
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION,
)
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.models.zeroshot.base_zs import evaluate_target
from scripts.experiments.run_joint_train import run_joint_train
from scripts.experiments._shared import split_origins

SEEDS = [0, 1, 2]
TARGETS = ["US_CISO", "US_ERCO", "US_MISO", "US_PJM", "US_ISNE",
           "US_FPL", "US_BPAT", "QLD1", "VIC1", "NSW1", "SA1",
           "UK_09_East_Midlands"]
N_TRAIN_SWEEP = [0, 3, 6, 12, 24]
N_EVAL = 12


def build_pool():
    discover_uk_regions()
    all_cfgs = {**AU_REGIONS, **US_REGIONS, **UK_REGIONS}
    return {n: load_region_data(n, all_cfgs) for n in all_cfgs}


def zs_plus_mae(target, regions, seed):
    """Pure ZS+ baseline (n_train=0): no target labels, test-time calibration."""
    r = evaluate_target(target, regions, seed=seed)
    return r["transcif_zs_plus"]["mae"]


def joint_mae(target, regions, seed, n_train, n_eval=N_EVAL, out_dir="/tmp/curve"):
    """Joint-trained MAE with n_train calibration origins, n_eval disjoint eval."""
    src_names = [n for n in regions if n != target][:3]
    small = {target: regions[target]}
    for n in src_names:
        small[n] = regions[n]
    rs = regions[target]["rs"]
    train_origins, eval_origins = split_origins(rs, n_train=n_train, n_eval=n_eval)
    if n_train == 0 or len(train_origins) == 0:
        # Fall back to pure ZS+ when no calibration labels are provided.
        return zs_plus_mae(target, regions, seed)
    summary = run_joint_train(
        small, target,
        stages=("stage1", "stage2"),
        n_origins=len(train_origins),
        out_dir=out_dir,
        seed=seed,
        n_steps_stage1=30,
        n_steps_stage2=30,
        margin=0.10,
        eval_origins=eval_origins,
    )
    return summary.get("held_out_mae", float("nan"))


def main():
    print("=" * 85)
    print("Calibration-data-amount curve: n_train origins → MAE")
    print("=" * 85)
    print(f"Targets: {len(TARGETS)} regions × {len(SEEDS)} seeds")
    print(f"n_train sweep: {N_TRAIN_SWEEP}  (n_train=0 = pure ZS+)")
    print(f"n_eval: {N_EVAL} disjoint origins")
    print(f"Calibration hours = n_train × {HORIZON}")

    regions = build_pool()

    results = {}  # results[n_train][target] = median MAE over seeds
    for n_train in N_TRAIN_SWEEP:
        calib_hours = n_train * HORIZON
        label = "ZS+(0h)" if n_train == 0 else f"Joint({calib_hours}h)"
        print(f"\n--- {label} (n_train={n_train}) ---")
        t0 = time.time()
        results[n_train] = {}
        for i, target in enumerate(TARGETS):
            if target not in regions:
                continue
            try:
                seed_maes = []
                for seed in SEEDS:
                    m = joint_mae(target, regions, seed, n_train)
                    seed_maes.append(m)
                med = float(np.median(seed_maes))
                results[n_train][target] = med
                print(f"  {i+1:2d}/{len(TARGETS)} {target:20s} "
                      f"median={med:.1f}  ({label})")
            except Exception as ex:
                print(f"  {i+1:2d}/{len(TARGETS)} {target:20s} FAIL: {ex}")
                results[n_train][target] = float("nan")
        elapsed = time.time() - t0
        med_all = np.nanmedian(list(results[n_train].values()))
        print(f"  [{label}] median across targets = {med_all:.2f}  ({elapsed:.0f}s)")

    # Summary table.
    print("\n" + "=" * 95)
    print("CALIBRATION CURVE: median MAE by n_train origins")
    print("=" * 95)
    header = f"{'Target':<22}"
    for n_train in N_TRAIN_SWEEP:
        hours = n_train * HORIZON
        header += f" {f'{hours}h':>8}"
    print(header)
    print("-" * len(header))
    for target in TARGETS:
        row = f"{target:<22}"
        for n_train in N_TRAIN_SWEEP:
            v = results[n_train].get(target, float("nan"))
            row += f" {v:>8.1f}" if not np.isnan(v) else f" {'--':>8}"
        print(row)
    # Aggregate row.
    row = f"{'MEDIAN':<22}"
    for n_train in N_TRAIN_SWEEP:
        med = np.nanmedian(list(results[n_train].values()))
        row += f" {med:>8.2f}"
    print(row)

    out = {
        "targets": TARGETS, "seeds": SEEDS,
        "n_train_sweep": N_TRAIN_SWEEP, "n_eval": N_EVAL,
        "calibration_hours": [n * HORIZON for n in N_TRAIN_SWEEP],
        "results": results,
    }
    out_path = RESULTS_DIR / "probe_calibration_curve.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[WRITE] {out_path}")


if __name__ == "__main__":
    main()
