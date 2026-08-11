"""Task 8.6 driver: full LORO evaluation for joint training.

For each (target, seed):
  1. Load 29 regions
  2. Train joint model on first half of test origins (Stage 1 + Stage 2)
  3. Evaluate on second half (held out)
  4. Write one row per pair to results/joint_train_full.json

Aggregates median/mean/std to results/joint_train_full_summary.json.

Comparison target: results/fused_five_full_summary.json BasisMix+ row
(median MAE 46.89). Apples-to-apples comparison requires same eval protocol,
but note that the joint model is supervised (uses target CIF labels for
calibration fine-tuning), unlike pure zero-shot BasisMix+.

Usage:
    .venv/bin/python scripts/experiments/run_joint_train_full.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Users/cyyc0310/Downloads/transcif")

from transcif.config import HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION
from transcif.data.loaders import load_region_data, all_region_configs
from scripts.experiments.run_joint_train import run_joint_train, _origins_from_series


def split_origins(rs: np.ndarray, n_train: int = 12, n_eval: int = 12):
    """Get disjoint train + eval origin lists from the test split."""
    split = int(len(rs) * TRAIN_FRACTION)
    all_origins = [
        split + st
        for st in range(0, len(rs) - split - HORIZON + 1, TEST_STRIDE)
    ]
    if len(all_origins) < n_train + n_eval:
        # Fall back to fewer if series is short
        n_train = max(2, len(all_origins) // 2)
        n_eval = len(all_origins) - n_train
    train_origins = all_origins[:n_train]
    eval_origins = all_origins[n_train : n_train + n_eval]
    return train_origins, eval_origins


def evaluate_target(target: str, all_regions: dict, seed: int,
                     n_train: int, n_eval: int, out_dir: str):
    """Train + eval one (target, seed) pair. Returns result dict."""
    src_names = [n for n in all_regions if n != target][:3]
    small_regions = {target: all_regions[target]}
    for n in src_names:
        small_regions[n] = all_regions[n]

    rs = all_regions[target]["rs"]
    train_origins, eval_origins = split_origins(rs, n_train=n_train, n_eval=n_eval)

    t0 = time.time()
    summary = run_joint_train(
        small_regions, target,
        stages=("stage1", "stage2"),
        n_origins=n_train,
        out_dir=out_dir,
        seed=seed,
        n_steps_stage1=30,
        n_steps_stage2=30,
        margin=0.10,
        eval_origins=eval_origins,
    )
    elapsed = time.time() - t0

    return {
        "target": target,
        "seed": seed,
        "sources": src_names,
        "n_train_origins": len(train_origins),
        "n_eval_origins": len(eval_origins),
        "stage1_train_mae": summary.get("stage1_final_mae"),
        "stage2_train_mae": summary.get("stage2_final_mae"),
        "held_out_mae": summary.get("held_out_mae"),
        "elapsed_seconds": elapsed,
    }


def main():
    ap = argparse.ArgumentParser(description="Joint train full LORO (Task 8.6)")
    ap.add_argument("--regions", nargs="+", default=None,
                    help="Specific regions; default = all 29")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-eval", type=int, default=12)
    ap.add_argument("--out", default="results/joint_train_full.json")
    ap.add_argument("--out-dir-root", default="results/joint_train_runs")
    args = ap.parse_args()

    print(f"[LOAD] loading regions...", flush=True)
    all_configs = all_region_configs()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}
    print(f"[LOAD] {len(all_regions)} regions", flush=True)

    targets = args.regions or list(all_regions.keys())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_dir_root = Path(args.out_dir_root)
    out_dir_root.mkdir(parents=True, exist_ok=True)

    # Resume support: load existing rows
    if out_path.exists():
        rows = json.loads(out_path.read_text())
        print(f"[RESUME] {len(rows)} existing rows", flush=True)
    else:
        rows = []

    completed_pairs = {(r["target"], r["seed"]) for r in rows}

    t_start = time.time()
    for target in targets:
        for seed in args.seeds:
            if (target, seed) in completed_pairs:
                continue
            print(f"\n=== {target} seed {seed} ===", flush=True)
            pair_out_dir = out_dir_root / f"{target}_seed{seed}"
            try:
                row = evaluate_target(
                    target, all_regions, seed,
                    n_train=args.n_train, n_eval=args.n_eval,
                    out_dir=str(pair_out_dir),
                )
                rows.append(row)
                completed_pairs.add((target, seed))
                # Incremental write for resume safety
                out_path.write_text(json.dumps(rows, indent=2))
                print(
                    f"  held_out_mae={row['held_out_mae']:.2f}  "
                    f"elapsed={row['elapsed_seconds']:.1f}s",
                    flush=True,
                )
            except Exception as e:
                print(f"  [ERROR] {target}/{seed}: {e}", flush=True)
                rows.append({
                    "target": target, "seed": seed, "error": str(e),
                })
                out_path.write_text(json.dumps(rows, indent=2))

    total_elapsed = time.time() - t_start
    print(f"\n[DONE] {total_elapsed:.1f}s total", flush=True)

    # Aggregate
    valid_rows = [r for r in rows if "held_out_mae" in r and r["held_out_mae"] is not None]
    if valid_rows:
        maes = np.array([r["held_out_mae"] for r in valid_rows])
        summary = {
            "joint_trained": {
                "mae": {
                    "median": float(np.median(maes)),
                    "mean": float(np.mean(maes)),
                    "std": float(np.std(maes)),
                    "n_pairs": int(len(maes)),
                },
            },
            "baseline_basisMix_plus": {
                "mae": {"median": 46.89, "source": "results/fused_five_full_summary.json"},
            },
            "headline_target": 41.0,
        }
        summary_path = out_path.parent / "joint_train_full_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2))
        print(f"\n[SUMMARY] median MAE = {summary['joint_trained']['mae']['median']:.2f}")
        print(f"  vs baseline 46.89, target < 41.0")


if __name__ == "__main__":
    main()
