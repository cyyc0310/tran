"""Full LORO driver for the torch-native joint pipeline (Phase 9, Step 5).

Mirrors ``run_joint_train_full`` but calls ``run_native_joint_train`` (3 live
directions + 2 frozen + LearnedFusion + ZS+, Stage 2 unfreezes direction heads).
Writes one row per (target, seed) to results/joint_train_native_full.json,
schema-compatible with joint_train_full.json for direct comparison.

Usage:
    .venv/bin/python scripts/experiments/run_joint_train_native_full.py
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transcif.config import SEEDS_FULL
from transcif.data.loaders import all_region_configs, load_region_data
from scripts.experiments.run_joint_train_native import run_native_joint_train


def main():
    ap = argparse.ArgumentParser(description="Native joint train full LORO (Phase 9)")
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--fusion", default="learned", choices=["learned", "softmax"])
    ap.add_argument("--gate", default=None, choices=[None, "internal_val"],
                    help="internal_val: skip Stage 2 head finetune when it "
                         "overfits the inner validation split (fixes easy-grid regressions)")
    ap.add_argument("--out", default="results/joint_train_native_full.json")
    ap.add_argument("--n-steps-s1", type=int, default=30)
    ap.add_argument("--n-steps-s2", type=int, default=30)
    args = ap.parse_args()

    print("[LOAD] regions...", flush=True)
    all_configs = all_region_configs()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}
    print(f"[LOAD] {len(all_regions)} regions", flush=True)

    targets = args.regions or list(all_regions.keys())
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = json.loads(out_path.read_text()) if out_path.exists() else []
    done = {(r["target"], r["seed"]) for r in rows if "held_out_mae" in r}

    t_start = time.time()
    for target in targets:
        for seed in args.seeds:
            if (target, seed) in done:
                continue
            if target not in all_regions:
                continue
            src_names = [n for n in all_regions if n != target][:3]
            small_regions = {}
            for n in [target] + src_names:
                rd = dict(all_regions[n])
                rd["config"] = np.asarray(rd["config"], dtype=np.float32)[:2]
                small_regions[n] = rd

            print(f"\n=== {target} seed{seed} fusion={args.fusion} ===", flush=True)
            t0 = time.time()
            try:
                summ = run_native_joint_train(
                    small_regions, target, seed=seed, fusion_kind=args.fusion,
                    n_steps_s1=args.n_steps_s1, n_steps_s2=args.n_steps_s2,
                    gate=args.gate)
                rows.append({
                    "target": target, "seed": seed, "sources": src_names,
                    "fusion": args.fusion, "gate": args.gate,
                    "stage1_train_mae": summ["stage1_train_mae"],
                    "stage2_train_mae": summ["stage2_train_mae"],
                    "held_out_mae": summ["held_out_mae"],
                    "n_eval_origins": summ["n_eval_origins"],
                    "gate_decision": summ.get("gate_decision"),
                    "elapsed_seconds": time.time() - t0,
                })
                out_path.write_text(json.dumps(rows, indent=2))
                print(f"  held_out_mae={summ['held_out_mae']:.2f}  "
                      f"({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(f"  [ERROR] {target}/{seed}: {e}", flush=True)
                traceback.print_exc()
                rows.append({"target": target, "seed": seed, "error": str(e)})
                out_path.write_text(json.dumps(rows, indent=2))

    valid = [r for r in rows if "held_out_mae" in r and r["held_out_mae"] is not None]
    if valid:
        maes = np.array([r["held_out_mae"] for r in valid])
        summary = {
            "native_joint_trained": {
                "fusion": args.fusion,
                "mae": {"median": float(np.median(maes)),
                        "mean": float(np.mean(maes)),
                        "std": float(np.std(maes)),
                        "n_pairs": int(len(maes))},
            },
            "baseline_frozen_proxy": {"mae": {"median": 40.53}},
        }
        (out_path.parent / "joint_train_native_full_summary.json").write_text(
            json.dumps(summary, indent=2))
        print(f"\n[DONE] {time.time()-t_start:.0f}s  median MAE = {np.median(maes):.2f} "
              f"(baseline frozen-proxy 40.53)", flush=True)


if __name__ == "__main__":
    main()
