"""Task 3.2 LOO-CV validation: run on real source stacks, report metrics.

Usage:
    .venv/bin/python scripts/experiments/run_fused_five_loo.py --target QLD1 --seed 0

Reports per-fold OOF MAE, in-fold MAE, weight vectors, and the R2 verdict
(OOF/in-fold gap < 20%, weight std < 0.15).
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from transcif.config import AU_REGIONS, UK_REGIONS, US_REGIONS
from transcif.data.loaders import load_region_data
from transcif.models.zeroshot.collector import collect_source_stacks
from transcif.models.zeroshot.fusion import loo_cv_train


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="QLD1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-sources", type=int, default=5,
                    help="number of source regions to use (LOO-CV cost is O(n^2))")
    ap.add_argument("--epochs", type=int, default=100)
    args = ap.parse_args()

    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}
    print(f"[LOAD] {len(all_regions)} regions", flush=True)

    source_names = [n for n in all_regions if n != args.target][:args.n_sources]
    print(f"[COLLECT] gathering stacks from {len(source_names)} sources: "
          f"{source_names}", flush=True)

    t0 = time.time()
    stacks, true_cif, names = collect_source_stacks(
        all_regions, args.target, seed=args.seed, device=None,
        source_names=source_names, progress=True,
    )
    print(f"[COLLECT] {len(stacks)} stacks in {time.time()-t0:.1f}s", flush=True)

    if len(stacks) < 2:
        print("[FAIL] need at least 2 valid sources for LOO-CV")
        sys.exit(1)

    print(f"[LOO] training {len(stacks)} folds x {args.epochs} epochs...",
          flush=True)
    t0 = time.time()
    result = loo_cv_train(stacks, true_cif, names,
                          epochs=args.epochs, seed=args.seed)
    print(f"[LOO] done in {time.time()-t0:.1f}s", flush=True)

    print("\n=== Per-fold results ===")
    print(f"{'fold':>4} {'name':>20} {'OOF MAE':>10} {'in-fold':>10} "
          f"{'weights':>40}")
    for rec in result["loo_per_fold"]:
        w_str = np.array2string(rec["weights"], precision=3, suppress_small=True)
        print(f"{rec['fold']:>4} {rec['name']:>20} {rec['oof_mae']:>10.3f} "
              f"{rec['in_fold_mae']:>10.3f} {w_str:>40}")

    print("\n=== Aggregate ===")
    print(f"OOF MAE mean ± std : {result['oof_mae_mean']:.3f} ± "
          f"{result['oof_mae_std']:.3f}")
    in_fold_mean = float(np.mean([r["in_fold_mae"] for r in result["loo_per_fold"]]))
    gap = abs(result["oof_mae_mean"] - in_fold_mean) / max(in_fold_mean, 1e-6)
    print(f"In-fold MAE mean   : {in_fold_mean:.3f}")
    print(f"OOF / in-fold gap  : {gap:.1%}  (R2 budget: <20%)")

    print(f"\nWeight std per direction:")
    for d, std in enumerate(result["weight_std_per_direction"]):
        flag = "OK" if std < 0.15 else "OVER"
        print(f"  dir {d}: std={std:.4f}  [{flag}]")

    r2_gap_ok = gap < 0.20
    r2_std_ok = (result["weight_std_per_direction"] < 0.15).all()
    verdict = "PASS" if (r2_gap_ok and r2_std_ok) else "FAIL"
    print(f"\n[R2 VERDICT] {verdict}  "
          f"(gap {'OK' if r2_gap_ok else 'OVER'}, "
          f"std {'OK' if r2_std_ok else 'OVER'})")

    os.makedirs("results", exist_ok=True)
    out_path = f"results/fused_five_loo_{args.target}_seed{args.seed}.json"
    summary = {
        "target": args.target,
        "seed": args.seed,
        "n_sources": len(stacks),
        "epochs": args.epochs,
        "oof_mae_mean": result["oof_mae_mean"],
        "oof_mae_std": result["oof_mae_std"],
        "in_fold_mae_mean": in_fold_mean,
        "oof_in_fold_gap": gap,
        "weight_std_per_direction": result["weight_std_per_direction"].tolist(),
        "weight_vectors": result["weight_vectors"].tolist(),
        "per_fold": [
            {"fold": r["fold"], "name": r["name"],
             "oof_mae": r["oof_mae"], "in_fold_mae": r["in_fold_mae"],
             "weights": r["weights"].tolist()}
            for r in result["loo_per_fold"]
        ],
        "r2_verdict": verdict,
    }
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[WRITE] {out_path}")


if __name__ == "__main__":
    main()
