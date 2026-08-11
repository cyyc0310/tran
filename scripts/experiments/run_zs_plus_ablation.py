"""ZS+ branch ablation: is the model contributing, or are persistence branches
doing all the work?

Hypothesis (from zs_plus.py:49-72): 4 of 6 ZS+ branches are pure persistence
or lag baselines (branches 1, 2, 3, 4). Backtesting picks the best fusion per
target. If the model is weak, the fusion down-weights it and ZS+ becomes a
tuned persistence baseline. This would explain why all 7 ZS+ methods in
``results/fused_five_full_summary.json`` collapse to MAE 46.5-47.0 regardless
of base direction.

This script monkey-patches ``FUSION_MENU`` and re-runs ZS+ on each direction
with several restricted branch sets. If MAE diverges across directions when
persistence branches are removed, the hypothesis is confirmed.

Variants:
  DEFAULT      : Original FUSION_MENU (branches 0/1/3, 0/1/3/4, 0/1).
  MODEL_DELTA  : Branches (0,) only — model output + delta anchor.
  MODEL_RAW    : Branches (5,) only — model output, no delta, no anchoring.
  MODEL_BOTH   : Branches (0, 5) — model with and without delta.
  LAG_ONLY     : Branches (1,) only — daily lag (control, no model).

Usage:
    .venv/bin/python scripts/experiments/run_zs_plus_ablation.py
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from transcif.config import (
    AU_REGIONS,
    HORIZON,
    SEQ_LEN,
    TEST_STRIDE,
    TRAIN_FRACTION,
)
from transcif.data.loaders import load_region_data, all_region_configs
from transcif.data.windows import build_windows
from transcif.evaluation.metrics import compute_metrics
from transcif.calibration import zs_plus as zs_module
from transcif.calibration.zs_plus import zs_plus_predict
from transcif.models.zeroshot.rag import train_rag_zero_shot, predict_rag_zs
from transcif.models.zeroshot.phys_irm import train_phys_irm, predict_phys_irm
from transcif.models.zeroshot.causal import train_causal_zero_shot, predict_causal_zs
from transcif.models.zeroshot.icl import train_icl, predict_icl_zs
from transcif.models.zeroshot.hier import train_hier, predict_hier_zs

DEVICE = "cuda" if torch.cuda.is_available() else None


FUSION_VARIANTS = {
    "DEFAULT": (
        dict(branches=(0, 1, 3), gamma=2.5, k_backtest=28),
        dict(branches=(0, 1, 3, 4), gamma=2.5, k_backtest=28),
        dict(branches=(0, 1), gamma=2.0, k_backtest=7),
    ),
    "MODEL_DELTA": (
        dict(branches=(0,), gamma=2.0, k_backtest=7),
    ),
    "MODEL_RAW": (
        dict(branches=(5,), gamma=2.0, k_backtest=7),
    ),
    "MODEL_BOTH": (
        dict(branches=(0, 5), gamma=2.0, k_backtest=7),
    ),
    "LAG_ONLY": (
        dict(branches=(1,), gamma=2.0, k_backtest=7),
    ),
}


def build_predictors(small_regions, target, seed):
    predictors = {}
    m, bank = train_rag_zero_shot(small_regions, target, seed=seed, device=DEVICE)
    predictors["rag"] = lambda x, cfg, ef_r, ef_nr, m=m, b=bank: predict_rag_zs(
        m, b, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
    m, _ = train_phys_irm(small_regions, target, seed=seed, gamma_irm=0.1,
                          lambda_cif=0.5, device=DEVICE)
    predictors["phys"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_phys_irm(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
    m, _ = train_causal_zero_shot(small_regions, target, seed=seed, device=DEVICE)
    predictors["causal"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_causal_zs(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
    m = train_icl(small_regions, target, seed=seed, device=DEVICE)
    predictors["icl"] = lambda x, cfg, ef_r, ef_nr, m=m, r=small_regions, t=target: (
        predict_icl_zs(m, r, t, x.astype(np.float32), ef_r, ef_nr))
    m = train_hier(small_regions, target, seed=seed, device=DEVICE)
    predictors["hier"] = lambda x, cfg, ef_r, ef_nr, m=m: predict_hier_zs(
        m, x.astype(np.float32), cfg.astype(np.float32), ef_r, ef_nr)
    return predictors


def zs_plus_origins(rs, cif):
    split = int(len(rs) * TRAIN_FRACTION)
    return [split + st for st in range(0, len(cif) - split - HORIZON + 1, TEST_STRIDE)]


class ShareWrapper:
    """Wrap a direction's predict_fn into the share_fn interface for zs_plus_predict."""

    def __init__(self, pred_fn, config, ef_r, ef_nr):
        self.pred_fn = pred_fn
        self.config = config
        self.ef_r = ef_r
        self.ef_nr = ef_nr

    def __call__(self, x_window_np):
        cif_pred = self.pred_fn(
            x_window_np[None, :], self.config, self.ef_r, self.ef_nr
        ).reshape(-1)
        share = (cif_pred - self.ef_nr) / (self.ef_r - self.ef_nr + 1e-8)
        return np.clip(share, 0.0, 1.0)


def evaluate_region(target, all_regions, seed, src_limit):
    """Train 5 directions on target, return MAE per (direction, fusion_variant)."""
    data = all_regions[target]
    config = data["config"].astype(np.float32)
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    rs, cif = data["rs"], data["cif"]

    split = int(len(rs) * TRAIN_FRACTION)
    x_test, _, y_true = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE,
    )

    src_names = [n for n in all_regions if n != target][:src_limit]
    small_regions = {target: all_regions[target]}
    for n in src_names:
        small_regions[n] = all_regions[n]

    print(f"  [train] training 5 directions on {target}...", flush=True)
    t0 = time.time()
    predictors = build_predictors(small_regions, target, seed)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    origins = zs_plus_origins(rs, cif)

    results = {}
    for direction in ["rag", "phys", "causal", "icl", "hier"]:
        pred_fn = predictors[direction]
        share_fn = ShareWrapper(pred_fn, config, ef_r, ef_nr)
        results[direction] = {}

        for variant_name, menu in FUSION_VARIANTS.items():
            zs_module.FUSION_MENU = menu
            try:
                cf = zs_plus_predict(
                    model=None, config=config, rs=rs, cif=cif,
                    ef_r=ef_r, ef_nr=ef_nr, origins=origins,
                    share_fn=share_fn,
                )
                m = compute_metrics(cf, y_true)
                results[direction][variant_name] = m["mae"]
            except Exception as e:
                results[direction][variant_name] = float("nan")
                print(f"    [WARN] {direction}/{variant_name}: {e}", flush=True)
        print(f"    {direction}: " + " ".join(
            f"{v}={results[direction][v]:.2f}" for v in FUSION_VARIANTS
        ), flush=True)

    # Also evaluate persistence baseline (no ZS+)
    last_window_rs = x_test[:, -HORIZON:].astype(np.float32)
    pred_persistence = last_window_rs * ef_r + (1.0 - last_window_rs) * ef_nr
    results["persistence"] = compute_metrics(pred_persistence, y_true)["mae"]

    # Restore default
    zs_module.FUSION_MENU = FUSION_VARIANTS["DEFAULT"]
    return results


def main():
    ap = argparse.ArgumentParser(description="ZS+ branch ablation")
    ap.add_argument("--regions", nargs="+", default=["QLD1", "NSW1", "VIC1", "SA1"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--src-limit", type=int, default=3)
    ap.add_argument("--out", default="results/zs_plus_ablation.json")
    args = ap.parse_args()

    print(f"[LOAD] Loading region configs...", flush=True)
    all_configs = all_region_configs()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}
    print(f"[LOAD] {len(all_regions)} regions loaded", flush=True)

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for target in args.regions:
        print(f"\n=== {target} seed {args.seed} ===", flush=True)
        all_results[target] = evaluate_region(
            target, all_regions, args.seed, args.src_limit
        )

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[WRITE] {output_path}", flush=True)

    # Summary table
    print("\n=== MAE by direction × fusion variant ===")
    header = f"{'Direction':<10}" + "".join(f"{v:<13}" for v in FUSION_VARIANTS) + "persistence"
    print(header)
    print("-" * len(header))
    for direction in ["rag", "phys", "causal", "icl", "hier"]:
        row = f"{direction:<10}"
        for variant_name in FUSION_VARIANTS:
            vals = [all_results[t].get(direction, {}).get(variant_name, float("nan"))
                    for t in args.regions]
            median = float(np.median([v for v in vals if not np.isnan(v)])) if vals else float("nan")
            row += f"{median:<13.3f}"
        persistence_vals = [all_results[t].get("persistence", float("nan")) for t in args.regions]
        persistence_med = float(np.median(persistence_vals))
        row += f"{persistence_med:.3f}"
        print(row)

    # Diagnostic: per-variant spread across directions
    print("\n=== Cross-direction spread (max - min MAE) ===")
    print("(Large spread = ZS+ is using the model. Small spread = ZS+ ignores model.)")
    for variant_name in FUSION_VARIANTS:
        medians = []
        for direction in ["rag", "phys", "causal", "icl", "hier"]:
            vals = [all_results[t].get(direction, {}).get(variant_name, float("nan"))
                    for t in args.regions]
            medians.append(float(np.median([v for v in vals if not np.isnan(v)])))
        spread = max(medians) - min(medians)
        print(f"  {variant_name:<12} spread = {spread:.2f}  "
              f"(range {min(medians):.2f} - {max(medians):.2f})")


if __name__ == "__main__":
    main()
