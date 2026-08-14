"""Run Task 5.1: 29-region LORO × 5 seeds full evaluation.

This orchestrator evaluates all five single-direction predictors, their
ZS+ calibrated versions, two fusion methods (equal-weight and BasisMix),
plus a persistence baseline across all regions and seeds.

Outputs:
  results/fused_five_full.json         — 145 rows (29 regions × 5 seeds)
  results/fused_five_full_summary.json — per-method median/mean/std

Usage:
    .venv/bin/python scripts/experiments/run_fused_five_full.py --smoke
    .venv/bin/python scripts/experiments/run_fused_five_full.py --workers 1
"""

import argparse
import json
import os
import sys
import time
from itertools import product
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from transcif.config import (
    AU_REGIONS,
    HORIZON,
    SEQ_LEN,
    TEST_STRIDE,
    TRAIN_FRACTION,
    UK_REGIONS,
    US_REGIONS,
    SEEDS_FULL,
)
from transcif.data.loaders import load_region_data, all_region_configs
from transcif.data.windows import build_windows
from transcif.evaluation.metrics import compute_metrics
from transcif.calibration.zs_plus import zs_plus_predict
from transcif.models.zeroshot.fusion import (
    BasisMixFusion,
    EqualWeightFusion,
    FusionModel,
    basis_mix_loss,
    train_fusion,
)
from scripts.experiments._shared import (
    build_direction_predictors as _build_predictors,
    zs_plus_origins as _zs_plus_origins,
)

DEVICE = "cuda" if torch.cuda.is_available() else None


# ---------------------------------------------------------------------------
# Single-direction evaluation
# ---------------------------------------------------------------------------

def _eval_single_direction(predictors, direction_name, x_test, config,
                           ef_r, ef_nr, rs, cif, y_true):
    """Evaluate a single direction with and without ZS+ calibration."""
    pred_fn = predictors[direction_name]
    cf_base = pred_fn(x_test, config, ef_r, ef_nr)
    metrics_base = compute_metrics(cf_base, y_true)

    # For ZS+, we need to wrap this single direction as a 1-element FusionModel
    # so we can reuse the share_fn mechanism
    from transcif.models.zeroshot.fusion import DIRECTION_ORDER

    class SingleDirectionHead(torch.nn.Module):
        def __init__(self, direction_idx):
            super().__init__()
            self.direction_idx = direction_idx

        def forward(self, cif_stack):
            # Extract just this direction's predictions
            return cif_stack[:, self.direction_idx, :]

    # Create a minimal wrapper that extracts just this direction
    class SingleDirectionFusion:
        def __init__(self, direction_name):
            self.direction_idx = DIRECTION_ORDER.index(direction_name)
            self.head = SingleDirectionHead(self.direction_idx)

        def predict_cif(self, x_rs, config, ef_r, ef_nr):
            # Call the predictor directly and return in the right shape
            pred = pred_fn(x_rs, config, ef_r, ef_nr)
            # pred is (n, HORIZON), return as-is for single direction
            return pred

    single_fusion = SingleDirectionFusion(direction_name)

    # Configure ZS+ for this single direction
    origins = _zs_plus_origins(rs, cif)

    # For single-direction ZS+, we use the predictor's share_fn
    # We need to create a minimal wrapper that implements the share_fn interface
    class SingleDirectionShareWrapper:
        def __init__(self, predictor_fn, config, ef_r, ef_nr):
            self.predictor_fn = predictor_fn
            self.config = config
            self.ef_r = ef_r
            self.ef_nr = ef_nr

        def __call__(self, x_window_np):
            # Predict CIF for this window
            cif_pred = self.predictor_fn(
                x_window_np[None, :], self.config, self.ef_r, self.ef_nr
            ).reshape(-1)

            # Convert to RenewShare using emission factors
            share = (cif_pred - self.ef_nr) / (self.ef_r - self.ef_nr + 1e-8)
            return np.clip(share, 0.0, 1.0)

    share_wrapper = SingleDirectionShareWrapper(pred_fn, config, ef_r, ef_nr)

    cf_plus = zs_plus_predict(
        model=None, config=config, rs=rs, cif=cif,
        ef_r=ef_r, ef_nr=ef_nr, origins=origins,
        share_fn=share_wrapper
    )
    metrics_plus = compute_metrics(cf_plus, y_true)

    return metrics_base, metrics_plus



# ---------------------------------------------------------------------------
# Fusion evaluation
# ---------------------------------------------------------------------------

def _eval_fusion_method(head_model, x_test, config, ef_r, ef_nr, rs, cif, y_true):
    """Evaluate a fusion method with and without ZS+ calibration."""
    # Base fusion (no ZS+)
    cf_base = head_model.predict_cif(x_test, config, ef_r, ef_nr)
    metrics_base = compute_metrics(cf_base, y_true)

    # ZS+ calibrated fusion
    head_model.configure_for_target(config, ef_r, ef_nr)
    origins = _zs_plus_origins(rs, cif)
    cf_plus = zs_plus_predict(
        model=None, config=config, rs=rs, cif=cif,
        ef_r=ef_r, ef_nr=ef_nr, origins=origins,
        share_fn=head_model.share_fn
    )
    metrics_plus = compute_metrics(cf_plus, y_true)

    return metrics_base, metrics_plus


def _train_basismix(src_stacks, src_true, predictors, seed,
                    epochs=300, lr=1e-2, l2=1e-4,
                    lambda_entropy=1e-2, lambda_diversity=1e-2):
    """Train BasisMixFusion with regularization (reuse from variants script)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    head = BasisMixFusion()
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=l2)

    X = np.concatenate(src_stacks, axis=0).astype(np.float32)
    Y = np.concatenate(src_true, axis=0).astype(np.float32)
    X_t = torch.as_tensor(X, dtype=torch.float32)
    Y_t = torch.as_tensor(Y, dtype=torch.float32)

    head.train()
    for _ in range(epochs):
        opt.zero_grad()
        loss = basis_mix_loss(head, X_t, Y_t,
                              lambda_l2=l2,
                              lambda_entropy=lambda_entropy,
                              lambda_diversity=lambda_diversity)
        loss.backward()
        opt.step()
    head.eval()
    return FusionModel(head, predictors=predictors)


# ---------------------------------------------------------------------------
# Persistence baseline
# ---------------------------------------------------------------------------

def _eval_persistence(x_test, y_true, ef_r, ef_nr):
    """Persistence baseline: predict the last HORIZON rs values converted to CIF.

    Standard "lag-24h" persistence used across the codebase (see
    scripts/benchmark/temporal_ood.py:142, run_supervised_baselines_v2.py).
    Captures diurnal cycle: hour t+24 ≈ hour t. The earlier implementation
    compared rs (renewable share, 0-1) directly to y_true (CIF), inflating
    MAE by ~10x.
    """
    last_window_rs = x_test[:, -HORIZON:].astype(np.float32)
    pred = last_window_rs * ef_r + (1.0 - last_window_rs) * ef_nr
    return compute_metrics(pred, y_true)


# ---------------------------------------------------------------------------
# Per-target evaluation
# ---------------------------------------------------------------------------

def evaluate_target(target, all_regions, seed, src_limit, output_json,
                    resume=False):
    """Evaluate all methods for one (region, seed) pair.

    Args:
        target: Target region name
        all_regions: Dict of all loaded region data
        seed: Random seed
        src_limit: Limit source regions (for faster smoke testing)
        output_json: Path to output JSON file
        resume: If True, skip if this row already exists in output

    Returns:
        dict with results for this (region, seed) pair, or None if skipped
    """
    # Check resume condition
    if resume and output_json.exists():
        try:
            with open(output_json) as f:
                existing = json.load(f)
            for row in existing:
                if row.get("target") == target and row.get("seed") == seed:
                    print(f"  [SKIP] {target} seed {seed} already in output")
                    return None
        except (json.JSONDecodeError, FileNotFoundError):
            pass  # File corrupt or missing, proceed normally

    data = all_regions[target]
    config = data["config"].astype(np.float32)
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    rs, cif = data["rs"], data["cif"]

    split = int(len(rs) * TRAIN_FRACTION)
    x_test, _, y_true = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        seq_len=SEQ_LEN, horizon=HORIZON, stride=TEST_STRIDE,
    )

    # Build a small donor pool: target + first ``src_limit`` other regions.
    # Train functions (rag/phys_irm/causal/icl/hier) iterate ``all_regions.items()``
    # for the auxiliary pool, so passing the full 29-region dict makes each train
    # call ~24x slower. With src_limit=3 the donor pool is 4 regions, matching
    # the Wave 3 evaluation that finished in ~22 min for 4 AU targets.
    src_names = [n for n in all_regions if n != target][:src_limit]
    small_regions = {target: all_regions[target]}
    for n in src_names:
        small_regions[n] = all_regions[n]
    print(f"  [donor] pool size: {len(small_regions)} regions "
          f"(target + {len(src_names)} sources)", flush=True)

    print(f"  [predictors] training 5 directions on {target}...", flush=True)
    t0 = time.time()
    predictors = _build_predictors(small_regions, target, seed, DEVICE)
    print(f"    done in {time.time()-t0:.1f}s", flush=True)

    print(f"  [collect] gathering source stacks (src_limit={src_limit})...",
          flush=True)
    t0 = time.time()
    from transcif.models.zeroshot.collector import collect_source_stacks
    src_stacks, src_true, src_names_used = collect_source_stacks(
        small_regions, target, seed=seed, device=DEVICE,
        source_names=src_names, progress=False,
    )
    print(f"    done in {time.time()-t0:.1f}s ({len(src_stacks)} sources)",
          flush=True)

    if not src_stacks:
        print(f"  [SKIP] {target} has no valid sources")
        return None

    # Result row
    row = {"target": target, "seed": seed}

    # Evaluate single directions
    print(f"  [single] evaluating 5 directions...", flush=True)
    for direction in ["rag", "phys", "causal", "icl", "hier"]:
        print(f"    {direction}...", flush=True)
        metrics_base, metrics_plus = _eval_single_direction(
            predictors, direction, x_test, config, ef_r, ef_nr, rs, cif, y_true
        )
        row[direction] = metrics_base
        row[f"{direction}_plus"] = metrics_plus
        print(f"      MAE={metrics_base['mae']:.3f}, "
              f"MAE+={metrics_plus['mae']:.3f}", flush=True)

    # Evaluate equal-weight fusion
    print(f"  [equal] evaluating equal-weight fusion...", flush=True)
    equal_head = EqualWeightFusion()
    equal_model = FusionModel(equal_head, predictors=predictors)
    metrics_eq, metrics_eq_plus = _eval_fusion_method(
        equal_model, x_test, config, ef_r, ef_nr, rs, cif, y_true
    )
    row["equal"] = metrics_eq
    row["equal_plus"] = metrics_eq_plus
    print(f"    MAE={metrics_eq['mae']:.3f}, MAE+={metrics_eq_plus['mae']:.3f}",
          flush=True)

    # Evaluate BasisMix fusion
    print(f"  [basismix] evaluating BasisMix fusion...", flush=True)
    bm_model = _train_basismix(src_stacks, src_true, predictors, seed)
    metrics_bm, metrics_bm_plus = _eval_fusion_method(
        bm_model, x_test, config, ef_r, ef_nr, rs, cif, y_true
    )
    row["basismix"] = metrics_bm
    row["basismix_plus"] = metrics_bm_plus
    print(f"    MAE={metrics_bm['mae']:.3f}, MAE+={metrics_bm_plus['mae']:.3f}",
          flush=True)

    # Evaluate persistence baseline
    print(f"  [persistence] evaluating persistence baseline...", flush=True)
    row["persistence"] = _eval_persistence(x_test, y_true, ef_r, ef_nr)
    print(f"    MAE={row['persistence']['mae']:.3f}", flush=True)

    return row


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------

def compute_summary_statistics(results):
    """Compute per-method median, mean, and std across all (region, seed) rows.

    Args:
        results: List of result rows from evaluate_target

    Returns:
        dict with per-method statistics
    """
    methods = [
        "rag", "rag_plus",
        "phys", "phys_plus",
        "causal", "causal_plus",
        "icl", "icl_plus",
        "hier", "hier_plus",
        "equal", "equal_plus",
        "basismix", "basismix_plus",
        "persistence"
    ]

    summary = {}

    for method in methods:
        maes = [r[method]["mae"] for r in results if method in r]
        rmses = [r[method]["rmse"] for r in results if method in r]
        smapes = [r[method]["smape"] for r in results if method in r]

        if not maes:
            continue

        summary[method] = {
            "mae": {
                "median": float(np.median(maes)),
                "mean": float(np.mean(maes)),
                "std": float(np.std(maes)),
            },
            "rmse": {
                "median": float(np.median(rmses)),
                "mean": float(np.mean(rmses)),
                "std": float(np.std(rmses)),
            },
            "smape": {
                "median": float(np.median(smapes)),
                "mean": float(np.mean(smapes)),
                "std": float(np.std(smapes)),
            },
        }

    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Task 5.1: Full 29-region × 5-seed evaluation"
    )
    ap.add_argument("--regions", nargs="+",
                    default=None,
                    help="Target regions (default: all 29 regions)")
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS_FULL,
                    help="Seeds to evaluate (default: 0 1 2 3 4)")
    ap.add_argument("--src-limit", type=int, default=3,
                    help="Limit source regions for faster smoke testing")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke test: 2 regions × 1 seed, write to *_smoke.json")
    ap.add_argument("--resume", action="store_true",
                    help="Skip (region, seed) pairs already in output JSON")
    ap.add_argument("--workers", type=int, default=1,
                    help="Number of parallel workers (default: 1 for safety)")
    ap.add_argument("--out", default="results/fused_five_full.json",
                    help="Output JSON path")

    args = ap.parse_args()

    # Smoke test overrides
    if args.smoke:
        args.regions = ["QLD1", "NSW1"]  # 2 AU regions
        args.seeds = [0]  # 1 seed
        args.out = "results/fused_five_full_smoke.json"
        args.src_limit = 1
        print("[SMOKE] Running smoke test: 2 regions × 1 seed")

    # Load regions
    print("[LOAD] Loading region configs...", flush=True)
    all_configs = all_region_configs()
    print(f"[LOAD] {len(all_configs)} regions found", flush=True)

    target_regions = args.regions if args.regions else list(all_configs.keys())
    print(f"[TARGET] {len(target_regions)} target regions", flush=True)
    print(f"[SEEDS] {args.seeds}", flush=True)

    # Load all region data once
    print("[LOAD] Loading region data...", flush=True)
    t0 = time.time()
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs}
    print(f"[LOAD] done in {time.time()-t0:.1f}s", flush=True)

    # Create output directory
    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize output file (for resume mode)
    if args.resume and not output_path.exists():
        with open(output_path, "w") as f:
            json.dump([], f)

    # Generate all (region, seed) pairs
    pairs = list(product(target_regions, args.seeds))
    total_pairs = len(pairs)
    print(f"[PLAN] {total_pairs} (region, seed) pairs to evaluate", flush=True)

    # Estimate runtime
    est_min_per_pair = 6.0
    est_total_hours = (total_pairs * est_min_per_pair) / 60.0
    print(f"[EST] Runtime estimate: ~{est_total_hours:.1f} hours "
          f"({est_min_per_pair} min per pair)", flush=True)

    # Evaluate all pairs
    results = []
    completed = 0
    start_time = time.time()

    for i, (target, seed) in enumerate(pairs, 1):
        print(f"\n[{i}/{total_pairs}] {target} seed {seed}", flush=True)

        # Check if region exists in loaded data
        if target not in all_regions:
            print(f"  [SKIP] {target} not in loaded regions")
            continue

        t0_pair = time.time()
        row = evaluate_target(
            target, all_regions, seed, args.src_limit,
            output_path, resume=args.resume
        )

        if row is not None:
            results.append(row)
            completed += 1

            # Incremental write (append to file)
            if args.resume and output_path.exists():
                try:
                    with open(output_path) as f:
                        existing = json.load(f)
                    existing.append(row)
                    with open(output_path, "w") as f:
                        json.dump(existing, f, indent=2)
                except (json.JSONDecodeError, FileNotFoundError):
                    # File corrupt, overwrite with current results
                    with open(output_path, "w") as f:
                        json.dump([row], f, indent=2)
            else:
                # First write, create new file
                with open(output_path, "w") as f:
                    json.dump([row], f, indent=2)

            pair_time = time.time() - t0_pair
            print(f"  [DONE] {target} seed {seed} in {pair_time:.1f}s", flush=True)

    # Final write (ensure all results are in file)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n[SUMMARY] {completed} pairs completed in {elapsed/60:.1f} min",
          flush=True)
    print(f"[WRITE] {output_path}", flush=True)

    # Compute and write summary statistics
    print("[SUMMARY] Computing per-method statistics...", flush=True)
    summary = compute_summary_statistics(results)

    summary_path = output_path.parent / (
        output_path.stem.replace("_full", "_full_summary") + ".json"
    )
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[WRITE] {summary_path}", flush=True)

    # Print summary table
    print("\n=== MAE Summary (median across all regions × seeds) ===")
    print(f"{'Method':<20} {'Median MAE':<12} {'Mean±Std':<20}")
    print("-" * 52)
    for method in ["rag", "phys", "causal", "icl", "hier",
                  "equal", "basismix", "persistence"]:
        if method in summary:
            mae_stats = summary[method]["mae"]
            mean_std = f"{mae_stats['mean']:.2f}±{mae_stats['std']:.2f}"
            print(f"{method:<20} {mae_stats['median']:>10.3f}  {mean_std:<20}")

    print("\n=== ZS+ Calibrated Methods ===")
    for method in ["rag_plus", "phys_plus", "causal_plus", "icl_plus",
                  "hier_plus", "equal_plus", "basismix_plus"]:
        if method in summary:
            mae_stats = summary[method]["mae"]
            mean_std = f"{mae_stats['mean']:.2f}±{mae_stats['std']:.2f}"
            print(f"{method:<20} {mae_stats['median']:>10.3f}  {mean_std:<20}")

    print("\n[DONE] Task 5.1 complete. For full 14.5h run, use:")
    print(f"  .venv/bin/python {sys.argv[0]} --workers 1")


if __name__ == "__main__":
    main()
