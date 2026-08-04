"""Debiased-Hier Experiment: Hierarchical Prediction with Physics Consistency.

Evaluates:
    1. TransCIF-ZS (baseline)
    2. Debiased-Hier (hierarchical + consistency loss)
    3. Debiased-Hier (hierarchical only, no consistency) — ablation

Ablations:
    --ablation-consist : sweep λ_consist

Usage:
    python scripts/run_hier_eval.py --quick
    python scripts/run_hier_eval.py --ablation-consist --target VIC1
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_unified_eval import (
    DATA_DIR, RESULTS_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE,
    TRAIN_FRACTION, AU_REGIONS, US_REGIONS, UK_REGIONS,
)
from transcif_pipeline import (
    discover_uk_regions, load_region_data, build_windows,
    cif_from_shares, train_zero_shot, compute_metrics,
)
from transcif_hier import (
    HierDLinear, train_hier, predict_hier_zs, compute_debias_metric,
)


def evaluate_hier_target(target_name, all_regions, seed=42, lambda_consist=0.3):
    """Compare ZS-ERM vs Hier vs Hier (no consistency)."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    data = all_regions[target_name]
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    x_rs_test, _, y_cif_test = build_windows(
        data["rs"][split - SEQ_LEN:], data["cif"][split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)
    if len(x_rs_test) == 0:
        return None

    result = {"target": target_name, "seed": seed}

    # Baseline
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    cfg_t = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_share = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), cfg_t).numpy()
    result["zs_erm"] = compute_metrics(
        cif_from_shares(zs_share, data["ef_r"], data["ef_nr"]), y_cif_test)

    # Hier (full: hierarchical + consistency)
    t0 = time.time()
    hier_model = train_hier(all_regions, target_name, seed=seed,
                             lambda_consist=lambda_consist)
    hier_cif = predict_hier_zs(hier_model, x_rs_test, data["config"],
                                data["ef_r"], data["ef_nr"])
    result["hier"] = compute_metrics(hier_cif, y_cif_test)
    result["hier_time"] = time.time() - t0

    # Hier (no consistency)
    hier_nc = train_hier(all_regions, target_name, seed=seed,
                          lambda_consist=0.0)
    nc_cif = predict_hier_zs(hier_nc, x_rs_test, data["config"],
                              data["ef_r"], data["ef_nr"])
    result["hier_noconsist"] = compute_metrics(nc_cif, y_cif_test)

    # Debias metric
    dm = compute_debias_metric(hier_model, x_rs_test, data["config"],
                                data["ef_r"], data["ef_nr"], y_cif_test)
    result.update(dm)

    result["ratio_vs_zs"] = result["hier"]["mae"] / max(result["zs_erm"]["mae"], 1e-6)
    result["consist_benefit"] = result["hier"]["mae"] / max(result["hier_noconsist"]["mae"], 1e-6)

    return result


def main():
    parser = argparse.ArgumentParser(description="Debiased-Hier Experiments")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--lambda-consist", type=float, default=0.3)
    parser.add_argument("--ablation-consist", action="store_true")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for n in all_configs:
        try:
            all_regions[n] = load_region_data(n, all_configs)
        except Exception as e:
            print(f"  [WARN] {n}: {e}")

    if args.ablation_consist:
        target = args.target or "VIC1"
        print(f"\nλ_consist ablation for {target}")
        for lam in [0.0, 0.01, 0.1, 0.3, 0.5, 1.0, 2.0]:
            r = evaluate_hier_target(target, all_regions, lambda_consist=lam)
            if r:
                print(f"  λ={lam:<6} ZS={r['zs_erm']['mae']:.1f} "
                      f"Hier={r['hier']['mae']:.1f} ×{r['ratio_vs_zs']:.3f} "
                      f"bias_red={r['bias_reduction']:.3f}")
        return

    targets = ["QLD1", "NSW1", "VIC1", "SA1"] if args.quick else sorted(all_regions)
    results = []
    t0 = time.time()

    for target in targets:
        print(f"\n[{target}]", flush=True)
        r = evaluate_hier_target(target, all_regions, lambda_consist=args.lambda_consist)
        if r:
            results.append(r)
            print(f"  ZS={r['zs_erm']['mae']:.1f} "
                  f"Hier={r['hier']['mae']:.1f} (×{r['ratio_vs_zs']:.3f}) "
                  f"NoConsist={r['hier_noconsist']['mae']:.1f} "
                  f"consistΔ={r['consist_benefit']:.3f} "
                  f"bias_red={r['bias_reduction']:.3f}")

    if results:
        ratios = [r["ratio_vs_zs"] for r in results]
        biases = [r["bias_reduction"] for r in results]
        print(f"\n{'='*60}")
        print(f"Debiased-Hier: {len(results)} evals, {time.time()-t0:.0f}s")
        print(f"  median ratio vs ZS: {np.median(ratios):.3f}")
        print(f"  median bias reduction: {np.median(biases):.3f}")
        better = sum(1 for r in ratios if r < 1)
        print(f"  Hier < ZS: {better}/{len(results)} ({100*better/max(len(results),1):.0f}%)")
        print(f"{'='*60}")

        out = args.out or str(RESULTS_DIR / "hier_eval.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w"), indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
