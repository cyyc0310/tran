"""Causal-ZS Experiment: Domain Disentanglement + Counterfactual Augmentation.

Compact evaluation script. Compares:
    1. TransCIF-ZS (baseline)
    2. Causal-ZS (VAE disentanglement + counterfactual augmentation)
    3. Causal-ZS (VAE only, no counterfactual augmentation) — ablation
    4. TransCIF-ZS+ (calibrated baseline)

Ablations:
    --ablation-beta : sweep beta_kl
    --ablation-cf   : sweep lambda_cf

Usage:
    python scripts/run_causal_eval.py --quick
    python scripts/run_causal_eval.py
    python scripts/run_causal_eval.py --ablation-beta --target VIC1
"""

import argparse
import json
import random
import time

import numpy as np
import torch

from transcif.config import (
    DATA_DIR, RESULTS_DIR, SEQ_LEN, HORIZON, TRAIN_STRIDE, TEST_STRIDE,
    TRAIN_FRACTION, AU_REGIONS, US_REGIONS, UK_REGIONS,
)
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.data.windows import build_windows
from transcif.physics.decompose import cif_from_shares
from transcif.models.zeroshot.base_zs import (
    train_zero_shot, compute_metrics, zs_plus_predict,
)
from transcif.models.zeroshot.causal import (
    CausalDomainVAE, train_causal_zero_shot, predict_causal_zs,
    disentanglement_quality,
)


def evaluate_causal_target(target_name, all_regions, seed=42,
                            beta_kl=0.01, lambda_cf=0.3):
    """Compare ZS-ERM vs Causal-ZS vs Causal-ZS (no CF)."""
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
    zs_cif = cif_from_shares(zs_share, data["ef_r"], data["ef_nr"])
    result["zs_erm"] = compute_metrics(zs_cif, y_cif_test)

    # Causal-ZS (full: VAE + CF)
    t0 = time.time()
    causal_model, _ = train_causal_zero_shot(
        all_regions, target_name, seed=seed, beta_kl=beta_kl, lambda_cf=lambda_cf)
    causal_cif = predict_causal_zs(causal_model, x_rs_test, data["config"],
                                    data["ef_r"], data["ef_nr"])
    result["causal_zs"] = compute_metrics(causal_cif, y_cif_test)
    result["causal_time"] = time.time() - t0

    # Causal-ZS (no CF)
    causal_nocf, _ = train_causal_zero_shot(
        all_regions, target_name, seed=seed, beta_kl=beta_kl, lambda_cf=0.0)
    nocf_cif = predict_causal_zs(causal_nocf, x_rs_test, data["config"],
                                  data["ef_r"], data["ef_nr"])
    result["causal_nocf"] = compute_metrics(nocf_cif, y_cif_test)

    # Disentanglement quality
    dq = disentanglement_quality(
        causal_model,
        torch.tensor(x_rs_test[:min(100, len(x_rs_test))], dtype=torch.float32),
        torch.tensor(data["config"]).unsqueeze(0).expand(min(100, len(x_rs_test)), -1))
    result["disentangle_corr"] = dq

    # ZS+
    origins = [split + st for st in range(
        0, len(data["cif"][split - SEQ_LEN:]) - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]
    zsp_pred = zs_plus_predict(zs_model, data["config"],
                               data["rs"], data["cif"], data["ef_r"], data["ef_nr"], origins)
    result["zs_plus"] = compute_metrics(zsp_pred, y_cif_test)

    result["ratio_vs_zs"] = result["causal_zs"]["mae"] / max(result["zs_erm"]["mae"], 1e-6)
    result["cf_benefit"] = result["causal_zs"]["mae"] / max(result["causal_nocf"]["mae"], 1e-6)

    return result


def main():
    parser = argparse.ArgumentParser(description="Causal-ZS Experiments")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--beta-kl", type=float, default=0.01)
    parser.add_argument("--lambda-cf", type=float, default=0.3)
    parser.add_argument("--ablation-beta", action="store_true")
    parser.add_argument("--target", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {n: load_region_data(n, all_configs) for n in all_configs
                   if n in all_configs}
    all_regions = {k: v for k, v in all_regions.items() if v is not None}

    if args.ablation_beta:
        target = args.target or "VIC1"
        print(f"\nbeta_kl ablation for {target}")
        for b in [0.0, 0.001, 0.01, 0.05, 0.1, 0.5]:
            r = evaluate_causal_target(target, all_regions, beta_kl=b)
            if r:
                print(f"  β={b:<6} ZS={r['zs_erm']['mae']:.1f} Causal={r['causal_zs']['mae']:.1f} "
                      f"×{r['ratio_vs_zs']:.3f} disent={r['disentangle_corr']:.3f}")
        return

    seeds = [0] if args.quick else list(range(args.seeds))
    targets = ["QLD1", "NSW1", "VIC1", "SA1"] if args.quick else sorted(all_regions)

    results = []
    t0 = time.time()
    for target in targets:
        print(f"\n[{target}]", flush=True)
        for seed in seeds:
            r = evaluate_causal_target(target, all_regions, seed=seed,
                                        beta_kl=args.beta_kl, lambda_cf=args.lambda_cf)
            if r:
                results.append(r)
                print(f"  s{seed}: ZS={r['zs_erm']['mae']:.1f} "
                      f"Causal={r['causal_zs']['mae']:.1f} (×{r['ratio_vs_zs']:.3f}) "
                      f"NoCF={r['causal_nocf']['mae']:.1f} CFΔ={r['cf_benefit']:.3f} "
                      f"disent={r['disentangle_corr']:.3f}")

    if results:
        ratios = [r["ratio_vs_zs"] for r in results]
        dqs = [r["disentangle_corr"] for r in results]
        print(f"\n{'='*60}")
        print(f"Causal-ZS: {len(results)} evals, {time.time()-t0:.0f}s")
        print(f"  median ratio vs ZS: {np.median(ratios):.3f}")
        print(f"  median disentangle corr: {np.median(dqs):.3f}")
        better = sum(1 for r in ratios if r < 1)
        print(f"  Causal < ZS: {better}/{len(results)} ({100*better/max(len(results),1):.0f}%)")
        print(f"{'='*60}")

        out = args.out or str(RESULTS_DIR / "causal_eval.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w"), indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
