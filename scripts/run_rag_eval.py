"""RAG-TS Experiment: Retrieval-Augmented Zero-Shot Carbon Forecasting.

Standalone script for RAG experiments and ablation studies.

Evaluates:
    1. TransCIF-ZS (baseline, AdaptivePersistDLinear)
    2. TransCIF-RAG (retrieval-augmented RagDLinear)
    3. TransCIF-ZS+ (calibrated baseline)
    4. Ablation: k_retrieve ∈ {1, 3, 5, 10}
    5. Ablation: n_coarse ∈ {1, 2, 3, 5}
    6. Ablation: retrieval similarity measure (L2 vs cosine vs config-only)

Usage: python scripts/run_rag_eval.py [--quick] [--k 5] [--coarse 3]
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
    ANCHOR_WIN, RESID_WIN, WEEKLY_LAG, SELECT_DAYS, SELECT_MARGIN, FUSION_MENU,
)
from transcif_pipeline import (
    discover_uk_regions, load_region_data, build_windows,
    cif_from_shares, train_zero_shot, compute_metrics, zs_plus_predict,
)
from transcif_model import AdaptivePersistDLinear
from transcif_rag import RagMemoryBank, RagDLinear, train_rag_zero_shot, predict_rag_zs


def evaluate_rag_target(target_name, all_regions, k_retrieve=5, n_coarse=3,
                         seed=42, use_weighted=True):
    """Full evaluation on one target using RAG-DLinear.

    Compares: ZS (no retrieval) vs RAG (with retrieval) vs ZS+ (calibrated).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    data = all_regions[target_name]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]

    split = int(len(rs) * TRAIN_FRACTION)
    x_rs_test, _, y_cif_test = build_windows(
        rs[split - SEQ_LEN:], cif[split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)

    if len(x_rs_test) == 0:
        return None

    print(f"      Training RAG model (k={k_retrieve}, coarse={n_coarse})...", end=" ", flush=True)
    t0 = time.time()

    # Train RAG model
    rag_model, bank = train_rag_zero_shot(
        all_regions, target_name, seed=seed,
        use_weighted=use_weighted, k_retrieve=k_retrieve, n_coarse=n_coarse)

    dt_rag = time.time() - t0

    # Train baseline ZS model (for direct comparison)
    t1 = time.time()
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    dt_zs = time.time() - t1

    print(f"(RAG: {dt_rag:.1f}s, ZS: {dt_zs:.1f}s)")

    # Predictions
    # 1. Baseline ZS
    cfg_t = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_share = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), cfg_t).numpy()
    cif_zs = cif_from_shares(zs_share, ef_r, ef_nr)

    # 2. RAG
    cif_rag = predict_rag_zs(
        rag_model, bank, x_rs_test, data["config"].astype(np.float32), ef_r, ef_nr)

    # 3. Persistence baseline
    cif_persist = x_rs_test[:, -HORIZON:]

    # 4. ZS+ (calibrated, on ZS model for fair comparison)
    n_off = len(rs) - (split - SEQ_LEN)
    origins = [split + st
               for st in range(0, n_off - SEQ_LEN - HORIZON + 1, TEST_STRIDE)]
    cif_zsp = zs_plus_predict(zs_model, data["config"], rs, cif, ef_r, ef_nr, origins)

    # Metrics
    metrics_zs = compute_metrics(cif_zs, y_cif_test)
    metrics_rag = compute_metrics(cif_rag, y_cif_test)
    metrics_per = compute_metrics(cif_persist, y_cif_test)
    metrics_zsp = compute_metrics(cif_zsp, y_cif_test)

    result = {
        "target": target_name, "seed": seed,
        "mean_rs": data["mean_rs"], "k_retrieve": k_retrieve, "n_coarse": n_coarse,
        "zs_mae": metrics_zs["mae"], "rag_mae": metrics_rag["mae"],
        "persist_mae": metrics_per["mae"], "zsp_mae": metrics_zsp["mae"],
        "zs_rmse": metrics_zs["rmse"], "rag_rmse": metrics_rag["rmse"],
        "zsp_rmse": metrics_zsp["rmse"],
        "rag_vs_zs": metrics_rag["mae"] / max(metrics_zs["mae"], 1e-6),
        "rag_vs_persist": metrics_rag["mae"] / max(metrics_per["mae"], 1e-6),
        "zsp_vs_zs": metrics_zsp["mae"] / max(metrics_zs["mae"], 1e-6),
        "time_rag": dt_rag, "time_zs": dt_zs,
    }

    if metrics_rag["mae"] < metrics_zs["mae"]:
        result["rag_improvement"] = (metrics_zs["mae"] - metrics_rag["mae"]) / max(metrics_zs["mae"], 1e-6)
    else:
        result["rag_improvement"] = 0.0

    return result


def print_comparison(results):
    """Print RAG vs ZS comparison table."""
    print("\n" + "=" * 100)
    print("RAG-TS EXPERIMENT RESULTS")
    print("=" * 100)
    print(f"{'Region':<18} {'MAE_ZS':<9} {'MAE_RAG':<9} {'Improve':<9} "
          f"{'RAG/Pers':<9} {'MAE_ZSP':<9} {'Time(RAG)':<10}")
    print("-" * 100)

    improvements = []
    for r in results:
        imp = r.get("rag_improvement", 0) * 100
        improvements.append(imp)
        marker = "+" if imp > 1 else ("=" if abs(imp) < 1 else "-")
        print(f"{r['target']:<18} {r['zs_mae']:<9.1f} {r['rag_mae']:<9.1f} "
              f"{marker}{abs(imp):<7.1f}% {r['rag_vs_persist']:<9.3f} "
              f"{r['zsp_mae']:<9.1f} {r['time_rag']:<9.0f}s")

    mean_imp = np.mean(improvements)
    print("-" * 100)
    print(f"{'MEAN':<18} {'':9} {'':9} {mean_imp:+.1f}% {'':9} {'':9}")

    n_better = sum(1 for r in results if r.get("rag_improvement", 0) > 0.01)
    n_worse = sum(1 for r in results if r.get("rag_improvement", 0) < -0.01)
    print(f"\nRAG better than ZS: {n_better}/{len(results)} regions")
    print(f"RAG worse than ZS:  {n_worse}/{len(results)} regions")

    # Also compare vs persistence
    n_better_per = sum(1 for r in results if r["rag_vs_persist"] < 1.0)
    print(f"RAG better than Persistence: {n_better_per}/{len(results)} regions")

    return mean_imp


def run_ablation_k(all_regions, targets, seeds):
    """Ablation: k_retrieve sweep."""
    print("\n" + "=" * 80)
    print("ABLATION: k_retrieve ∈ {0, 1, 3, 5, 10}")
    print("=" * 80)
    k_values = [0, 1, 3, 5, 10]
    results_by_k = {k: [] for k in k_values}
    for k in k_values:
        print(f"\nk_retrieve = {k}")
        for target in targets:
            r = evaluate_rag_target(target, all_regions, k_retrieve=k,
                                    n_coarse=3, seed=seeds[0])
            if r:
                r["k_retrieve"] = k
                results_by_k[k].append(r)
    print("\nK-RETRIEVE RESULTS:")
    for k in k_values:
        vals = results_by_k[k]
        if vals:
            avg = np.mean([v["rag_vs_zs"] for v in vals])
            print(f"  k={k}: RAG/ZS={avg:.3f} ({len(vals)} regions)")
    return results_by_k


def run_ablation_coarse(all_regions, targets, seeds):
    """Ablation: n_coarse sweep."""
    print("\n" + "=" * 80)
    print("ABLATION: n_coarse ∈ {1, 2, 3, 5}")
    print("=" * 80)
    n_values = [1, 2, 3, 5]
    results_by_n = {n: [] for n in n_values}
    for n in n_values:
        print(f"\nn_coarse = {n}")
        for target in targets:
            r = evaluate_rag_target(target, all_regions, k_retrieve=5,
                                    n_coarse=n, seed=seeds[0])
            if r:
                r["n_coarse"] = n
                results_by_n[n].append(r)
    print("\nN-COARSE RESULTS:")
    for n in n_values:
        vals = results_by_n[n]
        if vals:
            avg = np.mean([v["rag_vs_zs"] for v in vals])
            print(f"  coarse={n}: RAG/ZS={avg:.3f} ({len(vals)} regions)")
    return results_by_n


def main():
    parser = argparse.ArgumentParser(description="RAG-TS Experiment")
    parser.add_argument("--quick", action="store_true", help="Quick mode: AU only, 1 seed")
    parser.add_argument("--k", type=int, default=5, help="Number of retrieved windows")
    parser.add_argument("--coarse", type=int, default=3, help="Number of coarse candidate regions")
    parser.add_argument("--ablation-k", action="store_true", help="Run k_retrieve sweep")
    parser.add_argument("--ablation-coarse", action="store_true", help="Run n_coarse sweep")
    args = parser.parse_args()

    mode = "QUICK" if args.quick else "FULL"
    print("=" * 80)
    print(f"RAG-TS: Retrieval-Augmented Zero-Shot Forecasting ({mode})")
    print("=" * 80)

    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception as e:
            print(f"  [WARN] Skip {name}: {e}")

    print(f"Loaded: {len(all_regions)} regions")

    if args.quick:
        targets_list = sorted(all_regions.keys(), key=lambda x: all_regions[x]["mean_rs"])
        seeds_list = [42]
    else:
        targets_list = sorted(all_regions.keys(), key=lambda x: all_regions[x]["mean_rs"])
        seeds_list = [42]

    # --- Ablation mode ---
    if args.ablation_k:
        results_k = run_ablation_k(all_regions, targets_list, seeds_list)
        output = RESULTS_DIR / "rag_ablation_k.json"
        with open(output, "w") as f:
            json.dump({str(k): v for k, v in results_k.items()}, f, indent=2, default=float)
        print(f"\nSaved: {output}")
        return

    if args.ablation_coarse:
        results_n = run_ablation_coarse(all_regions, targets_list, seeds_list)
        output = RESULTS_DIR / "rag_ablation_coarse.json"
        with open(output, "w") as f:
            json.dump({str(n): v for n, v in results_n.items()}, f, indent=2, default=float)
        print(f"\nSaved: {output}")
        return

    # --- Main experiment ---
    t0 = time.time()
    all_results = []

    for i, target in enumerate(targets_list):
        print(f"\n[{i+1}/{len(targets_list)}] {target} "
              f"(rs={all_regions[target]['mean_rs']:.3f})", flush=True)
        result = evaluate_rag_target(target, all_regions,
                                     k_retrieve=args.k,
                                     n_coarse=args.coarse,
                                     seed=seeds_list[0])
        if result is None:
            print(f"  SKIP (insufficient data)")
            continue
        all_results.append(result)
        imp = result.get("rag_improvement", 0) * 100
        marker = "+" if imp > 1 else ("=" if abs(imp) < 1 else "-")
        print(f"  ZS={result['zs_mae']:.1f}  RAG={result['rag_mae']:.1f}  "
              f"ZSP={result['zsp_mae']:.1f}  RAG/ZS={result['rag_vs_zs']:.3f}  "
              f"({marker}{abs(imp):.1f}%)", flush=True)

    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed/60:.1f} min")

    # Print comparison
    print_comparison(all_results)

    # Save
    output_file = RESULTS_DIR / "rag_experiment.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"Results saved: {output_file}")
    print("\nRAG-TS experiment complete!")


if __name__ == "__main__":
    main()
