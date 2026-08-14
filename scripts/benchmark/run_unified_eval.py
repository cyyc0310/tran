"""Phase 1.3: Unified Evaluation Protocol (LORO, 5-seed, 29 regions).

This is the DEFINITIVE experiment for the paper. All methods evaluated under
identical conditions with statistical rigor.

Protocol:
- LORO: Leave-One-Region-Out (target has NO training data)
- 5 random seeds for variance estimation (3 seeds in --quick mode)
- Metrics: MAE, RMSE, sMAPE (point prediction)
- Time split: first 80% train, last 20% test
- All 29 regions: 4 AU + 17 UK + 8 US

Model implementations and pipeline utilities live in the ``transcif`` package
(src/transcif). This script is the DEFINITIVE experiment entry point.

Usage: python scripts/run_unified_eval.py [--quick]
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from transcif.config import AU_REGIONS, US_REGIONS, UK_REGIONS, SEEDS_FULL, SEEDS_QUICK
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.models.zeroshot.base_zs import evaluate_target

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _run_long_mode(use_rag=False, use_phys_irm=False,
                    use_causal=False, use_icl=False, use_hier=False):
    """Resume-safe LORO over all 29 regions with 5 seeds."""
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception as e:
            print(f"  [WARN] Skip {name}: {e}")

    seeds = SEEDS_FULL
    targets = sorted(all_regions.keys(), key=lambda x: all_regions[x]["mean_rs"])
    run_loop(all_regions, seeds, targets, "full",
             use_rag=use_rag, use_phys_irm=use_phys_irm,
             use_causal=use_causal, use_icl=use_icl, use_hier=use_hier)


def _run_quick_mode(use_rag=False, use_phys_irm=False,
                    use_causal=False, use_icl=False, use_hier=False):
    """Quick mode: 3 seeds, AU regions only."""
    all_configs = {**AU_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception as e:
            print(f"  [WARN] Skip {name}: {e}")

    seeds = SEEDS_QUICK
    targets = ["QLD1", "NSW1", "VIC1", "SA1"]
    run_loop(all_regions, seeds, targets, "quick",
             use_rag=use_rag, use_phys_irm=use_phys_irm,
             use_causal=use_causal, use_icl=use_icl, use_hier=use_hier)


def run_loop(all_regions, seeds, targets, tag, use_rag=False, use_phys_irm=False,
             use_causal=False, use_icl=False, use_hier=False):
    """Core LORO loop with resume support.

    Args:
        use_rag : if True, also evaluate RagDLinear
        use_phys_irm : if True, also evaluate PhysIRM
        use_causal : if True, also evaluate CausalDomainVAE
        use_icl : if True, also evaluate ICTransformer
        use_hier : if True, also evaluate HierDLinear
    """
    print(f"Loaded: {len(all_regions)} regions")
    print(f"Targets: {len(targets)} regions, Seeds: {seeds}")
    extras = []
    for flag, name in [("rag", use_rag), ("phys-irm", use_phys_irm),
                       ("causal", use_causal), ("icl", use_icl), ("hier", use_hier)]:
        if name:
            extras.append(flag)
    extra_str = " +" + "+".join(extras) if extras else ""
    print(f"Total evaluations: {len(targets) * len(seeds)}{extra_str}")
    t0 = time.time()

    partial_file = RESULTS_DIR / f"unified_eval_{tag}.partial.json"
    all_results = []
    done_targets = set()
    if partial_file.exists():
        with open(partial_file) as f:
            all_results = json.load(f)
        for t in {r["target"] for r in all_results}:
            t_seeds = {r["seed"] for r in all_results if r["target"] == t}
            if set(seeds) <= t_seeds:
                done_targets.add(t)
        print(f"Resuming: {len(done_targets)} targets already complete")
        all_results = [r for r in all_results if r["target"] in done_targets]

    for i, target in enumerate(targets):
        print(f"\n[{i+1}/{len(targets)}] {target} "
              f"(rs={all_regions[target]['mean_rs']:.3f})", flush=True)
        if target in done_targets:
            print("  (cached from partial results)", flush=True)
            continue
        for seed in seeds:
            r = evaluate_target(target, all_regions, seed=seed,
                                 use_rag=use_rag, use_phys_irm=use_phys_irm,
                                 use_causal=use_causal, use_icl=use_icl,
                                 use_hier=use_hier)
            if r is None:
                continue
            all_results.append(r)
            rag_str = ""
            phys_str = ""
            causal_str = ""
            icl_str = ""
            hier_str = ""
            if use_rag and r.get("transcif_rag"):
                rag_str = (f" RAG={r['transcif_rag']['mae']:.1f} "
                           f"r_rag={r['ratio_rag_vs_zs']:.3f}")
            if use_phys_irm and r.get("transcif_phys_irm"):
                phys_str = (f" PhysIRM={r['transcif_phys_irm']['mae']:.1f} "
                            f"r_irm={r['ratio_phys_irm_vs_zs']:.3f}")
            if use_causal and r.get("transcif_causal"):
                causal_str = (f" Causal={r['transcif_causal']['mae']:.1f} "
                              f"r_c={r['ratio_causal_vs_zs']:.3f}")
            if use_icl and r.get("transcif_icl"):
                icl_str = (f" ICL={r['transcif_icl']['mae']:.1f} "
                           f"r_i={r['ratio_icl_vs_zs']:.3f}")
            if use_hier and r.get("transcif_hier"):
                hier_str = (f" Hier={r['transcif_hier']['mae']:.1f} "
                            f"r_h={r['ratio_hier_vs_zs']:.3f}")
            print(f"  s{seed}: Persist={r['persistence']['mae']:.1f} "
                  f"PatchTST={r['patchtst_sup']['mae']:.1f} "
                  f"TransCIF={r['transcif_zs']['mae']:.1f} "
                  f"ZS+={r['transcif_zs_plus']['mae']:.1f} "
                  f"ratio={r['ratio_vs_patchtst']:.3f} "
                  f"ratio+={r['ratio_plus_vs_patchtst']:.3f}"
                  f"{rag_str}{phys_str}{causal_str}{icl_str}{hier_str}", flush=True)

        with open(partial_file, "w") as f:
            json.dump(all_results, f)

    elapsed = time.time() - t0
    print(f"\n\nTotal time: {elapsed/60:.1f} min")

    # Save final results
    results_file = RESULTS_DIR / f"unified_eval_{tag}.json"
    with open(results_file, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved: {results_file}")
    if partial_file.exists():
        partial_file.unlink()

    print_summary(all_results, targets)


def print_summary(all_results, targets):
    """Print formatted summary table."""
    print("\n\n" + "=" * 100)
    print("SUMMARY TABLE (Mean ± Std across seeds)")
    print("=" * 100)
    print(f"{'Region':<25} {'mean_rs':<8} {'Persist':<12} {'PatchTST-S':<12} "
          f"{'TransCIF-ZS':<12} {'Ratio':<8} {'ZS/P':<6}")
    print("-" * 100)

    for target in targets:
        tr = [r for r in all_results if r["target"] == target]
        if not tr:
            continue
        persist_vals = [r["persistence"]["mae"] for r in tr]
        ptst_vals = [r["patchtst_sup"]["mae"] for r in tr]
        zs_vals = [r["transcif_zs"]["mae"] for r in tr]
        ratio_vals = [r["ratio_vs_patchtst"] for r in tr]
        zsp_vals = [r["ratio_vs_persist"] for r in tr]
        print(f"{target:<25} {tr[0]['mean_rs']:<8.3f} "
              f"{np.mean(persist_vals):.1f}±{np.std(persist_vals):.1f}  "
              f"{np.mean(ptst_vals):.1f}±{np.std(ptst_vals):.1f}  "
              f"{np.mean(zs_vals):.1f}±{np.std(zs_vals):.1f}  "
              f"{np.mean(ratio_vals):<8.3f} {np.mean(zsp_vals):<6.3f}")

    all_ratios = [r["ratio_vs_patchtst"] for r in all_results]
    all_zsp = [r["ratio_vs_persist"] for r in all_results]
    all_ratios_p = [r["ratio_plus_vs_patchtst"] for r in all_results]
    all_zspp = [r["ratio_plus_vs_persist"] for r in all_results]
    print(f"\n{'OVERALL':<25} {'':8} {'':12} {'':12} {'':12} "
          f"{np.mean(all_ratios):<8.3f} {np.mean(all_zsp):<6.3f}")
    print(f"\nMedian Ratio vs PatchTST: ZS={np.median(all_ratios):.3f}  "
          f"ZS+={np.median(all_ratios_p):.3f}")
    print(f"Mean Ratio vs PatchTST:   ZS={np.mean(all_ratios):.3f}  "
          f"ZS+={np.mean(all_ratios_p):.3f}")
    print(f"Regions where ZS  < Persist: "
          f"{sum(1 for r in all_zsp if r < 1)}/{len(all_zsp)}")
    print(f"Regions where ZS+ < Persist: "
          f"{sum(1 for r in all_zspp if r < 1)}/{len(all_zspp)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: 3 seeds, AU only")
    parser.add_argument("--rag", action="store_true",
                        help="Include RAG-DLinear comparison")
    parser.add_argument("--phys-irm", action="store_true",
                        help="Include Phys-IRM comparison")
    parser.add_argument("--causal", action="store_true",
                        help="Include Causal-VAE comparison")
    parser.add_argument("--icl", action="store_true",
                        help="Include IC-Transformer comparison")
    parser.add_argument("--hier", action="store_true",
                        help="Include Hierarchical Debiased comparison")
    args = parser.parse_args()

    mode_str = "QUICK (AU only, 3 seeds)" if args.quick else "FULL (29 regions, 5 seeds)"
    flags = []
    if args.rag: flags.append("RAG")
    if args.phys_irm: flags.append("Phys-IRM")
    if args.causal: flags.append("Causal")
    if args.icl: flags.append("ICL")
    if args.hier: flags.append("Hier")
    if flags:
        mode_str += " +" + "+".join(flags)
    print("=" * 80)
    print(f"Phase 1.3: Unified Evaluation Protocol — {mode_str}")
    print("=" * 80)

    if args.quick:
        _run_quick_mode(use_rag=args.rag, use_phys_irm=args.phys_irm,
                        use_causal=args.causal, use_icl=args.icl, use_hier=args.hier)
    else:
        _run_long_mode(use_rag=args.rag, use_phys_irm=args.phys_irm,
                       use_causal=args.causal, use_icl=args.icl, use_hier=args.hier)


if __name__ == "__main__":
    main()
