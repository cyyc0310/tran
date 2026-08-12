"""Stage D/C POC: compare RevIN and Regime-MoE variants against the flagship.

Runs a focused LORO (12 regions × 3 seeds) comparing up to 4 model variants:

  - AdaptivePersistDLinear   (flagship baseline, Stage 0)
  - RevINAdaptivePersistDLinear  (Stage D — RevIN on DLinear branch)
  - RegimeMoEAdaptivePersist (Stage C — K-expert softmax routing)

All variants share the same train/test protocol and fuel-augmented 12-D config,
so the only difference is the model architecture.  Hypothesis:
  - RevIN helps seasonal-OOD regions (SA1, BPAT — high seasonal swing).
  - MoE helps false-neighbour pairs (ERCO vs MISO — fuel-divergent).

Output: results/probe_models.json + console summary.

Usage:
    PYTHONPATH=src python scripts/experiments/probe_models.py
"""

import json
import time
from pathlib import Path

import numpy as np
import torch

from transcif.config import RESULTS_DIR, AU_REGIONS, US_REGIONS, UK_REGIONS
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.models.base import (
    AdaptivePersistDLinear,
    RevINAdaptivePersistDLinear,
    RegimeMoEAdaptivePersist,
)
from transcif.models.zeroshot.base_zs import evaluate_target

SEEDS = [0, 1, 2]
TARGETS = ["US_CISO", "US_ERCO", "US_MISO", "US_PJM", "US_ISNE",
           "US_NYIS", "US_FPL", "US_BPAT", "QLD1", "VIC1", "NSW1", "SA1"]

VARIANTS = {
    "flagship": AdaptivePersistDLinear,
    "revin": RevINAdaptivePersistDLinear,
    "moe3": RegimeMoEAdaptivePersist,
}


def build_pool():
    discover_uk_regions()
    all_cfgs = {**AU_REGIONS, **US_REGIONS, **UK_REGIONS}
    return {n: load_region_data(n, all_cfgs) for n in all_cfgs}


def run_variant(model_class, label, regions):
    results = {}
    t0 = time.time()
    for i, target in enumerate(TARGETS):
        if target not in regions:
            continue
        seed_mae = []
        for seed in SEEDS:
            r = evaluate_target(target, regions, seed=seed, model_class=model_class)
            seed_mae.append({
                "persistence": r["persistence"]["mae"],
                "transcif_zs": r["transcif_zs"]["mae"],
                "transcif_zs_plus": r["transcif_zs_plus"]["mae"],
            })
        agg = {m: float(np.median([s[m] for s in seed_mae]))
               for m in ("persistence", "transcif_zs", "transcif_zs_plus")}
        results[target] = agg
        print(f"  [{label:9s}] {i+1:2d}/{len(TARGETS)} {target:10s}  "
              f"persist={agg['persistence']:6.1f}  ZS={agg['transcif_zs']:6.1f}  "
              f"ZS+={agg['transcif_zs_plus']:6.1f}")
    print(f"  [{label}] done in {time.time()-t0:.0f}s")
    return results


def main():
    print("=" * 80)
    print("Stage D/C POC: RevIN vs Regime-MoE vs flagship")
    print("=" * 80)
    regions = build_pool()

    all_results = {}
    for label, cls in VARIANTS.items():
        print(f"\n--- {label} ({cls.__name__}) ---")
        all_results[label] = run_variant(cls, label, regions)

    # Summary table.
    print("\n" + "=" * 90)
    print("RESULTS: median MAE across 12 targets")
    print("=" * 90)
    header = f"{'Target':<12} {'persist':>8}"
    for label in VARIANTS:
        header += f" {label+' ZS':>10} {label+' ZS+':>10}"
    print(header)
    print("-" * len(header))
    for t in TARGETS:
        if t not in all_results["flagship"]:
            continue
        row = f"{t:<12} {all_results['flagship'][t]['persistence']:>8.1f}"
        for label in VARIANTS:
            zs = all_results[label][t]["transcif_zs"]
            zsp = all_results[label][t]["transcif_zs_plus"]
            row += f" {zs:>10.1f} {zsp:>10.1f}"
        print(row)

    # Aggregate + per-variant delta vs flagship.
    print()
    base_zsp = np.median([all_results["flagship"][t]["transcif_zs_plus"] for t in all_results["flagship"]])
    print(f"Median ZS+ (flagship): {base_zsp:.2f}")
    for label in VARIANTS:
        zs = np.median([all_results[label][t]["transcif_zs"] for t in all_results[label]])
        zsp = np.median([all_results[label][t]["transcif_zs_plus"] for t in all_results[label]])
        dzsp = zsp - base_zsp
        print(f"  {label:9s}: ZS median={zs:.2f}  ZS+ median={zsp:.2f}  Δ vs flagship={dzsp:+.2f}")

    # Highlight key regions.
    print("\n--- Key region diagnostics ---")
    for t in ["US_ERCO", "US_MISO", "VIC1", "SA1", "US_BPAT"]:
        if t not in all_results["flagship"]:
            continue
        print(f"  {t}:")
        for label in VARIANTS:
            zsp = all_results[label][t]["transcif_zs_plus"]
            print(f"    {label:9s}: ZS+={zsp:.1f}")

    out = {"targets": TARGETS, "seeds": SEEDS, "variants": list(VARIANTS), "results": all_results}
    out_path = RESULTS_DIR / "probe_models.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[WRITE] {out_path}")


if __name__ == "__main__":
    main()
