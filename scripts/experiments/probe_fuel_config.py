"""A.6 POC: verify that multi-fuel config improves zero-shot transfer.

Runs a focused LORO on US (8) + AU (4) regions with the AdaptivePersistDLinear
flagship model, comparing:

  (a) legacy 2-D config  [mean_rs, ef_nr/1000]
  (b) multi-fuel config  [mean_rs, ef_nr/1000, coal, gas, nuclear, ...] (Stage A)

The comparison isolates the config-representation effect because the only
difference between arms is the config vector width.  Hypothesis: regions that
were "false neighbours" under mean_rs (e.g. ERCO wind-heavy vs MISO coal-heavy,
both ~0.2-0.3 mean_rs) should diverge more in config distance and transfer
better when fuel breakdown is visible.

Output: results/probe_fuel_config.json + console summary.

Usage:
    PYTHONPATH=src python scripts/experiments/probe_fuel_config.py
"""

import json
import time
from pathlib import Path

import numpy as np

from transcif.config import (
    DATA_DIR, RESULTS_DIR, AU_REGIONS, US_REGIONS, UK_REGIONS,
    get_fuel_shares, get_fuel_order,
)
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.models.zeroshot.base_zs import evaluate_target

SEEDS = [0, 1, 2]
TARGETS = ["US_CISO", "US_ERCO", "US_MISO", "US_PJM", "US_ISNE",
           "US_NYIS", "US_FPL", "US_BPAT", "QLD1", "VIC1", "NSW1", "SA1"]


def build_region_pool(use_fuel: bool):
    """Load all regions; optionally strip fuel dims to force 2-D config."""
    discover_uk_regions()
    all_cfgs = {**AU_REGIONS, **US_REGIONS, **UK_REGIONS}
    regions = {}
    for name in all_cfgs:
        d = load_region_data(name, all_cfgs)
        if not use_fuel:
            # Force legacy 2-D config for the control arm.
            d["config"] = d["config"][:2].copy()
            d["fuel_shares"] = None
        regions[name] = d
    return regions


def config_distance_matrix(regions):
    """Pairwise |Δmean_rs| matrix for diagnostic."""
    names = sorted(regions.keys())
    means = np.array([regions[n]["mean_rs"] for n in names])
    D = np.abs(means[:, None] - means[None, :])
    return names, D


def fuel_distance_matrix(regions):
    """Pairwise L1 distance in fuel-share space (US regions only)."""
    fuel_names = [n for n in sorted(regions) if regions[n].get("fuel_shares")]
    if not fuel_names:
        return [], np.zeros((0, 0))
    order = get_fuel_order()
    vecs = []
    for n in fuel_names:
        shares = regions[n]["fuel_shares"]
        vecs.append([shares.get(f, 0.0) for f in order])
    V = np.array(vecs)
    D = np.abs(V[:, None, :] - V[None, :, :]).sum(axis=2)
    return fuel_names, D


def run_arm(use_fuel: bool, label: str):
    """Run LORO ZS+ZS+ on all targets; return per-target MAE dict."""
    regions = build_region_pool(use_fuel=use_fuel)
    results = {}
    t0 = time.time()
    for i, target in enumerate(TARGETS):
        if target not in regions:
            continue
        seed_mae = []
        for seed in SEEDS:
            r = evaluate_target(target, regions, seed=seed)
            seed_mae.append({
                "persistence": r["persistence"]["mae"],
                "transcif_zs": r["transcif_zs"]["mae"],
                "transcif_zs_plus": r["transcif_zs_plus"]["mae"],
            })
        agg = {
            m: float(np.median([s[m] for s in seed_mae]))
            for m in ("persistence", "transcif_zs", "transcif_zs_plus")
        }
        results[target] = agg
        print(f"  [{label}] {i+1}/{len(TARGETS)} {target:10s}  "
              f"persist={agg['persistence']:6.1f}  ZS={agg['transcif_zs']:6.1f}  "
              f"ZS+={agg['transcif_zs_plus']:6.1f}")
    print(f"  [{label}] done in {time.time()-t0:.0f}s")
    return results


def main():
    print("=" * 70)
    print("A.6 POC: Multi-fuel config vs legacy 2-D config")
    print("=" * 70)
    print(f"Targets: {TARGETS}")
    print(f"Seeds: {SEEDS}")

    # Diagnostic: config distance matrices.
    print("\n--- Config distance diagnostics ---")
    legacy_pool = build_region_pool(use_fuel=False)
    fuel_pool = build_region_pool(use_fuel=True)
    _, D_rs = config_distance_matrix(fuel_pool)
    fnames, D_fuel = fuel_distance_matrix(fuel_pool)
    print(f"US mean_rs range: {min(fuel_pool[n]['mean_rs'] for n in fnames):.3f} "
          f"→ {max(fuel_pool[n]['mean_rs'] for n in fnames):.3f}")
    # Highlight ERCO vs MISO (the suspected false neighbours).
    for a, b in [("US_ERCO", "US_MISO"), ("US_CISO", "US_PJM"), ("US_FPL", "US_BPAT")]:
        i, j = fnames.index(a), fnames.index(b)
        print(f"  {a} vs {b}: |Δmean_rs|={abs(fuel_pool[a]['mean_rs']-fuel_pool[b]['mean_rs']):.3f}  "
              f"fuel_L1={D_fuel[i,j]:.3f}")

    # Run both arms.
    print("\n--- Control arm: legacy 2-D config ---")
    legacy = run_arm(use_fuel=False, label="2D")

    print("\n--- Treatment arm: multi-fuel config ---")
    fuel = run_arm(use_fuel=True, label="ND")

    # Summary.
    print("\n" + "=" * 70)
    print("RESULTS: median MAE across all targets")
    print("=" * 70)
    print(f"{'Target':<12} {'persist':>8} {'ZS(2D)':>8} {'ZS(ND)':>8} {'Δ':>7} "
          f"{'ZS+(2D)':>8} {'ZS+(ND)':>8} {'Δ':>7}")
    print("-" * 76)
    for t in TARGETS:
        if t not in legacy:
            continue
        l, f = legacy[t], fuel[t]
        dz = f["transcif_zs"] - l["transcif_zs"]
        dzp = f["transcif_zs_plus"] - l["transcif_zs_plus"]
        print(f"{t:<12} {l['persistence']:>8.1f} {l['transcif_zs']:>8.1f} "
              f"{f['transcif_zs']:>8.1f} {dz:>+7.1f} {l['transcif_zs_plus']:>8.1f} "
              f"{f['transcif_zs_plus']:>8.1f} {dzp:>+7.1f}")

    # Aggregate.
    med_legacy = np.median([legacy[t]["transcif_zs_plus"] for t in legacy])
    med_fuel = np.median([fuel[t]["transcif_zs_plus"] for t in fuel])
    print(f"\nMedian ZS+ MAE: 2D={med_legacy:.2f}  ND={med_fuel:.2f}  "
          f"Δ={med_fuel-med_legacy:+.2f}")

    out = {
        "targets": TARGETS, "seeds": SEEDS,
        "legacy_2d": legacy, "multi_fuel": fuel,
        "median_zs_plus_2d": med_legacy, "median_zs_plus_nd": med_fuel,
    }
    out_path = RESULTS_DIR / "probe_fuel_config.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[WRITE] {out_path}")


if __name__ == "__main__":
    main()
