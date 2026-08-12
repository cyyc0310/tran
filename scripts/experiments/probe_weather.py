"""Stage B.4 POC: does a weather side channel improve zero-shot CIF?

Compares the flagship AdaptivePersistDLinear against WeatherAdaptivePersistDLinear
(which adds temperature / solar radiation / wind speed as a side input) on the
12-region LORO × 3-seed protocol.

Hypothesis: weather helps solar-dominated (QLD1, CISO) and wind-dominated
(SA1, ERCO) regions where renewable generation is physically driven by
irradiance/wind.  UK DNO regions may benefit less (power is imported).

Usage:
    PYTHONPATH=src python scripts/experiments/probe_weather.py
"""

import json
import time
from pathlib import Path

import numpy as np

from transcif.config import RESULTS_DIR, AU_REGIONS, US_REGIONS, UK_REGIONS
from transcif.data.loaders import discover_uk_regions, load_region_data
from transcif.models.base import AdaptivePersistDLinear, WeatherAdaptivePersistDLinear
from transcif.models.zeroshot.base_zs import evaluate_target

SEEDS = [0, 1, 2]
TARGETS = ["US_CISO", "US_ERCO", "US_MISO", "US_PJM", "US_ISNE",
           "US_NYIS", "US_FPL", "US_BPAT", "QLD1", "VIC1", "NSW1", "SA1"]

VARIANTS = {
    "flagship": AdaptivePersistDLinear,
    "weather": WeatherAdaptivePersistDLinear,
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
        has_w = regions[target].get("weather") is not None
        print(f"  [{label:9s}] {i+1:2d}/{len(TARGETS)} {target:10s}  "
              f"w={'Y' if has_w else 'N'}  "
              f"persist={agg['persistence']:6.1f}  ZS={agg['transcif_zs']:6.1f}  "
              f"ZS+={agg['transcif_zs_plus']:6.1f}")
    print(f"  [{label}] done in {time.time()-t0:.0f}s")
    return results


def main():
    print("=" * 80)
    print("Stage B.4 POC: Weather side channel vs flagship")
    print("=" * 80)
    regions = build_pool()
    n_with_weather = sum(1 for r in regions.values() if r.get("weather") is not None)
    print(f"Regions with weather data: {n_with_weather}/{len(regions)}")

    all_results = {}
    for label, cls in VARIANTS.items():
        print(f"\n--- {label} ({cls.__name__}) ---")
        all_results[label] = run_variant(cls, label, regions)

    # Summary.
    print("\n" + "=" * 80)
    print("RESULTS: median MAE across 12 targets")
    print("=" * 80)
    print(f"{'Target':<12} {'persist':>8} {'flag ZS':>8} {'wx ZS':>8} {'Δ':>6} "
          f"{'flag ZS+':>9} {'wx ZS+':>8} {'Δ':>6}")
    print("-" * 75)
    for t in TARGETS:
        if t not in all_results["flagship"]:
            continue
        f, w = all_results["flagship"][t], all_results["weather"][t]
        dz = w["transcif_zs"] - f["transcif_zs"]
        dzp = w["transcif_zs_plus"] - f["transcif_zs_plus"]
        print(f"{t:<12} {f['persistence']:>8.1f} {f['transcif_zs']:>8.1f} "
              f"{w['transcif_zs']:>8.1f} {dz:>+6.1f} {f['transcif_zs_plus']:>9.1f} "
              f"{w['transcif_zs_plus']:>8.1f} {dzp:>+6.1f}")

    base_zs = np.median([all_results["flagship"][t]["transcif_zs"] for t in all_results["flagship"]])
    wx_zs = np.median([all_results["weather"][t]["transcif_zs"] for t in all_results["weather"]])
    base_zsp = np.median([all_results["flagship"][t]["transcif_zs_plus"] for t in all_results["flagship"]])
    wx_zsp = np.median([all_results["weather"][t]["transcif_zs_plus"] for t in all_results["weather"]])
    print(f"\nMedian ZS:  flagship={base_zs:.2f}  weather={wx_zs:.2f}  Δ={wx_zs-base_zs:+.2f}")
    print(f"Median ZS+: flagship={base_zsp:.2f}  weather={wx_zsp:.2f}  Δ={wx_zsp-base_zsp:+.2f}")

    out = {"targets": TARGETS, "seeds": SEEDS, "variants": list(VARIANTS), "results": all_results}
    out_path = RESULTS_DIR / "probe_weather.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[WRITE] {out_path}")


if __name__ == "__main__":
    main()
