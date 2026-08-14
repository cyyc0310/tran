"""IC-TSF Experiment: In-Context Learning for Zero-Shot Time Series Forecasting.

Evaluates:
    1. TransCIF-ZS (baseline)
    2. IC-TSF (causal Transformer ICL, m ∈ {1, 3, 5} examples)
    3. Ablation: n_examples sweep
    4. Ablation: example selection strategy (config-only vs combined vs random)

Usage:
    python scripts/run_icl_eval.py --quick
    python scripts/run_icl_eval.py --ablation-m
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
    train_zero_shot, compute_metrics,
)
from transcif.models.zeroshot.icl import (
    ICTransformer, build_context, select_examples,
    train_icl, predict_icl_zs,
)


def evaluate_icl_target(target_name, all_regions, seed=42, n_examples=3):
    """Compare ZS-ERM vs IC-TSF."""
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

    data = all_regions[target_name]
    split = int(len(data["rs"]) * TRAIN_FRACTION)
    x_rs_test, _, y_cif_test = build_windows(
        data["rs"][split - SEQ_LEN:], data["cif"][split - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)
    if len(x_rs_test) == 0:
        return None

    result = {"target": target_name, "seed": seed, "n_examples": n_examples}

    # Baseline
    zs_model = train_zero_shot(all_regions, target_name, seed=seed)
    cfg_t = torch.tensor(data["config"]).unsqueeze(0).expand(len(x_rs_test), -1)
    with torch.no_grad():
        zs_share = zs_model(torch.tensor(x_rs_test, dtype=torch.float32), cfg_t).numpy()
    result["zs_erm"] = compute_metrics(
        cif_from_shares(zs_share, data["ef_r"], data["ef_nr"]), y_cif_test)

    # IC-TSF
    t0 = time.time()
    icl_model = train_icl(all_regions, target_name, seed=seed,
                           n_examples=n_examples)
    icl_cif = predict_icl_zs(icl_model, all_regions, target_name,
                              x_rs_test, data["ef_r"], data["ef_nr"],
                              n_examples=n_examples)
    result["icl_tsf"] = compute_metrics(icl_cif, y_cif_test)
    result["icl_time"] = time.time() - t0
    result["ratio_vs_zs"] = result["icl_tsf"]["mae"] / max(result["zs_erm"]["mae"], 1e-6)

    return result


def main():
    parser = argparse.ArgumentParser(description="IC-TSF Experiments")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--n-examples", type=int, default=3)
    parser.add_argument("--ablation-m", action="store_true")
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

    if args.ablation_m:
        targets = args.target.split(",") if args.target else ["VIC1", "SA1"]
        print(f"\nn_examples ablation for {targets}")
        for tgt in targets:
            print(f"\n--- {tgt} ---")
            for m in [0, 1, 2, 3, 5]:
                r = evaluate_icl_target(tgt, all_regions, n_examples=m) if m > 0 else None
                if m == 0:
                    r2 = evaluate_icl_target(tgt, all_regions, n_examples=1)
                    print(f"  m={m} (ZS) MAE={r2['zs_erm']['mae']:.1f}")
                elif r:
                    print(f"  m={m} MAE={r['icl_tsf']['mae']:.1f} ×{r['ratio_vs_zs']:.3f} "
                          f"time={r['icl_time']:.1f}s")
        return

    targets = ["QLD1", "NSW1", "VIC1", "SA1"] if args.quick else sorted(all_regions)
    results = []
    t0 = time.time()

    for target in targets:
        print(f"\n[{target}]", flush=True)
        r = evaluate_icl_target(target, all_regions, n_examples=args.n_examples)
        if r:
            results.append(r)
            print(f"  ZS={r['zs_erm']['mae']:.1f} ICL={r['icl_tsf']['mae']:.1f} "
                  f"×{r['ratio_vs_zs']:.3f} time={r['icl_time']:.1f}s")

    if results:
        ratios = [r["ratio_vs_zs"] for r in results]
        print(f"\n{'='*60}")
        print(f"IC-TSF: {len(results)} evals, m={args.n_examples}, {time.time()-t0:.0f}s")
        print(f"  median ratio vs ZS: {np.median(ratios):.3f}")
        better = sum(1 for r in ratios if r < 1)
        print(f"  ICL < ZS: {better}/{len(results)} ({100*better/max(len(results),1):.0f}%)")
        print(f"{'='*60}")

        out = args.out or str(RESULTS_DIR / "icl_eval.json")
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(results, open(out, "w"), indent=2)
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
