"""Multi-seed probe of the two finalist selector configs on knife-edge regions.

D3m15 : menu (C0, C1, LEG), dual metric, margin 0.015
W3t15 : + W4 config with its own recruitment margin 0.03

Usage: PYTHONPATH=scripts python scripts/probe_final_multiseed.py
"""

import json

import numpy as np

import run_unified_eval as rue
from run_unified_eval import (
    AU_REGIONS, UK_REGIONS, US_REGIONS, RESULTS_DIR,
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION,
    discover_uk_regions, load_region_data, build_windows,
    train_zero_shot, compute_metrics, zs_plus_predict,
)

FOCUS = ["UK_13_London", "UK_14_South_East_England", "US_MISO", "SA1",
         "UK_01_North_Scotland", "VIC1"]
SEEDS = [0, 1, 2, 3, 4]

C0 = dict(branches=(0, 1, 3), gamma=2.5, k_backtest=28)
C1 = dict(branches=(0, 1, 3, 4), gamma=2.5, k_backtest=28)
LEG = dict(branches=(0, 1), gamma=2.0, k_backtest=7)
W4 = dict(branches=(0, 1, 2, 3, 4), gamma=2.5, k_backtest=28)

VARIANTS = {
    "D3m15": dict(_menu=(C0, C1, LEG), _margin=0.015, _tol=None),
    "W3t15": dict(_menu=(C0, C1, LEG, dict(W4, margin=0.03)),
                  _margin=0.015, _tol=0.015),
}
# external leaderboard rivals (5-seed means from stored archives)
CC_ZS = {"US_MISO": 46.19, "SA1": 60.57}


def main():
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception as e:
            print(f"  [WARN] Skip {name}: {e}")

    with open(RESULTS_DIR / "unified_eval_full.json") as f:
        stored = json.load(f)

    acc = {v: {t: [] for t in FOCUS} for v in VARIANTS}
    for target in FOCUS:
        data = all_regions[target]
        rs, cif = data["rs"], data["cif"]
        ef_r, ef_nr = data["ef_r"], data["ef_nr"]
        split_hour = int(len(rs) * TRAIN_FRACTION)
        _, _, y_cif_test = build_windows(
            rs[split_hour - SEQ_LEN:], cif[split_hour - SEQ_LEN:],
            SEQ_LEN, HORIZON, TEST_STRIDE)
        origins = [split_hour + st
                   for st in range(0, len(y_cif_test) * TEST_STRIDE, TEST_STRIDE)]
        for seed in SEEDS:
            model = train_zero_shot(all_regions, target, seed=seed)
            for vname, kw in VARIANTS.items():
                rue.SELECT_METRIC = "dual"
                rue.SELECT_DAYS = 56
                rue.SELECT_MARGIN = kw["_margin"]
                rue.SELECT_TOL = kw["_tol"]
                rue.FUSION_MENU = kw["_menu"]
                zsp = zs_plus_predict(model, data["config"], rs, cif,
                                      ef_r, ef_nr, origins, fusion=None)
                acc[vname][target].append(compute_metrics(zsp, y_cif_test)["mae"])
            print(f"  {target} s{seed}: "
                  + "  ".join(f"{v}={acc[v][target][-1]:.2f}" for v in VARIANTS),
                  flush=True)

    print("\n" + "=" * 78)
    print(f"{'Region':<26} {'stored':>8} " + "".join(f"{v:>14}" for v in VARIANTS)
          + f" {'Persist':>8} {'rival':>7}")
    print("-" * 78)
    for target in FOCUS:
        tr = [r for r in stored if r["target"] == target]
        st = np.mean([r["transcif_zs_plus"]["mae"] for r in tr])
        pe = np.mean([r["persistence"]["mae"] for r in tr])
        rival = CC_ZS.get(target, np.nan)
        cells = "".join(
            f"  {np.mean(acc[v][target]):6.2f}±{np.std(acc[v][target]):4.2f}"
            for v in VARIANTS)
        print(f"{target:<26} {st:>8.2f}{cells} {pe:>8.2f} {rival:>7.2f}")


if __name__ == "__main__":
    main()
