"""Probe: fusion variants for ZS+ on 10 focus regions (seed 0).

Variants (each region's zero-shot model is trained once and shared):
  M0..M3 : the four FUSION_MENU configs, forced
  AUTO   : regional self-selection over the menu (pre-test observed days only)

Usage: PYTHONPATH=src:scripts python scripts/probe_multibranch_fusion.py
"""

import json

import numpy as np

import run_unified_eval as rue
from run_unified_eval import (
    AU_REGIONS, UK_REGIONS, US_REGIONS, RESULTS_DIR, FUSION_MENU,
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION,
    discover_uk_regions, load_region_data, build_windows,
    train_zero_shot, compute_metrics, zs_plus_predict,
)

FOCUS = [
    "UK_13_London", "UK_14_South_East_England",          # persistence losses
    "US_MISO", "VIC1", "SA1",                            # CC-ZS losses
    "UK_01_North_Scotland", "US_ERCO",                   # new-fusion regressions
    "QLD1", "US_FPL", "UK_11_South_West_England",        # guards (beat PatchTST)
    "UK_07_South_Wales", "US_CISO",
]
SEED = 0

C0 = dict(branches=(0, 1, 3), gamma=2.5, k_backtest=28)
C1 = dict(branches=(0, 1, 3, 4), gamma=2.5, k_backtest=28)
LEG = dict(branches=(0, 1), gamma=2.0, k_backtest=7)
RAW = dict(branches=(5, 1, 3), gamma=2.5, k_backtest=28)
W4 = dict(branches=(0, 1, 2, 3, 4), gamma=2.5, k_backtest=28)
RAWP = dict(branches=(5,), gamma=2.5, k_backtest=28)  # pure raw model (=ZS)

VARIANTS = {
    "D3 m15":     dict(fusion=None, _menu=(C0, C1, LEG), _metric="dual", _margin=0.015),
    "W3 t15":     dict(fusion=None, _menu=(C0, C1, LEG, dict(W4, margin=0.03)),
                       _metric="dual", _margin=0.015, _tol=0.015),
    "W3 t05":     dict(fusion=None, _menu=(C0, C1, LEG, dict(W4, margin=0.03)),
                       _metric="dual", _margin=0.015, _tol=0.005),
    "W25 t1":     dict(fusion=None, _menu=(C0, C1, LEG, dict(W4, margin=0.025)),
                       _metric="dual", _margin=0.015, _tol=0.01),
    "W2 t1":      dict(fusion=None, _menu=(C0, C1, LEG, dict(W4, margin=0.02)),
                       _metric="dual", _margin=0.015, _tol=0.01),
}


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
    ref = {r["target"]: r for r in stored if r["seed"] == SEED}

    header = f"{'Region':<26} {'stored':>7} " + "".join(
        f"{v:>10}" for v in VARIANTS) + f" {'Persist':>8}"
    print(header)
    print("-" * len(header))
    table = {v: [] for v in VARIANTS}
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
        zs_model = train_zero_shot(all_regions, target, seed=SEED)

        maes = {}
        for vname, kw in VARIANTS.items():
            kw = dict(kw)
            rue.SELECT_MARGIN = kw.pop("_margin", 0.02)
            rue.SELECT_DAYS = kw.pop("_days", 56)
            rue.SELECT_METRIC = kw.pop("_metric", "mean")
            rue.SELECT_TOL = kw.pop("_tol", None)
            rue.FUSION_MENU = kw.pop("_menu", (C0, C1))
            zsp = zs_plus_predict(zs_model, data["config"], rs, cif,
                                  ef_r, ef_nr, origins, **kw)
            maes[vname] = compute_metrics(zsp, y_cif_test)["mae"]
            table[vname].append((target, maes[vname]))

        r = ref[target]
        print(f"{target:<26} {r['transcif_zs_plus']['mae']:>7.2f} "
              + "".join(f"{maes[v]:>10.2f}" for v in VARIANTS)
              + f" {r['persistence']['mae']:>8.2f}", flush=True)

    print("-" * len(header))
    nf = len(FOCUS)
    for vname in VARIANTS:
        beat_p = sum(1 for t, m in table[vname]
                     if m < ref[t]["persistence"]["mae"])
        better = sum(1 for t, m in table[vname]
                     if m < ref[t]["transcif_zs_plus"]["mae"])
        mean_ratio = np.mean([m / ref[t]["transcif_zs_plus"]["mae"]
                              for t, m in table[vname]])
        print(f"{vname:<12} beat-persist {beat_p}/{nf}  "
              f"improved-vs-stored {better}/{nf}  mean-ratio-vs-stored {mean_ratio:.4f}")


if __name__ == "__main__":
    main()
