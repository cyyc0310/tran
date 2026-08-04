"""Diagnose why rolling self-selection ignores LEG in UK_01_North_Scotland.

For every test origin, compare:
  - the 56d replayed scores of C0 / C1 / LEG (what the selector sees)
  - the realized daily MAE of each forced config at that origin (ground truth)

Usage: PYTHONPATH=scripts python scripts/probe_uk01_diagnosis.py
"""

import numpy as np

import run_unified_eval as rue
from run_unified_eval import (
    AU_REGIONS, UK_REGIONS, US_REGIONS,
    SEQ_LEN, HORIZON, TEST_STRIDE, TRAIN_FRACTION,
    discover_uk_regions, load_region_data, build_windows,
    train_zero_shot, zs_plus_predict,
)

TARGET = "UK_01_North_Scotland"
SEED = 0

C0 = dict(branches=(0, 1, 3), gamma=2.5, k_backtest=28)
C1 = dict(branches=(0, 1, 3, 4), gamma=2.5, k_backtest=28)
LEG = dict(branches=(0, 1), gamma=2.0, k_backtest=7)
MENU = (C0, C1, LEG)
NAMES = ["C0", "C1", "LEG"]


def main():
    discover_uk_regions()
    all_configs = {**AU_REGIONS, **UK_REGIONS, **US_REGIONS}
    all_regions = {}
    for name in all_configs:
        try:
            all_regions[name] = load_region_data(name, all_configs)
        except Exception as e:
            print(f"  [WARN] Skip {name}: {e}")

    data = all_regions[TARGET]
    rs, cif = data["rs"], data["cif"]
    ef_r, ef_nr = data["ef_r"], data["ef_nr"]
    split_hour = int(len(rs) * TRAIN_FRACTION)
    _, _, y_cif_test = build_windows(
        rs[split_hour - SEQ_LEN:], cif[split_hour - SEQ_LEN:],
        SEQ_LEN, HORIZON, TEST_STRIDE)
    origins = [split_hour + st
               for st in range(0, len(y_cif_test) * TEST_STRIDE, TEST_STRIDE)]
    model = train_zero_shot(all_regions, TARGET, seed=SEED)

    # forced predictions per config -> realized daily MAE per origin
    daily = {}
    for nm, cfg in zip(NAMES, MENU):
        preds = zs_plus_predict(model, data["config"], rs, cif,
                                ef_r, ef_nr, origins, fusion=cfg)
        daily[nm] = np.abs(preds - y_cif_test).mean(axis=1)

    # replicate the selector's replay scores at each origin
    rue.FUSION_MENU = MENU
    sim_cache = {}

    def sim_mae(o, ci):
        if (o, ci) not in sim_cache:
            p = zs_plus_predict(model, data["config"], rs, cif,
                                ef_r, ef_nr, [o], fusion=MENU[ci])[0]
            sim_cache[(o, ci)] = np.abs(p[:24] - cif[o:o + 24]).mean()
        return sim_cache[(o, ci)]

    print(f"{'i':>3} {'sc_C0':>8} {'sc_C1':>8} {'sc_LEG':>8} {'sel':>4} "
          f"{'d_C0':>7} {'d_C1':>7} {'d_LEG':>7} {'best':>5}")
    sel_counts = {nm: 0 for nm in NAMES}
    win_counts = {nm: 0 for nm in NAMES}
    margins = []
    for i, t0 in enumerate(origins):
        scores = np.zeros(len(MENU))
        n_sel = 0
        for j in range(1, rue.SELECT_DAYS + 1):
            o_s = t0 - j * 24
            if o_s - SEQ_LEN - 24 < 0:
                break
            for ci in range(len(MENU)):
                scores[ci] += sim_mae(o_s, ci)
            n_sel += 1
        scores /= max(n_sel, 1)
        best_i = int(np.argmin(scores))
        sel = best_i if scores[best_i] < scores[0] * (1 - rue.SELECT_MARGIN) else 0
        sel_counts[NAMES[sel]] += 1
        winner = NAMES[int(np.argmin([daily[nm][i] for nm in NAMES]))]
        win_counts[winner] += 1
        margins.append(scores[2] / scores[0] - 1.0)
        if i % 5 == 0 or NAMES[sel] != "C0":
            print(f"{i:>3} {scores[0]:>8.2f} {scores[1]:>8.2f} {scores[2]:>8.2f} "
                  f"{NAMES[sel]:>4} {daily['C0'][i]:>7.2f} {daily['C1'][i]:>7.2f} "
                  f"{daily['LEG'][i]:>7.2f} {winner:>5}")

    print("-" * 66)
    print(f"selected: {sel_counts}   daily-winner: {win_counts}")
    print(f"test-period mean MAE: "
          + "  ".join(f"{nm}={daily[nm].mean():.2f}" for nm in NAMES))
    m = np.array(margins)
    print(f"LEG replay score vs C0: mean {m.mean()*100:+.2f}%  "
          f"median {np.median(m)*100:+.2f}%  min {m.min()*100:+.2f}%  "
          f"frac<-2% {np.mean(m < -0.02):.2f}  frac<0 {np.mean(m < 0):.2f}")


if __name__ == "__main__":
    main()
