#!/usr/bin/env python
"""Paired comparison of two fuel-decomp eval runs (same region-seed pairs).

Usage:
    .venv/bin/python scripts/analysis/compare_eval_runs.py NEW.json BASE.json
"""

import json
import sys
from pathlib import Path

from scipy.stats import wilcoxon

TIERS = ["fuel_i0", "fuel_i_cfg", "fuel_i_plus",
         "fuel_i0_phys", "fuel_i_cfg_phys"]


def load(path):
    rows = json.load(open(path))["rows"]
    return {(r["target"], r["seed"]): r for r in rows}


def main(new_path, base_path):
    new, base = load(new_path), load(base_path)
    pairs = sorted(set(new) & set(base))
    print(f"pairs: {len(pairs)}  (new-only {len(set(new)-set(base))}, "
          f"base-only {len(set(base)-set(new))})")
    for tier in TIERS:
        dn = [(base[k][tier]["mae"] - new[k][tier]["mae"], k) for k in pairs]
        dn = [x for x in dn if x[0] == x[0]]
        if not dn:
            continue
        wins = sum(1 for d, _ in dn if d > 0)
        med_b = sorted(base[k][tier]["mae"] for k in pairs)
        med_n = sorted(new[k][tier]["mae"] for k in pairs)
        p = wilcoxon([d for d, _ in dn]).pvalue if len(dn) >= 6 else float("nan")
        print(f"{tier:18s} base {med_b[len(med_b)//2]:6.2f} -> new "
              f"{med_n[len(med_n)//2]:6.2f}  win {wins}/{len(dn)}  p={p:.3g}")
        worst = sorted(dn)[:3]
        for d, k in worst:
            if d < -2:
                print(f"    regression: {k[0]} seed{k[1]} {d:+.1f}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
