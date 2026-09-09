#!/usr/bin/env python
"""FD-46: event-day conservative shrinkage, offline prototype on FD-45 dumps.

Rule (deployment-legal):
    On wind_lull days (detectable from day-ahead weather), shrink the
    I_cfg prediction toward the config-constant prediction:
        pred' = w * pred_i_cfg + (1-w) * pred_config_constant
    Regular days keep the raw I_cfg prediction.
    w is selected LEAVE-ONE-REGION-OUT: for each target region, choose w
    from a grid by minimising the mean MAE over the OTHER regions' dumps
    (each with its own lull labels).  No target test labels are touched.

Usage:
    .venv-nemed/bin/python scripts/experiments/fd46_event_shrinkage.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
DUMP_DIR = RESULTS / "dunkelflaute"
W_GRID = np.round(np.arange(0.0, 1.01, 0.1), 2)


def load_region(target):
    path = DUMP_DIR / f"{target}_seed0.npz"
    if not path.exists():
        return None
    d = np.load(path)
    if len(d["wx_day_ord"]) == 0:
        return None
    origin_ts = pd.to_datetime(d["origin_hours"] * 3600, unit="s")
    w_thr = np.quantile(d["wx_w_mean"], 0.2)
    wx_days = pd.to_datetime(d["wx_day_ord"] - 719163, unit="D")
    lull = pd.Series(np.asarray(d["wx_w_mean"]) < w_thr, index=wx_days)
    day = origin_ts.normalize()
    is_lull = np.array([bool(lull.get(dd, False)) for dd in day])
    return {"y": d["y_true"], "i_cfg": d["pred_i_cfg"],
            "cfg_const": d["pred_config_constant"], "is_lull": is_lull}


def region_mae(r, w):
    pred = r["i_cfg"].copy()
    if r["is_lull"].any():
        pred[r["is_lull"]] = (w * r["i_cfg"][r["is_lull"]]
                              + (1 - w) * r["cfg_const"][r["is_lull"]])
    return float(np.abs(pred - r["y"]).mean())


def main():
    regions = sorted(p.name.replace("_seed0.npz", "")
                     for p in DUMP_DIR.glob("*_seed0.npz"))
    data = {}
    for t in regions:
        r = load_region(t)
        if r is not None:
            data[t] = r
    print(f"[fd46] {len(data)} regions with weather + dumps")

    rows = []
    for tgt in sorted(data):
        others = [t for t in data if t != tgt]
        # LORO weight selection: minimise mean MAE over the OTHER regions
        grid_mae = []
        for w in W_GRID:
            maes = [region_mae(data[t], w) for t in others]
            grid_mae.append(float(np.mean(maes)))
        w_star = float(W_GRID[int(np.argmin(grid_mae))])
        base = region_mae(data[tgt], 1.0)   # no shrinkage
        shr = region_mae(data[tgt], w_star)  # LORO-selected shrinkage
        rows.append({"target": tgt, "w_star": w_star,
                     "mae_base": round(base, 2),
                     "mae_shrunk": round(shr, 2),
                     "delta": round(shr - base, 2)})
        # in-sample oracle w for diagnostics only (upper bound of the rule)
        or_mae = [region_mae(data[tgt], w) for w in W_GRID]
        rows[-1]["w_oracle"] = float(W_GRID[int(np.argmin(or_mae))])
        rows[-1]["mae_oracle"] = round(float(np.min(or_mae)), 2)

    df = pd.DataFrame(rows)
    med_base = df["mae_base"].median()
    med_shr = df["mae_shrunk"].median()
    print(f"\n[fd46] median MAE: base {med_base:.2f} -> shrunk {med_shr:.2f} "
          f"(delta {med_shr - med_base:+.2f})")
    hurt = ["VIC1", "UK_09_East_Midlands", "UK_15_England",
            "UK_16_Scotland", "US_ISNE", "US_CISO"]
    print("\n[fd46] FD-45 significantly-hurt regions:")
    print(df[df["target"].isin(hurt)].to_string(index=False))
    print("\n[fd46] all regions:")
    print(df.to_string(index=False))

    out = RESULTS / "fd46_event_shrinkage.json"
    with open(out, "w") as f:
        json.dump({"rows": rows,
                   "median_base": float(med_base),
                   "median_shrunk": float(med_shr)}, f, indent=1)
    print(f"\n[fd46] -> {out}")


if __name__ == "__main__":
    main()
