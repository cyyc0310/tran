#!/usr/bin/env python3
"""FD-50c: event-conditional transform oracle + deployable donor version.

Hypothesis (from FD-50): right-tail regions lose disproportionately on
wind-lull days (UK_08 129 vs 70, UK_01 101 vs 37, SA1 102 vs 85). The event
label is zero-telemetry (ERA5 20-pct wind threshold, FD-45 convention).
If the model's failure on event days is directional (under-predicts the
thermal-backup CIF rise), a donor-learned event-day transform could capture
it without target labels.

Oracle per region (cheating upper bounds):
  a_evt*, a_reg*   : separate within-window amplitude on event/regular
                     windows (window labelled by origin-day wind_lull)
  b_evt            : additive level offset on event windows
Deployable versions:
  donor median of a_evt*, a_reg*, b_evt (global + family LOO)

Output: results/fd50c_event.json + fd50c_event.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DUMP = ROOT / "results" / "dunkelflaute"
RESULTS = ROOT / "results"


def family_of(t: str) -> str:
    if t in ("VIC1", "NSW1", "SA1", "QLD1"):
        return "AU"
    if t.startswith("UK"):
        return "UK"
    return "US"


def amp(pred: np.ndarray, a: float, b: float = 0.0) -> np.ndarray:
    pm = pred.mean(axis=1, keepdims=True)
    return pm + b + a * (pred - pm)


def mae(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(p - y).mean())


def fit_a(pred: np.ndarray, y: np.ndarray) -> float:
    best_a, best_m = 1.0, mae(pred, y)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(amp(pred, a), y)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


def fit_b(pred: np.ndarray, y: np.ndarray) -> float:
    """Additive offset minimising MAE (median of window-mean errors)."""
    return float(np.median(y.mean(axis=1) - pred.mean(axis=1)))


def event_mask(d: dict, origin_ts: pd.DatetimeIndex) -> np.ndarray:
    """Window is 'event' if its origin day is a wind-lull day (20 pct)."""
    if len(d["wx_day_ord"]) == 0:
        return np.zeros(len(origin_ts), dtype=bool)
    w_thr = np.quantile(d["wx_w_mean"], 0.2)
    wx_days = pd.to_datetime(d["wx_day_ord"] - 719163, unit="D")
    lull = pd.Series(np.asarray(d["wx_w_mean"]) < w_thr, index=wx_days)
    return lull.reindex(origin_ts.normalize()).fillna(False).values


def main() -> None:
    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))
    data, evt = {}, {}
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        o = pd.to_datetime(d["origin_hours"] * 3600, unit="s")
        data[t] = {"y": d["y_true"].astype(np.float64),
                   "pred": d["pred_i_cfg"].astype(np.float64)}
        evt[t] = event_mask(d, o)

    # oracle fits
    fit = {}
    for t in targets:
        y, p, m = data[t]["y"], data[t]["pred"], evt[t]
        pe, pr, ye, yr = p[m], p[~m], y[m], y[~m]
        fit[t] = {
            "base": mae(p, y),
            "base_evt": mae(pe, ye), "base_reg": mae(pr, yr),
            "n_evt": int(m.sum()), "n_reg": int((~m).sum()),
            "a_evt": fit_a(pe, ye) if m.sum() >= 5 else None,
            "a_reg": fit_a(pr, yr),
            "b_evt": fit_b(pe, ye) if m.sum() >= 5 else None,
            "bias_evt": float((pe - ye).mean()) if m.sum() >= 5 else None,
            "bias_reg": float((pr - yr).mean()),
        }

    # deployable: donor medians (global LOO + family LOO)
    rows = []
    for t in targets:
        y, p, m = data[t]["y"], data[t]["pred"], evt[t]
        f = fit[t]
        others = [o for o in targets if o != t]
        fam = [o for o in others if family_of(o) == family_of(t)]

        def dmed(key, pool):
            vals = [fit[o][key] for o in pool if fit[o][key] is not None]
            return float(np.median(vals)) if vals else None

        g_ae, g_ar, g_b = (dmed("a_evt", others), dmed("a_reg", others),
                           dmed("b_evt", others))
        f_ae, f_ar, f_b = (dmed("a_evt", fam), dmed("a_reg", fam),
                           dmed("b_evt", fam))

        # combined deployable: a on regular + (a,b) on event
        def apply(a_r, a_e, b_e):
            out = amp(p, a_r)
            if a_e is not None and m.sum() > 0:
                out[m] = amp(p[m], a_e, b_e or 0.0)
            return out

        row = {
            "target": t, "family": family_of(t),
            "base": round(f["base"], 2),
            "base_evt": round(f["base_evt"], 2),
            "base_reg": round(f["base_reg"], 2),
            "n_evt": f["n_evt"],
            "a_evt": f["a_evt"], "a_reg": round(f["a_reg"], 2),
            "b_evt": round(f["b_evt"], 1) if f["b_evt"] is not None else None,
            "bias_evt": round(f["bias_evt"], 1) if f["bias_evt"] is not None else None,
            "bias_reg": round(f["bias_reg"], 1),
            # deployable
            "dep_g_mae": round(mae(apply(g_ar, g_ae, g_b), y), 2),
            "dep_f_mae": round(mae(apply(f_ar, f_ae, f_b), y), 2),
            # oracles
            "orc_a_only": round(mae(apply(f["a_reg"], f["a_evt"], 0.0), y), 2),
            "orc_ab": round(mae(apply(f["a_reg"], f["a_evt"], f["b_evt"]), y), 2),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    out = RESULTS / "fd50c_event.json"
    out.write_text(json.dumps({"rows": rows}, indent=1))

    md = ["# FD-50c 事件日条件化变换（wind-lull origin 日，零遥测标签）", "",
          "oracle：a_reg/a_evt/b_evt 用目标自身标签拟合（作弊上界）；",
          "dep_g/dep_f：donor 中位（全局/家族 LOO）代入，零目标标签", "",
          "## 汇总（29 区 pooled）", ""]
    base_p = df["base"].mean()
    md.append(f"- 基线：{base_p:.2f}")
    for col, name in [("dep_g_mae", "dep 全局 donor"),
                      ("dep_f_mae", "dep 家族 donor"),
                      ("orc_a_only", "oracle 分日幅度"),
                      ("orc_ab", "oracle 分日幅度+事件偏移")]:
        v = df[col].mean()
        md.append(f"- {name}：{v:.2f} ({v - base_p:+.2f})")
    md += ["", "## 事件日方向一致性（b_evt 与 bias_evt 符号）", ""]
    for _, r in df[df["base"] > 40].sort_values("base", ascending=False).iterrows():
        md.append(f"- {r['target']}: base {r['base']:.1f} "
                  f"(evt {r['base_evt']:.1f}/reg {r['base_reg']:.1f}, "
                  f"n_evt {r['n_evt']}), a_reg {r['a_reg']:.2f} "
                  f"a_evt {r['a_evt']}, b_evt {r['b_evt']}, "
                  f"bias evt {r['bias_evt']} / reg {r['bias_reg']}, "
                  f"dep_g {r['dep_g_mae']:.1f} dep_f {r['dep_f_mae']:.1f} "
                  f"orc_ab {r['orc_ab']:.1f}")
    md_path = RESULTS / "fd50c_event.md"
    md_path.write_text("\n".join(md))
    print(f"[fd50c] -> {md_path}")
    cols = ["target", "base", "base_evt", "base_reg", "n_evt", "a_reg",
            "a_evt", "b_evt", "bias_evt", "bias_reg", "dep_g_mae",
            "dep_f_mae", "orc_ab"]
    print(df[cols].sort_values("base", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
