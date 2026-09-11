#!/usr/bin/env python3
"""FD-50f: event-conditioned delayed-label calibration.

FD-50c oracle (fit on own full test): -4.08 pooled, mostly from b_evt.
FD-50d R5 (delayed labels, amplitude only): -1.99 late-half.
This experiment combines them deployably: fit a_reg, a_evt, b_evt on
the EARLY half (delayed labels), apply to the LATE half, with a
sign-stability gate on b_evt (split early into quarters).

Arms on the late half:
  base       : a=1, b=0
  R5         : a_reg only (from fd50d, reproduced here)
  E1         : a_reg + b_evt (gated: b sign stable across early quarters)
  E2         : a_reg + a_evt + b_evt (a_evt only if n_evt_early >= 8)
  oracle-E   : a_reg/a_evt/b_evt refit on the late half itself (cheating)

Output: results/fd50f_event_delayed.json + fd50f_event_delayed.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DUMP = ROOT / "results" / "dunkelflaute"
RESULTS = ROOT / "results"


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
    return float(np.median(y.mean(axis=1) - pred.mean(axis=1)))


def event_mask(d: dict, origin_ts: pd.DatetimeIndex) -> np.ndarray:
    if len(d["wx_day_ord"]) == 0:
        return np.zeros(len(origin_ts), dtype=bool)
    w_thr = np.quantile(d["wx_w_mean"], 0.2)
    wx_days = pd.to_datetime(d["wx_day_ord"] - 719163, unit="D")
    lull = pd.Series(np.asarray(d["wx_w_mean"]) < w_thr, index=wx_days)
    return lull.reindex(origin_ts.normalize()).fillna(False).values


def main() -> None:
    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))

    data = {}
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        o = pd.to_datetime(d["origin_hours"] * 3600, unit="s")
        o_ord = np.asarray(d["origin_hours"])
        order = o_ord.argsort()
        n = len(order)
        cut = n // 2
        early = np.zeros(n, dtype=bool)
        early[order[:cut]] = True
        q1 = np.zeros(n, dtype=bool)
        q1[order[: n // 4]] = True
        data[t] = {
            "y": d["y_true"].astype(np.float64),
            "pred": d["pred_i_cfg"].astype(np.float64),
            "early": early, "q1": q1,
            "evt": event_mask(d, o),
        }

    rows = []
    for t in targets:
        p, y, e, q1, ev = (data[t]["pred"], data[t]["y"],
                           data[t]["early"], data[t]["q1"],
                           data[t]["evt"])
        pe, ye, ee = p[e], y[e], e
        ev_e, ev_l = ev[e], ev[~e]
        base_l = mae(p[~e], y[~e])

        # ---- fits on early half (delayed labels) ----
        a_reg = fit_a(p[e & ~ev], y[e & ~ev])
        n_ev_e = int((e & ev).sum())
        a_evt = fit_a(p[e & ev], y[e & ev]) if n_ev_e >= 8 else None
        b_evt = fit_b(p[e & ev], y[e & ev]) if n_ev_e >= 4 else None
        # sign stability of b_evt across early quarters
        bq1 = fit_b(p[q1 & ev], y[q1 & ev]) if (q1 & ev).sum() >= 2 else None
        bq2 = fit_b(p[e & ~q1 & ev], y[e & ~q1 & ev]) \
            if (e & ~q1 & ev).sum() >= 2 else None
        b_stable = (bq1 is not None and bq2 is not None
                    and np.sign(bq1) == np.sign(bq2))

        # ---- apply on late half ----
        late_pred = {k: None for k in
                     ("base", "r5", "e1", "e2", "orc_e")}

        # base
        late_pred["base"] = amp(p[~e], 1.0)
        # R5: a_reg everywhere
        late_pred["r5"] = amp(p[~e], a_reg)
        # E1: a_reg + gated b_evt on event windows
        pl = np.array(late_pred["r5"])
        if b_evt is not None and b_stable:
            tmp = amp(p[~e][ev_l], a_reg, b_evt)
            pl[ev_l] = tmp
        late_pred["e1"] = pl
        # E2: E1 + a_evt on event windows (if estimated)
        pl2 = np.array(late_pred["e1"])
        if a_evt is not None and b_stable:
            tmp = amp(p[~e][ev_l], a_evt, b_evt or 0.0)
            pl2[ev_l] = tmp
        late_pred["e2"] = pl2
        # oracle-E: refit everything on late half
        p_l, y_l = p[~e], y[~e]
        a_reg_l = fit_a(p_l[~ev_l], y_l[~ev_l])
        a_evt_l = fit_a(p_l[ev_l], y_l[ev_l]) if ev_l.sum() >= 4 else None
        b_evt_l = fit_b(p_l[ev_l], y_l[ev_l]) if ev_l.sum() >= 4 else None
        plo = amp(p_l, a_reg_l)
        if a_evt_l is not None or b_evt_l is not None:
            tmp = amp(p_l[ev_l], a_evt_l or a_reg_l, b_evt_l or 0.0)
            plo[ev_l] = tmp
        late_pred["orc_e"] = plo

        row = {
            "target": t, "base_late": round(base_l, 2),
            "a_reg": round(a_reg, 2),
            "n_evt_early": n_ev_e, "n_evt_late": int(ev_l.sum()),
            "b_evt": round(b_evt, 1) if b_evt is not None else None,
            "b_stable": bool(b_stable),
        }
        for k in ("base", "r5", "e1", "e2", "orc_e"):
            row[f"{k}_mae"] = round(mae(late_pred[k], y[~e]), 2)
        # event-day MAE on late half for each arm
        for k in ("base", "r5", "e1", "e2", "orc_e"):
            if ev_l.sum() > 0:
                row[f"{k}_evt"] = round(
                    mae(late_pred[k][ev_l], y[~e][ev_l]), 2)
        rows.append(row)

    df = pd.DataFrame(rows)
    (RESULTS / "fd50f_event_delayed.json").write_text(
        json.dumps({"rows": rows}, indent=1))

    bl = df["base_mae"].mean()
    md = ["# FD-50f 事件条件化 × 延迟标签（late 半段）", "",
          "early 半段拟合 a_reg/a_evt/b_evt（延迟标签，b 需跨季符号稳定）→ late 半段应用", "",
          "## 汇总（29 区 late 半段 pooled）", "",
          f"- 基线：{bl:.2f}"]
    for k, name in (("r5", "R5 仅幅度"), ("e1", "E1 幅度+事件偏移"),
                    ("e2", "E2 幅度+事件幅度+偏移"),
                    ("orc_e", "oracle-E（late 自拟合）")):
        v = df[f"{k}_mae"].mean()
        md.append(f"- {name}：{v:.2f} ({v - bl:+.2f})")
    rt = df[df["base_late"] > 40]
    md += ["", f"## 右尾区（n={len(rt)}）", ""]
    md.append(f"- 基线 {rt['base_mae'].mean():.2f} → "
              f"R5 {rt['r5_mae'].mean():.2f} | E1 {rt['e1_mae'].mean():.2f} "
              f"| E2 {rt['e2_mae'].mean():.2f} "
              f"| oracle {rt['orc_e_mae'].mean():.2f}")
    md += ["", "## 右尾区明细", "",
           "| target | late基线 | R5 | E1 | E2 | oracle | n_evt(e/l) | b_evt |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in rt.sort_values("base_late", ascending=False).iterrows():
        md.append(
            f"| {r['target']} | {r['base_mae']:.1f} | {r['r5_mae']:.1f} "
            f"| {r['e1_mae']:.1f} | {r['e2_mae']:.1f} | {r['orc_e_mae']:.1f} "
            f"| {r['n_evt_early']}/{r['n_evt_late']} | {r['b_evt']} |")
    (RESULTS / "fd50f_event_delayed.md").write_text("\n".join(md))
    print(f"[fd50f] -> results/fd50f_event_delayed.md")
    cols = ["target", "base_mae", "r5_mae", "e1_mae", "e2_mae",
            "orc_e_mae", "n_evt_early", "b_evt", "b_stable"]
    print(df[cols].sort_values("base_mae", ascending=False)
          .to_string(index=False))


if __name__ == "__main__":
    main()
