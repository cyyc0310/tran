#!/usr/bin/env python3
"""FD-50e: gated delayed-label self-calibration (deployment recipe).

Problem (FD-50d): R5 (a* from early half) is -1.99 pooled but hurts
regions whose a* drifts temporally (UK_09 late base 60.8 -> R5 68.3).
b-offset helps SA1/US_CISO but hurts NSW1 badly.

Gate idea: the early half itself tells us whether a* != 1 is a real
signal — measure relative improvement of a* over a=1 on the EARLY half:
    imp_e = (mae_e(1.0) - mae_e(a_e)) / mae_e(1.0)
Apply a_e only when imp_e is large enough; otherwise fall back to 1.0
or to the family donor median (R2b). Same gate for b (plus a
sign-stability check inside the early half: split early into two
quarters, require same sign of b).

Strategies:
  G0  no gate (R5 / R5b from fd50d)
  G1  gate a on imp_e >= tau_a; else a=1
  G2  gate a on imp_e >= tau_a; else family median (R2b value)
  G3  G2 + gated b (imp_b >= tau_b AND sign stable across early quarters)
tau swept on a small grid; report pooled late-half MAE.

Output: results/fd50e_gate.json + fd50e_gate.md
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


def main() -> None:
    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))

    def family_of(t: str) -> str:
        if t in ("VIC1", "NSW1", "SA1", "QLD1"):
            return "AU"
        if t.startswith("UK"):
            return "UK"
        return "US"

    data = {}
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        o = np.asarray(d["origin_hours"])
        order = o.argsort()
        n = len(order)
        cut = n // 2
        early = np.zeros(n, dtype=bool)
        early[order[:cut]] = True
        # quarters inside early half (for b sign-stability)
        q1 = np.zeros(n, dtype=bool)
        q1[order[: n // 4]] = True
        data[t] = {
            "y": d["y_true"].astype(np.float64),
            "pred": d["pred_i_cfg"].astype(np.float64),
            "early": early, "q1": q1,
        }

    # precompute per-region quantities
    info = {}
    for t in targets:
        p, y, e = data[t]["pred"], data[t]["y"], data[t]["early"]
        q1 = data[t]["q1"]
        a_e = fit_a(p[e], y[e])
        m1 = mae(amp(p[e], 1.0), y[e])
        ma = mae(amp(p[e], a_e), y[e])
        imp_a = (m1 - ma) / m1
        b_e = fit_b(p[e], y[e])
        m_b0 = mae(amp(p[e], a_e), y[e])
        m_b = mae(amp(p[e], a_e, b_e), y[e])
        imp_b = (m_b0 - m_b) / m_b0
        b_q1 = fit_b(p[q1], y[q1])           # first quarter offset
        b_q2 = fit_b(p[e & ~q1], y[e & ~q1])  # second quarter offset
        b_stable = (np.sign(b_q1) == np.sign(b_q2)) and \
            abs(b_q1 - b_q2) <= max(3.0, 0.5 * abs(b_e) + 3.0)
        info[t] = {
            "a_e": a_e, "imp_a": imp_a, "b_e": b_e, "imp_b": imp_b,
            "b_stable": bool(b_stable),
            "base_late": mae(p[~e], y[~e]),
            "fam": family_of(t),
        }

    fam_med = {}
    for fam in ("AU", "UK", "US"):
        vals = [info[t]["a_e"] for t in targets if info[t]["fam"] == fam]
        fam_med[fam] = float(np.median(vals))

    # ---- sweep ------------------------------------------------------------
    results = {}
    best = None
    for tau_a in (0.0, 0.02, 0.05, 0.08, 0.12):
        for mode in ("identity", "family"):
            for tau_b in (0.99, 0.02, 0.05):   # 0.99 = b disabled
                late_map = {}
                for t in targets:
                    p, y, e = (data[t]["pred"], data[t]["y"],
                               data[t]["early"])
                    i = info[t]
                    if i["imp_a"] >= tau_a:
                        a_use = i["a_e"]
                    else:
                        a_use = fam_med[i["fam"]] if mode == "family" else 1.0
                    b_use = 0.0
                    if tau_b < 0.99 and i["imp_b"] >= tau_b and i["b_stable"]:
                        b_use = i["b_e"]
                    late_map[t] = mae(amp(p[~e], a_use, b_use), y[~e])
                pooled = float(np.mean(list(late_map.values())))
                key = f"tau_a={tau_a:.2f}/{mode}/tau_b={tau_b:.2f}"
                results[key] = round(pooled, 2)
                if best is None or pooled < best[1]:
                    best = (key, pooled, late_map)

    base_late_pooled = float(np.mean([info[t]["base_late"]
                                      for t in targets]))
    out = {
        "base_late_pooled": round(base_late_pooled, 2),
        "sweep": results,
        "best": {"config": best[0], "pooled": round(best[1], 2),
                 "per_region": best[2]},
    }
    (RESULTS / "fd50e_gate.json").write_text(json.dumps(out, indent=1))

    # reference arms on the same late halves
    r5 = float(np.mean([mae(amp(data[t]["pred"][~data[t]["early"]],
                                info[t]["a_e"]),
                           data[t]["y"][~data[t]["early"]]) for t in targets]))

    md = ["# FD-50e 门控延迟自校准（部署配方）", "",
          f"late 半段 pooled 基线：{base_late_pooled:.2f}",
          f"R5 无门控参照：{r5:.2f} ({r5 - base_late_pooled:+.2f})", "",
          "## tau 扫描（pooled late MAE）", ""]
    for k, v in sorted(results.items()):
        md.append(f"- {k}: {v:.2f} ({v - base_late_pooled:+.2f})")
    md += ["", f"## 最优配置：{best[0]}",
           f"- pooled：{best[1]:.2f} ({best[1] - base_late_pooled:+.2f})", "",
           "## 右尾区明细（最优配置 vs 无门控 R5）", ""]
    r5_map = {t: mae(amp(data[t]["pred"][~data[t]["early"]], info[t]["a_e"]),
                     data[t]["y"][~data[t]["early"]]) for t in targets}
    md += ["| target | late基线 | R5 | gated | Δ(gated−R5) |",
           "|---|---|---|---|---|"]
    for t in sorted(targets, key=lambda x: -info[x]["base_late"]):
        i = info[t]
        if i["base_late"] > 40:
            md.append(f"| {t} | {i['base_late']:.1f} | {r5_map[t]:.1f} "
                      f"| {best[2][t]:.1f} "
                      f"| {best[2][t] - r5_map[t]:+.1f} |")
    (RESULTS / "fd50e_gate.md").write_text("\n".join(md))
    print(f"[fd50e] best: {best[0]} pooled {best[1]:.2f} "
          f"({best[1] - base_late_pooled:+.2f})")
    for k, v in sorted(results.items())[:8]:
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
