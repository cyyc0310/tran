#!/usr/bin/env python3
"""FD-50i: event-window source fallback (persistence blend) — final recipe.

FD-50h diagnostic: on late-half event windows (zero-telemetry wx q20
label) `pred_persistence` (previous-day truth carried forward —
deployable delayed label) crushes `pred_i_cfg` in several right-tail
regions (UK_08 103->61, UK_12 73->39, UK_05 58->37, NSW1 83->37, SA1
97->67, US_CISO 44->25). Early-half event preference is 9/9 stable
where it says "pers", flips only in the other direction.

Arms (late half, non-event windows always get FD-50g B4@0.60):
  R0    B4@0.60 everywhere (reference)
  A     event windows switch to persistence if early-half event MAE
        favors pers
  B     event windows always blend 50/50 i_cfg+persistence
  ORC   event windows switch per late-half truth (cheating bound)

Final recipe candidate: B (simplest, no gate needed).

Output: results/fd50i_event_fallback.json / .md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DUMP = ROOT / "results" / "dunkelflaute"
RESULTS = ROOT / "results"

NB = 3
TAU_S = 0.60
OFF = 718872


def mae(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(p - y).mean())


def fit_a_pivot(ps: np.ndarray, pm: np.ndarray,
                ys: np.ndarray) -> float:
    best_a, best_m = 1.0, mae(pm + 1.0 * (ps - pm), ys)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(pm + a * (ps - pm), ys)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


def main() -> None:
    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))

    rows = []
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        o = np.asarray(d["origin_hours"])
        order = o.argsort()
        n = len(order)
        y = d["y_true"].astype(np.float64)[order]
        p = d["pred_i_cfg"].astype(np.float64)[order]
        per = np.asarray(d["pred_persistence"]).astype(np.float64)[order]
        w = np.asarray(d["wx_w_mean"])
        wo = np.asarray(d["wx_day_ord"])
        wmap = dict(zip(wo.tolist(), w.tolist()))
        ev = np.array([wmap[dd] for dd in (o // 24 + OFF)]) \
            <= np.quantile(w, 0.20)
        h = p.shape[1]
        cuts = [(i * h // NB, (i + 1) * h // NB) for i in range(NB)]
        e, l = slice(0, n // 2), slice(n // 2, n)
        q1, q2 = slice(0, n // 4), slice(n // 4, n // 2)

        # ---- B4@0.60 recipe ----
        pm_e = p[e].mean(axis=1, keepdims=True)
        fit_e = [fit_a_pivot(p[e, s:e2], pm_e, y[e, s:e2])
                 for s, e2 in cuts]
        a_g = fit_a_pivot(p[e], pm_e, y[e])
        pm1 = p[q1].mean(axis=1, keepdims=True)
        pm2 = p[q2].mean(axis=1, keepdims=True)
        ok = [abs(fit_a_pivot(p[q1, s:e2], pm1, y[q1, s:e2])
                  - fit_a_pivot(p[q2, s:e2], pm2, y[q2, s:e2])) <= TAU_S
              for s, e2 in cuts]
        a_b4 = [fit_e[b] if ok[b] else a_g for b in range(NB)]
        pm_l = p[l].mean(axis=1, keepdims=True)
        out_r0 = np.empty_like(p[l])
        for (s, e2), a in zip(cuts, a_b4):
            out_r0[:, s:e2] = pm_l + a * (p[l, s:e2] - pm_l)
        m_r0 = mae(out_r0, y[l])
        base_l = mae(p[l], y[l])

        evl = ev[l]
        eve = ev[e]
        if eve.sum() >= 1 and (~eve).sum() >= 1:
            m_cfg_e = mae(p[e][eve], y[e][eve])
            m_per_e = mae(per[e][eve], y[e][eve])
        else:
            m_cfg_e = m_per_e = float("nan")

        # ---- A: gated switch ----
        out_a = out_r0.copy()
        if evl.sum() > 0 and m_per_e < m_cfg_e:
            out_a[evl] = per[l][evl]
        m_a = mae(out_a, y[l])

        # ---- B: unconditional blend on event windows ----
        out_b = out_r0.copy()
        if evl.sum() > 0:
            out_b[evl] = 0.5 * (p[l][evl] + per[l][evl])
        m_b = mae(out_b, y[l])

        # ---- ORC: late-truth switch ----
        out_o = out_r0.copy()
        if evl.sum() > 0 and \
                mae(per[l][evl], y[l][evl]) < mae(p[l][evl], y[l][evl]):
            out_o[evl] = per[l][evl]
        m_o = mae(out_o, y[l])

        rows.append({
            "target": t, "n_evt_late": int(evl.sum()),
            "base_late": round(base_l, 2), "R0": round(m_r0, 2),
            "A": round(m_a, 2), "B": round(m_b, 2),
            "ORC": round(m_o, 2),
            "evt_cfg": round(mae(p[l][evl], y[l][evl]), 1)
            if evl.sum() else None,
            "evt_per": round(mae(per[l][evl], y[l][evl]), 1)
            if evl.sum() else None,
        })

    df = pd.DataFrame(rows)
    (RESULTS / "fd50i_event_fallback.json").write_text(
        json.dumps({"rows": rows}, indent=1))

    rt = df[df["base_late"] > 40]
    md = ["# FD-50i 事件窗源回退（persistence 混合）— final recipe",
          "",
          "非事件窗：FD-50g B4@0.60；事件窗（wx q20 零遥测标签）策略对比：",
          "",
          f"## pooled（29 区）",
          f"- R0(B4) {df['R0'].mean():.2f} | A 门控切换 {df['A'].mean():.2f} "
          f"| B 全混合 {df['B'].mean():.2f} | ORC(cheat) {df['ORC'].mean():.2f}",
          "",
          f"## 右尾区（n={len(rt)}）",
          f"- R0(B4) {rt['R0'].mean():.2f} | A {rt['A'].mean():.2f} "
          f"| B {rt['B'].mean():.2f} | ORC {rt['ORC'].mean():.2f}",
          "",
          "## 全区明细（按 base_late 降序）", "",
          "| target | nE | base | R0 | A | B | ORC | evt cfg/per |",
          "|---|---|---|---|---|---|---|---|"]
    for _, r in df.sort_values("base_late", ascending=False).iterrows():
        md.append(
            f"| {r['target']} | {r['n_evt_late']} | {r['base_late']:.1f} "
            f"| {r['R0']:.1f} | {r['A']:.1f} | {r['B']:.1f} "
            f"| {r['ORC']:.1f} | {r['evt_cfg']}/{r['evt_per']} |")
    (RESULTS / "fd50i_event_fallback.md").write_text("\n".join(md))

    cols = ["target", "n_evt_late", "base_late", "R0", "A", "B", "ORC"]
    print(df[cols].sort_values("base_late", ascending=False)
          .to_string(index=False))
    print()
    print(f"pooled: R0 {df['R0'].mean():.2f} | A {df['A'].mean():.2f} "
          f"| B {df['B'].mean():.2f} | ORC {df['ORC'].mean():.2f}")
    print(f"RT    : R0 {rt['R0'].mean():.2f} | A {rt['A'].mean():.2f} "
          f"| B {rt['B'].mean():.2f} | ORC {rt['ORC'].mean():.2f}")
    # B vs R0 regressions
    hurt = [(r["target"], r["R0"], r["B"])
            for _, r in df.iterrows() if r["B"] - r["R0"] > 1.0]
    print()
    print("B 相对 R0 恶化>1 的区：", hurt if hurt else "无")


if __name__ == "__main__":
    main()
