#!/usr/bin/env python3
"""FD-50h: event-conditional per-bucket amplification, family-pooled delta.

FD-50g diagnostic (late-half oracle, cheating): event-day (wx wind <=
per-region q20, zero-telemetry label) bucket a* diverges upward from
regular-day a* in the mid/late buckets for the UK right-tail regions
(UK_08/12/06/17/03/09: delta up to +1.9 on b2) while SA1/UK_07 show
almost no divergence. Event windows carry the dominant excess error
(UK_08: 103 vs 63).

Per-region event fitting fails (n_evt=2..10, FD-50f lesson), so the
deployable arm pools the evt-reg delta per bucket ACROSS the family,
fit on the early half only:

  a_evt_pred(t,b) = a_B4recipe(t,b) + median_{s in fam} delta(s,b)
  delta(s,b) = a_evt_early(s,b) - a_reg_early(s,b)

Arms (late half, event windows only get the event path):
  R0   FD-50g B4@0.60 applied to all windows (reference)
  H1   per-region early event a* on event windows (expected unstable)
  H2   family-pooled early delta on event windows
  H4   H2 gated by region's early event-excess ratio maeE/maeR >= 1.25
  H3   family-pooled LATE delta (cheating)
  ORC  per-region late event a* (cheating, unstable)

Output: results/fd50h_event_bucket.json / .md
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
TAU_S = 0.60          # B4 stability gate from FD-50g
OFF = 718872          # origin_hours//24 + OFF == wx_day_ord
GATE_RATIO = 1.25     # H4 event-excess gate


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


def family_of(t: str) -> str:
    if t in ("VIC1", "NSW1", "SA1", "QLD1"):
        return "AU"
    if t.startswith("UK"):
        return "UK"
    return "US"


def clip(a: float) -> float:
    return float(np.clip(a, 0.30, 2.30))


def main() -> None:
    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))

    data = {}
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        o = np.asarray(d["origin_hours"])
        order = o.argsort()
        n = len(order)
        y = d["y_true"].astype(np.float64)[order]
        p = d["pred_i_cfg"].astype(np.float64)[order]
        w = np.asarray(d["wx_w_mean"])
        wo = np.asarray(d["wx_day_ord"])
        wmap = dict(zip(wo.tolist(), w.tolist()))
        w_win = np.array([wmap[dd] for dd in (o // 24 + OFF)])
        evt = w_win <= np.quantile(w, 0.20)
        h = p.shape[1]
        cuts = [(i * h // NB, (i + 1) * h // NB) for i in range(NB)]
        data[t] = {
            "y": y, "p": p, "evt": evt, "cuts": cuts,
            "e": slice(0, n // 2), "l": slice(n // 2, n),
            "q1": slice(0, n // 4), "q2": slice(n // 4, n // 2),
            "fam": family_of(t),
        }

    # ---------- early-half fits ----------
    fit_e = {}      # per-bucket a on early (all windows)  == FD-50g fit_e
    fit_e_evt = {}  # per-bucket a on early event windows
    a_glob_e = {}
    imp_ok = {}     # q1/q2 stability per bucket (B4 gate)
    for t in targets:
        dd = data[t]
        p, y, cuts, evt = dd["p"], dd["y"], dd["cuts"], dd["evt"]
        e, q1, q2 = dd["e"], dd["q1"], dd["q2"]
        pm_e = p[e].mean(axis=1, keepdims=True)
        fit_e[t] = [fit_a_pivot(p[e, s:e2], pm_e, y[e, s:e2])
                    for s, e2 in cuts]
        a_glob_e[t] = fit_a_pivot(p[e], pm_e, y[e])
        if evt[e].sum() >= 2:
            pm_ee = p[e][evt[e]].mean(axis=1, keepdims=True)
            fit_e_evt[t] = [fit_a_pivot(p[e][evt[e]][:, s:e2], pm_ee,
                                        y[e][evt[e]][:, s:e2])
                            for s, e2 in cuts]
        else:
            fit_e_evt[t] = None
        pm1 = p[q1].mean(axis=1, keepdims=True)
        pm2 = p[q2].mean(axis=1, keepdims=True)
        imp_ok[t] = [abs(
            fit_a_pivot(p[q1, s:e2], pm1, y[q1, s:e2])
            - fit_a_pivot(p[q2, s:e2], pm2, y[q2, s:e2])) <= TAU_S
            for s, e2 in cuts]

    # ---------- family-pooled deltas ----------
    def pooled_delta(fit_evt, half) -> dict:
        out = {}
        for fam in ("AU", "UK", "US"):
            rows = []
            for t in targets:
                if data[t]["fam"] != fam:
                    continue
                dd = data[t]
                p, y, cuts, evt = (dd["p"], dd["y"], dd["cuts"],
                                   dd["evt"])
                sl = dd["e"] if half == "early" else dd["l"]
                if evt[sl].sum() < 2:
                    continue
                pm_s = p[sl][evt[sl]].mean(axis=1, keepdims=True)
                a_ev = [fit_a_pivot(p[sl][evt[sl]][:, s:e2], pm_s,
                                    y[sl][evt[sl]][:, s:e2])
                        for s, e2 in cuts]
                if fit_evt == "pooled":
                    # regular a on the same half (all windows)
                    pm_a = p[sl].mean(axis=1, keepdims=True)
                    a_rg = [fit_a_pivot(p[sl, s:e2], pm_a,
                                        y[sl, s:e2])
                            for s, e2 in cuts]
                else:
                    a_rg = fit_e[t]
                rows.append([a_ev[b] - a_rg[b] for b in range(NB)])
            out[fam] = (np.median(np.array(rows), axis=0).tolist()
                        if rows else [0.0] * NB)
        return out

    delta_e = pooled_delta("pooled", "early")   # deployable
    delta_l = pooled_delta("pooled", "late")    # cheating

    # ---------- late-half evaluation ----------
    rows = []
    for t in targets:
        dd = data[t]
        p, y, cuts, evt = dd["p"], dd["y"], dd["cuts"], dd["evt"]
        e, l = dd["e"], dd["l"]
        pm_l = p[l].mean(axis=1, keepdims=True)
        ev_l = evt[l]

        # R0: FD-50g B4@0.60 recipe on all late windows
        a_b4 = [fit_e[t][b] if imp_ok[t][b] else a_glob_e[t]
                for b in range(NB)]
        out_r0 = np.empty_like(p[l])
        for (s, e2), a in zip(cuts, a_b4):
            out_r0[:, s:e2] = pm_l + a * (p[l, s:e2] - pm_l)
        m_r0 = mae(out_r0, y[l])

        # event-window subset MAE of R0
        m_r0_evt = (mae(out_r0[ev_l], y[l][ev_l])
                    if ev_l.sum() >= 2 else float("nan"))

        def evt_apply(a_evt_list) -> float:
            out = out_r0.copy()
            if ev_l.sum() == 0:
                return m_r0
            pe = p[l][ev_l]
            pm_e = pe.mean(axis=1, keepdims=True)
            oe = np.empty_like(pe)
            for (s, e2), a in zip(cuts, a_evt_list):
                oe[:, s:e2] = pm_e + a * (pe[:, s:e2] - pm_e)
            out[ev_l] = oe
            return mae(out, y[l])

        fam = dd["fam"]

        # H1: per-region early event a*
        m_h1 = (evt_apply(fit_e_evt[t])
                if fit_e_evt[t] is not None else m_r0)

        # H2: B4 + family-pooled early delta (event windows only)
        m_h2 = evt_apply([clip(a_b4[b] + delta_e[fam][b])
                          for b in range(NB)])

        # H4: H2 gated by early event-excess ratio
        if evt[e].sum() >= 2 and (~evt[e]).sum() >= 2:
            ratio = mae(p[e][evt[e]], y[e][evt[e]]) / \
                mae(p[e][~evt[e]], y[e][~evt[e]])
        else:
            ratio = float("nan")
        m_h4 = (m_h2 if ratio >= GATE_RATIO else m_r0)

        # H3: cheating late delta
        m_h3 = evt_apply([clip(a_b4[b] + delta_l[fam][b])
                          for b in range(NB)])

        # ORC: per-region late event a* (cheating, unstable)
        if ev_l.sum() >= 2:
            pm_el = p[l][ev_l].mean(axis=1, keepdims=True)
            a_orc = [fit_a_pivot(p[l][ev_l][:, s:e2], pm_el,
                                 y[l][ev_l][:, s:e2])
                     for s, e2 in cuts]
            m_orc = evt_apply(a_orc)
            m_orc_evt = mae(pm_el + 0 * p[l][ev_l], y[l][ev_l]) \
                if False else None
        else:
            m_orc = m_r0

        rows.append({
            "target": t, "n_evt_late": int(ev_l.sum()),
            "ratio_e": round(ratio, 2) if ratio == ratio else None,
            "R0": round(m_r0, 2), "H1": round(m_h1, 2),
            "H2": round(m_h2, 2), "H4": round(m_h4, 2),
            "H3": round(m_h3, 2), "ORC": round(m_orc, 2),
            "r0_evt": round(m_r0_evt, 2) if m_r0_evt == m_r0_evt
            else None,
            "delta_fam": [round(v, 2) for v in delta_e[fam]],
        })

    df = pd.DataFrame(rows)
    (RESULTS / "fd50h_event_bucket.json").write_text(
        json.dumps({"rows": rows,
                    "delta_e": delta_e, "delta_l": delta_l},
                   indent=1))

    rt = df[df["R0"] > 40]
    md = ["# FD-50h 事件条件化 × 分桶放大（家族池化 delta）— late 半段",
          "",
          f"R0 = FD-50g B4@0.60；事件标签 = wx_w_mean ≤ 区内 q20（零遥测）",
          "",
          f"## pooled（29 区）",
          f"- R0 {df['R0'].mean():.2f} | H1 {df['H1'].mean():.2f} "
          f"| H2 {df['H2'].mean():.2f} | H4 {df['H4'].mean():.2f} "
          f"| H3(cheat) {df['H3'].mean():.2f} | ORC {df['ORC'].mean():.2f}",
          "",
          f"## 右尾区（n={len(rt)}）",
          f"- R0 {rt['R0'].mean():.2f} | H1 {rt['H1'].mean():.2f} "
          f"| H2 {rt['H2'].mean():.2f} | H4 {rt['H4'].mean():.2f} "
          f"| H3(cheat) {rt['H3'].mean():.2f} | ORC {rt['ORC'].mean():.2f}",
          "",
          "## 家族池化 delta（early 拟合，部署版）",
          f"- UK {delta_e['UK']} | AU {delta_e['AU']} "
          f"| US {delta_e['US']}",
          f"- （cheating late 版：UK {delta_l['UK']} "
          f"| AU {delta_l['AU']} | US {delta_l['US']}）",
          "",
          "## 右尾区明细", "",
          "| target | nE | ratio | R0 | H1 | H2 | H4 | H3 | ORC |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in rt.sort_values("R0", ascending=False).iterrows():
        md.append(
            f"| {r['target']} | {r['n_evt_late']} | {r['ratio_e']} "
            f"| {r['R0']:.1f} | {r['H1']:.1f} | {r['H2']:.1f} "
            f"| {r['H4']:.1f} | {r['H3']:.1f} | {r['ORC']:.1f} |")
    (RESULTS / "fd50h_event_bucket.md").write_text("\n".join(md))

    cols = ["target", "n_evt_late", "ratio_e", "R0", "H1", "H2",
            "H4", "H3", "ORC"]
    print(df[cols].sort_values("R0", ascending=False)
          .to_string(index=False))
    print()
    print(f"pooled: R0 {df['R0'].mean():.2f} | H1 {df['H1'].mean():.2f} "
          f"| H2 {df['H2'].mean():.2f} | H4 {df['H4'].mean():.2f} "
          f"| H3 {df['H3'].mean():.2f} | ORC {df['ORC'].mean():.2f}")
    print(f"RT    : R0 {rt['R0'].mean():.2f} | H1 {rt['H1'].mean():.2f} "
          f"| H2 {rt['H2'].mean():.2f} | H4 {rt['H4'].mean():.2f} "
          f"| H3 {rt['H3'].mean():.2f} | ORC {rt['ORC'].mean():.2f}")
    print()
    print("delta_e:", delta_e)
    print("delta_l:", delta_l)


if __name__ == "__main__":
    main()
