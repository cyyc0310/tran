"""FD-50j: online recursive residual tracking on top of the FD-50i recipe.

Information tier: identical to pred_persistence (yesterday's truth = 1-day
delayed label).  Nothing else is added.

Base recipe R (FD-50g B4@0.60 + FD-50i event 50/50 persistence mix) is
recomputed here; the online layers are stacked on top of R:

  J1  global EWMA level bias      b_t = (1-beta) b_{t-1} + beta * mean(y_{t-1} - r_{t-1})
  J2  hour-of-day EWMA bias       24-vector, same recursion per hour slot
  J3  all-window convex blend with persistence, weight w fitted on early half
  J4  J1 + J3
  J5  J2 + J3

Hyper-parameter selection for generalisability:
  fixed  : one beta / one w for all 29 regions (grid), reported per value
  LORO   : for each target region, pick beta on the pooled EARLY halves of the
           other 28 regions (zero target-region tuning), apply to target late.

Evaluation: late half (honest, no target-late information), plus full period.
Subsets: all 29, MAE>30 on base late, MAE>40 on base late.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUMP = Path("results/dunkelflaute")
OUT_JSON = Path("results/fd50j_online_bias.json")
OUT_MD = Path("results/fd50j_online_bias.md")
NB = 3
TAU_S = 0.60
OFFD = 718872
W_EVT = 0.5
BETAS = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7]
WS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]


def mae(p, y):
    return float(np.abs(p - y).mean())


def fit_a_pivot(ps, pm, ys):
    best_a, best_m = 1.0, mae(pm + 1.0 * (ps - pm), ys)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(pm + a * (ps - pm), ys)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


def load_region(t):
    d = np.load(DUMP / f"{t}_seed0.npz")
    o = np.asarray(d["origin_hours"])
    order = o.argsort()
    o = o[order]
    y = d["y_true"].astype(float)[order]
    p = d["pred_i_cfg"].astype(float)[order]
    per = np.asarray(d["pred_persistence"]).astype(float)[order]
    w = np.asarray(d["wx_w_mean"])
    wo = np.asarray(d["wx_day_ord"])
    wmap = dict(zip(wo.tolist(), w.tolist()))
    ev = np.array([wmap[dd] for dd in (o // 24 + OFFD)]) <= np.quantile(w, 0.20)
    return o, y, p, per, ev


def recipe_R(y, p, per, ev):
    """FD-50g B4@0.60 (early-fit) + FD-50i event 50/50 mix, applied to full period."""
    nn, h = p.shape
    cuts = [(i * h // NB, (i + 1) * h // NB) for i in range(NB)]
    e = slice(0, nn // 2)
    q1 = slice(0, nn // 4)
    q2 = slice(nn // 4, nn // 2)
    pm_e = p[e].mean(axis=1, keepdims=True)
    fit_e = [fit_a_pivot(p[e, s:ee], pm_e, y[e, s:ee]) for s, ee in cuts]
    a_g = fit_a_pivot(p[e], pm_e, y[e])
    pm1 = p[q1].mean(axis=1, keepdims=True)
    pm2 = p[q2].mean(axis=1, keepdims=True)
    ok = [
        abs(fit_a_pivot(p[q1, s:ee], pm1, y[q1, s:ee]) - fit_a_pivot(p[q2, s:ee], pm2, y[q2, s:ee])) <= TAU_S
        for s, ee in cuts
    ]
    a_b4 = [fit_e[b] if ok[b] else a_g for b in range(NB)]
    pm = p.mean(axis=1, keepdims=True)
    out = np.empty_like(p)
    for (s, ee), a in zip(cuts, a_b4):
        out[:, s:ee] = pm + a * (p[:, s:ee] - pm)
    out[ev] = (1 - W_EVT) * p[ev] + W_EVT * per[ev]
    return out


def ewma_global(r, y, beta):
    """Sequential: correction for window t uses residuals of windows < t only."""
    out = np.empty_like(r)
    b = 0.0
    for t in range(len(r)):
        out[t] = r[t] + b
        b = (1 - beta) * b + beta * float((y[t] - r[t]).mean())
    return out


def ewma_hod(r, y, beta):
    out = np.empty_like(r)
    b = np.zeros(r.shape[1])
    for t in range(len(r)):
        out[t] = r[t] + b
        b = (1 - beta) * b + beta * (y[t] - r[t])
    return out


def blend_per(r, per, w):
    return (1 - w) * r + w * per


def fit_w_early(r, per, y, nn):
    e = slice(0, nn // 2)
    best_w, best_m = 0.0, mae(r[e], y[e])
    for w in np.arange(0.0, 0.71, 0.05):
        m = mae(blend_per(r[e], per[e], w), y[e])
        if m < best_m:
            best_w, best_m = float(w), m
    return best_w


def main():
    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
    data = {t: load_region(t) for t in targets}
    R = {t: recipe_R(data[t][1], data[t][2], data[t][3], data[t][4]) for t in targets}

    # --- per region, per beta/w: early & late MAE for each arm -------------
    rows = {}
    for t in targets:
        o, y, p, per, ev = data[t]
        r = R[t]
        nn = len(y)
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)
        row = {
            "base_early": mae(p[e], y[e]), "base_late": mae(p[l], y[l]), "base_full": mae(p, y),
            "R_early": mae(r[e], y[e]), "R_late": mae(r[l], y[l]), "R_full": mae(r, y),
            "per_late": mae(per[l], y[l]),
            "J1": {}, "J2": {}, "J3": {}, "J4": {}, "J5": {},
        }
        for beta in BETAS:
            g = ewma_global(r, y, beta)
            hd = ewma_hod(r, y, beta)
            row["J1"][beta] = (mae(g[e], y[e]), mae(g[l], y[l]), mae(g, y))
            row["J2"][beta] = (mae(hd[e], y[e]), mae(hd[l], y[l]), mae(hd, y))
        w_e = fit_w_early(r, per, y, nn)
        row["w_early"] = w_e
        for w in WS:
            bl = blend_per(r, per, w)
            row["J3"][w] = (mae(bl[e], y[e]), mae(bl[l], y[l]), mae(bl, y))
        for beta in BETAS:
            g = ewma_global(r, y, beta)
            hd = ewma_hod(r, y, beta)
            row["J4"][beta] = (mae(blend_per(g[l], per[l], w_e), y[l]), mae(blend_per(g, per, w_e), y))
            row["J5"][beta] = (mae(blend_per(hd[l], per[l], w_e), y[l]), mae(blend_per(hd, per, w_e), y))
        rows[t] = row

    # --- LORO beta selection (early halves of other regions) --------------
    for arm in ("J1", "J2"):
        for t in targets:
            others = [u for u in targets if u != t]
            best_b, best_m = None, 1e9
            for beta in BETAS:
                m = np.mean([rows[u][arm][beta][0] for u in others])
                if m < best_m:
                    best_b, best_m = beta, m
            rows[t][f"{arm}_loro_beta"] = best_b
            rows[t][f"{arm}_loro_late"] = rows[t][arm][best_b][1]
            rows[t][f"{arm}_loro_full"] = rows[t][arm][best_b][2]
    # LORO for J4/J5 (beta chosen on others' early J1/J2; w is own early)
    for arm, src in (("J4", "J1"), ("J5", "J2")):
        for t in targets:
            b = rows[t][f"{src}_loro_beta"]
            rows[t][f"{arm}_loro_late"] = rows[t][arm][b][0]
            rows[t][f"{arm}_loro_full"] = rows[t][arm][b][1]

    # --- aggregates ---------------------------------------------------------
    def subset(th):
        return [t for t in targets if rows[t]["base_late"] > th]

    subsets = {"all": targets, "gt30": subset(30), "gt40": subset(40)}
    agg = {}
    for name, S in subsets.items():
        a = {"n": len(S)}
        a["base_late"] = float(np.mean([rows[t]["base_late"] for t in S]))
        a["R_late"] = float(np.mean([rows[t]["R_late"] for t in S]))
        a["base_full"] = float(np.mean([rows[t]["base_full"] for t in S]))
        a["R_full"] = float(np.mean([rows[t]["R_full"] for t in S]))
        for arm in ("J1", "J2"):
            a[f"{arm}_fixed_late"] = {b: float(np.mean([rows[t][arm][b][1] for t in S])) for b in BETAS}
            a[f"{arm}_loro_late"] = float(np.mean([rows[t][f"{arm}_loro_late"] for t in S]))
            a[f"{arm}_loro_full"] = float(np.mean([rows[t][f"{arm}_loro_full"] for t in S]))
        a["J3_fixed_late"] = {w: float(np.mean([rows[t]["J3"][w][1] for t in S])) for w in WS}
        a["J3_wearly_late"] = float(np.mean([rows[t]["J3"][min(WS, key=lambda w: abs(w - rows[t]["w_early"]))][1] for t in S]))
        for arm in ("J4", "J5"):
            a[f"{arm}_loro_late"] = float(np.mean([rows[t][f"{arm}_loro_late"] for t in S]))
            a[f"{arm}_loro_full"] = float(np.mean([rows[t][f"{arm}_loro_full"] for t in S]))
        agg[name] = a

    # --- write --------------------------------------------------------------
    def ser(o):
        if isinstance(o, dict):
            return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [ser(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 3)
        return o

    OUT_JSON.write_text(json.dumps({"rows": ser(rows), "agg": ser(agg), "subsets": subsets}, indent=1, ensure_ascii=False))

    L = ["# FD-50j 在线残差跟踪（EWMA）叠加 FD-50i 配方 — late 半段（诚实）", ""]
    L.append("信息层级与 pred_persistence 完全相同（前日真值 = 1 天延迟标签）。")
    L.append("")
    for name, a in agg.items():
        L.append(f"## {name}（n={a['n']}）")
        L.append(f"- 基线 late {a['base_late']:.2f} | R(B4+事件混合) late {a['R_late']:.2f}")
        L.append("- J1 全局 EWMA 固定 beta: " + " | ".join(f"β{b}: {v:.2f}" for b, v in a["J1_fixed_late"].items()))
        L.append("- J2 分时 EWMA 固定 beta: " + " | ".join(f"β{b}: {v:.2f}" for b, v in a["J2_fixed_late"].items()))
        L.append("- J3 全窗 persistence 凸组合固定 w: " + " | ".join(f"w{w}: {v:.2f}" for w, v in a["J3_fixed_late"].items()))
        L.append(f"- J3 w 由 early 拟合: {a['J3_wearly_late']:.2f}")
        L.append(f"- LORO 选参（零目标区调参）: J1 {a['J1_loro_late']:.2f} | J2 {a['J2_loro_late']:.2f} | J4=J1+J3 {a['J4_loro_late']:.2f} | J5=J2+J3 {a['J5_loro_late']:.2f}")
        L.append(f"- 全周期: 基线 {a['base_full']:.2f} | R {a['R_full']:.2f} | J1 {a['J1_loro_full']:.2f} | J2 {a['J2_loro_full']:.2f} | J4 {a['J4_loro_full']:.2f} | J5 {a['J5_loro_full']:.2f}")
        L.append("")
    L.append("## 全区明细（late，按基线降序）")
    L.append("")
    L.append("| target | base | R | J1 loro | J2 loro | J4 loro | J5 loro | β(J2) | w_e | per |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(targets, key=lambda u: -rows[u]["base_late"]):
        r = rows[t]
        L.append(
            f"| {t} | {r['base_late']:.1f} | {r['R_late']:.1f} | {r['J1_loro_late']:.1f} | {r['J2_loro_late']:.1f} "
            f"| {r['J4_loro_late']:.1f} | {r['J5_loro_late']:.1f} | {r['J2_loro_beta']} | {r['w_early']:.2f} | {r['per_late']:.1f} |"
        )
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
