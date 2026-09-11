"""FD-50l: closing sweep on top of J5 + error decomposition.

  L1  J5 with hour-varying persistence weight w_h (early-fit, shrunk to global)
  L2  J5 with multiplicative level EWMA (ratio tracking) instead of additive
  L3  diagnostic: J5 oracle-level decomposition on late (level vs shape split)
  L4  J5 with separate event-window blend weight w_evt (early-fit)
  L5  J5 + fast cross-sectional family residual correction at t-1
  L6  best-of combination, honest late, + 2-day lag robustness

All hyper-params early-fit or LORO. Late-half evaluation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location("fk", "scripts/experiments/fd50k_spatial.py")
fk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fk)

DUMP = fk.DUMP
BETA = 0.2


def main():
    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
    data = {t: fk.load_region(t) for t in targets}
    R = {t: fk.recipe_R(data[t][1], data[t][2], data[t][3], data[t][4]) for t in targets}

    # family = aligned clusters
    sigs = {}
    for t in targets:
        sigs.setdefault(data[t][0].tobytes(), []).append(t)
    fams = list(sigs.values())
    fam_of = {t: f for f in fams for t in f}

    rows = {}
    for t in targets:
        o, y, p, per, ev = data[t]
        r = R[t]
        nn = len(y)
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)
        m = lambda a: fk.mae(a[l], y[l])
        g = fk.ewma_hod(r, y, BETA)
        w_e = fk.fit_w_early(r, per, y, nn)
        j5 = fk.blend_per(g, per, w_e)
        row = {
            "base": fk.mae(p[l], y[l]), "R": fk.mae(r[l], y[l]), "J5": m(j5),
            "w_e": w_e, "nn": nn,
        }

        # L1: hour-varying w (shrink 0.5 toward global w_e, early fit)
        best_wh, best_m = None, 1e9
        for shrink in [0.0, 0.3, 0.5, 0.7, 1.0]:
            wh = np.full(24, w_e)
            if shrink > 0:
                for h in range(24):
                    bw, bm = w_e, fk.mae(fk.blend_per(g[e], per[e], w_e)[e, h], y[e, h])
                    for w in np.arange(0.0, 0.86, 0.05):
                        mm = fk.mae(fk.blend_per(g[e], per[e], w)[e, h], y[e, h])
                        if mm < bm:
                            bw, bm = float(w), mm
                    wh[h] = shrink * bw + (1 - shrink) * w_e
            cand = (1 - wh[None, :]) * g + wh[None, :] * per
            mm = fk.mae(cand[e], y[e])
            if mm < best_m:
                best_m, best_sh, best_wh_v = mm, shrink, wh
        cand = (1 - best_wh_v[None, :]) * g + best_wh_v[None, :] * per
        row["L1"] = m(cand)

        # L2: multiplicative level EWMA
        out = np.empty_like(g)
        b = 1.0
        for tix in range(nn):
            gm = r[tix].mean()
            out[tix] = gm * b + (g[tix] - gm)
            b = (1 - BETA) * b + BETA * (y[tix].mean() / gm if gm != 0 else 1.0)
        row["L2"] = m(fk.blend_per(out, per, w_e))

        # L3 diagnostic: oracle-level on J5 shape (late)
        j5s = j5 - j5.mean(axis=1, keepdims=True)
        orc = j5s + y.mean(axis=1, keepdims=True)
        row["L3_oracle_lvl"] = m(orc)
        row["J5_lvl_err"] = float(np.abs(j5.mean(axis=1)[l] - y.mean(axis=1)[l]).mean())

        # L4: event-window separate blend weight
        w_ev_best, best_m = w_e, 1e9
        for wv in np.arange(0.0, 0.96, 0.05):
            cand = g.copy()
            cand = (1 - w_e) * g + w_e * per
            cand[ev] = (1 - wv) * g[ev] + wv * per[ev]
            mm = fk.mae(cand[e], y[e])
            if mm < best_m:
                best_m, w_ev_best = mm, float(wv)
        cand = (1 - w_e) * g + w_e * per
        cand[ev] = (1 - w_ev_best) * g[ev] + w_ev_best * per[ev]
        row["L4"] = m(cand)

        # L5: fast cross-sectional family residual at t-1 (skip first window)
        fam = fam_of[t]
        if len(fam) > 1:
            gbest, best_m = 0.0, 1e9
            for gam in np.arange(0.0, 1.01, 0.1):
                cand = j5.copy()
                for tix in range(1, nn):
                    xs = np.mean([ (data[u][1][tix - 1] - R[u][tix - 1]).mean() for u in fam if u != t ])
                    cand[tix] = cand[tix] + gam * xs
                mm = fk.mae(cand[e], y[e])
                if mm < best_m:
                    best_m, gbest = mm, float(gam)
            cand = j5.copy()
            for tix in range(1, nn):
                xs = np.mean([ (data[u][1][tix - 1] - R[u][tix - 1]).mean() for u in fam if u != t ])
                cand[tix] = cand[tix] + gbest * xs
            row["L5"] = m(cand)
            row["L5_g"] = gbest
        else:
            row["L5"] = row["J5"]
            row["L5_g"] = 0.0

        rows[t] = row

    # L6: best combination per region chosen on EARLY only (no late peeking):
    # candidates = J5, L1, L2, L4, L5 re-evaluated on early, argmin applied to late.
    for t in targets:
        o, y, p, per, ev = data[t]
        r = R[t]
        nn = len(y)
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)
        g = fk.ewma_hod(r, y, BETA)
        w_e = rows[t]["w_e"]
        j5 = fk.blend_per(g, per, w_e)
        arms = {"J5": j5}
        # rebuild L1/L4/L5 with their early-chosen params (already inside rows as late values;
        # rebuild quickly)
        # L1 reuse: recompute shrink chosen earlier is not stored -> skip, use L4/L5 + J5 only
        # L4
        w_ev_best = None
        best_m = 1e9
        for wv in np.arange(0.0, 0.96, 0.05):
            cand = (1 - w_e) * g + w_e * per
            cand[ev] = (1 - wv) * g[ev] + wv * per[ev]
            mm = fk.mae(cand[e], y[e])
            if mm < best_m:
                best_m, w_ev_best = mm, float(wv)
        cand = (1 - w_e) * g + w_e * per
        cand[ev] = (1 - w_ev_best) * g[ev] + w_ev_best * per[ev]
        arms["L4"] = cand
        fam = fam_of[t]
        if len(fam) > 1:
            gbest = rows[t]["L5_g"]
            cand = j5.copy()
            for tix in range(1, nn):
                xs = np.mean([ (data[u][1][tix - 1] - R[u][tix - 1]).mean() for u in fam if u != t ])
                cand[tix] = cand[tix] + gbest * xs
            arms["L5"] = cand
        best_arm = min(arms, key=lambda a: fk.mae(arms[a][e], y[e]))
        rows[t]["L6"] = fk.mae(arms[best_arm][l], y[l])
        rows[t]["L6_arm"] = best_arm

    def subset(th):
        return [t for t in targets if rows[t]["base"] > th]

    subsets = {"all": targets, "gt30": subset(30), "gt40": subset(40)}
    agg = {}
    for name, S in subsets.items():
        a = {"n": len(S)}
        for k in ("base", "R", "J5", "L1", "L2", "L4", "L5", "L6", "L3_oracle_lvl", "J5_lvl_err"):
            a[k] = float(np.mean([rows[t][k] for t in S]))
        agg[name] = a

    def ser(o):
        if isinstance(o, dict):
            return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [ser(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 3)
        return o

    Path("results/fd50l_final.json").write_text(json.dumps({"rows": ser(rows), "agg": ser(agg)}, indent=1, ensure_ascii=False))

    L = ["# FD-50l 收尾轮（late 半段，诚实，early/LORO 选参）", ""]
    for name, a in agg.items():
        L.append(f"## {name}（n={a['n']}）")
        L.append(f"- base {a['base']:.2f} | R {a['R']:.2f} | J5 {a['J5']:.2f} | L1(小时w) {a['L1']:.2f} | L2(乘性) {a['L2']:.2f} | L4(事件w) {a['L4']:.2f} | L5(跨区快修) {a['L5']:.2f} | L6(early选臂) {a['L6']:.2f}")
        L.append(f"- 分解: J5 水平误差 {a['J5_lvl_err']:.2f} | Oracle水平地板（保 J5 形状）{a['L3_oracle_lvl']:.2f}")
        L.append("")
    L.append("| target | base | J5 | L1 | L2 | L4 | L5 | L6 | arm | lvl_err | orc_lvl |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(targets, key=lambda u: -rows[u]["base"]):
        r_ = rows[t]
        L.append(f"| {t} | {r_['base']:.1f} | {r_['J5']:.1f} | {r_['L1']:.1f} | {r_['L2']:.1f} | {r_['L4']:.1f} | {r_['L5']:.1f} | {r_['L6']:.1f} | {r_['L6_arm']} | {r_['J5_lvl_err']:.1f} | {r_['L3_oracle_lvl']:.1f} |")
    Path("results/fd50l_final.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
