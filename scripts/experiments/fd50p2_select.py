"""FD-50p2: honest per-region arm selection + finer best3 grid.

All selection on EARLY half; late untouched. Arms:
  best3f   3-way (g, per, C) grid step 0.05 fitted early
  J5OC     FD-50o outer blend (reproduced identically)
  select   per-region pick of argmin(early) among {best3f, J5OC}
  meta     convex blend of (best3f, J5OC), weight on early grid 0.05
Uses the FD-50p cache; no new Chronos inference.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUMP = Path("results/dunkelflaute")
CACHE = Path("results/fd50p_cache")
OUT_JSON = Path("results/fd50p2_select.json")
OFFD = 718872
NB = 3
TAU_S = 0.60
W_EVT = 0.5
BETA = 0.2


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
    w = np.asarray(d["wx_w_mean"], dtype=float)
    wo = np.asarray(d["wx_day_ord"])
    wmap = dict(zip(wo.tolist(), w.tolist()))
    ev = np.array([wmap[dd] for dd in (o // 24 + OFFD)]) <= np.quantile(w, 0.20)
    return o, y, p, per, ev


def recipe_R(y, p, per, ev):
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


def ewma_hod(r, ys, beta):
    out = np.empty_like(r)
    b = np.zeros(r.shape[1])
    for t in range(len(r)):
        out[t] = r[t] + b
        b = (1 - beta) * b + beta * (ys[t] - r[t])
    return out


def fit_w_early(r, per, y, nn, lo=0.0, hi=0.7, step=0.05):
    e = slice(0, nn // 2)
    best_w, best_m = 0.0, mae(r[e], y[e])
    for w in np.arange(lo, hi + 1e-9, step):
        m = mae((1 - w) * r[e] + w * per[e], y[e])
        if m < best_m:
            best_w, best_m = float(w), m
    return best_w


def main():
    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
    rows = {}
    for t in targets:
        o, y, p, per, ev = load_region(t)
        nn, h = y.shape
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)

        r = recipe_R(y, p, per, ev)
        g = ewma_hod(r, y, BETA)
        w_e = fit_w_early(r, per, y, nn)
        j5 = (1 - w_e) * g + w_e * per

        c = np.load(CACHE / f"{t}_c.npz")
        valid = np.asarray(c["valid"], dtype=bool)
        med = c["med2048"].astype(float)
        med[~valid] = per[~valid]

        # best3f: fine grid 0.05
        best_w, best_m = (1.0, 0.0, 0.0), mae(g[e], y[e])
        for wg in np.arange(0.0, 1.001, 0.05):
            for wp in np.arange(0.0, 1.001 - wg + 1e-9, 0.05):
                wc = round(1.0 - wg - wp, 2)
                if wc < -1e-9:
                    continue
                m = mae(wg * g[e] + wp * per[e] + wc * med[e], y[e])
                if m < best_m:
                    best_w, best_m = (round(float(wg), 2), round(float(wp), 2), wc), m
        wg, wp, wc = best_w
        best3f = wg * g + wp * per + wc * med

        # J5OC (identical to FD-50o)
        w2 = fit_w_early(j5, med, y, nn, lo=0.0, hi=1.0, step=0.1)
        j5oc = (1 - w2) * j5 + w2 * med

        # select: per-region argmin on early
        m_b3, m_j5oc = mae(best3f[e], y[e]), mae(j5oc[e], y[e])
        sel = best3f if m_b3 <= m_j5oc else j5oc
        sel_name = "best3f" if m_b3 <= m_j5oc else "J5OC"

        # meta: convex blend of best3f and J5OC
        best_wm, best_mm = 0.0, mae(best3f[e], y[e])
        for wm in np.arange(0.0, 1.001, 0.05):
            m = mae((1 - wm) * best3f[e] + wm * j5oc[e], y[e])
            if m < best_mm:
                best_wm, best_mm = float(wm), m
        meta = (1 - best_wm) * best3f + best_wm * j5oc

        rows[t] = {
            "J5_late": mae(j5[l], y[l]),
            "J5OC_late": mae(j5oc[l], y[l]),
            "best3f_late": mae(best3f[l], y[l]),
            "select_late": mae(sel[l], y[l]),
            "meta_late": mae(meta[l], y[l]),
            "sel": sel_name, "w3f": best_w, "w2": w2, "wm": best_wm,
            "select_full": mae(sel, y), "meta_full": mae(meta, y),
            "best3f_full": mae(best3f, y), "J5OC_full": mae(j5oc, y), "J5_full": mae(j5, y),
        }
        print(f"[{t}] J5 {rows[t]['J5_late']:.1f} J5OC {rows[t]['J5OC_late']:.1f} best3f {rows[t]['best3f_late']:.1f} "
              f"select({sel_name}) {rows[t]['select_late']:.1f} meta {rows[t]['meta_late']:.1f}")

    key_sets = {
        "all": sorted(rows),
        "gt30": [t for t in rows if rows[t]["J5_late"] > 30],
        "gt40": [t for t in rows if rows[t]["J5_late"] > 40],
    }
    agg = {}
    for name, S in key_sets.items():
        a = {"n": len(S)}
        for k in ["J5_late", "J5OC_late", "best3f_late", "select_late", "meta_late",
                  "J5_full", "J5OC_full", "best3f_full", "select_full", "meta_full"]:
            a[k] = float(np.mean([rows[t][k] for t in S]))
        a["select_wins_vs_best3f"] = int(sum(rows[t]["select_late"] <= rows[t]["best3f_late"] + 1e-9 for t in S))
        a["n_sel_best3f"] = int(sum(rows[t]["sel"] == "best3f" for t in S))
        agg[name] = a

    def ser(o):
        if isinstance(o, dict):
            return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [ser(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 3)
        return o

    OUT_JSON.write_text(json.dumps({"rows": ser(rows), "agg": ser(agg)}, indent=1, ensure_ascii=False))
    for name, a in agg.items():
        print(f"\n[{name}] n={a['n']} J5 {a['J5_late']:.2f} | J5OC {a['J5OC_late']:.2f} | best3f {a['best3f_late']:.2f} "
              f"| select {a['select_late']:.2f} | meta {a['meta_late']:.2f} (sel best3f {a['n_sel_best3f']}/{a['n']})")


if __name__ == "__main__":
    main()
