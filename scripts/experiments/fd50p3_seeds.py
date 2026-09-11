"""FD-50p3: 5-seed stability of best3 (g/per/C convex blend, early-fitted weights).

Chronos context depends only on y_true, identical across seeds — cache reused.
Variation across seeds comes from pred_i_cfg (model init) through R, g, and the
fitted blend weights. Reports per-seed late MAE for J5, J5||C, best3f and the
seed-spread of the best3f gain vs J5.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUMP = Path("results/dunkelflaute")
CACHE = Path("results/fd50p_cache")
OUT_JSON = Path("results/fd50p3_seeds.json")
OFFD = 718872
NB = 3
TAU_S = 0.60
W_EVT = 0.5
BETA = 0.2
SEEDS = [0, 1, 2, 3, 4]


def mae(p, y):
    return float(np.abs(p - y).mean())


def fit_a_pivot(ps, pm, ys):
    best_a, best_m = 1.0, mae(pm + 1.0 * (ps - pm), ys)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(pm + a * (ps - pm), ys)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


def load_region(t, seed):
    d = np.load(DUMP / f"{t}_seed{seed}.npz")
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
    per_seed = {}
    for seed in SEEDS:
        rows = {}
        for t in targets:
            o, y, p, per, ev = load_region(t, seed)
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

            w2 = fit_w_early(j5, med, y, nn, lo=0.0, hi=1.0, step=0.1)
            j5oc = (1 - w2) * j5 + w2 * med

            rows[t] = {
                "J5_late": mae(j5[l], y[l]),
                "J5OC_late": mae(j5oc[l], y[l]),
                "best3f_late": mae(best3f[l], y[l]),
                "w3f": best_w,
            }
        a = {
            "J5_late": float(np.mean([rows[t]["J5_late"] for t in targets])),
            "J5OC_late": float(np.mean([rows[t]["J5OC_late"] for t in targets])),
            "best3f_late": float(np.mean([rows[t]["best3f_late"] for t in targets])),
            "best3f_wins": int(sum(rows[t]["best3f_late"] < rows[t]["J5_late"] for t in targets)),
        }
        per_seed[seed] = {"agg": a, "rows": rows}
        print(f"seed {seed}: J5 {a['J5_late']:.3f} | J5OC {a['J5OC_late']:.3f} | best3f {a['best3f_late']:.3f} "
              f"| best3f wins {a['best3f_wins']}/29")

    gains = [per_seed[s]["agg"]["best3f_late"] for s in SEEDS]
    j5s = [per_seed[s]["agg"]["J5_late"] for s in SEEDS]
    j5ocs = [per_seed[s]["agg"]["J5OC_late"] for s in SEEDS]
    print(f"\nbest3f 5-seed: mean {np.mean(gains):.3f} ± {np.std(gains):.3f} (min {min(gains):.3f}, max {max(gains):.3f})")
    print(f"J5     5-seed: mean {np.mean(j5s):.3f} ± {np.std(j5s):.3f}")
    print(f"J5OC   5-seed: mean {np.mean(j5ocs):.3f} ± {np.std(j5ocs):.3f}")
    print(f"gain best3f-J5 per seed: {[round(g-j,3) for g, j in zip(gains, j5s)]}")

    OUT_JSON.write_text(json.dumps({str(s): {"agg": per_seed[s]["agg"],
                                             "rows": {t: {k: (v if not isinstance(v, tuple) else list(v))
                                                          for k, v in per_seed[s]["rows"][t].items()}
                                                      for t in targets}}
                                     for s in SEEDS}, indent=1, default=float))


if __name__ == "__main__":
    main()
