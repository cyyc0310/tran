"""把 B4+事件混合配方（early 拟合）逐区对齐官方 FD-41 口径，澄清 39 vs 40.14 的口径差异。"""
import json
from pathlib import Path

import numpy as np

DUMP = Path("results/dunkelflaute")
NB = 3
TAU_S = 0.60
OFFD = 718872


def mae(p, y):
    return float(np.abs(p - y).mean())


def fit_a_pivot(ps, pm, ys):
    best_a, best_m = 1.0, mae(pm + 1.0 * (ps - pm), ys)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(pm + a * (ps - pm), ys)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


OFF = json.load(open("results/fd41_ep900.json"))
off = {r["target"]: r["mae_cfg"] for r in OFF}
off0 = {r["target"]: r["mae_i0"] for r in OFF}

targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
rows = []
for t in targets:
    d = np.load(DUMP / f"{t}_seed0.npz")
    o = np.asarray(d["origin_hours"])
    order = o.argsort()
    nn = len(order)
    y = d["y_true"].astype(float)[order]
    p = d["pred_i_cfg"].astype(float)[order]
    per = np.asarray(d["pred_persistence"]).astype(float)[order]
    w = np.asarray(d["wx_w_mean"])
    wo = np.asarray(d["wx_day_ord"])
    wmap = dict(zip(wo.tolist(), w.tolist()))
    ev = np.array([wmap[dd] for dd in (o // 24 + OFFD)]) <= np.quantile(w, 0.20)
    h = p.shape[1]
    cuts = [(i * h // NB, (i + 1) * h // NB) for i in range(NB)]
    e = slice(0, nn // 2)
    l = slice(nn // 2, nn)
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

    def apply(sl):
        pm = p[sl].mean(axis=1, keepdims=True)
        out = np.empty_like(p[sl])
        for (s, ee), a in zip(cuts, a_b4):
            out[:, s : ee] = pm + a * (p[sl, s:ee] - pm)
        evl = ev[sl]
        if evl.sum() > 0:
            out[evl] = 0.5 * (p[sl][evl] + per[sl][evl])
        return out

    out_full = np.concatenate([apply(e), apply(l)], 0)
    rows.append((t, off.get(t, float("nan")), off0.get(t, float("nan")), mae(p, y), mae(out_full, y)))

rows.sort(key=lambda r: r[4])
print("%-26s %8s %8s %9s %8s" % ("region", "off_cfg", "off_i0", "dumpbase", "B4"))
for t, a, b, c, dd in rows:
    print("%-26s %8.2f %8.2f %9.2f %8.2f" % (t, a, b, c, dd))
g = lambda i: np.array([r[i] for r in rows])
print()
print("mean: off_cfg %.2f | off_i0 %.2f | dumpbase %.2f | B4 %.2f" % (np.nanmean(g(1)), np.nanmean(g(2)), g(3).mean(), g(4).mean()))
print("median: off_cfg %.2f | off_i0 %.2f | dumpbase %.2f | B4 %.2f" % (np.nanmedian(g(1)), np.nanmedian(g(2)), np.median(g(3)), np.median(g(4))))
