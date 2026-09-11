"""FD-50n analysis: multi-year-source arm vs base arm, with J5 stack.

Reads results/dunkelflaute_my/{tgt}_seed0.npz (multi-year source training,
2023 target holdout — produced by fd50n_multiyear.py) and compares against
results/dunkelflaute/{tgt}_seed0.npz (official single-year).

Reports honest late-half MAE for:
  base I_cfg | my I_cfg | base+J5 | my+J5
J5 recipe identical to FD-50j (recipe_R + hod-EWMA beta 0.2 + persistence
blend, w early-fit). Event mask from the official dump (same windows).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUNK = Path("results/dunkelflaute")
MY = Path("results/dunkelflaute_my")
NB = 3
TAU_S = 0.60
W_EVT = 0.5
BETA = 0.2
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


def load_dunk(t):
    a = np.load(DUNK / f"{t}_seed0.npz")
    o = np.asarray(a["origin_hours"])
    order = o.argsort()
    y = np.asarray(a["y_true"], dtype=float)[order]
    per = np.asarray(a["pred_persistence"], dtype=float)[order]
    w = np.asarray(a["wx_w_mean"], dtype=float)
    wo = np.asarray(a["wx_day_ord"])
    wmap = dict(zip(wo.tolist(), w.tolist()))
    oo = o[order]
    ev = np.array([wmap[dd] for dd in (oo // 24 + OFFD)]) <= np.quantile(w, 0.20)
    return oo, y, per, ev


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


def fit_w_early(r, per, y, nn):
    e = slice(0, nn // 2)
    best_w, best_m = 0.0, mae(r[e], y[e])
    for w in np.arange(0.0, 0.71, 0.05):
        m = mae((1 - w) * r[e] + w * per[e], y[e])
        if m < best_m:
            best_w, best_m = float(w), m
    return best_w


def main():
    targets = sorted(p.name[: -len("_seed0.npz")] for p in MY.glob("*_seed0.npz"))
    rows = {}
    for t in targets:
        o, y, per, ev = load_dunk(t)
        b = np.load(MY / f"{t}_seed0.npz")
        om = np.asarray(b["origin_hours"])
        order = om.argsort()
        ym = np.asarray(b["y_true"], dtype=float)[order]
        pm = np.asarray(b["pred_i_cfg"], dtype=float)[order]
        n = min(len(y), len(ym))
        assert np.array_equal(o[:n], om[order][:n]), f"window mismatch {t}"
        l = slice(n // 2, n)
        y_, per_, ev_ = y[:n], per[:n], ev[:n]
        base = np.asarray(np.load(DUNK / f"{t}_seed0.npz")["pred_i_cfg"], dtype=float)
        base = base[np.asarray(np.load(DUNK / f"{t}_seed0.npz")["origin_hours"]).argsort()][:n]
        my = pm[:n]
        r_base = recipe_R(y_, base, per_, ev_)
        r_my = recipe_R(y_, my, per_, ev_)
        g_base = ewma_hod(r_base, y_, BETA)
        g_my = ewma_hod(r_my, y_, BETA)
        w_b = fit_w_early(g_base, per_, y_, n)
        w_m = fit_w_early(g_my, per_, y_, n)
        j5_base = (1 - w_b) * g_base + w_b * per_
        j5_my = (1 - w_m) * g_my + w_m * per_
        rows[t] = {
            "n": int(n),
            "base_late": round(mae(base[l], y_[l]), 3),
            "my_late": round(mae(my[l], y_[l]), 3),
            "j5_base_late": round(mae(j5_base[l], y_[l]), 3),
            "j5_my_late": round(mae(j5_my[l], y_[l]), 3),
        }

    def agg(key):
        return float(np.mean([rows[t][key] for t in rows]))

    summary = {k: round(agg(k), 3) for k in
               ("base_late", "my_late", "j5_base_late", "j5_my_late")}
    out = {"summary": summary, "rows": rows}
    Path("results/fd50n_analysis.json").write_text(json.dumps(out, indent=1, ensure_ascii=False))

    L = ["# FD-50n 多年数据通道分析（late 半段，诚实）", ""]
    L.append(f"- n regions: {len(rows)}")
    L.append(f"- base I_cfg: {summary['base_late']:.2f} | multi-year I_cfg: {summary['my_late']:.2f}")
    L.append(f"- base+J5: {summary['j5_base_late']:.2f} | multi-year+J5: {summary['j5_my_late']:.2f}")
    L.append("")
    L.append("| target | base | my | J5 base | J5 my | Δ(my-base) |")
    L.append("|---|---|---|---|---|---|")
    for t in sorted(rows, key=lambda u: -rows[u]["base_late"]):
        r_ = rows[t]
        L.append(f"| {t} | {r_['base_late']:.1f} | {r_['my_late']:.1f} | {r_['j5_base_late']:.1f} | {r_['j5_my_late']:.1f} | {r_['my_late']-r_['base_late']:+.1f} |")
    Path("results/fd50n_analysis.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
