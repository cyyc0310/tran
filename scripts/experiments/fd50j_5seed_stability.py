"""FD-50j 5-seed stability: run J5 on every seed dump and report mean±std late MAE.

Reads results/dunkelflaute/{tgt}_seed{0..4}.npz (5-seed rerun) and evaluates
the J5 recipe (recipe_R + hod-EWMA beta 0.2 + persistence blend, w early-fit)
per seed per region.  Honest late-half MAE.  Event mask from each dump's own
wx stats (20 pct threshold, OFFD=718872).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUNK = Path("results/dunkelflaute")
NB = 3
TAU_S = 0.60
W_EVT = 0.5
BETA = 0.2
OFFD = 718872
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


def load_dunk(t, seed):
    a = np.load(DUNK / f"{t}_seed{seed}.npz")
    o = np.asarray(a["origin_hours"])
    order = o.argsort()
    y = np.asarray(a["y_true"], dtype=float)[order]
    per = np.asarray(a["pred_persistence"], dtype=float)[order]
    p = np.asarray(a["pred_i_cfg"], dtype=float)[order]
    w = np.asarray(a["wx_w_mean"], dtype=float)
    wo = np.asarray(a["wx_day_ord"])
    wmap = dict(zip(wo.tolist(), w.tolist()))
    oo = o[order]
    ev = np.array([wmap[dd] for dd in (oo // 24 + OFFD)]) <= np.quantile(w, 0.20)
    return oo, y, per, p, ev


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


def j5_late(t, seed):
    """Return (n, late MAE of J5, late MAE of raw I_cfg) for one region+seed."""
    oo, y, per, p, ev = load_dunk(t, seed)
    r = recipe_R(y, p, per, ev)
    g = ewma_hod(r, y, BETA)
    w = fit_w_early(g, per, y, len(y))
    j5 = (1 - w) * g + w * per
    l = slice(len(y) // 2, len(y))
    return len(y), mae(j5[l], y[l]), mae(p[l], y[l])


def main():
    # region list from seed0 dumps
    targets = sorted(
        p.name[: -len("_seed0.npz")] for p in DUNK.glob("*_seed0.npz"))
    rows = {}
    for t in targets:
        per_seed = {}
        for s in SEEDS:
            f = DUNK / f"{t}_seed{s}.npz"
            if not f.exists():
                continue
            n, j5m, rawm = j5_late(t, s)
            per_seed[s] = {"n": int(n), "j5_late": round(j5m, 3),
                           "raw_late": round(rawm, 3)}
        if not per_seed:
            continue
        j5s = np.array([per_seed[s]["j5_late"] for s in sorted(per_seed)])
        raws = np.array([per_seed[s]["raw_late"] for s in sorted(per_seed)])
        rows[t] = {
            "n_seeds": len(j5s),
            "j5_mean": round(float(j5s.mean()), 3),
            "j5_std": round(float(j5s.std(ddof=1)) if len(j5s) > 1 else 0.0, 3),
            "raw_mean": round(float(raws.mean()), 3),
            "raw_std": round(float(raws.std(ddof=1)) if len(raws) > 1 else 0.0, 3),
            "per_seed": per_seed,
        }

    j5_means = np.array([rows[t]["j5_mean"] for t in rows])
    raw_means = np.array([rows[t]["raw_mean"] for t in rows])
    summary = {
        "n_regions": len(rows),
        "j5_late_mean": round(float(j5_means.mean()), 3),
        "j5_late_std_across_regions": round(float(j5_means.std()), 3),
        "raw_late_mean": round(float(raw_means.mean()), 3),
        "raw_late_std_across_regions": round(float(raw_means.std()), 3),
        "median_region_j5_std": round(float(np.median(
            [rows[t]["j5_std"] for t in rows])), 3),
        "max_region_j5_std": round(float(max(
            rows[t]["j5_std"] for t in rows)), 3),
    }
    out = {"summary": summary, "rows": rows}
    Path("results/fd50j_5seed_stability.json").write_text(
        json.dumps(out, indent=1, ensure_ascii=False))

    L = ["# FD-50j 5-seed 稳定性（J5 late MAE，诚实口径）", ""]
    L.append(f"- regions: {summary['n_regions']}  seeds: 0-4")
    L.append(f"- 29 区 J5 late 均值: {summary['j5_late_mean']:.2f} "
             f"(跨区 std {summary['j5_late_std_across_regions']:.2f})")
    L.append(f"- raw I_cfg late 均值: {summary['raw_late_mean']:.2f} "
             f"(跨区 std {summary['raw_late_std_across_regions']:.2f})")
    L.append(f"- 区内 seed std: 中位 {summary['median_region_j5_std']:.2f} / "
             f"最大 {summary['max_region_j5_std']:.2f}")
    L.append("")
    L.append("| target | J5 mean±std | raw mean±std | seeds |")
    L.append("|---|---|---|---|")
    for t in sorted(rows, key=lambda u: -rows[u]["j5_mean"]):
        r_ = rows[t]
        L.append(f"| {t} | {r_['j5_mean']:.1f}±{r_['j5_std']:.1f} | "
                 f"{r_['raw_mean']:.1f}±{r_['raw_std']:.1f} | {r_['n_seeds']} |")
    Path("results/fd50j_5seed_stability.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
