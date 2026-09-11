"""FD-50m: NWP-arm x J5 stack (channel 1).

For each region: load FD-47 four-arm predictions (era5/gfs/icon/ensemble at
I_cfg layer), apply J5 (recipe_R + hod-EWMA beta0.2 + persistence blend) on
each arm's predictions, compare honest late-half MAE.

Honesty: J5 hyper-params (beta, w) are the ones proven optimal on the era5 arm
of the same dump family (FD-50j/l); we re-fit w on early half of each arm as
J5 does (no late peeking). Event mask from dunkelflaute npz (same windows,
verified aligned).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUNK = Path("results/dunkelflaute")
NWP = Path("results/fd47_nwp")
ARMS = ["era5", "gfs", "icon", "ensemble"]
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


def load_pair(t):
    a = np.load(DUNK / f"{t}_seed0.npz")
    b = np.load(NWP / f"{t}_seed0.npz")
    assert np.array_equal(np.asarray(a["origin_hours"]), np.asarray(b["origin_hours"]))
    o = np.asarray(b["origin_hours"])
    order = o.argsort()
    o = o[order]
    y = np.asarray(b["y_true"], dtype=float)[order]
    P = {arm: np.asarray(b[f"pred_{arm}_icfg"], dtype=float)[order] for arm in ARMS}
    w = np.asarray(a["wx_w_mean"], dtype=float)
    wo = np.asarray(a["wx_day_ord"])
    wmap = dict(zip(wo.tolist(), w.tolist()))
    ev = np.array([wmap[dd] for dd in (o // 24 + OFFD)]) <= np.quantile(w, 0.20)
    per = np.asarray(a["pred_persistence"], dtype=float)[order]
    return o, y, P, per, ev


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
    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUNK.glob("*_seed0.npz"))
    rows = {}
    for t in targets:
        o, y, P, per, ev = load_pair(t)
        nn = len(y)
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)
        row = {}
        for arm in ARMS:
            p = P[arm]
            base = mae(p[l], y[l])
            r = recipe_R(y, p, per, ev)
            g = ewma_hod(r, y, BETA)
            w_e = fit_w_early(g, per, y, nn)
            j5 = (1 - w_e) * g + w_e * per
            row[arm] = {
                "base_late": round(base, 3),
                "R_late": round(mae(r[l], y[l]), 3),
                "J5_late": round(mae(j5[l], y[l]), 3),
                "w_e": w_e,
            }
        rows[t] = row

    agg = {}
    for arm in ARMS:
        S = targets
        agg[arm] = {
            "base": float(np.mean([rows[t][arm]["base_late"] for t in S])),
            "J5": float(np.mean([rows[t][arm]["J5_late"] for t in S])),
        }
    # family split: era5-real (21) vs no-era5 fallback (9)
    fam_no_era5 = ["UK_01", "UK_02", "UK_03", "UK_05_Yorkshire", "UK_06", "UK_07", "UK_08", "US_FPL", "US_BPAT"]
    for arm in ARMS:
        for fam, S in [("era5real", [t for t in targets if t not in fam_no_era5]),
                       ("no_era5", [t for t in targets if t in fam_no_era5])]:
            agg[f"{arm}_{fam}"] = {
                "n": len(S),
                "base": float(np.mean([rows[t][arm]["base_late"] for t in S])),
                "J5": float(np.mean([rows[t][arm]["J5_late"] for t in S])),
            }

    def ser(o):
        if isinstance(o, dict):
            return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [ser(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 3)
        return o

    Path("results/fd50m_nwp_j5.json").write_text(json.dumps({"rows": ser(rows), "agg": ser(agg)}, indent=1, ensure_ascii=False))

    L = ["# FD-50m NWP 臂 × J5 叠加（late 半段，诚实）", ""]
    L.append("| arm | all base | all J5 | era5real base | era5real J5 | no_era5 base | no_era5 J5 |")
    L.append("|---|---|---|---|---|---|---|")
    for arm in ARMS:
        a = agg[arm]
        ar = agg[f"{arm}_era5real"]
        ne = agg[f"{arm}_no_era5"]
        L.append(f"| {arm} | {a['base']:.2f} | {a['J5']:.2f} | {ar['base']:.2f} | {ar['J5']:.2f} | {ne['base']:.2f} | {ne['J5']:.2f} |")
    L.append("")
    L.append("## 逐区（J5_late，按 era5 J5 降序）")
    L.append("")
    L.append("| target | era5 J5 | gfs J5 | icon J5 | ens J5 |")
    L.append("|---|---|---|---|---|")
    for t in sorted(targets, key=lambda u: -rows[u]["era5"]["J5_late"]):
        L.append(f"| {t} | {rows[t]['era5']['J5_late']:.1f} | {rows[t]['gfs']['J5_late']:.1f} | {rows[t]['icon']['J5_late']:.1f} | {rows[t]['ensemble']['J5_late']:.1f} |")
    Path("results/fd50m_nwp_j5.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
