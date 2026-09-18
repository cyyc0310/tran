"""FD-50q: TimesFM 2.5-200M zero-shot — same honest protocol as FD-50o/p.

Direction 1 from FD-50p verdict: swap Chronos-2 (ctx 2048h) for TimesFM 2.5
(ctx up to 5120h), everything else identical (I_lag tier, 1-day-delayed label
context, early-only weight fitting, late = honest).

Smoke test first: single region (US_CISO, largest J5 gain in FD-50o) to
validate the pipeline end-to-end before the 29-region run.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

DUMP = Path("results/dunkelflaute")
OUT = Path("results/fd50q_smoke.json")
OFFD = 718872
NB = 3
TAU_S = 0.60
W_EVT = 0.5
BETA = 0.2
MAX_CTX = 5120      # TimesFM 2.5 context cap
HORIZON = 24
MODEL = "google/timesfm-2.5-200m-pytorch"


def mae(p, y):
    return float(np.abs(p - y).mean())


def fit_a_pivot(ps, pm, ys):
    best_a, best_m = 1.0, mae(pm + 1.0 * (ps - pm), ys)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(pm + a * (ps - pm), ys)
        if m < best_a:
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
        if m < best_w:
            best_w, best_m = float(w), m
    return best_w


def timesfm_forecast(tfm, o, y, ctx_len):
    """Batched TimesFM zero-shot 24h forecasts from strictly-past truths."""
    n = len(o)
    ends = o + 24
    inputs, valid = [], np.zeros(n, dtype=bool)
    for t in range(n):
        idx = np.nonzero(ends <= o[t])[0]
        if len(idx) == 0:
            inputs.append(np.zeros(3))
            continue
        ctx = np.concatenate([y[j] for j in idx])[-ctx_len:]
        inputs.append(np.ascontiguousarray(ctx, dtype=np.float32))
        valid[t] = True
    pts, qts = tfm.forecast(horizon=HORIZON, inputs=inputs)
    med = np.stack([np.asarray(q).reshape(-1)[:HORIZON] for q in qts])
    pt = np.stack([np.asarray(q).reshape(-1)[:HORIZON] for q in pts])
    return pt, med, valid


def main():
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

    tfm = TimesFM_2p5_200M_torch.from_pretrained(MODEL)
    tfm.compile(ForecastConfig(max_context=MAX_CTX, horizon=HORIZON))

    t = "US_CISO"
    t0 = time.time()
    o, y, p, per, ev = load_region(t)
    nn, h = y.shape
    e = slice(0, nn // 2)
    l = slice(nn // 2, nn)

    r = recipe_R(y, p, per, ev)
    g = ewma_hod(r, y, BETA)
    w_e = fit_w_early(r, per, y, nn)
    j5 = (1 - w_e) * g + w_e * per

    pt, med, valid = timesfm_forecast(tfm, o, y, MAX_CTX)
    c = med.copy()
    c[~valid] = per[~valid]

    # arms: pure T, T||J5 outer blend (w2 early), best3-T (swap Chronos→TimesFM)
    w2 = fit_w_early(j5, c, y, nn, lo=0.0, hi=1.0, step=0.1)
    j5ot = (1 - w2) * j5 + w2 * c
    best_w, best_m = (1.0, 0.0, 0.0), mae(g[e], y[e])
    for wg in np.arange(0.0, 1.01, 0.1):
        for wp in np.arange(0.0, 1.01 - wg + 1e-9, 0.1):
            wc = round(1.0 - wg - wp, 2)
            if wc < -1e-9:
                continue
            m = mae(wg * g[e] + wp * per[e] + wc * c[e], y[e])
            if m < best_m:
                best_w, best_m = (float(wg), float(wp), float(wc)), m
    wg, wp, wc = best_w
    best3T = wg * g + wp * per + wc * c

    rows = {
        "base_late": mae(p[l], y[l]),
        "per_late": mae(per[l], y[l]),
        "J5_late": mae(j5[l], y[l]),
        "T_late": mae(c[l], y[l]),
        "J5OT_late": mae(j5ot[l], y[l]),
        "BEST3T_late": mae(best3T[l], y[l]),
        "n_windows": int(nn),
        "n_valid_ctx": int(valid.sum()),
        "w_e": w_e, "w2": w2, "best3T_w": best_w,
    }
    import json
    OUT.write_text(json.dumps({t: {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows.items()}},
                              indent=1, ensure_ascii=False))
    print(f"[{t}] base {rows['base_late']:.1f} | per {rows['per_late']:.1f} | J5 {rows['J5_late']:.1f}")
    print(f"[{t}] T(TimesFM) {rows['T_late']:.1f} | J5||T {rows['J5OT_late']:.1f} | best3T {rows['BEST3T_late']:.1f}")
    print(f"weights: w_e={w_e:.2f} w2={w2:.2f} best3T=({wg:.1f},{wp:.1f},{wc:.1f})  {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
