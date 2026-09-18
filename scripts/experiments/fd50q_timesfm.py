"""FD-50q: TimesFM 2.5-200M zero-shot — full 29-region run, honest protocol.

Same as FD-50o (Chronos-2) and FD-50p (best3) but the TSFM arm is TimesFM
2.5-200M with a 5120h context cap (vs Chronos-2's 2048h).

Arms (late half = honest, all weights fitted on EARLY only):
  base      raw pred_i_cfg
  per       persistence
  J5        recipe_R + hod-EWMA(0.2) + persistence blend (official)
  T         pure TimesFM median from delayed-label context
  J5||T     outer convex blend J5/T (w2 fitted early)
  best3T    3-way blend (g_EWMA, persistence, T) grid-fitted early
  best3     reference: Chronos-2 3-way blend from FD-50p cache
  best4     4-way blend (g_EWMA, per, Chronos, TimesFM) grid-fitted early

TimesFM outputs cached per region to results/fd50q_cache/ (median + point at
ctx 5120) so blend arms iterate without re-running the model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

DUMP = Path("results/dunkelflaute")
CACHE = Path("results/fd50q_cache")
CHRONOS_CACHE = Path("results/fd50p_cache")
OUT_JSON = Path("results/fd50q_timesfm.json")
OUT_MD = Path("results/fd50q_timesfm.md")
OFFD = 718872
NB = 3
TAU_S = 0.60
W_EVT = 0.5
BETA = 0.2
MAX_CTX = 5120
HORIZON = 24
MODEL = "google/timesfm-2.5-200m-pytorch"


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


def timesfm_ctx(tfm, o, y, ctx_len):
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


def build_cache(tfm, targets):
    CACHE.mkdir(exist_ok=True)
    for t in targets:
        f = CACHE / f"{t}_t.npz"
        if f.exists():
            continue
        t0 = time.time()
        o, y, p, per, ev = load_region(t)
        pt, med, valid = timesfm_ctx(tfm, o, y, MAX_CTX)
        np.savez_compressed(f, pt=pt, med=med, valid=valid, per_fb=per)
        print(f"[cache] {t} {time.time()-t0:.0f}s", flush=True)


def main():
    from timesfm import TimesFM_2p5_200M_torch, ForecastConfig

    tfm = TimesFM_2p5_200M_torch.from_pretrained(MODEL)
    tfm.compile(ForecastConfig(max_context=MAX_CTX, horizon=HORIZON))

    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
    build_cache(tfm, targets)
    del tfm

    rows = {}
    for t in targets:
        t0 = time.time()
        o, y, p, per, ev = load_region(t)
        nn, h = y.shape
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)

        # J5 official
        r = recipe_R(y, p, per, ev)
        g = ewma_hod(r, y, BETA)
        w_e = fit_w_early(r, per, y, nn)
        j5 = (1 - w_e) * g + w_e * per

        # TimesFM (from cache)
        cf = np.load(CACHE / f"{t}_t.npz")
        valid = np.asarray(cf["valid"], dtype=bool)
        t_med = cf["med"].astype(float)
        t_med[~valid] = per[~valid]

        # Chronos-2 reference (from FD-50p cache)
        ch = np.load(CHRONOS_CACHE / f"{t}_c.npz")
        cval = np.asarray(ch["valid"], dtype=bool)
        c_med = ch["med2048"].astype(float)
        c_med[~cval] = per[~cval]

        # best3 reference (Chronos 3-way, official FD-50p recipe)
        best3_w, best3_m = (1.0, 0.0, 0.0), mae(g[e], y[e])
        for wg in np.arange(0.0, 1.01, 0.1):
            for wp in np.arange(0.0, 1.01 - wg + 1e-9, 0.1):
                wc = round(1.0 - wg - wp, 2)
                if wc < -1e-9:
                    continue
                m = mae(wg * g[e] + wp * per[e] + wc * c_med[e], y[e])
                if m < best3_m:
                    best3_w, best3_m = (float(wg), float(wp), float(wc)), m
        wg, wp, wc = best3_w
        best3 = wg * g + wp * per + wc * c_med

        # best3T: TimesFM 3-way
        b3t_w, b3t_m = (1.0, 0.0, 0.0), mae(g[e], y[e])
        for wg in np.arange(0.0, 1.01, 0.1):
            for wp in np.arange(0.0, 1.01 - wg + 1e-9, 0.1):
                wtt = round(1.0 - wg - wp, 2)
                if wtt < -1e-9:
                    continue
                m = mae(wg * g[e] + wp * per[e] + wtt * t_med[e], y[e])
                if m < b3t_m:
                    b3t_w, b3t_m = (float(wg), float(wp), float(wtt)), m
        wg2, wp2, wt2 = b3t_w
        best3T = wg2 * g + wp2 * per + wt2 * t_med

        # best4: 4-way (g, per, Chronos, TimesFM)
        b4_w, b4_m = (1.0, 0.0, 0.0, 0.0), mae(g[e], y[e])
        for wg in np.arange(0.0, 1.01, 0.2):
            for wp in np.arange(0.0, 1.01 - wg + 1e-9, 0.2):
                for wcx in np.arange(0.0, 1.01 - wg - wp + 1e-9, 0.2):
                    wtx = round(1.0 - wg - wp - wcx, 2)
                    if wtx < -1e-9:
                        continue
                    m = mae(wg * g[e] + wp * per[e] + wcx * c_med[e] + wtx * t_med[e], y[e])
                    if m < b4_m:
                        b4_w, b4_m = (float(wg), float(wp), float(wcx), float(wtx)), m
        wg4, wp4, wc4, wt4 = b4_w
        best4 = wg4 * g + wp4 * per + wc4 * c_med + wt4 * t_med

        # J5||T outer blend
        w2 = fit_w_early(j5, t_med, y, nn, lo=0.0, hi=1.0, step=0.1)
        j5ot = (1 - w2) * j5 + w2 * t_med

        row = {
            "base_late": mae(p[l], y[l]),
            "per_late": mae(per[l], y[l]),
            "J5_late": mae(j5[l], y[l]),
            "T_late": mae(t_med[l], y[l]),
            "C_late": mae(c_med[l], y[l]),
            "J5OT_late": mae(j5ot[l], y[l]),
            "BEST3T_late": mae(best3T[l], y[l]),
            "BEST3_late": mae(best3[l], y[l]),
            "BEST4_late": mae(best4[l], y[l]),
            "base_full": mae(p, y),
            "J5_full": mae(j5, y),
            "T_full": mae(t_med, y),
            "BEST3T_full": mae(best3T, y),
            "BEST3_full": mae(best3, y),
            "BEST4_full": mae(best4, y),
            "w_e": w_e, "w2": w2,
            "best3T_w": b3t_w, "best3_w": best3_w, "best4_w": b4_w,
        }
        rows[t] = row
        print(f"[{t}] base {row['base_late']:.1f} | J5 {row['J5_late']:.1f} | T {row['T_late']:.1f} "
              f"| C {row['C_late']:.1f} | J5||T {row['J5OT_late']:.1f} | best3T {row['BEST3T_late']:.1f} "
              f"| best3 {row['BEST3_late']:.1f} | best4 {row['BEST4_late']:.1f} ({time.time()-t0:.0f}s)", flush=True)

    key_sets = {
        "all": sorted(rows),
        "gt30_J5": [t for t in rows if rows[t]["J5_late"] > 30],
        "gt40_J5": [t for t in rows if rows[t]["J5_late"] > 40],
    }
    agg = {}
    for name, S in key_sets.items():
        a = {"n": len(S)}
        for k in rows[sorted(rows)[0]]:
            if k.endswith("_late") or k.endswith("_full"):
                a[k] = float(np.mean([rows[t][k] for t in S]))
        for arm in ["T", "J5OT", "BEST3T", "BEST4"]:
            a[f"{arm}_wins_vs_best3"] = int(sum(rows[t][f"{arm}_late"] < rows[t]["BEST3_late"] for t in S))
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

    L = ["# FD-50q TimesFM 2.5-200M 零样本 — late 半段（诚实）", ""]
    L.append("协议与 FD-50o/p 完全一致：I_lag 层（1 日延迟标签上下文），权重仅 early 拟合。")
    L.append(f"模型：{MODEL}，上下文上限 {MAX_CTX}h（Chronos-2 为 2048h）。")
    L.append("参照：FD-50p best3（Chronos 三分量）late all = 27.95。")
    L.append("")
    for name, a in agg.items():
        L.append(f"## {name}（n={a['n']}）")
        L.append(f"- base {a['base_late']:.2f} | per {a['per_late']:.2f} | J5 {a['J5_late']:.2f}")
        L.append(f"- T(纯TimesFM) {a['T_late']:.2f} | C(纯Chronos) {a['C_late']:.2f} | J5||T {a['J5OT_late']:.2f}")
        L.append(f"- best3T {a['BEST3T_late']:.2f} | best3(Chronos版) {a['BEST3_late']:.2f} | best4(双TSFM) {a['BEST4_late']:.2f}")
        wins = " | ".join(f"{arm} {a[f'{arm}_wins_vs_best3']}/{a['n']}" for arm in ["T", "J5OT", "BEST3T", "BEST4"])
        L.append(f"- 胜场 vs best3: {wins}")
        L.append(f"- 全周期: J5 {a['J5_full']:.2f} | T {a['T_full']:.2f} | best3T {a['BEST3T_full']:.2f} | best3 {a['BEST3_full']:.2f} | best4 {a['BEST4_full']:.2f}")
        L.append("")
    L.append("## 逐区明细（late，按 J5 降序）")
    L.append("")
    L.append("| target | base | J5 | T | C | J5||T | best3T | best3 | best4 | b3T权重(g/p/T) | b4权重(g/p/C/T) |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(rows, key=lambda u: -rows[u]["J5_late"]):
        r = rows[t]
        L.append(f"| {t} | {r['base_late']:.1f} | {r['J5_late']:.1f} | {r['T_late']:.1f} | {r['C_late']:.1f} "
                 f"| {r['J5OT_late']:.1f} | {r['BEST3T_late']:.1f} | {r['BEST3_late']:.1f} | {r['BEST4_late']:.1f} "
                 f"| {r['best3T_w'][0]:.1f}/{r['best3T_w'][1]:.1f}/{r['best3T_w'][2]:.1f} "
                 f"| {r['best4_w'][0]:.1f}/{r['best4_w'][1]:.1f}/{r['best4_w'][2]:.1f}/{r['best4_w'][3]:.1f} |")
    OUT_MD.write_text("\n".join(L) + "\n")
    print("DONE", flush=True)
    print("\n".join(L[:32]))


if __name__ == "__main__":
    main()
