"""FD-50p: second-round TSFM squeeze — online level correction on Chronos + 3-way blend.

Levers tested (all honest: weights/betas fitted on EARLY half only):
  CE        EWMA(beta)-corrected Chronos: g_C[t] = C[t] + b, b = (1-β)b + β(y[t-1]-C[t-1])
             (J5's online level machinery applied to TSFM output; beta from small scan)
  CEp       CE + persistence blend, w fitted early  (full J5-analogue on Chronos)
  best3     3-way convex blend (g_EWMA, persistence, C) weights grid-fitted early
  CExJ5     outer blend J5||CE, w3 fitted early
  ctx ablation: C computed at context lengths 2048 / 720 / 336 — tests whether
             short contexts fix the regions where C loses badly (US_ERCO, UK_10)
  qtrim     (q25+q50+q75)/3 trimmed median from the same call — minor smoothing arm

Chronos outputs are cached per region to results/fd50p_cache/ (median + q25 + q75
at each context length) so blend arms iterate without re-running the model.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

DUMP = Path("results/dunkelflaute")
CACHE = Path("results/fd50p_cache")
OUT_JSON = Path("results/fd50p_blend2.json")
OUT_MD = Path("results/fd50p_blend2.md")
MODEL = "amazon/chronos-2"
OFFD = 718872
NB = 3
TAU_S = 0.60
W_EVT = 0.5
BETA = 0.2
CTX_LENS = [2048, 720, 336]


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


def chronos_ctx(pipe, o, y, ctx_len, qlevels=(0.25, 0.5, 0.75)):
    """Batched Chronos-2 with context truncated to last ctx_len hours."""
    n = len(o)
    ends = o + 24
    series, valid = [], np.zeros(n, dtype=bool)
    for t in range(n):
        idx = np.nonzero(ends <= o[t])[0]
        if len(idx) == 0:
            series.append(torch.zeros(1))
            continue
        ctx = np.concatenate([y[j] for j in idx])[-ctx_len:]
        series.append(torch.tensor(np.ascontiguousarray(ctx, dtype=np.float32)))
        valid[t] = True
    _, m = pipe.predict_quantiles(series, prediction_length=24, quantile_levels=list(qlevels))
    med = np.stack([np.asarray(m[i])[0] for i in range(n)])
    q25 = np.stack([np.asarray(m[i])[0] for i in range(n)])  # replaced below
    # m is list over batch, each (1, h, nq)
    arr = [np.asarray(mi) for mi in m]  # each (1, h, nq) presumably
    return med, valid, arr


def build_cache(pipe, targets):
    CACHE.mkdir(exist_ok=True)
    for t in targets:
        f = CACHE / f"{t}_c.npz"
        if f.exists():
            continue
        t0 = time.time()
        o, y, p, per, ev = load_region(t)
        out = {}
        for L in CTX_LENS:
            n = len(o)
            ends = o + 24
            series, valid = [], np.zeros(n, dtype=bool)
            for tt in range(n):
                idx = np.nonzero(ends <= o[tt])[0]
                if len(idx) == 0:
                    series.append(torch.zeros(1))
                    continue
                ctx = np.concatenate([y[j] for j in idx])[-L:]
                series.append(torch.tensor(np.ascontiguousarray(ctx, dtype=np.float32)))
                valid[tt] = True
            qs, _ = pipe.predict_quantiles(series, prediction_length=24,
                                           quantile_levels=[0.25, 0.5, 0.75])
            # qs: list over batch; each element (1, h, nq) — quantiles along axis 2
            A = np.stack([np.asarray(qi).reshape(24, -1) for qi in qs])  # (n, h, nq)
            out[f"med{L}"] = A[:, :, 1]
            out[f"q25{L}"] = A[:, :, 0]
            out[f"q75{L}"] = A[:, :, 2]
        out["valid"] = valid
        out["per_fallback"] = per
        np.savez_compressed(f, **out)
        print(f"[cache] {t} {time.time()-t0:.0f}s")


def main():
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cpu", dtype=torch.float32)

    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
    build_cache(pipe, targets)
    del pipe

    rows = {}
    for t in targets:
        t0 = time.time()
        o, y, p, per, ev = load_region(t)
        nn, h = y.shape
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)

        # J5 (official convention)
        r = recipe_R(y, p, per, ev)
        g = ewma_hod(r, y, BETA)
        w_e = fit_w_early(r, per, y, nn)
        j5 = (1 - w_e) * g + w_e * per

        c = np.load(CACHE / f"{t}_c.npz")
        valid = np.asarray(c["valid"], dtype=bool)
        med = c["med2048"].astype(float)
        med[~valid] = per[~valid]
        med720 = c["med720"].astype(float)
        med720[~valid] = per[~valid]
        med336 = c["med336"].astype(float)
        med336[~valid] = per[~valid]
        q25 = c["q252048"].astype(float)
        q75 = c["q752048"].astype(float)
        q25[~valid] = per[~valid]
        q75[~valid] = per[~valid]
        qtrim = (q25 + med + q75) / 3.0

        # CE: EWMA level correction on Chronos (beta scan on early only)
        best_beta, best_ce = 0.2, None
        for beta in (0.1, 0.2, 0.3, 0.4):
            ce_b = ewma_hod(med, y, beta)
            m = mae(ce_b[e], y[e])
            if best_ce is None or m < mae(best_ce[e], y[e]):
                best_beta, best_ce = beta, ce_b
        ce = best_ce

        # CEp: CE + persistence blend (J5-analogue on Chronos)
        w_ce = fit_w_early(ce, per, y, nn)
        cep = (1 - w_ce) * ce + w_ce * per

        # best3: 3-way (g, per, C) grid on early
        best3_w, best3_m = (1.0, 0.0, 0.0), mae(g[e], y[e])
        for wg in np.arange(0.0, 1.01, 0.1):
            for wp in np.arange(0.0, 1.01 - wg + 1e-9, 0.1):
                wc = round(1.0 - wg - wp, 2)
                if wc < -1e-9:
                    continue
                blend = wg * g[e] + wp * per[e] + wc * med[e]
                m = mae(blend, y[e])
                if m < best3_m:
                    best3_w, best3_m = (float(wg), float(wp), float(wc)), m
        wg, wp, wc = best3_w
        best3 = wg * g + wp * per + wc * med

        # CExJ5: outer blend J5 || CE, w3 on early
        w3 = fit_w_early(j5, ce, y, nn, lo=0.0, hi=1.0, step=0.1)
        cexj5 = (1 - w3) * j5 + w3 * ce

        # ctx ablation arms (EWMA-corrected short contexts, beta=best_beta)
        ce720 = ewma_hod(med720, y, best_beta)
        ce336 = ewma_hod(med336, y, best_beta)

        row = {
            "J5_late": mae(j5[l], y[l]),
            "C_late": mae(med[l], y[l]),
            "CE_late": mae(ce[l], y[l]),
            "CEp_late": mae(cep[l], y[l]),
            "best3_late": mae(best3[l], y[l]),
            "CExJ5_late": mae(cexj5[l], y[l]),
            "CE720_late": mae(ce720[l], y[l]),
            "CE336_late": mae(ce336[l], y[l]),
            "qtrim_late": mae(qtrim[l], y[l]),
            "J5_full": mae(j5, y),
            "CE_full": mae(ce, y),
            "CEp_full": mae(cep, y),
            "best3_full": mae(best3, y),
            "CExJ5_full": mae(cexj5, y),
            "w_e": w_e, "w_ce": w_ce, "w3": w3,
            "best3_w": best3_w, "beta_ce": best_beta,
        }
        rows[t] = row
        print(f"[{t}] J5 {row['J5_late']:.1f} | C {row['C_late']:.1f} | CE {row['CE_late']:.1f} "
              f"| CEp {row['CEp_late']:.1f} | best3 {row['best3_late']:.1f} | CExJ5 {row['CExJ5_late']:.1f} "
              f"| ce720 {row['CE720_late']:.1f} | ce336 {row['CE336_late']:.1f} ({time.time()-t0:.0f}s)")

    def subset(th, key="J5_late"):
        return [t for t in rows if rows[t][key] > th]

    key_sets = {
        "all": sorted(rows),
        "gt30_J5": [t for t in rows if rows[t]["J5_late"] > 30],
        "gt40_J5": [t for t in rows if rows[t]["J5_late"] > 40],
    }
    agg = {}
    for name, S in key_sets.items():
        a = {"n": len(S)}
        for k in rows["SA1"]:
            if k.endswith("_late") or k.endswith("_full"):
                a[k] = float(np.mean([rows[t][k] for t in S]))
        for arm in ["CE", "CEp", "best3", "CExJ5", "CE720", "CE336", "qtrim"]:
            a[f"{arm}_wins_vs_J5"] = int(sum(rows[t][f"{arm}_late"] < rows[t]["J5_late"] for t in S))
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

    arms = ["C", "CE", "CEp", "best3", "CExJ5", "CE720", "CE336", "qtrim"]
    L = ["# FD-50p 二轮 TSFM 压降 — late 半段（诚实）", ""]
    L.append("基线 J5（官方口径复现）与 FD-50o 的 J5||C 参照：all 30.76 → 28.27。")
    L.append("")
    for name, a in agg.items():
        L.append(f"## {name}（n={a['n']}）")
        L.append(f"- J5 {a['J5_late']:.2f} | C {a['C_late']:.2f} | CE {a['CE_late']:.2f} | CEp {a['CEp_late']:.2f}")
        L.append(f"- best3 {a['best3_late']:.2f} | CExJ5 {a['CExJ5_late']:.2f} | ce720 {a['CE720_late']:.2f} | ce336 {a['CE336_late']:.2f} | qtrim {a['qtrim_late']:.2f}")
        wins = " | ".join(f"{arm} {a[f'{arm}_wins_vs_J5']}/{a['n']}" for arm in ["CE", "CEp", "best3", "CExJ5"])
        L.append(f"- 胜场 vs J5: {wins}")
        L.append(f"- 全周期: J5 {a['J5_full']:.2f} | CE {a['CE_full']:.2f} | CEp {a['CEp_full']:.2f} | best3 {a['best3_full']:.2f} | CExJ5 {a['CExJ5_full']:.2f}")
        L.append("")
    L.append("## 逐区明细（late，按 J5 降序）")
    L.append("")
    L.append("| target | J5 | C | CE | CEp | best3 | CExJ5 | ce720 | ce336 | w3 | best3_w |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(rows, key=lambda u: -rows[u]["J5_late"]):
        r = rows[t]
        L.append(f"| {t} | {r['J5_late']:.1f} | {r['C_late']:.1f} | {r['CE_late']:.1f} | {r['CEp_late']:.1f} "
                 f"| {r['best3_late']:.1f} | {r['CExJ5_late']:.1f} | {r['CE720_late']:.1f} | {r['CE336_late']:.1f} "
                 f"| {r['w3']:.1f} | {r['best3_w'][0]:.1f}/{r['best3_w'][1]:.1f}/{r['best3_w'][2]:.1f} |")
    OUT_MD.write_text("\n".join(L) + "\n")
    print("DONE")
    print("\n".join(L[:36]))


if __name__ == "__main__":
    main()
