"""FD-50o: TSFM (Chronos-2) zero-shot fusion into the J5 recipe (channel P3).

Motivation: CarbonX (arXiv 2510.01521) shows pretrained time-series foundation
models match specialised SOTA on carbon-intensity forecasting using only the
historical CI series.  Our 20-unreachable proof (FD-50l L3) fixes the *shape*
to J5's shape and cheats the *level* perfectly — the only lever that can move
that floor is a better shape from the delayed-label series itself.  Chronos-2
consumes the same 1-day-delayed hourly truths as persistence/J5 (I_lag tier),
so any gain is attributable to the pretrained temporal prior, not extra data.

Arms (all on the official dump, late half = honest):
  base      raw pred_i_cfg
  J5        recipe_R + hour-of-day EWMA(beta=0.2, LORO-proven) + persistence blend w_early
  C         Chronos-2 median zero-shot from delayed-label context (pure TSFM)
  J5+C      J5's persistence component replaced by Chronos-2 prediction
             (1-w_e)*g + w_e*C        [component swap]
  J5||C     convex blend of J5 and C  (1-w2)*J5 + w2*C, w2 fitted on early
  scale     Chronos-2 output affine-corrected to J5 level (a*C+b, early fit)
             then used as persistence replacement  — tests whether TSFM wins
             on shape only (level already carried by EWMA)

Honesty protocol:
  - Chronos context for window t uses ONLY windows with origin+24 <= origin_t
    (strictly past truths, same 1-day delay as persistence).
  - All blend weights are fitted on the EARLY half only; late is untouched.
  - Chronos-2 is used zero-shot; no fine-tuning on any region.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

DUMP = Path("results/dunkelflaute")
OUT_JSON = Path("results/fd50o_tsfm.json")
OUT_MD = Path("results/fd50o_tsfm.md")
MODEL = "amazon/chronos-2"
OFFD = 718872
NB = 3
TAU_S = 0.60
W_EVT = 0.5
BETA = 0.2          # LORO-proven on era5 arm (FD-50j/l/m), 29/29 regions
MAX_CTX = 2048      # Chronos-2 context cap (tokens == hours here)


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


def chronos_predictions(pipe, o, y):
    """Zero-shot Chronos-2 24h medians from strictly-past truths.

    Context for window t = concat(y[j] for j with o[j]+24 <= o[t]), truncated
    to the last MAX_CTX hours.  Gaps (DST / missing) are tolerated — the
    series is simply the available delayed labels in time order.
    """
    n = len(o)
    # Build the continuous delayed-label series piecewise: available history
    # before window t is the concat of all windows ending at/before o[t].
    # Since gaps only shrink context, we rebuild context per window cheaply:
    # prefix arrays of windows sorted by origin.
    ends = o + 24  # window j's truths cover [o[j], o[j]+24)
    preds = np.empty((n, y.shape[1]), dtype=float)
    for t in range(n):
        # windows fully in the past: end <= origin_t  => available labels
        idx = np.nonzero(ends <= o[t])[0]
        if len(idx) == 0:
            # no past labels: fall back to model-free persistence-of-climatology
            preds[t] = y[t] * np.nan  # placeholder, handled by caller
            continue
        ctx = np.concatenate([y[j] for j in idx])[-MAX_CTX:]
        x = torch.tensor(np.ascontiguousarray(ctx, dtype=np.float32)).reshape(1, 1, -1)
        _, m = pipe.predict_quantiles(x, prediction_length=24, quantile_levels=[0.5])
        preds[t] = np.asarray(m[0])[0]
    return preds


def chronos_batch(pipe, o, y):
    """Batched version: one call with n series (n,1,ctx) — same contexts."""
    n = len(o)
    ends = o + 24
    series, valid = [], np.zeros(n, dtype=bool)
    for t in range(n):
        idx = np.nonzero(ends <= o[t])[0]
        if len(idx) == 0:
            series.append(torch.zeros(1))
            continue
        ctx = np.concatenate([y[j] for j in idx])[-MAX_CTX:]
        series.append(torch.tensor(np.ascontiguousarray(ctx, dtype=np.float32)))
        valid[t] = True
    q, m = pipe.predict_quantiles(series, prediction_length=24, quantile_levels=[0.5])
    preds = np.stack([np.asarray(m[i])[0] for i in range(n)])
    return preds, valid


def affine_to_level(src, ref, mask):
    """Least-squares a*src+b fitted on masked (early) windows to match ref level."""
    s = src[mask].ravel()
    r = ref[mask].ravel()
    A = np.stack([s, np.ones_like(s)], axis=1)
    coef, *_ = np.linalg.lstsq(A, r, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    return a * src + b, a, b


def main():
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained(MODEL, device_map="cpu", dtype=torch.float32)

    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
    rows = {}
    t_start = time.time()
    for t in targets:
        t0 = time.time()
        o, y, p, per, ev = load_region(t)
        nn, h = y.shape
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)
        em = np.zeros(nn, dtype=bool)
        em[: nn // 2] = True

        # J5 reproduction (OFFICIAL fd50j convention: w fitted on R pre-EWMA,
        # beta=0.2 LORO-proven; verified to reproduce fd50j J5_loro_late exactly)
        r = recipe_R(y, p, per, ev)
        g = ewma_hod(r, y, BETA)
        w_e = fit_w_early(r, per, y, nn)
        j5 = (1 - w_e) * g + w_e * per

        # Chronos-2 zero-shot from delayed labels
        c_raw, valid = chronos_batch(pipe, o, y)
        c = c_raw.copy()
        # window 0 has no context: fall back to persistence (same tier info)
        c[~valid] = per[~valid]

        # Arm C (pure Chronos)
        arm_c = c

        # Arm J5+C (component swap)
        j5c = (1 - w_e) * g + w_e * c

        # Arm J5||C (outer blend, w2 on early)
        w2 = fit_w_early(j5, c, y, nn, lo=0.0, hi=1.0, step=0.1)
        j5oc = (1 - w2) * j5 + w2 * c

        # Arm scale (Chronos shape, EWMA/J5 level anchor)
        c_lv, a_cf, b_cf = affine_to_level(c, g, em)
        scale = (1 - w_e) * g + w_e * c_lv

        row = {
            "base_late": mae(p[l], y[l]),
            "R_late": mae(r[l], y[l]),
            "per_late": mae(per[l], y[l]),
            "J5_late": mae(j5[l], y[l]),
            "C_late": mae(c[l], y[l]),
            "J5C_late": mae(j5c[l], y[l]),
            "J5OC_late": mae(j5oc[l], y[l]),
            "SCALE_late": mae(scale[l], y[l]),
            "base_full": mae(p, y),
            "J5_full": mae(j5, y),
            "C_full": mae(c, y),
            "J5C_full": mae(j5c, y),
            "J5OC_full": mae(j5oc, y),
            "SCALE_full": mae(scale, y),
            "w_e": w_e, "w2": w2, "affine": [a_cf, b_cf],
            "n_valid_ctx": int(valid.sum()),
        }
        rows[t] = row
        print(f"[{t}] base {row['base_late']:.1f} | J5 {row['J5_late']:.1f} | C {row['C_late']:.1f} "
              f"| J5+C {row['J5C_late']:.1f} | J5||C {row['J5OC_late']:.1f} | scale {row['SCALE_late']:.1f} "
              f"({time.time()-t0:.0f}s)")

    def subset(th, key="base_late"):
        return [t for t in rows if rows[t][key] > th]

    subsets = {"all": sorted(rows), "gt30": subset(30), "gt40": subset(40)}
    agg = {}
    for name, S in subsets.items():
        a = {"n": len(S)}
        for k in ["base_late", "R_late", "per_late", "J5_late", "C_late", "J5C_late", "J5OC_late", "SCALE_late",
                  "base_full", "J5_full", "C_full", "J5C_full", "J5OC_full", "SCALE_full"]:
            a[k] = float(np.mean([rows[t][k] for t in S]))
        # win counts vs J5
        a["J5C_wins"] = int(sum(rows[t]["J5C_late"] < rows[t]["J5_late"] for t in S))
        a["J5OC_wins"] = int(sum(rows[t]["J5OC_late"] < rows[t]["J5_late"] for t in S))
        a["SCALE_wins"] = int(sum(rows[t]["SCALE_late"] < rows[t]["J5_late"] for t in S))
        a["best_wins"] = int(sum(min(rows[t]["J5C_late"], rows[t]["J5OC_late"], rows[t]["SCALE_late"]) < rows[t]["J5_late"] for t in S))
        agg[name] = a

    def ser(o):
        if isinstance(o, dict):
            return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [ser(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 3)
        return o

    OUT_JSON.write_text(json.dumps({"rows": ser(rows), "agg": ser(agg), "subsets": {k: v for k, v in subsets.items()}},
                                    indent=1, ensure_ascii=False))

    L = ["# FD-50o TSFM（Chronos-2 零样本）融合 J5 — late 半段（诚实）", ""]
    L.append("信息层级：与 persistence/J5 相同（1 日延迟标签，I_lag）。Chronos-2 零样本、无微调。")
    L.append(f"模型：{MODEL}，上下文 = 全部 origin+24h ≤ 当窗 origin 的真值（最长 {MAX_CTX}h）。")
    L.append("")
    for name, a in agg.items():
        L.append(f"## {name}（n={a['n']}）")
        L.append(f"- base {a['base_late']:.2f} | R {a['R_late']:.2f} | persistence {a['per_late']:.2f}")
        L.append(f"- J5 {a['J5_late']:.2f} | C(纯Chronos) {a['C_late']:.2f} | J5+C(分量替换) {a['J5C_late']:.2f} "
                 f"| J5||C(外混合) {a['J5OC_late']:.2f} | scale(水平锚定) {a['SCALE_late']:.2f}")
        L.append(f"- 胜场 vs J5: J5+C {a['J5C_wins']}/{a['n']} | J5||C {a['J5OC_wins']}/{a['n']} | scale {a['SCALE_wins']}/{a['n']} | 任一 {a['best_wins']}/{a['n']}")
        L.append(f"- 全周期: base {a['base_full']:.2f} | J5 {a['J5_full']:.2f} | C {a['C_full']:.2f} | J5+C {a['J5C_full']:.2f} | J5||C {a['J5OC_full']:.2f} | scale {a['SCALE_full']:.2f}")
        L.append("")
    L.append("## 逐区明细（late，按 base 降序）")
    L.append("")
    L.append("| target | base | per | J5 | C | J5+C | J5||C | scale | w_e | w2 |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(rows, key=lambda u: -rows[u]["base_late"]):
        r = rows[t]
        L.append(f"| {t} | {r['base_late']:.1f} | {r['per_late']:.1f} | {r['J5_late']:.1f} | {r['C_late']:.1f} "
                 f"| {r['J5C_late']:.1f} | {r['J5OC_late']:.1f} | {r['SCALE_late']:.1f} | {r['w_e']:.2f} | {r['w2']:.2f} |")
    OUT_MD.write_text("\n".join(L) + "\n")
    print(f"\nTOTAL {time.time()-t_start:.0f}s")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
