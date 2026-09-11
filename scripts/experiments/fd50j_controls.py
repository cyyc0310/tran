"""FD-50j controls: is the online gain real model value or just delayed labels?

C0  base i_cfg (no post-proc)                  late MAE
C1  R recipe (B4 + event mix)
C2  raw i_cfg + J2 hod-EWMA (beta .2)         -> is R still needed?
C3  persistence + J2 hod-EWMA (beta .2)       -> labels-only control
C4  persistence + J5 (own EWMA + blend w/ i_cfg? no) -> keep pure: per + hod-EWMA
C5  R + J5 (beta .2 fixed, w early-fit)       -> headline
C6  i0 layer + same J5 stack                  -> does it transfer to I_0
C7  C5 with beta chosen LORO (already .2 everywhere) == C5
C8  C5 with 2-day label delay (EWMA uses residuals of windows <= t-2)
      -> robustness to realistic label latency
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import importlib.util

spec = importlib.util.spec_from_file_location("fj", "scripts/experiments/fd50j_online_bias.py")
fj = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fj)

DUMP = fj.DUMP
BETA = 0.2


def ewma_hod_lag(r, y, beta, lag):
    out = np.empty_like(r)
    b = np.zeros(r.shape[1])
    hist = []
    for t in range(len(r)):
        out[t] = r[t] + b
        hist.append(y[t] - r[t])
        if len(hist) >= lag:
            b = (1 - beta) * b + beta * hist[-lag]
    return out


def main():
    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
    res = {k: [] for k in ("C0", "C1", "C2", "C3", "C5", "C6b", "C6", "C8", "C5w0")}
    base_late = {}
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        o = np.asarray(d["origin_hours"])
        order = o.argsort()
        y = d["y_true"].astype(float)[order]
        p = d["pred_i_cfg"].astype(float)[order]
        p0 = d["pred_i0"].astype(float)[order]
        per = np.asarray(d["pred_persistence"]).astype(float)[order]
        w = np.asarray(d["wx_w_mean"])
        wo = np.asarray(d["wx_day_ord"])
        wmap = dict(zip(wo.tolist(), w.tolist()))
        ev = np.array([wmap[dd] for dd in (o[order] // 24 + fj.OFFD)]) <= np.quantile(w, 0.20)
        nn = len(y)
        l = slice(nn // 2, nn)
        m = lambda a: fj.mae(a[l], y[l])
        r = fj.recipe_R(y, p, per, ev)
        base_late[t] = m(p)
        res["C0"].append(m(p))
        res["C1"].append(m(r))
        res["C2"].append(m(fj.ewma_hod(p, y, BETA)))
        res["C3"].append(m(fj.ewma_hod(per, y, BETA)))
        g = fj.ewma_hod(r, y, BETA)
        w_e = fj.fit_w_early(r, per, y, nn)
        res["C5"].append(m(fj.blend_per(g, per, w_e)))
        res["C5w0"].append(m(g))
        r0 = fj.recipe_R(y, p0, per, ev)
        res["C6b"].append(m(p0))
        g0 = fj.ewma_hod(r0, y, BETA)
        w0 = fj.fit_w_early(r0, per, y, nn)
        res["C6"].append(m(fj.blend_per(g0, per, w0)))
        g2 = ewma_hod_lag(r, y, BETA, 2)
        res["C8"].append(m(fj.blend_per(g2, per, w_e)))

    names = {
        "C0": "base i_cfg", "C1": "R (B4+evt)", "C2": "raw i_cfg + hodEWMA", "C3": "persistence + hodEWMA (labels-only)",
        "C5w0": "R + hodEWMA (no blend)", "C5": "R + hodEWMA + per-blend  [J5]",
        "C6b": "base I_0", "C6": "I_0 + R + J5", "C8": "J5 with 2-day label delay",
    }
    S = {"all": targets, "gt30": [t for t in targets if base_late[t] > 30], "gt40": [t for t in targets if base_late[t] > 40]}
    lines = ["# FD-50j 对照组（late 半段，β=0.2 固定）", ""]
    hdr = "| arm | " + " | ".join(f"{k}(n={len(v)})" for k, v in S.items()) + " |"
    lines += [hdr, "|---|" + "---|" * len(S)]
    idx = {t: i for i, t in enumerate(targets)}
    for k in ("C0", "C1", "C2", "C3", "C5w0", "C5", "C8", "C6b", "C6"):
        vals = [np.mean([res[k][idx[t]] for t in sub]) for sub in S.values()]
        lines.append(f"| {names[k]} | " + " | ".join(f"{v:.2f}" for v in vals) + " |")
    lines.append("")
    wins = sum(1 for t in targets if res["C5"][idx[t]] < res["C3"][idx[t]])
    lines.append(f"J5 胜 labels-only 对照的区数: {wins}/29")
    wins2 = sum(1 for t in targets if res["C5"][idx[t]] < res["C1"][idx[t]])
    lines.append(f"J5 胜 R 配方的区数: {wins2}/29")
    lines.append("")
    lines.append("| target | base | R | labels-only | J5 | J5 lag2 | I_0 | I_0+J5 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for t in sorted(targets, key=lambda u: -base_late[u]):
        i = idx[t]
        lines.append(f"| {t} | {res['C0'][i]:.1f} | {res['C1'][i]:.1f} | {res['C3'][i]:.1f} | {res['C5'][i]:.1f} | {res['C8'][i]:.1f} | {res['C6b'][i]:.1f} | {res['C6'][i]:.1f} |")
    Path("results/fd50j_controls.md").write_text("\n".join(lines) + "\n")
    Path("results/fd50j_controls.json").write_text(json.dumps({"targets": targets, "res": res}, indent=1))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
