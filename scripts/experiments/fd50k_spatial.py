"""FD-50k: spatial + multi-scale + amplitude + Oracle-floor stack on FD-50j.

Levers (all delayed-label tier, no new information beyond yesterday's truth):
  K1  family-shared hod-EWMA: one 24-vector per aligned cluster (UK14/US8/AU3/GB1/...),
      fitted on the pooled residuals of ALL family members (y_t - r_t, strictly causal).
  K2  multi-scale EWMA: slow beta_s + fast beta_f additive, hod vector each.
  K2e multi-scale with event-window beta separation.
  K3  amplitude online calibration (window-mean ratio tracking).
  K4  K1 + persistence blend.
  K5  K4 + K3 amplitude.
  Orl Oracle constant per target computed WITHOUT target-late truth.

Selection: all hyper-params LORO (picked on pooled EARLY halves of other
regions / family members), applied to target late. Honest late eval.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

DUMP = Path("results/dunkelflaute")
OUT_JSON = Path("results/fd50k_spatial.json")
OUT_MD = Path("results/fd50k_spatial.md")
NB = 3
TAU_S = 0.60
OFFD = 718872
W_EVT = 0.5
BETAS = [0.1, 0.2, 0.3]
BFAS = [0.3, 0.4, 0.5, 0.7]
BSLO = [0.03, 0.05, 0.10]
B_EV = [0.5, 0.7, 0.9]


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
    w = np.asarray(d["wx_w_mean"])
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


def ewma_family(rD, yD, beta, fam):
    """Family-shared hod-EWMA: same time index across members; b updated with
    pooled family residuals of windows <= t (strictly causal per window)."""
    ref = fam[0]
    T = len(rD[ref])
    H = rD[ref].shape[1]
    out = {t: np.empty_like(rD[t]) for t in fam}
    b = np.zeros(H)
    for i in range(T):
        res_all = []
        for t in fam:
            out[t][i] = rD[t][i] + b
            res_all.append(yD[t][i] - rD[t][i])
        b = (1 - beta) * b + beta * np.mean(res_all, axis=0)
    return out


def ewma_multi(r, y, bs, bf):
    out = np.empty_like(r)
    bs_v = np.zeros(r.shape[1])
    bf_v = np.zeros(r.shape[1])
    for t in range(len(r)):
        out[t] = r[t] + bs_v + bf_v
        res = y[t] - r[t]
        bs_v = (1 - bs) * bs_v + bs * res
        bf_v = (1 - bf) * bf_v + bf * res
    return out


def ewma_multi_evt(r, y, bs, bf, evt, bf_ev):
    out = np.empty_like(r)
    bs_v = np.zeros(r.shape[1])
    bf_v = np.zeros(r.shape[1])
    for t in range(len(r)):
        out[t] = r[t] + bs_v + bf_v
        res = y[t] - r[t]
        bs_v = (1 - bs) * bs_v + bs * res
        bb = bf if not evt[t] else bf_ev
        bf_v = (1 - bb) * bf_v + bb * res
    return out


def amp_track(r, y, a0=1.0, beta_a=0.3):
    """Online amplitude ratio tracking on window means."""
    out = np.empty_like(r)
    a = a0
    for t in range(len(r)):
        m_r = r[t].mean()
        out[t] = m_r + a * (r[t] - m_r)
        ratio = (y[t].mean() / m_r) if m_r != 0 else 1.0
        a = (1 - beta_a) * a + beta_a * ratio
    return out


def blend_per(r, per, w):
    return (1 - w) * r + w * per


def fit_w_early(r, per, y, nn):
    e = slice(0, nn // 2)
    best_w, best_m = 0.0, mae(r[e], y[e])
    for w in np.arange(0.0, 0.71, 0.05):
        m = mae(blend_per(r[e], per[e], w), y[e])
        if m < best_m:
            best_w, best_m = float(w), m
    return best_w


def oracle_const_floor(data, targets):
    """Per target: best constant on late windows computed WITHOUT any
    target-late info (candidates: target early mean + all other regions'
    full-period means). Late MAE of best candidate."""
    res = {}
    for t in targets:
        y_ = data[t][1]
        nn = len(y_)
        e = slice(0, nn // 2)
        l = slice(nn // 2, nn)
        cands = [float(y_[e].mean())] + [float(data[u][1].mean()) for u in targets if u != t]
        best_c, best_m = None, 1e9
        for c in cands:
            m = mae(np.full_like(y_[l], c), y_[l])
            if m < best_m:
                best_c, best_m = c, m
        res[t] = (best_c, best_m)
    return res


def main():
    targets = sorted(p.name[: -len("_seed0.npz")] for p in DUMP.glob("*_seed0.npz"))
    data = {t: load_region(t) for t in targets}
    R = {t: recipe_R(data[t][1], data[t][2], data[t][3], data[t][4]) for t in targets}

    sigs = {}
    for t in targets:
        sigs.setdefault(data[t][0].tobytes(), []).append(t)
    fams = {f"cluster{i+1}": v for i, v in enumerate(sorted(sigs.values(), key=lambda v: v[0]))}
    fam_of = {t: f for f, v in fams.items() for t in v}

    rows = {}
    for t in targets:
        y = data[t][1]
        r = R[t]
        nn = len(y)
        l = slice(nn // 2, nn)
        rows[t] = {
            "base_late": mae(data[t][2][l], y[l]),
            "R_late": mae(r[l], y[l]),
            "per_late": mae(data[t][3][l], y[l]),
            "w_early": fit_w_early(r, data[t][3], y, nn),
        }

    # J5 arm (beta LORO on other regions early, w early own)
    for t in targets:
        y = data[t][1]
        r = R[t]
        per = data[t][3]
        nn = len(y)
        l = slice(nn // 2, nn)
        others = [u for u in targets if u != t]
        best_b, best_m = None, 1e9
        for b in BETAS:
            m = np.mean([
                mae(ewma_hod(R[u], data[u][1], b)[slice(0, len(data[u][1]) // 2)],
                    data[u][1][slice(0, len(data[u][1]) // 2)])
                for u in others
            ])
            if m < best_m:
                best_b, best_m = b, m
        g = ewma_hod(r, y, best_b)
        w_e = rows[t]["w_early"]
        rows[t]["J5"] = mae(blend_per(g[l], per[l], w_e), y[l])
        rows[t]["beta_loro"] = best_b

    # K1 family EWMA, beta picked on pooled early of family members (honest: early only)
    fam_beta = {}
    for cname, fam in fams.items():
        rD = {t: R[t] for t in fam}
        yD = {t: data[t][1] for t in fam}
        T = len(data[fam[0]][1])
        e = slice(0, T // 2)
        best_b, best_m = None, 1e9
        for b in BETAS:
            oD = ewma_family({t: rD[t][e] for t in fam}, {t: yD[t][e] for t in fam}, b, fam)
            m = np.mean([mae(oD[t], yD[t][e]) for t in fam])
            if m < best_m:
                best_b, best_m = b, m
        fam_beta[cname] = best_b
        oD = ewma_family(rD, yD, best_b, fam)
        for t in fam:
            l = slice(T // 2, T)
            rows[t]["K1"] = (mae(oD[t][l], yD[t][l]), best_b)

    # K2 multi-scale, LORO beta grid
    for t in targets:
        y = data[t][1]
        r = R[t]
        nn = len(y)
        l = slice(nn // 2, nn)
        others = [u for u in targets if u != t]
        best, best_m = None, 1e9
        for bs in BSLO:
            for bf in BFAS:
                m = np.mean([
                    mae(ewma_multi(R[u], data[u][1], bs, bf)[slice(0, len(data[u][1]) // 2)],
                        data[u][1][slice(0, len(data[u][1]) // 2)])
                    for u in others
                ])
                if m < best_m:
                    best, best_m = (bs, bf), m
        g = ewma_multi(r, y, best[0], best[1])
        rows[t]["K2"] = (mae(g[l], y[l]), best)

    # K2e multi-scale with event beta
    for t in targets:
        y = data[t][1]
        r = R[t]
        ev = data[t][4]
        nn = len(y)
        l = slice(nn // 2, nn)
        others = [u for u in targets if u != t]
        best, best_m = None, 1e9
        for bs in BSLO:
            for bf in BFAS:
                for bev in B_EV:
                    m = np.mean([
                        mae(ewma_multi_evt(R[u], data[u][1], bs, bf, data[u][4], bev)[slice(0, len(data[u][1]) // 2)],
                            data[u][1][slice(0, len(data[u][1]) // 2)])
                        for u in others
                    ])
                    if m < best_m:
                        best, best_m = (bs, bf, bev), m
        g = ewma_multi_evt(r, y, best[0], best[1], ev, best[2])
        rows[t]["K2e"] = (mae(g[l], y[l]), best)

    # K3 amplitude
    for t in targets:
        y = data[t][1]
        r = R[t]
        nn = len(y)
        l = slice(nn // 2, nn)
        others = [u for u in targets if u != t]
        best_ba, best_m = None, 1e9
        for ba in [0.1, 0.2, 0.3, 0.5]:
            m = np.mean([
                mae(amp_track(R[u], data[u][1], beta_a=ba)[slice(0, len(data[u][1]) // 2)],
                    data[u][1][slice(0, len(data[u][1]) // 2)])
                for u in others
            ])
            if m < best_m:
                best_ba, best_m = ba, m
        g = amp_track(r, y, beta_a=best_ba)
        rows[t]["K3"] = (mae(g[l], y[l]), best_ba)

    # K4 = K1 + persistence blend
    K1_out = {}
    for cname, fam in fams.items():
        rD = {t: R[t] for t in fam}
        yD = {t: data[t][1] for t in fam}
        K1_out[cname] = ewma_family(rD, yD, fam_beta[cname], fam)
    for t in targets:
        y = data[t][1]
        per = data[t][3]
        nn = len(y)
        l = slice(nn // 2, nn)
        g_t = K1_out[fam_of[t]][t]
        rows[t]["K4"] = mae(blend_per(g_t[l], per[l], rows[t]["w_early"]), y[l])

    # K5 = K4 + amplitude
    for t in targets:
        y = data[t][1]
        per = data[t][3]
        nn = len(y)
        l = slice(nn // 2, nn)
        g_t = amp_track(K1_out[fam_of[t]][t], y, beta_a=rows[t]["K3"][1])
        rows[t]["K5"] = mae(blend_per(g_t[l], per[l], rows[t]["w_early"]), y[l])

    # Oracle floor
    orc = oracle_const_floor(data, targets)
    for t in targets:
        rows[t]["oracle_floor"] = orc[t][1]
        rows[t]["oracle_c"] = orc[t][0]

    def subset(th):
        return [t for t in targets if rows[t]["base_late"] > th]

    subsets = {"all": targets, "gt30": subset(30), "gt40": subset(40)}
    agg = {}
    for name, S in subsets.items():
        a = {"n": len(S)}
        a["base"] = float(np.mean([rows[t]["base_late"] for t in S]))
        a["per"] = float(np.mean([rows[t]["per_late"] for t in S]))
        a["R"] = float(np.mean([rows[t]["R_late"] for t in S]))
        a["J5"] = float(np.mean([rows[t]["J5"] for t in S]))
        a["K1"] = float(np.mean([rows[t]["K1"][0] for t in S]))
        a["K2"] = float(np.mean([rows[t]["K2"][0] for t in S]))
        a["K2e"] = float(np.mean([rows[t]["K2e"][0] for t in S]))
        a["K3"] = float(np.mean([rows[t]["K3"][0] for t in S]))
        a["K4"] = float(np.mean([rows[t]["K4"] for t in S]))
        a["K5"] = float(np.mean([rows[t]["K5"] for t in S]))
        a["oracle_floor"] = float(np.mean([rows[t]["oracle_floor"] for t in S]))
        agg[name] = a

    def ser(o):
        if isinstance(o, dict):
            return {str(k): ser(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [ser(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return round(float(o), 3)
        return o

    OUT_JSON.write_text(json.dumps({"rows": ser(rows), "agg": ser(agg), "fams": fams, "fam_beta": fam_beta}, indent=1, ensure_ascii=False))

    L = ["# FD-50k 空间/多尺度/幅度 + Oracle 地板（late 半段，诚实）", ""]
    L.append("信息层级：昨日真值（I_lag）。所有 β/权重 LORO 或 early 拟合。")
    L.append("")
    for name, a in agg.items():
        L.append(f"## {name}（n={a['n']}）")
        L.append(f"- base {a['base']:.2f} | persistence {a['per']:.2f} | R {a['R']:.2f} | J5 {a['J5']:.2f} | K1(family) {a['K1']:.2f} | K2(ms) {a['K2']:.2f} | K2e {a['K2e']:.2f} | K3(amp) {a['K3']:.2f} | K4 {a['K4']:.2f} | K5 {a['K5']:.2f} | Oracle地板 {a['oracle_floor']:.2f}")
        L.append("")
    L.append("## 全区明细（late，按 base_late 降序）")
    L.append("")
    L.append("| target | base | R | J5 | K1 | K2 | K2e | K3 | K4 | K5 | oracle |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in sorted(targets, key=lambda u: -rows[u]["base_late"]):
        r_ = rows[t]
        L.append(
            f"| {t} | {r_['base_late']:.1f} | {r_['R_late']:.1f} | {r_['J5']:.1f} "
            f"| {r_['K1'][0]:.1f} | {r_['K2'][0]:.1f} | {r_['K2e'][0]:.1f} | {r_['K3'][0]:.1f} "
            f"| {r_['K4']:.1f} | {r_['K5']:.1f} | {r_['oracle_floor']:.1f} |"
        )
    OUT_MD.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
