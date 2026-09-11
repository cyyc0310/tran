#!/usr/bin/env python3
"""FD-50g: per-lead-bucket delayed calibration.

Diagnostic finding: optimal a* varies systematically WITHIN the 24h
window — UK regions need mid-window (h8-16) amplification up to 2.3
while AU regions need mid-window suppression to 0.3. A single global
a* averages this away. Buckets have plenty of samples per window
(each window contributes h/3 hours to each bucket), so per-bucket
delayed labels are as deployable as the global version.

Arms on the late half (all zero-telemetry, delayed labels only):
  G    FD-50e reference: global gated a (tau=0.05, family fallback)
  B1   per-bucket a from early half, ungated
  B2   per-bucket gated (imp_b >= 0.05), fallback = family per-bucket
       median, then 1.0
  B3   B2 + rolling: for the last quarter of the late half, refit the
       per-bucket a on the previous quarter (captures a* drift like
       UK_09 -0.96)
  orc  per-bucket a refit on the late half itself (cheating upper bound)

Output: results/fd50g_bucket.json + fd50g_bucket.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DUMP = ROOT / "results" / "dunkelflaute"
RESULTS = ROOT / "results"

TAU = 0.05
NB = 3  # lead buckets


def amp(pred: np.ndarray, a: float) -> np.ndarray:
    pm = pred.mean(axis=1, keepdims=True)
    return pm + a * (pred - pm)


def mae(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(p - y).mean())


def fit_a_pivot(p_slice: np.ndarray, pm: np.ndarray,
                y_slice: np.ndarray) -> float:
    """Fit a for slice, rotating around the given window-mean pivot pm."""
    best_a, best_m = 1.0, mae(pm + 1.0 * (p_slice - pm), y_slice)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(pm + a * (p_slice - pm), y_slice)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


def apply_pivot(p_slice: np.ndarray, pm: np.ndarray, a: float) -> np.ndarray:
    return pm + a * (p_slice - pm)


def fit_a(pred: np.ndarray, y: np.ndarray) -> float:
    if len(pred) == 0:
        return 1.0
    best_a, best_m = 1.0, mae(pred, y)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(amp(pred, a), y)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


def family_of(t: str) -> str:
    if t in ("VIC1", "NSW1", "SA1", "QLD1"):
        return "AU"
    if t.startswith("UK"):
        return "UK"
    return "US"


def main() -> None:
    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))

    data = {}
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        o = np.asarray(d["origin_hours"])
        order = o.argsort()
        n = len(order)
        y = d["y_true"].astype(np.float64)[order]
        p = d["pred_i_cfg"].astype(np.float64)[order]
        h = p.shape[1]
        cuts = [(i * h // NB, (i + 1) * h // NB) for i in range(NB)]
        data[t] = {
            "y": y, "p": p, "h": h, "cuts": cuts,
            "q": [slice(i * n // 4, (i + 1) * n // 4) for i in range(4)],
            "fam": family_of(t),
        }

    # ---- per-region fits on EARLY half (q1+q2) ----
    fit_e = {}   # target -> [a_b0, a_b1, a_b2] fitted on q1+q2
    imp_e = {}   # target -> [imp_b0, imp_b1, imp_b2]
    a_glob_e = {}
    for t in targets:
        p, y, cuts = data[t]["p"], data[t]["y"], data[t]["cuts"]
        e = slice(0, len(y) // 2)
        pm_e = p[e].mean(axis=1, keepdims=True)
        ab, ib = [], []
        for s, e2 in cuts:
            a = fit_a_pivot(p[e, s:e2], pm_e, y[e, s:e2])
            m1 = mae(pm_e + 1.0 * (p[e, s:e2] - pm_e), y[e, s:e2])
            ma = mae(pm_e + a * (p[e, s:e2] - pm_e), y[e, s:e2])
            ab.append(a)
            ib.append((m1 - ma) / m1 if m1 > 0 else 0.0)
        fit_e[t] = ab
        imp_e[t] = ib
        a_glob_e[t] = fit_a_pivot(p[e], pm_e, y[e])

    # family per-bucket medians (LOO for the target itself)
    fam_med = {}
    for t in targets:
        fam = data[t]["fam"]
        row = []
        for b in range(NB):
            vals = [fit_e[o][b] for o in targets
                    if data[o]["fam"] == fam and o != t]
            row.append(float(np.median(vals)) if vals else 1.0)
        fam_med[t] = row

    rows = []
    for t in targets:
        p, y, cuts, q = (data[t]["p"], data[t]["y"],
                         data[t]["cuts"], data[t]["q"])
        n = len(y)
        e = slice(0, n // 2)
        l = slice(n // 2, n)

        base_l = mae(p[l], y[l])

        # ---- G: FD-50e global gated ----
        pm_l = p[l].mean(axis=1, keepdims=True)
        pm_e = p[e].mean(axis=1, keepdims=True)
        m1e = mae(p[e], y[e])
        ma_e = mae(apply_pivot(p[e], pm_e, a_glob_e[t]), y[e])
        if (m1e - ma_e) / m1e >= TAU:
            a_g = a_glob_e[t]
        else:
            vals = [a_glob_e[o] for o in targets
                    if data[o]["fam"] == data[t]["fam"] and o != t]
            a_g = float(np.median(vals))
        m_g = mae(apply_pivot(p[l], pm_l, a_g), y[l])

        # ---- bucket arms: build per-bucket transformed late pred ----
        def bucket_apply(a_list: list[float]) -> np.ndarray:
            out = np.empty_like(p[l])
            for (s, e2), a in zip(cuts, a_list):
                out[:, s:e2] = apply_pivot(p[l, s:e2], pm_l, a)
            return out

        # B1 ungated
        m_b1 = mae(bucket_apply(fit_e[t]), y[l])

        # B2 gated w/ own-global fallback
        a_b2 = [fit_e[t][b] if imp_e[t][b] >= TAU else a_g
                for b in range(NB)]
        m_b2 = mae(bucket_apply(a_b2), y[l])

        # B3: full rolling — q3 uses per-bucket a refit on q2, q4 on q3;
        # gates fall back to B2 values
        n_l = n - n // 2
        n3 = 3 * n // 4 - n // 2        # len of q3 inside late
        n4 = n - 3 * n // 4             # len of q4 inside late
        out_b3 = bucket_apply(a_b2)     # start from B2
        prev = None                     # (a_list, imp_list) of prev quarter
        for qi, (q_fit, q_app) in enumerate((
                (q[1], slice(0, n3)),           # q2 fit -> q3 apply
                (q[2], slice(n3, n_l)),         # q3 fit -> q4 apply
        )):
            pm_f = p[q_fit].mean(axis=1, keepdims=True)
            a_f = [fit_a_pivot(p[q_fit, s:e2], pm_f, y[q_fit, s:e2])
                   for s, e2 in cuts]
            i_f = []
            for s, e2 in cuts:
                m1q = mae(p[q_fit, s:e2], y[q_fit, s:e2])
                maq = mae(apply_pivot(p[q_fit, s:e2], pm_f,
                                      fit_a_pivot(p[q_fit, s:e2], pm_f,
                                                  y[q_fit, s:e2])),
                          y[q_fit, s:e2])
                i_f.append((m1q - maq) / m1q if m1q > 0 else 0.0)
            a_use = [a_f[b] if i_f[b] >= TAU else a_b2[b] for b in range(NB)]
            p_app = p[l][q_app]
            pm_a = p_app.mean(axis=1, keepdims=True)
            out_app = np.empty_like(p_app)
            for (s, e2), a in zip(cuts, a_use):
                out_app[:, s:e2] = apply_pivot(p_app[:, s:e2], pm_a, a)
            out_b3[q_app] = out_app
        m_b3 = mae(out_b3, y[l])

        # B4: per-bucket q1/q2 stability gate, fallback to global a_g
        row = {
            "target": t, "base_late": round(base_l, 2),
            "a_glob": round(a_g, 2),
            "a_b_early": [round(a, 2) for a in fit_e[t]],
            "a_b_used": [round(a, 2) for a in a_b2],
            "G": round(m_g, 2), "B1": round(m_b1, 2),
            "B2": round(m_b2, 2), "B3": round(m_b3, 2),
        }
        pm1 = p[q[0]].mean(axis=1, keepdims=True)
        pm2 = p[q[1]].mean(axis=1, keepdims=True)
        a_q1b = [fit_a_pivot(p[q[0], s:e2], pm1, y[q[0], s:e2])
                 for s, e2 in cuts]
        a_q2b = [fit_a_pivot(p[q[1], s:e2], pm2, y[q[1], s:e2])
                 for s, e2 in cuts]
        for tau_s in (0.30, 0.45, 0.60):
            a_b4 = [fit_e[t][b]
                    if abs(a_q1b[b] - a_q2b[b]) <= tau_s
                    else a_glob_e[t]        # raw early global, un-gated
                    for b in range(NB)]
            row[f"B4@{tau_s:.2f}"] = round(mae(bucket_apply(a_b4), y[l]), 2)

        # oracle per-bucket on late
        a_late = [fit_a_pivot(p[l, s:e2], pm_l, y[l, s:e2])
                  for s, e2 in cuts]
        m_orc = mae(bucket_apply(a_late), y[l])
        row["orc"] = round(m_orc, 2)

        rows.append(row)

    df = pd.DataFrame(rows)
    (RESULTS / "fd50g_bucket.json").write_text(
        json.dumps({"rows": rows}, indent=1))

    bl = df["base_late"].mean()
    b4cols = [c for c in df.columns if c.startswith("B4@")]
    md = ["# FD-50g 分桶（lead-bucket）延迟校准 — late 半段", "",
          f"基线：{bl:.2f}", ""]
    for k, nm in (("G", "G FD-50e 全局门控"), ("B1", "B1 分桶无门控"),
                  ("B2", "B2 分桶+门控+全局回退"),
                  ("B3", "B3 B2+全滚动(q3←q2, q4←q3)")):
        v = df[k].mean()
        md.append(f"- {nm}：{v:.2f} ({v - bl:+.2f})")
    for c in b4cols:
        v = df[c].mean()
        md.append(f"- B4 分桶+q1/q2稳定门控 tau={c[3:]}（回退 raw 全局）："
                  f"{v:.2f} ({v - bl:+.2f})")
    orc = df["orc"].mean()
    md.append(f"- oracle 分桶（late 自拟合）：{orc:.2f} ({orc - bl:+.2f})")
    md += ["", "## 配方定型（FD-50g final）", "",
           "分桶 a\*（early 半段拟合，围绕整窗 pivot）+ q1/q2 分桶稳定性门控",
           "（|Δa|≤0.60，不稳定桶回退 raw early 全局 a\*，无第二重门控）；",
           "收缩 λ=1.0（不收缩）优于 0.85/0.70。"]
    rt = df[df["base_late"] > 40]
    md += ["", f"## 右尾区（n={len(rt)}）", "",
           f"- 基线 {rt['base_late'].mean():.2f} → "
           f"G {rt['G'].mean():.2f} | B1 {rt['B1'].mean():.2f} "
           f"| B2 {rt['B2'].mean():.2f} | B3 {rt['B3'].mean():.2f} "
           f"| oracle {rt['orc'].mean():.2f}",
           "", "## 右尾区明细", "",
           "| target | late基线 | G | B1 | B2 | B3 | orc | a_b2 |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in rt.sort_values("base_late", ascending=False).iterrows():
        md.append(
            f"| {r['target']} | {r['base_late']:.1f} | {r['G']:.1f} "
            f"| {r['B1']:.1f} | {r['B2']:.1f} | {r['B3']:.1f} "
            f"| {r['orc']:.1f} | {r['a_b_used']} |")
    (RESULTS / "fd50g_bucket.md").write_text("\n".join(md))
    print(f"[fd50g] -> results/fd50g_bucket.md")
    cols = ["target", "base_late", "G", "B1", "B2", "B3", "orc"]
    print(df[cols].sort_values("base_late", ascending=False)
          .to_string(index=False))
    print()
    print(f"pooled: base {bl:.2f} | G {df['G'].mean():.2f} "
          f"| B1 {df['B1'].mean():.2f} | B2 {df['B2'].mean():.2f} "
          f"| B3 {df['B3'].mean():.2f} | orc {df['orc'].mean():.2f}")


if __name__ == "__main__":
    main()
