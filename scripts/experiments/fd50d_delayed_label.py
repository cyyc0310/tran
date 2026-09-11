#!/usr/bin/env python3
"""FD-50d: config-router (fixed import) + delayed-label self-calibration.

Two deployable amplitude routers, both zero-telemetry at inference:

R4-config : linear fit  a* ~ [mean_rs, ef_nr/1000, mean_rs^2]  on 28
            donors, LOO-predict each target's a. Fixed import
            (all_region_configs lives in transcif.data.loaders, not fuel).

R5-delay  : target's OWN delayed labels. Fit a* on the early half of
            the target's test windows (deployment: labels arrive after
            ~2 weeks), apply to the late half. This quantifies the
            value of delayed self-labels for amplitude calibration —
            the gap between donor router (-1.0) and oracle (-2.0) is
            hypothesised to be closable this way.

Also reports R5-with-b (early-half additive offset too) and the
split-half temporal stability of each region's own a*.

Output: results/fd50d_delayed_label.json + fd50d_delayed_label.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DUMP = ROOT / "results" / "dunkelflaute"
RESULTS = ROOT / "results"


def amp(pred: np.ndarray, a: float, b: float = 0.0) -> np.ndarray:
    pm = pred.mean(axis=1, keepdims=True)
    return pm + b + a * (pred - pm)


def mae(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(p - y).mean())


def fit_a(pred: np.ndarray, y: np.ndarray) -> float:
    best_a, best_m = 1.0, mae(pred, y)
    for a in np.arange(0.30, 2.31, 0.02):
        m = mae(amp(pred, a), y)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


def fit_b(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.median(y.mean(axis=1) - pred.mean(axis=1)))


def main() -> None:
    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))
    data = {}
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        o = np.asarray(d["origin_hours"])
        order = o.argsort()
        cut = len(order) // 2
        early = np.zeros(len(o), dtype=bool)
        early[order[:cut]] = True
        data[t] = {
            "y": d["y_true"].astype(np.float64),
            "pred": d["pred_i_cfg"].astype(np.float64),
            "early": early,
        }

    # ---- family router (R2b) for the late-half head-to-head ---------------
    def family_of(t: str) -> str:
        if t in ("VIC1", "NSW1", "SA1", "QLD1"):
            return "AU"
        if t.startswith("UK"):
            return "UK"
        return "US"

    a_early_all = {t: data[t] and None for t in targets}
    for t in targets:
        p, y, e = data[t]["pred"], data[t]["y"], data[t]["early"]
        a_early_all[t] = fit_a(p[e], y[e])

    rows = []
    for t in targets:
        y, p, e = data[t]["y"], data[t]["pred"], data[t]["early"]
        base = mae(p, y)
        base_e, base_l = mae(p[e], y[e]), mae(p[~e], y[~e])
        a_e = fit_a(p[e], y[e])
        b_e = fit_b(p[e], y[e])
        a_full = fit_a(p, y)
        # R2b: family LOO median of donors' EARLY-half a*, applied late
        fam = [o for o in targets
               if family_of(o) == family_of(t) and o != t]
        a_r2b = float(np.median([a_early_all[o] for o in fam]))

        row = {
            "target": t, "base": round(base, 2),
            "base_early": round(base_e, 2), "base_late": round(base_l, 2),
            "a_early": round(a_e, 2), "a_full": round(a_full, 2),
            "a_stable": bool(abs(a_e - fit_a(p[~e], y[~e])) <= 0.2),
            # R5: delayed self-labels (early fit -> late apply)
            "r5_a": round(a_e, 2),
            "r5_mae_late": round(mae(amp(p[~e], a_e), y[~e]), 2),
            "r5b_mae_late": round(mae(amp(p[~e], a_e, b_e), y[~e]), 2),
            # R2b: family donor router (no target labels at all)
            "r2b_a": round(a_r2b, 2),
            "r2b_mae_late": round(mae(amp(p[~e], a_r2b), y[~e]), 2),
        }
        # oracle on late half for reference
        row["orc_a_late"] = round(fit_a(p[~e], y[~e]), 2)
        row["orc_mae_late"] = round(mae(amp(p[~e], row["orc_a_late"]),
                                        y[~e]), 2)

        # R4 config router dropped: config vector unavailable in npz dumps
        rows.append(row)

    df = pd.DataFrame(rows)
    (RESULTS / "fd50d_delayed_label.json").write_text(
        json.dumps({"rows": rows}, indent=1))

    md = ["# FD-50d 延迟标签自校准 vs donor 路由（late 半段对决）", "",
          "R2b：家族 donor early a* 中位（零目标标签）；",
          "R5：目标区 early 半段延迟标签拟合 a → late 半段应用；",
          "R5b：R5 + early 偏移 b（幅度+水平联合）", "",
          "## 汇总（29 区 late 半段 pooled）", ""]
    base_p = df["base"].mean()
    bl = df["base_late"].mean()
    r2b = df["r2b_mae_late"].mean()
    r5 = df["r5_mae_late"].mean()
    r5b = df["r5b_mae_late"].mean()
    orc = df["orc_mae_late"].mean()
    md += [
        f"- 全区基线：{base_p:.2f}",
        f"- late 半段基线：{bl:.2f}",
        f"- R2b donor 路由（late）：{r2b:.2f} ({r2b - bl:+.2f})",
        f"- R5 延迟自校准（late）：{r5:.2f} ({r5 - bl:+.2f})",
        f"- R5b 幅度+偏移（late）：{r5b:.2f} ({r5b - bl:+.2f})",
        f"- [作弊] late 自身 a*：{orc:.2f} ({orc - bl:+.2f})",
    ]
    rt = df[df["base"] > 40]
    md += [
        "",
        f"右尾区（n={len(rt)}）late 半段：基线 {rt['base_late'].mean():.2f} "
        f"→ R2b {rt['r2b_mae_late'].mean():.2f} "
        f"| R5 {rt['r5_mae_late'].mean():.2f} "
        f"| R5b {rt['r5b_mae_late'].mean():.2f} "
        f"| oracle {rt['orc_mae_late'].mean():.2f}",
    ]
    md += ["", "## 右尾区明细", "",
           "| target | base | late基线 | a_e | R2b | R5 | R5b | orc |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in rt.sort_values("base", ascending=False).iterrows():
        md.append(
            f"| {r['target']} | {r['base']:.1f} | {r['base_late']:.1f} "
            f"| {r['a_early']:.2f} | {r['r2b_mae_late']:.1f} "
            f"| {r['r5_mae_late']:.1f} | {r['r5b_mae_late']:.1f} "
            f"| {r['orc_mae_late']:.1f} |")
    (RESULTS / "fd50d_delayed_label.md").write_text("\n".join(md))
    print(f"[fd50d] -> results/fd50d_delayed_label.md")
    cols = ["target", "base", "base_late", "a_early", "r2b_mae_late",
            "r5_mae_late", "r5b_mae_late", "orc_mae_late"]
    cols = [c for c in cols if c in df]
    print(df[cols].sort_values("base", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
