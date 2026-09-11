#!/usr/bin/env python3
"""FD-50b: deployable amplitude router validation on FD-45 dump assets.

Question: how much of the amplitude lever (FD-50 oracle: pooled -2.0,
LOO-median: -1.05) survives WITHOUT touching target labels?

Routers tested (all zero-telemetry legal, target labels never used):
  R0  a=1.0 (no-op baseline)
  R1  global median of donors' a*                (label-free for target)
  R2  family LOO median (AU/UK/US market family) (label-free for target)
  R3  split-half self-fit: a* on early half of the target's OWN test
      windows, applied to late half  — uses only the target's own
      PREDICTIONS vs nothing? NO: a* needs truth. -> replaced by:
  R3' split-half of R1/R2 (temporal robustness of the router itself)
  R4  config-router: fit a* ~ linear(mean_rs, wind_share, ef_nr) on
      28 donors, predict target's a  (label-free for target)

Also: per-region temporal split-half self-fit using the target's own
EARLY test truth (deployment-legal only if online labels exist; reported
separately as "online-calibrated upper reference").

Output: results/fd50b_router.json + fd50b_router.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DUMP = ROOT / "results" / "dunkelflaute"
RESULTS = ROOT / "results"
FD48 = RESULTS / "fd48_repro_ep900_full.json"


def family_of(t: str) -> str:
    if t in ("VIC1", "NSW1", "SA1", "QLD1"):
        return "AU"
    if t.startswith("UK"):
        return "UK"
    return "US"


def amp_transform(pred: np.ndarray, a: float) -> np.ndarray:
    pm = pred.mean(axis=1, keepdims=True)
    return pm + a * (pred - pm)


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


def a_star(pred: np.ndarray, y: np.ndarray) -> float:
    best_a, best_m = 1.0, mae(pred, y)
    for a in np.arange(0.40, 2.21, 0.02):
        m = mae(amp_transform(pred, a), y)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a


def main() -> None:
    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))
    data = {}
    for t in targets:
        d = np.load(DUMP / f"{t}_seed0.npz")
        data[t] = {
            "y": d["y_true"].astype(np.float64),
            "pred": d["pred_i_cfg"].astype(np.float64),
            "origin": pd.to_datetime(d["origin_hours"] * 3600, unit="s"),
        }

    # per-region a* on full test (cheating reference) and on halves
    info = {}
    for t, dd in data.items():
        y, p, o = dd["y"], dd["pred"], dd["origin"]
        o_sorted = o.argsort()
        cut = len(o_sorted) // 2
        mA = np.zeros(len(o), dtype=bool)
        mA[o_sorted[:cut]] = True
        mB = ~mA
        info[t] = {
            "a_full": a_star(p, y),
            "a_early": a_star(p[mA], y[mA]),
            "a_late": a_star(p[mB], y[mB]),
            "base_full": mae(p, y),
            "n": len(y),
        }

    # ---- routers ----------------------------------------------------------
    rows = []
    a_full = {t: info[t]["a_full"] for t in targets}
    a_early = {t: info[t]["a_early"] for t in targets}

    # config features for R4 — from fd50 json (mean_rs etc. not stored there;
    # reuse config from src if importable, else skip R4)
    feats = None
    try:
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from transcif.data.fuel import all_region_configs  # type: ignore
        cfgs = all_region_configs()
        feats = {}
        for t in targets:
            c = cfgs[t]
            feats[t] = [
                float(np.asarray(c.get("mean_rs", 0.5))),
                float(np.asarray(c.get("wind_share", 0.0))),
                float(np.asarray(c.get("ef_nr", 500.0)) / 1000.0),
            ]
    except Exception as e:  # noqa: BLE001
        print(f"[fd50b] config router unavailable: {e}")

    for t in targets:
        y, p = data[t]["y"], data[t]["pred"]
        base = info[t]["base_full"]

        # R1 global LOO (leave-one-out median of full-test a*)
        r1 = float(np.median([a_full[o] for o in targets if o != t]))
        # R2 family LOO
        fam = [o for o in targets
               if family_of(o) == family_of(t) and o != t]
        r2 = float(np.median([a_full[o] for o in fam])) if fam else 1.0
        # R2b family LOO from EARLY halves only (deployable if early
        # online labels of donors exist; still no target labels)
        fam_early = [a_early[o] for o in fam]
        r2b = float(np.median(fam_early)) if fam_early else 1.0

        row = {
            "target": t, "family": family_of(t), "base": round(base, 2),
            "a_full": round(info[t]["a_full"], 2),
            "a_early": round(info[t]["a_early"], 2),
            "a_late": round(info[t]["a_late"], 2),
            "r1_a": round(r1, 2), "r2_a": round(r2, 2),
            "r2b_a": round(r2b, 2),
            "r1_mae": round(mae(amp_transform(p, r1), y), 2),
            "r2_mae": round(mae(amp_transform(p, r2), y), 2),
            "r2b_mae": round(mae(amp_transform(p, r2b), y), 2),
        }
        if feats is not None:
            # R4: linear fit of a* on donors' features, LOO predict
            X = np.array([feats[o] for o in targets if o != t])
            Y = np.array([a_full[o] for o in targets if o != t])
            try:
                Xb = np.hstack([X, np.ones((len(X), 1))])
                beta, *_ = np.linalg.lstsq(Xb, Y, rcond=None)
                x = np.array(feats[t] + [1.0])
                r4 = float(np.clip(x @ beta, 0.4, 2.2))
                row["r4_a"] = round(r4, 2)
                row["r4_mae"] = round(mae(amp_transform(p, r4), y), 2)
            except Exception:  # noqa: BLE001
                pass
        rows.append(row)

    df = pd.DataFrame(rows)
    # temporal robustness: apply routers fitted on EARLY to LATE half
    late_rows = []
    for t in targets:
        dd = data[t]
        o = dd["origin"]
        o_sorted = o.argsort()
        cut = len(o_sorted) // 2
        mB = np.zeros(len(o), dtype=bool)
        mB[o_sorted[cut:]] = True
        pB, yB = dd["pred"][mB], dd["y"][mB]
        a_late_true = info[t]["a_late"]
        # R2b uses early-half donor a*: apply its value to late half
        fam = [x for x in targets
               if family_of(x) == family_of(t) and x != t]
        r2b_early = float(np.median([info[x]["a_early"] for x in fam]))
        late_rows.append({
            "target": t,
            "base_late": round(mae(pB, yB), 2),
            "a_late_true": round(a_late_true, 2),
            "r2b_late_mae": round(mae(amp_transform(pB, r2b_early), yB), 2),
        })
    dfl = pd.DataFrame(late_rows)

    out = RESULTS / "fd50b_router.json"
    out.write_text(json.dumps(
        {"rows": rows, "late": late_rows}, indent=1))

    md = ["# FD-50b 幅度路由器部署化验证（零目标标签）", "",
          "R1=全局 LOO 中位 a*；R2=市场家族 LOO 中位；"
          "R2b=家族 early 半段 a* 中位（部署形态）；"
          "R4=config 线性路由（mean_rs/wind_share/ef_nr，LOO 预测）", "",
          "## 汇总（29 区 pooled 均值 MAE）", ""]
    base_p = df["base"].mean()
    md.append(f"- 基线：{base_p:.2f}")
    for col, name in [("r1_mae", "R1 全局 LOO"), ("r2_mae", "R2 家族 LOO"),
                      ("r2b_mae", "R2b 家族 early"), ("r4_mae", "R4 config")]:
        if col in df:
            v = df[col].mean()
            md.append(f"- {name}：{v:.2f} ({v - base_p:+.2f})")
    # cheating references
    cheat = np.mean([mae(amp_transform(data[t]["pred"], info[t]["a_full"]),
                         data[t]["y"]) for t in targets])
    md.append(f"- [作弊参照] 各区自身 a*：{cheat:.2f} ({cheat - base_p:+.2f})")
    bl = dfl["base_late"].mean()
    rl = dfl["r2b_late_mae"].mean()
    md += ["", "## 时序外推（early 半段拟合路由 → late 半段应用）", "",
           f"- late 半段基线 pooled：{bl:.2f}",
           f"- R2b（家族 early a*）late pooled：{rl:.2f} ({rl - bl:+.2f})"]

    md += ["", "## 右尾区明细", "",
           "| target | base | a* | R1 | R2 | R2b | R4 | a_early→a_late |",
           "|---|---|---|---|---|---|---|---|"]
    for _, r in df[df["base"] > 40].sort_values("base", ascending=False).iterrows():
        md.append(
            f"| {r['target']} | {r['base']:.1f} | {r['a_full']:.2f} "
            f"| {r['r1_mae']:.1f} | {r['r2_mae']:.1f} | {r['r2b_mae']:.1f} "
            f"| {r.get('r4_mae', float('nan')):.1f} "
            f"| {r['a_early']:.2f}→{r['a_late']:.2f} |")
    md_path = RESULTS / "fd50b_router.md"
    md_path.write_text("\n".join(md))
    print(f"[fd50b] -> {md_path}")
    print(df[["target", "base", "a_full", "r1_mae", "r2_mae", "r2b_mae",
              "r4_mae" if "r4_mae" in df else "r1_a"]]
          .sort_values("base", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
