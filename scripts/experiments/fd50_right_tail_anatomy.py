#!/usr/bin/env python3
"""FD-50: right-tail region anatomy (MAE>40) on FD-45 dump assets.

Post-hoc analysis on results/dunkelflaute/*_seed0.npz (restored-asset rerun,
2026-09-10, protocol-identical to FD-41/FD-48 official). No training.

Per region, for I_cfg (zero-telemetry tier) and reference tiers:
  1. base MAE (sanity vs fd48_repro_ep900_full.json)
  2. oracle-level floor: per-window mean swapped to truth window mean
     -> how much of MAE is LEVEL (day-level) error
  3. shape-amplitude oracle: per-window own-mean centered, 1-D search of
     scale a minimizing MAE  (pred' = pbar_w + a*(pred - pbar_w))
     -> upper bound of the under-dispersion lever (FD-35: ratios 0.35-0.74)
  4. LOO-deployable amplitude: a = median of other regions' oracle a
     -> zero-telemetry legal (donor labels only), leave-one-out
  5. top-10% window error concentration
  6. monthly MAE slice (Q4 drift check)
  7. event-day vs regular MAE (FD-45 ERA5 labels, per-region 20pct)
  8. I_cfg - I_0 gap (value of fuel telemetry)
  9. bias + dispersion ratios (global std ratio, within-window shape std ratio)

Output: results/fd50_right_tail.json + results/fd50_right_tail.md
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

TIER = "i_cfg"          # primary anatomy tier
REF_TIERS = ["i0", "persistence"]
RIGHT_TAIL_THRESH = 40.0


def load_region(target: str) -> dict | None:
    p = DUMP / f"{target}_seed0.npz"
    if not p.exists():
        return None
    return {k: d for k, d in np.load(p).items()}


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.abs(pred - y).mean())


def oracle_level(pred: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Replace each window's mean with the truth window mean."""
    pm = pred.mean(axis=1, keepdims=True)
    ym = y.mean(axis=1, keepdims=True)
    return pred - pm + ym


def amp_transform(pred: np.ndarray, a: float) -> np.ndarray:
    pm = pred.mean(axis=1, keepdims=True)
    return pm + a * (pred - pm)


def oracle_amp(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """1-D golden-ish grid search of amplitude a in [0.4, 2.2]."""
    best_a, best_m = 1.0, mae(pred, y)
    for a in np.arange(0.40, 2.21, 0.02):
        m = mae(amp_transform(pred, a), y)
        if m < best_m:
            best_a, best_m = float(a), m
    return best_a, best_m


def event_labels(d: dict) -> pd.Series | None:
    if len(d["wx_day_ord"]) == 0:
        return None
    w_thr = np.quantile(d["wx_w_mean"], 0.2)
    s_thr = np.quantile(d["wx_swr_mean"], 0.2)
    solar_meaningful = (d["wx_swr_mean"] > 1.0).mean() > 0.5
    wx_days = pd.to_datetime(d["wx_day_ord"] - 719163, unit="D")
    df = pd.DataFrame({"w": d["wx_w_mean"], "s": d["wx_swr_mean"]},
                      index=wx_days)
    lab = pd.Series(False, index=wx_days, name="wind_lull")
    lab[df["w"] < w_thr] = True
    return lab


def main() -> None:
    fd48 = {}
    if FD48.exists():
        fd48 = {r["target"]: r["mae_cfg"] for r in json.loads(FD48.read_text())}

    targets = sorted(p.name[:-len("_seed0.npz")]
                     for p in DUMP.glob("*_seed0.npz"))
    print(f"[fd50] {len(targets)} region dumps found")

    rows = []
    for tgt in targets:
        d = load_region(tgt)
        if d is None or "pred_i_cfg" not in d:
            continue
        y = d["y_true"].astype(np.float64)
        pred = d[f"pred_{TIER}"].astype(np.float64)
        origin_ts = pd.to_datetime(d["origin_hours"] * 3600, unit="s")

        base = mae(pred, y)
        lvl = mae(oracle_level(pred, y), y)
        a_star, amp_m = oracle_amp(pred, y)
        # combined oracle: level-perfect then amplitude-optimal
        lvl_pred = oracle_level(pred, y)
        a2, comb_m = oracle_amp(lvl_pred, y)

        # top-k concentration (per-window MAE)
        err_w = np.abs(pred - y).mean(axis=1)
        k = max(1, int(np.ceil(0.10 * len(err_w))))
        top_share = float(np.sort(err_w)[-k:].sum() / err_w.sum())

        # monthly slice
        mon = pd.Series(err_w, index=origin_ts).groupby(
            origin_ts.month).mean()
        mon_out = {int(m): round(float(v), 1) for m, v in mon.items()}

        # event-day slice (wind_lull, FD-45 convention)
        lab = event_labels(d)
        evt_mae = reg_mae = None
        n_evt = 0
        if lab is not None:
            daily = pd.Series(err_w, index=origin_ts.normalize())
            daily = daily.groupby(level=0).mean()
            joined = daily.to_frame("mae").join(lab, how="inner")
            evt = joined[joined["wind_lull"]]
            reg = joined[~joined["wind_lull"]]
            if len(evt) and len(reg):
                evt_mae = float(evt["mae"].mean())
                reg_mae = float(reg["mae"].mean())
                n_evt = int(len(evt))

        # I_0 gap + reference tiers
        i0 = float(np.abs(d["pred_i0"].astype(np.float64) - y).mean()) \
            if "pred_i0" in d else None
        persist = float(np.abs(d["pred_persistence"].astype(np.float64)
                               - y).mean()) if "pred_persistence" in d else None

        # bias & dispersion
        bias = float(pred.mean() - y.mean())
        disp_global = float(pred.std() / y.std())
        shape_ratio = float(np.mean(pred.std(axis=1) / y.std(axis=1)))

        rows.append({
            "target": tgt,
            "mae_cfg_dump": round(base, 2),
            "mae_cfg_fd48": round(fd48.get(tgt, float("nan")), 2),
            "oracle_level_mae": round(lvl, 2),
            "level_share_pct": round(100 * (1 - lvl / base), 1),
            "amp_star": round(a_star, 2),
            "amp_oracle_mae": round(amp_m, 2),
            "amp_gain_pct": round(100 * (1 - amp_m / base), 1),
            "comb_oracle_mae": round(comb_m, 2),
            "top10_share_pct": round(100 * top_share, 1),
            "monthly": mon_out,
            "n_wind_lull_days": n_evt,
            "mae_event": round(evt_mae, 2) if evt_mae else None,
            "mae_regular": round(reg_mae, 2) if reg_mae else None,
            "mae_i0": round(i0, 2) if i0 else None,
            "i0_gap": round(base - i0, 2) if i0 else None,
            "persistence": round(persist, 2) if persist else None,
            "bias": round(bias, 2),
            "disp_global": round(disp_global, 2),
            "disp_shape": round(shape_ratio, 2),
            "right_tail": base > RIGHT_TAIL_THRESH,
        })

    # LOO-deployable amplitude: median of other regions' a_star
    a_stars = np.array([r["amp_star"] for r in rows])
    for i, r in enumerate(rows):
        loo_a = float(np.median(np.delete(a_stars, i)))
        d = load_region(r["target"])
        y = d["y_true"].astype(np.float64)
        pred = d["pred_i_cfg"].astype(np.float64)
        r["amp_loo"] = round(loo_a, 2)
        r["amp_loo_mae"] = round(mae(amp_transform(pred, loo_a), y), 2)
        r["amp_loo_delta"] = round(r["amp_loo_mae"] - r["mae_cfg_dump"], 2)

    out = RESULTS / "fd50_right_tail.json"
    out.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"[fd50] -> {out}")

    # ---- markdown ----------------------------------------------------------
    df = pd.DataFrame(rows).sort_values("mae_cfg_dump", ascending=False)
    md = ["# FD-50 右尾区解剖（FD-45 恢复资产 dump，seed 0，ep900 官方口径）", "",
          "分解口径：",
          "- oracle_level：每窗均值换真值窗均值后的 MAE（水平完美地板，纯诊断）",
          "- amp_star：窗内幅度最优缩放 a（pred' = pbar_w + a·(pred−pbar_w)，1-D 搜索，标签作弊上界）",
          "- amp_loo：其余 28 区 amp_star 的中位数（donor 标签合法的部署版）",
          "- top10：最差 10% 窗口的绝对误差占比",
          "- disp_shape：窗内形状 std 比 std_h(pred)/std_h(y)（FD-35 欠离散口径）", "",
          "## 右尾区（MAE>40）按 dump 口径降序", ""]
    cols = ["target", "mae_cfg_dump", "oracle_level_mae", "level_share_pct",
            "amp_star", "amp_loo", "amp_loo_mae", "amp_loo_delta",
            "disp_shape", "top10_share_pct", "mae_event", "mae_regular",
            "mae_i0"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df[df["right_tail"]].iterrows():
        lines.append("| " + " | ".join(
            (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
            for c in cols) + " |")
    md += ["\n".join(lines), "", "## 非右尾区（对照）", ""]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df[~df["right_tail"]].iterrows():
        lines.append("| " + " | ".join(
            (f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]))
            for c in cols) + " |")
    md += ["\n".join(lines), ""]

    rt = df[df["right_tail"]]
    pooled_before = df["mae_cfg_dump"].mean()
    pooled_after_loo = df["amp_loo_mae"].mean()
    pooled_after_loo_rt = (pd.concat([
        df[~df["right_tail"]]["mae_cfg_dump"],
        rt["amp_loo_mae"]]).mean())
    md += [
        "## 汇总", "",
        f"- pooled 均值（29 区，dump 口径）：{pooled_before:.2f}",
        f"- amp_loo 全区套用后 pooled：{pooled_after_loo:.2f} "
        f"({pooled_after_loo - pooled_before:+.2f})",
        f"- amp_loo 仅右尾区套用后 pooled：{pooled_after_loo_rt:.2f} "
        f"({pooled_after_loo_rt - pooled_before:+.2f})",
        f"- 右尾区数（>40）：{len(rt)}/29",
        f-"" if False else "",
        f"- amp_loo 右尾区逐区 delta："
        + ", ".join(f"{r['target'].split('_')[0]} {r['amp_loo_delta']:+.1f}"
                    for _, r in rt.iterrows()),
    ]
    md_path = RESULTS / "fd50_right_tail.md"
    md_path.write_text("\n".join(str(x) for x in md))
    print(f"[fd50] -> {md_path}")

    show = df[["target", "mae_cfg_dump", "mae_cfg_fd48", "oracle_level_mae",
               "amp_star", "amp_loo_mae", "amp_loo_delta", "disp_shape"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
