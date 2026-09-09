#!/usr/bin/env python
"""Dunkelflaute-conditioned LORO error decomposition (FD-45).

Protocol (pre-registered before seeing any result):
    1. Re-run the FD-41 official pipeline (ep900, p_cold 0.3, monthly config,
       tau 1.1, seed 0) per target region, but DUMP per-window (n, HORIZON)
       predictions + truths + origin timestamps to npz.
    2. Label each forecast origin day with a *meteorological* Dunkelflaute
       index computed ONLY from the region's ERA5 weather (zero-telemetry
       reproducible): wind-lull + solar-dim compound stress.
    3. Bucket per-hour abs errors into event vs regular days; report
       bucketed MAE, event-day error inflation ratio, and a paired
       permutation test on daily MAE.

Usage:
    .venv-nemed/bin/python scripts/experiments/dunkelflaute_buckets.py \
        --regions QLD1 NSW1 --dump-only
    .venv-nemed/bin/python scripts/experiments/dunkelflaute_buckets.py --all
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from transcif.config import HORIZON, SEQ_LEN, TEST_STRIDE, TRAIN_FRACTION
from transcif.data.loaders import all_region_configs
from transcif.models.zeroshot.fuel import (
    train_fuel_zero_shot,
    predict_fuel_windows,
)

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
DUMP_DIR = RESULTS / "dunkelflaute"
DUMP_DIR.mkdir(parents=True, exist_ok=True)

# FD-22 official route table (see run_fuel_decomp_eval.ROUTE_TABLE)
ROUTE_TABLE = {"UK_01_North_Scotland": 1.1, "UK_02_South_Scotland": 1.1,
               "UK_16_Scotland": 1.1}


# ---------------------------------------------------------------------------
# 1. Per-window prediction dump (FD-41 official protocol, seed 0)
# ---------------------------------------------------------------------------
def _safe_prepare_fd_region(name, cfgs, **kw):
    """prepare_fd_region wrapper: AU regions (no fuel telemetry) return a
    None monthly_table, which crashes ``anchor_trust`` in the base module
    (pre-existing bug, documented in CN_DEPLOYMENT.md).  Mirror the demo's
    workaround: compute anchor_trust only when a table exists."""
    from transcif.models.zeroshot.fuel import anchor_trust, build_fd_config
    from transcif.data.loaders import load_region_data
    from transcif.data.fuel import (attach_fuel_and_exog,
                                    build_monthly_config_table)
    data = load_region_data(name, cfgs, data_dir=kw.get("data_dir"),
                            multi_year=kw.get("multi_year", False))
    attach_fuel_and_exog(data, name, cfgs, data_dir=kw.get("data_dir"),
                         use_au_state=kw.get("use_au_state", False))
    data["fd_config"] = build_fd_config(data, name)
    data["monthly_table"] = build_monthly_config_table(
        data, name, history_only=kw.get("monthly_history_only", False))
    if data["monthly_table"] is not None:
        data["anchor_trust"] = anchor_trust(data, data["monthly_table"])
    else:
        data["anchor_trust"] = 0.0
    import os
    mode = os.environ.get("ANCHOR_TRUST", "0")
    if mode in ("1", "2") and data["monthly_table"] is not None:
        t = data["anchor_trust"]
        blended = (data["fd_config"][None, :]
                   + t * (data["monthly_table"] - data["fd_config"][None, :]))
        if mode == "1":
            data["monthly_table"] = blended
        else:
            data["monthly_table_target"] = blended
    return data


def dump_region(target, fd_regions, device, seed=0, epochs=900):
    """Train on 28 sources, predict target test windows, dump npz."""
    t0 = time.time()
    model = train_fuel_zero_shot(
        fd_regions, target, seed=seed, epochs=epochs, device=device,
        p_cold=0.3, use_monthly=True, lag_months=1,
        wind_route_tau=1.1)
    data = fd_regions[target]
    tau = ROUTE_TABLE.get(target, 1.1)
    model.wind_route_tau = float(tau)

    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(split - SEQ_LEN, None)
    sliced = {**data,
              "rs": data["rs"][sl], "cif": data["cif"][sl],
              "fuel_shares": data["fuel_shares"][sl],
              "hours": data["hours"][sl],
              "exog": {k: v[sl] for k, v in data["exog"].items()}}
    from transcif.data.fuel import build_fd_windows
    w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                         stride=TEST_STRIDE,
                         monthly_table=(data.get("monthly_table_target",
                                                 data.get("monthly_table"))),
                         lag_months=1)
    n = len(w["x_rs"])
    if n == 0:
        return None
    y = w["y_cif"]
    ef_vec = data["ef_vec"].astype(np.float32)
    fd_cfg = data["fd_config"]

    # origin_hours is a pd.DatetimeIndex; store as epoch-hours (int64).
    # dtype may be us/s/ns depending on the source CSV — normalise via the
    # pandas Timestamp value (second precision is plenty for daily buckets).
    oidx = pd.DatetimeIndex(w["origin_hours"])
    origin_hours = (oidx.astype("datetime64[s]").astype(np.int64)
                    // 3600)
    out = {"origin_hours": origin_hours,
           "y_true": y.astype(np.float32)}

    for tier, cold in (("i_cfg", True), ("i0", False)):
        cif, _, _ = predict_fuel_windows(model, w, fd_cfg, ef_vec,
                                         cold=cold, device=device)
        out[f"pred_{tier}"] = cif.astype(np.float32)

    # persistence and config-constant baselines for reference
    cif_off = data["cif"][split - SEQ_LEN:]
    persist = np.stack([
        cif_off[s + SEQ_LEN - HORIZON:s + SEQ_LEN]
        for s in range(0, len(cif_off) - SEQ_LEN - HORIZON + 1, TEST_STRIDE)
    ])[:n]
    out["pred_persistence"] = persist.astype(np.float32)
    cfg_const = np.full_like(y, data["mean_rs"] * data["ef_r"]
                             + (1 - data["mean_rs"]) * data["ef_nr"])
    out["pred_config_constant"] = cfg_const.astype(np.float32)

    # Daily weather stats on the region's OWN timeline (data["hours"] —
    # local for AU NEM, UTC for US/UK — same axis the model windows live
    # on). exog["weather"] is already timestamp-joined, so the event labels
    # are aligned with the prediction origins by construction.
    wx = data["exog"]["weather"]
    hrs = pd.DatetimeIndex(data["hours"])
    has_real_weather = bool(np.std(wx[:, 2]) > 1e-6)
    if has_real_weather:
        wdf = pd.DataFrame({"w100": wx[:, 2], "swr": wx[:, 1]},
                           index=hrs)
        daily_wx = wdf.groupby(wdf.index.normalize()).agg(
            w_mean=("w100", "mean"), swr_mean=("swr", "mean"))
        out["wx_day_ord"] = np.array(
            [d.date().toordinal() for d in daily_wx.index], dtype=np.int64)
        out["wx_w_mean"] = daily_wx["w_mean"].values.astype(np.float32)
        out["wx_swr_mean"] = daily_wx["swr_mean"].values.astype(np.float32)
    else:
        out["wx_day_ord"] = np.array([], dtype=np.int64)

    path = DUMP_DIR / f"{target}_seed{seed}.npz"
    np.savez_compressed(path, **out)
    print(f"[dump] {target}: n={n} windows, weather={'ok' if has_real_weather else 'MISSING'} "
          f"-> {path.name} ({time.time()-t0:.0f}s)")
    return path


# ---------------------------------------------------------------------------
# 2. Meteorological Dunkelflaute labels live in the npz dump (wx_* arrays):
#    daily mean 100 m wind and daily mean shortwave on the region's own
#    timeline — the SAME joined arrays the model consumes, so labels and
#    prediction origins are aligned by construction.  Thresholds are
#    per-region annual 20th percentiles; a day is Dunkelflaute when the
#    wind-lull AND solar-dim conditions co-occur (compound event).  Dark
#    winter regions (mean SWR <= 1 W/m2 on >50% of days) fall back to a
#    wind-only definition.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 3. Bucketed evaluation lives inline in main() — per-window npz dumps are
#    joined with the daily Dunkelflaute index there.
# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--dump-only", action="store_true")
    ap.add_argument("--analyze", action="store_true",
                    help="skip training; analyze existing dumps")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import torch
    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[fd45] device={device}")

    cfgs = all_region_configs()
    data_dir = Path(__file__).resolve().parent.parent.parent / "data_2023"

    if not args.analyze:
        pool_names = list(cfgs)
        targets = args.regions or pool_names
        print(f"[fd45] preparing {len(pool_names)} FD regions ...")
        fd_regions = {}
        for name in pool_names:
            fd_regions[name] = _safe_prepare_fd_region(name, cfgs)
        print(f"[fd45] {len(fd_regions)} regions ready")
        for tgt in targets:
            for seed in args.seeds:
                try:
                    dump_region(tgt, fd_regions, device, seed=seed,
                                epochs=args.epochs)
                except Exception as e:  # noqa: BLE001
                    print(f"[fd45][ERR] {tgt} seed {seed}: {e}")

    if args.dump_only:
        print("[fd45] dump-only mode: analysis skipped")
        return

    # ---- analysis pass --------------------------------------------------
    rows = []
    for tgt in (args.regions or list(cfgs)):
        path = DUMP_DIR / f"{tgt}_seed0.npz"
        if not path.exists():
            continue
        d = np.load(path)
        if len(d["wx_day_ord"]) == 0:
            rows.append({"target": tgt, "status": "weather_missing"})
            continue
        origin_ts = pd.to_datetime(d["origin_hours"] * 3600, unit="s")
        y = d["y_true"]
        # per-region annual climatology thresholds (20th percentile)
        w_thr = np.quantile(d["wx_w_mean"], 0.2)
        s_thr = np.quantile(d["wx_swr_mean"], 0.2)
        solar_meaningful = (d["wx_swr_mean"] > 1.0).mean() > 0.5
        wx_days = pd.to_datetime(
            d["wx_day_ord"] - 719163, unit="D")  # ordinal -> date
        wx_df = pd.DataFrame(
            {"w_mean": d["wx_w_mean"], "swr_mean": d["wx_swr_mean"]},
            index=wx_days)
        wx_df["wind_lull"] = wx_df["w_mean"] < w_thr
        if solar_meaningful:
            wx_df["solar_dim"] = wx_df["swr_mean"] < s_thr
        else:
            wx_df["solar_dim"] = False  # dark-winter regions: wind-only
        wx_df["dunkelflaute"] = wx_df["wind_lull"] & wx_df["solar_dim"]
        for key in ("i_cfg", "i0", "persistence", "config_constant"):
            pk = f"pred_{key}"
            if pk not in d:
                continue
            err = np.abs(d[pk] - y).mean(axis=1)
            df_err = pd.DataFrame(
                {"day": origin_ts.normalize(), "mae": err})
            daily_mae = df_err.groupby("day")["mae"].mean()
            joined = daily_mae.to_frame("mae").join(
                wx_df[["wind_lull", "solar_dim", "dunkelflaute"]],
                how="left")
            joined = joined.dropna(subset=["dunkelflaute"])
            for bucket_name in ("dunkelflaute", "wind_lull", "solar_dim"):
                evt = joined[joined[bucket_name]]
                reg = joined[~joined[bucket_name]]
                if len(evt) == 0 or len(reg) == 0:
                    rows.append({"target": tgt, "method": key,
                                 "bucket": bucket_name,
                                 "status": "no_event_days"})
                    continue
                evt_mae = evt["mae"].mean()
                reg_mae = reg["mae"].mean()
                # paired permutation test on daily MAE (10k shuffles)
                rng = np.random.default_rng(0)
                obs = evt_mae - reg_mae
                vals = joined["mae"].values
                labels = joined[bucket_name].values
                n_evt = int(labels.sum())
                count = 0
                B = 10000
                for _ in range(B):
                    perm = rng.permutation(vals)
                    pe = perm[:n_evt].mean()
                    pr = perm[n_evt:].mean()
                    if abs(pe - pr) >= abs(obs):
                        count += 1
                pval = (count + 1) / (B + 1)
                rows.append({
                    "target": tgt, "method": key, "bucket": bucket_name,
                    "n_event_days": n_evt,
                    "n_regular_days": int(len(vals) - n_evt),
                    "mae_event": round(float(evt_mae), 2),
                    "mae_regular": round(float(reg_mae), 2),
                    "ratio": round(float(evt_mae / reg_mae), 3),
                    "p_perm": round(pval, 4),
                })
    out = RESULTS / "dunkelflaute_buckets.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=1)
    print(f"[fd45] -> {out}")

    # ---- markdown summary -------------------------------------------------
    df_rows = pd.DataFrame([r for r in rows if "method" in r])
    if len(df_rows):
        md = ["# FD-45 Dunkelflaute 分桶评估（29 区, seed 0, ep900 官方口径）", "",
              "事件定义（零遥测可复现，纯 ERA5）：",
              "- wind_lull：日均 100m 风速 < 该区全年 20 分位",
              "- solar_dim：日均短波 < 该区全年 20 分位",
              "- dunkelflaute：两者同时成立（复合事件）；暗冬区退化为 wind-only",
              "", "## I_cfg 层分桶（compound 口径）", ""]
        ok = df_rows[(df_rows["method"] == "i_cfg")
                     & (df_rows["bucket"] == "dunkelflaute")
                     & df_rows["mae_event"].notna()]
        cols = ["target", "n_event_days", "mae_event", "mae_regular",
                "ratio", "p_perm"]
        lines = ["| " + " | ".join(cols) + " |",
                 "|" + "|".join(["---"] * len(cols)) + "|"]
        for _, r in ok[cols].iterrows():
            lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
        md.append("\n".join(lines))
        noev = df_rows[(df_rows["method"] == "i_cfg")
                       & (df_rows["bucket"] == "dunkelflaute")
                       & df_rows["mae_event"].isna()]
        if len(noev):
            md += ["", "复合事件为 0 的区域：" + ", ".join(noev["target"])]
        md += ["", "## 全部明细（含 i0/persistence/config_constant 与 "
                "wind_lull/solar_dim 口径）见 dunkelflaute_buckets.json", ""]
        md_path = RESULTS / "dunkelflaute_buckets.md"
        md_path.write_text("\n".join(str(x) for x in md))
        print(f"[fd45] -> {md_path}")
        # console summary: i_cfg compound + wind_lull buckets only (readable)
        show = df_rows[(df_rows["method"] == "i_cfg")
                       & (df_rows["bucket"].isin(["dunkelflaute", "wind_lull"]))]
        print(show.to_string(index=False))


if __name__ == "__main__":
    main()
