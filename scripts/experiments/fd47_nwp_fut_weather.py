#!/usr/bin/env python
"""FD-47: deployment-real fut_weather — operational NWP vs ERA5 proxy.

Question (pre-registered before any run): the FD stack feeds the 24 h
horizon with an ERA5 *reanalysis proxy* for day-ahead weather.  A
deployer would instead receive an operational NWP forecast.  How much
does the proxy optimism cost — i.e. what does the FD-41/45 headline MAE
look like when the horizon sees the weather a real deployment gets?

Design (mirrors the FD-41/45 official protocol exactly):
    1. Same training: train_fuel_zero_shot(28 sources, target, seed 0,
       ep900, p_cold 0.3, monthly config lag 1, tau 1.1 + ROUTE_TABLE).
       The model is IDENTICAL across arms — only the TARGET's
       inference-time future weather changes.
    2. Same test windows (official slice, TEST_STRIDE, monthly table).
    3. Arms (target test windows only; the 336 h history stays ERA5 in
       every arm — a deployer's past is observed, not forecast):
         era5      FD-41 status quo (the baseline arm)
         gfs       future weather = gfs_seamless operational forecast
         icon      future = icon_seamless
         ensemble  future = (gfs + icon) / 2
    4. Channel layout (see attach_fuel_and_exog / build_fd_windows):
         exog["weather"] is (T, 10): [temp, swr, w100, wind_cf, csi,
             gust, pres, demand_z, regime24, tend6]
         fut_exog cols: 0-1 astro, 2 wind_cf, 3 csi, 4-9 calendar,
             10 hdh, 11 cdh, 12 coal, 13 gas, 14 dem, 15 regime24,
             16 tend6, 17 inter (+2 env-gated nwp_spread).
       Arm patching overwrites the weather-derived future channels:
         fut_weather cols [0,1,2,3,4,8,9]  (temp/swr/w100/wind_cf/csi
             + composite regime24/tend6)
         fut_exog    cols [2,3,10,11,15,16]
       Gust/pressure (fut_weather 5-6) keep the ERA5 proxy — the
       Open-Meteo historical-forecast archive has no gust/MSLP columns;
       documented approximation, secondary features.  Demand/
       interchange/astro/calendar/prices are identical across arms by
       construction.
       regime24/tend6 are recomputed on a COMPOSITE wind-CF axis: ERA5
       for t < split (history unchanged), arm wind CF for t >= split —
       so trailing means at horizon hours blend observed past with
       forecast future, exactly as at deployment.
    5. Positional alignment: window i's horizon is full-axis positions
       [split + s_i, split + s_i + HORIZON) — no timestamp round-trips,
       so the AU DST duplicated-hour axis cannot misalign.
    6. Regions WITHOUT their own ERA5 (has_real_weather False, zeros
       today) get the NWP arms as a *fallback upgrade* — a separate
       family in the analysis (public operational forecasts as the
       zero-telemetry weather source when no reanalysis infrastructure
       exists).  The has_real_weather flag separates the families.

NWP join convention: valid-time UTC grid reindexed onto the region's
join axis (hours_utc for AU with the same month-based DST correction as
attach_fuel_and_exog; hours as-is for US/UK) — the same axis the ERA5
join used, so arm-vs-baseline differences are attributable to forecast
skill alone.

Usage:
    env PYTHONPATH=src .venv-nemed/bin/python \
        scripts/experiments/fd47_nwp_fut_weather.py --regions NSW1
    env PYTHONPATH=src .venv-nemed/bin/python \
        scripts/experiments/fd47_nwp_fut_weather.py --all
    env PYTHONPATH=src .venv-nemed/bin/python \
        scripts/experiments/fd47_nwp_fut_weather.py --analyze
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
from transcif.data.fuel import (
    build_fd_windows,
    build_monthly_config_table,
    wind_capacity_factor,
)
from transcif.models.zeroshot.fuel import (
    anchor_trust,
    build_fd_config,
    train_fuel_zero_shot,
    predict_fuel_windows,
)

RESULTS = Path(__file__).resolve().parent.parent.parent / "results"
DUMP_DIR = RESULTS / "fd47_nwp"
DUMP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data_2023"
ROUTE_TABLE = {"UK_01_North_Scotland": 1.1, "UK_02_South_Scotland": 1.1,
               "UK_16_Scotland": 1.1}

ARMS = ("era5", "gfs", "icon", "ensemble")
TIERS = ("icfg", "i0")

# exog["weather"] column map (attach_fuel_and_exog)
WX_COLS = {"temp": 0, "swr": 1, "w100": 2, "wind_cf": 3, "csi": 4,
           "gust": 5, "pres": 6, "demand_z": 7, "regime24": 8, "tend6": 9}
# fut_exog weather-derived columns
FE_WIND_CF, FE_CSI, FE_HDH, FE_CDH, FE_REGIME, FE_TEND = 2, 3, 10, 11, 15, 16


# ---------------------------------------------------------------------------
# Region prep — FD-45 safe wrapper (AU anchor_trust None-table bug)
# ---------------------------------------------------------------------------
def _safe_prepare_fd_region(name, cfgs, data_dir):
    from transcif.data.loaders import load_region_data
    from transcif.data.fuel import attach_fuel_and_exog
    data = load_region_data(name, cfgs, data_dir=data_dir)
    attach_fuel_and_exog(data, name, cfgs, data_dir=data_dir)
    data["fd_config"] = build_fd_config(data, name)
    data["monthly_table"] = build_monthly_config_table(data, name)
    if data["monthly_table"] is not None:
        data["anchor_trust"] = anchor_trust(data, data["monthly_table"])
    else:
        data["anchor_trust"] = 0.0
    return data


# ---------------------------------------------------------------------------
# Join axis (replicates attach_fuel_and_exog's AU timeline conversion)
# ---------------------------------------------------------------------------
def region_utc_axis(data, region_name):
    from transcif.config.region_meta import get_region_meta
    from transcif.data.fuel import jurisdiction_of
    hours = pd.DatetimeIndex(data["hours"])
    lat, lon, tz = get_region_meta(region_name)
    if jurisdiction_of(region_name) == "au":
        dst_regions = {"NSW1", "VIC1", "SA1"}
        in_dst = np.isin(np.asarray(hours.month), [10, 11, 12, 1, 2, 3])
        off = tz + (in_dst.astype(float)
                    if region_name in dst_regions else 0.0)
        hours_utc = hours - pd.to_timedelta(off, unit="h")
    else:
        hours_utc = hours
    return hours_utc


# ---------------------------------------------------------------------------
# NWP archive -> T-axis arm arrays
# ---------------------------------------------------------------------------
def load_nwp_table(region_name, all_configs, data_dir):
    info = all_configs.get(region_name)
    if info is None:
        return None
    stem = info["file"].replace("_2023_hourly.csv", "")
    paths = sorted((data_dir / "nwp").glob(f"{stem}_nwp_*_hourly.csv"))
    if not paths:
        return None
    frames = [pd.read_csv(p, parse_dates=["hour"]) for p in paths]
    df = (pd.concat(frames, ignore_index=True)
          .drop_duplicates("hour", keep="last")
          .sort_values("hour").reset_index(drop=True))
    need = {"gfs_wind100", "gfs_temp", "gfs_swr",
            "icon_wind100", "icon_temp", "icon_swr"}
    if not need.issubset(df.columns):
        return None
    return df


def build_arm_arrays(df_nwp, model, hours_utc, clearsky):
    """Assemble T-axis arrays for one arm.

    model in {"gfs", "icon", "ensemble"}.  Returns dict with weather
    (T, 3) [temp, swr, wind100], wind_cf, csi, hdh, cdh and the join
    coverage fraction (non-missing fraction on the T axis)."""
    if model == "ensemble":
        temp = df_nwp[["gfs_temp", "icon_temp"]].mean(axis=1)
        swr = df_nwp[["gfs_swr", "icon_swr"]].mean(axis=1)
        w100 = df_nwp[["gfs_wind100", "icon_wind100"]].mean(axis=1)
    else:
        temp = df_nwp[f"{model}_temp"]
        swr = df_nwp[f"{model}_swr"]
        w100 = df_nwp[f"{model}_wind100"]
    j = pd.DataFrame({"temp": temp.values, "swr": swr.values,
                      "w100": w100.values},
                     index=pd.DatetimeIndex(df_nwp["hour"]))
    j = j[~j.index.duplicated(keep="first")].reindex(hours_utc)
    coverage = float(j.notna().all(axis=1).mean())
    arr = np.nan_to_num(j.values, nan=0.0).astype(np.float32)
    wind_cf = wind_capacity_factor(arr[:, 2]).astype(np.float32)
    csi = np.clip(arr[:, 1] / clearsky, 0.0, 1.3).astype(np.float32)
    hdh = np.clip(15.5 - arr[:, 0], 0.0, None).astype(np.float32)
    cdh = np.clip(arr[:, 0] - 22.0, 0.0, None).astype(np.float32)
    return {"weather": arr, "wind_cf": wind_cf, "csi": csi,
            "hdh": hdh, "cdh": cdh, "coverage": coverage}


def composite_regime(era5_wind_cf, arm_wind_cf, split):
    """regime24/tend6 on the composite axis: ERA5 before split, arm CF
    from split on.  History hours therefore stay bit-identical to the
    baseline; horizon hours blend observed past with forecast future."""
    cf = np.asarray(era5_wind_cf, dtype=np.float32).copy()
    cf[split:] = np.asarray(arm_wind_cf, dtype=np.float32)[split:]
    regime24 = (pd.Series(cf).rolling(24, min_periods=1).mean()
                .values.astype(np.float32))
    tend6 = np.zeros_like(regime24)
    tend6[6:] = regime24[6:] - regime24[:-6]
    return regime24, tend6


# ---------------------------------------------------------------------------
# Per-region dump: train once, predict every arm x tier
# ---------------------------------------------------------------------------
def dump_region(target, fd_regions, cfgs, device, seed=0, epochs=900):
    t0 = time.time()
    model = train_fuel_zero_shot(
        fd_regions, target, seed=seed, epochs=epochs, device=device,
        p_cold=0.3, use_monthly=True, lag_months=1, wind_route_tau=1.1)
    data = fd_regions[target]
    model.wind_route_tau = float(ROUTE_TABLE.get(target, 1.1))

    split = int(len(data["rs"]) * TRAIN_FRACTION)
    sl = slice(split - SEQ_LEN, None)
    sliced = {**data,
              "rs": data["rs"][sl], "cif": data["cif"][sl],
              "fuel_shares": data["fuel_shares"][sl],
              "hours": data["hours"][sl],
              "exog": {k: v[sl] for k, v in data["exog"].items()}}
    w = build_fd_windows(sliced, seq_len=SEQ_LEN, horizon=HORIZON,
                         stride=TEST_STRIDE,
                         monthly_table=(data.get("monthly_table_target",
                                                 data.get("monthly_table"))),
                         lag_months=1)
    n = len(w["x_rs"])
    if n == 0:
        print(f"[fd47] {target}: no windows, skipped")
        return None

    # reproduce build_fd_windows' start positions (sliced coordinates)
    starts = list(range(0, len(sliced["rs"]) - (SEQ_LEN + HORIZON) + 1,
                        TEST_STRIDE))
    assert len(starts) == n, (len(starts), n)

    T = len(data["rs"])
    hours_utc = region_utc_axis(data, target)
    astro = data["exog"]["astro"]
    clearsky = np.maximum(astro[:, 1], 1.0).astype(np.float32)
    wx_era5 = data["exog"]["weather"]
    has_real_weather = bool(np.std(wx_era5[:, 2]) > 1e-6)
    W = wx_era5.shape[1]
    FE_W = w["fut_exog"].shape[2]

    df_nwp = load_nwp_table(target, cfgs, DATA_DIR)
    arms = {"era5": None}
    cov = {}
    if df_nwp is not None:
        for m in ("gfs", "icon", "ensemble"):
            a = build_arm_arrays(df_nwp, m, hours_utc, clearsky)
            if has_real_weather:
                # history = ERA5 (deployer's observed past), horizon = arm
                reg24, tend6 = composite_regime(
                    data["exog"]["wind_cf"], a["wind_cf"], split)
            else:
                # no-ERA5 fallback: the PUBLIC NWP archive is the region's
                # only weather source — regime stats on the arm axis alone
                reg24 = (pd.Series(a["wind_cf"]).rolling(
                    24, min_periods=1).mean().values.astype(np.float32))
                tend6 = np.zeros_like(reg24)
                tend6[6:] = reg24[6:] - reg24[:-6]
            a["regime24"], a["tend6"] = reg24, tend6
            arms[m] = a
            cov[m] = a["coverage"]

    oidx = pd.DatetimeIndex(w["origin_hours"])
    origin_hours = (oidx.astype("datetime64[s]").astype(np.int64) // 3600)
    out = {"origin_hours": origin_hours,
           "y_true": w["y_cif"].astype(np.float32)}
    ef_vec = data["ef_vec"].astype(np.float32)
    fd_cfg = data["fd_config"]

    def predict(windows, tier):
        cif, _, _ = predict_fuel_windows(
            model, windows, fd_cfg, ef_vec, cold=(tier == "icfg"),
            device=device)
        return cif.astype(np.float32)

    for arm, a in arms.items():
        if a is None:
            w_arm = w  # baseline arm: untouched windows
        else:
            w_arm = {k: (v.copy() if hasattr(v, "copy") else v)
                     for k, v in w.items()}
            if not has_real_weather:
                # no-ERA5 fallback family: history comes from the same
                # public NWP archive — a coherent all-NWP deployment.
                # sliced coords s -> full-axis [split - SEQ_LEN + s,
                # split + s): the 336 h history window position.
                base = split - SEQ_LEN
                for i, s in enumerate(starts):
                    x0 = base + s
                    x1 = x0 + SEQ_LEN
                    w_arm["x_weather"][i, :, 0] = a["weather"][x0:x1, 0]
                    w_arm["x_weather"][i, :, 1] = a["weather"][x0:x1, 1]
                    w_arm["x_weather"][i, :, 2] = a["weather"][x0:x1, 2]
                    w_arm["x_weather"][i, :, 3] = a["wind_cf"][x0:x1]
                    w_arm["x_weather"][i, :, 4] = a["csi"][x0:x1]
                    if w_arm["x_weather"].shape[2] > WX_COLS["regime24"]:
                        w_arm["x_weather"][i, :, 8] = a["regime24"][x0:x1]
                        w_arm["x_weather"][i, :, 9] = a["tend6"][x0:x1]
            for i, s in enumerate(starts):
                h0 = split + s          # full-axis horizon start
                h1 = min(h0 + HORIZON, T)
                span = h1 - h0
                if span <= 0:
                    continue
                # fut_weather: [temp, swr, w100, wind_cf, csi, regime, tend]
                wx_src = a["weather"][h0:h1]
                w_arm["fut_weather"][i, :span, 0] = wx_src[:, 0]
                w_arm["fut_weather"][i, :span, 1] = wx_src[:, 1]
                w_arm["fut_weather"][i, :span, 2] = wx_src[:, 2]
                w_arm["fut_weather"][i, :span, 3] = a["wind_cf"][h0:h1]
                w_arm["fut_weather"][i, :span, 4] = a["csi"][h0:h1]
                if W > WX_COLS["regime24"]:
                    w_arm["fut_weather"][i, :span, 8] = a["regime24"][h0:h1]
                    w_arm["fut_weather"][i, :span, 9] = a["tend6"][h0:h1]
                # fut_exog weather-derived columns
                fe = w_arm["fut_exog"]
                fe[i, :span, FE_WIND_CF] = a["wind_cf"][h0:h1]
                fe[i, :span, FE_CSI] = a["csi"][h0:h1]
                fe[i, :span, FE_HDH] = a["hdh"][h0:h1]
                fe[i, :span, FE_CDH] = a["cdh"][h0:h1]
                if FE_W > FE_TEND:
                    fe[i, :span, FE_REGIME] = a["regime24"][h0:h1]
                    fe[i, :span, FE_TEND] = a["tend6"][h0:h1]
        for tier in TIERS:
            out[f"pred_{arm}_{tier}"] = predict(w_arm, tier)

    out["nwp_coverage"] = np.array(
        [cov.get("gfs", np.nan), cov.get("icon", np.nan),
         cov.get("ensemble", np.nan)], dtype=np.float32)
    out["has_real_weather"] = np.array(has_real_weather, dtype=np.int8)

    path = DUMP_DIR / f"{target}_seed{seed}.npz"
    np.savez_compressed(path, **out)
    arms_ok = "/".join(arms.keys())
    print(f"[fd47] {target}: n={n}, arms={arms_ok}, "
          f"wx={'era5' if has_real_weather else 'zeros'} "
          f"({time.time()-t0:.0f}s)")
    return path


# ---------------------------------------------------------------------------
# Analysis: per-region + pooled paired deltas vs the era5 baseline
# ---------------------------------------------------------------------------
def analyze():
    rows = []
    dump_order = []
    pooled = {a: [] for a in ARMS if a != "era5"}
    pooled_i0 = {a: [] for a in ARMS if a != "era5"}
    pooled_fallback = {a: [] for a in ARMS if a != "era5"}
    for path in sorted(DUMP_DIR.glob("*_seed0.npz")):
        target = path.name.replace("_seed0.npz", "")
        d = np.load(path)
        dump_order.append(target)
        y = d["y_true"]
        n = len(d["origin_hours"])
        has_wx = int(d["has_real_weather"])
        row = {"target": target, "n_windows": n,
               "has_real_weather": has_wx}
        for m in ("gfs", "icon", "ensemble"):
            k = ["gfs", "icon", "ensemble"].index(m)
            row[f"cov_{m}"] = round(float(d["nwp_coverage"][k]), 4)
        base_icfg = np.abs(d["pred_era5_icfg"] - y).mean()
        base_i0 = np.abs(d["pred_era5_i0"] - y).mean()
        row["mae_era5_icfg"] = round(float(base_icfg), 2)
        row["mae_era5_i0"] = round(float(base_i0), 2)
        for arm in ARMS:
            if arm == "era5":
                continue
            pk = f"pred_{arm}_icfg"
            if pk not in d:
                row[f"mae_{arm}_icfg"] = None
                row[f"delta_{arm}_icfg"] = None
                continue
            mae = float(np.abs(d[pk] - y).mean())
            row[f"mae_{arm}_icfg"] = round(mae, 2)
            row[f"delta_{arm}_icfg"] = round(mae - float(base_icfg), 2)
            mae_i0 = float(np.abs(d[f"pred_{arm}_i0"] - y).mean())
            row[f"mae_{arm}_i0"] = round(mae_i0, 2)
            # window-level paired samples for pooled stats
            w_err_a = np.abs(d[pk] - y).mean(axis=1)
            w_err_b = np.abs(d["pred_era5_icfg"] - y).mean(axis=1)
            pooled[arm].append(w_err_a - w_err_b)
            w_err_a0 = np.abs(d[f"pred_{arm}_i0"] - y).mean(axis=1)
            w_err_b0 = np.abs(d["pred_era5_i0"] - y).mean(axis=1)
            pooled_i0[arm].append(w_err_a0 - w_err_b0)
            if not has_wx:
                pooled_fallback[arm].append(w_err_a - w_err_b)
        rows.append(row)

    out = RESULTS / "fd47_nwp_fut_weather.json"
    with open(out, "w") as f:
        json.dump({"rows": rows}, f, indent=1)
    print(f"[fd47] -> {out}")

    df = pd.DataFrame(rows)
    md = ["# FD-47 部署真实 fut_weather：GFS/ICON 业务预报 vs ERA5 代理", "",
          "同一模型（28 源训练，seed 0，ep900 官方口径），仅目标区推理时的"
          "未来 24h 天气来源不同：", "",
          "- era5：ERA5 再分析代理（FD-41 基线臂）",
          "- gfs / icon / ensemble：Open-Meteo 业务预报归档"
          "（gfs_seamless / icon_seamless / 双模型均值）", "",
          "gust/pressure 未来通道保留 ERA5 代理（归档无此二列，次要特征，"
          "文档化近似）；regime24/tend6 用 ERA5 历史 + NWP 未来的复合轴重算；"
          "336h 历史在所有臂中恒为 ERA5。", "",
          "## I_cfg 层（零遥测，论文核心口径）", ""]
    cols = ["target", "has_real_weather", "mae_era5_icfg",
            "mae_gfs_icfg", "mae_icon_icfg", "mae_ensemble_icfg"]
    cols = [c for c in cols if c in df.columns]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df[cols].iterrows():
        lines.append("| " + " | ".join(
            ("%.2f" % r[c]) if isinstance(r[c], float) else str(r[c])
            for c in cols) + " |")
    md.append("\n".join(lines))

    md += ["", "## 汇总（窗口级配对 ΔMAE，正值 = NWP 臂更差）", "",
           "era5-real 家族 = 原本有 ERA5 天气的区（配对差 = 预报技巧损失）；"
           "no-era5 fallback 家族 = 原本天气全零的区（差值 = 获得公开业务预报"
           "基础设施的收益）", ""]
    summary_lines = ["| arm | family | pooled ΔMAE (icfg) | better | worse |",
                     "|---|---|---|---|---|"]

    def fam_stats(arm, deltas_list, rows_sel, fam):
        if not deltas_list:
            return
        deltas = np.concatenate(deltas_list)
        per_region = [r.get(f"delta_{arm}_icfg") for r in rows_sel
                      if r.get(f"delta_{arm}_icfg") is not None]
        better = sum(1 for x in per_region if x < 0)
        worse = sum(1 for x in per_region if x > 0)
        summary_lines.append(
            f"| {arm} | {fam} | {deltas.mean():+.2f} | {better} | {worse} |")

    wx_rows = [r for r in rows if r["has_real_weather"]]
    no_rows = [r for r in rows if not r["has_real_weather"]]
    wx_targets = {r["target"] for r in wx_rows}
    for arm in ("gfs", "icon", "ensemble"):
        # pooled[arm] is aligned with dump_order; keep wx-family only
        deltas_list = [x for x, tgt in zip(pooled[arm], dump_order)
                       if tgt in wx_targets]
        if deltas_list:
            fam_stats(arm, deltas_list, wx_rows, "era5-real")
        if pooled_fallback[arm]:
            fam_stats(arm, pooled_fallback[arm], no_rows,
                      "no-era5 fallback")
    md.append("\n".join(summary_lines))

    md += ["", "## 覆盖率异常区（join 覆盖 < 99%）", ""]
    bad = []
    for r in rows:
        for m in ("gfs", "icon", "ensemble"):
            c = r.get(f"cov_{m}")
            if c is not None and c < 0.99:
                bad.append(f"{r['target']} {m}: {c:.3f}")
    md.append(", ".join(bad) if bad else "无")
    md_path = RESULTS / "fd47_nwp_fut_weather.md"
    md_path.write_text("\n".join(str(x) for x in md) + "\n")
    print(f"[fd47] -> {md_path}")
    show_cols = [c for c in ["target", "has_real_weather", "mae_era5_icfg",
                             "mae_gfs_icfg", "mae_ensemble_icfg"]
                 if c in df.columns]
    print(df[show_cols].to_string(index=False))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=None)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=900)
    ap.add_argument("--analyze", action="store_true",
                    help="skip training; analyze existing dumps")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    if args.analyze:
        analyze()
        return

    import torch
    device = args.device or (
        "cuda" if torch.cuda.is_available() else
        ("mps" if torch.backends.mps.is_available() else "cpu"))
    print(f"[fd47] device={device}")

    cfgs = all_region_configs()
    pool_names = list(cfgs)
    targets = args.regions if args.regions else pool_names
    print(f"[fd47] preparing {len(pool_names)} FD regions ...")
    fd_regions = {}
    for name in pool_names:
        fd_regions[name] = _safe_prepare_fd_region(name, cfgs, DATA_DIR)
    print(f"[fd47] {len(fd_regions)} regions ready")
    for tgt in targets:
        for seed in args.seeds:
            try:
                dump_region(tgt, fd_regions, cfgs, device, seed=seed,
                            epochs=args.epochs)
            except Exception as e:  # noqa: BLE001
                print(f"[fd47][ERR] {tgt} seed {seed}: {e}")


if __name__ == "__main__":
    main()
