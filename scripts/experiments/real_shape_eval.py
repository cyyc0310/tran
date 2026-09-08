#!/usr/bin/env python
"""真实日内 CIF 形状 vs 模型形状层 — 形状层真值审计 (zero-telemetry audit).

背景 (2026-09-08 用户问题): 5 分钟产品的「小时形状层」来自 FuelDecompNet
零遥测管线 (UK donor 训练 + FD 月表合成配置), 从未与真实调度形状对照 —
国外场景有 ENTSO-E 小时级真值 diurnal MAE, 中国场景此前缺位。本脚本用
真实运行数据计算真实日内形状并对照评估, 补上论文形状层质量证据。

真实数据源 (运行值, 非假设):
  * 山西: ESPS 山西电力现货 2024 全年 15-min 真实运行值
    (esps_shanxi_tmp/pd/pd_train_origin.csv, 35136 行 96 点/天)。
    恒等式 TBS = PDL + TLP − NEL 全年精确成立 (脚本内复核) →
    thermal_share(t) = TBS/(PDL+TLP) = 1 − NEL/(PDL+TLP),
    真实 CIF 形状 ∝ thermal_share(t) (EF 日内近似常数, 归一后消除)。
    诚实注记: TBS 为「火电竞价空间」, 隐含把水电/非市场化机组并入
    热电 (山西水电 ~1-3%, 量级影响小, 形状影响更小)。
  * 上海: 无公开逐时/15min 运行数据。用 NDRC《各省级电网典型电力负荷
    曲线》官方工作日/节假日典型曲线 (ndrc_figs/ 像素数字化, 纵轴经
    网格线+OCR 双重标定) + 2023 官方受电结构, 推断 CIF 形状的包络:
    上界 = 受电平坦 (HVDC 计划曲线, 日内恒定) → CIF 随负荷起伏;
    下界 = 受电完全跟随负荷 → CIF 恒平。真实形状在包络内。

对照基准 (模型侧):
  * 模型形状层 = cif_5min_daily.predict_day (ERA5 真实天气基线,
    正常日 phys_delta≡0) 的 hourly_anchored / month_level,
    即纯形状层 (日均=1 口径), 与真实形状同口径归一。

指标:
  * 月均形状 MAE (‰ 与 % of daily mean 双报), 峰/谷时刻, 峰谷幅
  * 逐日形状 MAE 分布 (含日内天气特异性误差)
  * FD 月表风/光份额 vs ESPS 真实月度份额交叉核对 (2023 表 vs 2024 真值)

Usage:
    .venv-nemed/bin/python scripts/experiments/real_shape_eval.py \
        [--province shanxi|shanghai|both] [--day-step 1] [--device mps]
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "src"))

from cif_5min_daily import (  # noqa: E402
    build_forecaster, era5_weather,
    SHANXI_GEN_2023, SHANXI_THERMAL_EFS,
    SHANGHAI_SENDER_CIF, SHANGHAI_FLOW_SHARES,
)
from transcif.data.cn_deploy import cn_monthly_table  # noqa: E402

ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
ESPS_CSV = os.path.join(ROOT, "data_2023", "esps_shanxi_tmp", "pd",
                        "pd_train_origin.csv")
NDRC_DIR = os.path.join(ROOT, "data_2023", "ndrc_figs")
OUT_DIR = os.path.join(ROOT, "results_5min", "real_shape")

SEASONS = [("DJF 冬", [12, 1, 2]), ("MAM 春", [3, 4, 5]),
           ("JJA 夏", [6, 7, 8]), ("SON 秋", [9, 10, 11])]


# ---------------------------------------------------------------------------
# 山西: ESPS 真实运行数据 → 真实形状
# ---------------------------------------------------------------------------

def load_esps():
    df = pd.read_csv(ESPS_CSV)
    cols = ["TBS_DI", "PDL_DI", "TLP_DI", "NEL_DI", "WPO_DI", "PVO_DI",
            "TBS_DA", "PDL_DA", "TLP_DA", "NEL_DA"]
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["TBS_DI", "PDL_DI", "TLP_DI", "NEL_DI"]).copy()
    df["dt"] = pd.to_datetime(df["Date"])
    df = df[df["dt"].dt.year == 2024].copy()
    hh = df["TP"].str.split(":").str[0].astype(int)
    mm = df["TP"].str.split(":").str[1].astype(int)
    df["tp_idx"] = hh * 4 + mm // 15                     # 0..95
    df["gen"] = df["PDL_DI"] + df["TLP_DI"]              # 平衡总出力 MW
    df["thermal_share"] = df["TBS_DI"] / df["gen"]       # 真实热电份额
    df["ren_share"] = df["NEL_DI"] / df["gen"]
    df["solar_share"] = df["PVO_DI"] / df["gen"]
    df["wind_share"] = df["WPO_DI"] / df["gen"]
    df["thermal_share_da"] = df["TBS_DA"] / (df["PDL_DA"] + df["TLP_DA"])
    # 恒等式复核 (数据完整性守卫, 与手工验证一致)
    ident = float((df["PDL_DI"] + df["TLP_DI"] - df["NEL_DI"]
                   - df["TBS_DI"]).abs().max())
    assert ident < 1.0, f"identity TBS=PDL+TLP-NEL broken: {ident}"
    print(f"[esps] 2024 rows={len(df)} days={df['dt'].dt.date.nunique()} "
          f"| identity max|err|={ident:.4f} MW")
    return df


def real_day_shapes(df):
    """逐日真实形状 (24h, 日均=1)。dict date -> (24,)。"""
    out = {}
    for d, g in df.groupby("dt"):
        if len(g) < 90:
            continue
        arr = np.full(96, np.nan)
        arr[g["tp_idx"].values] = g["thermal_share"].values
        s96 = arr / np.nanmean(arr)
        out[d.date()] = s96.reshape(24, 4).mean(axis=1)
    return out


def real_monthly_from_days(day_shapes):
    monthly = {}
    for m in range(1, 13):
        mats = [s for d, s in day_shapes.items() if d.month == m]
        monthly[m] = np.nanmean(np.array(mats), axis=0)
    return monthly


def real_monthly_shares(df):
    """真实月度风/光/新能源份额 (日内平均, FD 表交叉核对用)。"""
    out = {}
    for m in range(1, 13):
        sub = df[df["dt"].dt.month == m]
        out[m] = {
            "wind": float(sub["wind_share"].mean()),
            "solar": float(sub["solar_share"].mean()),
            "ren": float(sub["ren_share"].mean()),
            "gen_TWh": float(sub["gen"].sum() * 0.25 / 1e6),
            "solar_TWh": float(sub["PVO_DI"].sum() * 0.25 / 1e6),
            "wind_TWh": float(sub["WPO_DI"].sum() * 0.25 / 1e6),
        }
    return out


def real_ren_duck(df):
    """月度真实新能源份额日内曲线 (24h, 未归一) — 真实 duck。"""
    duck = {}
    for m in range(1, 13):
        sub = df[df["dt"].dt.month == m]
        arr = np.full(96, np.nan)
        # 按 tp_idx 聚合当日均值
        g = sub.groupby("tp_idx")["ren_share"].mean()
        arr[g.index.values] = g.values
        duck[m] = arr.reshape(24, 4).mean(axis=1)
    return duck


def da_di_robustness(df):
    """同日 DA 口径 vs DI 口径形状差 (数据口径敏感性)。"""
    maes = []
    for d, g in df.groupby("dt"):
        if len(g) < 90:
            continue
        a = np.full(96, np.nan)
        b = np.full(96, np.nan)
        a[g["tp_idx"].values] = g["thermal_share"].values
        b[g["tp_idx"].values] = g["thermal_share_da"].values
        sa, sb = a / np.nanmean(a), b / np.nanmean(b)
        maes.append(np.nanmean(np.abs(sa - sb)))
    return float(np.mean(maes))


# ---------------------------------------------------------------------------
# 模型形状层 (ERA5 真实天气, 正常日)
# ---------------------------------------------------------------------------

def model_shapes(province, day_step=1, device="mps"):
    predict_day, meta = build_forecaster(province, device=device)
    daily = {}
    n_fail = 0
    dates = pd.date_range("2024-01-01", "2024-12-31", freq="D")[::day_step]
    for ts in dates:
        d = ts.date()
        try:
            _, hourly, pred_month = predict_day(
                d, base_weather_fn=era5_weather)
        except Exception as e:
            n_fail += 1
            if n_fail <= 3:
                print(f"  [warn] {d} predict failed: {e}")
            continue
        lvl = pred_month[d.month - 1]
        s = np.asarray(hourly, float) / lvl   # phys_delta≡0 → 纯形状层
        if s.size != 24:
            n_fail += 1
            continue
        daily[d] = s
    monthly = {m: np.mean([s for d, s in daily.items() if d.month == m],
                          axis=0) for m in range(1, 13)}
    print(f"[model:{province}] {len(daily)} days ok, {n_fail} failed "
          f"(day_step={day_step})")
    return monthly, daily


# ---------------------------------------------------------------------------
# 指标与汇总
# ---------------------------------------------------------------------------

def shape_metrics(s):
    s = np.asarray(s, float)
    return {"peak_h": int(np.argmax(s)), "trough_h": int(np.argmin(s)),
            "amp": float(s.max() - s.min())}


def shape_mae(a, b):
    return float(np.abs(np.asarray(a, float) - np.asarray(b, float)).mean())


def setup_plot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Hiragino Sans GB", "Arial Unicode MS",
                                       "PingFang SC", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt


# ---------------------------------------------------------------------------
# 山西主分析
# ---------------------------------------------------------------------------

def run_shanxi(day_step, device):
    print("=" * 72)
    print("山西: 真实形状 (ESPS 2024 15-min 运行值) vs 模型形状层 (ERA5)")
    print("=" * 72)
    df = load_esps()
    real_days = real_day_shapes(df)
    real_monthly = real_monthly_from_days(real_days)
    real_shares = real_monthly_shares(df)
    duck = real_ren_duck(df)
    da_mae = da_di_robustness(df)
    print(f"[esps] DA-vs-DI 同日形状 MAE (口径敏感性): {da_mae*1000:.1f} ‰ "
          f"({da_mae*100:.2f} % of daily mean)")

    # 真实形状基本量 (供报告)
    r0 = shape_metrics(real_monthly[6])
    print(f"[esps] 6月真实形状: 谷 {r0['trough_h']}:00 / 峰 {r0['peak_h']}:00 "
          f"/ 峰谷幅 {r0['amp']*100:.1f}% (日均=1 口径)")
    ann_ren = np.mean([real_shares[m]["ren"] for m in range(1, 13)])
    print(f"[esps] 2024 真实新能源份额 (发电口径, 年均): {ann_ren*100:.1f}% "
          f"| 风光年电量 "
          f"{np.mean([real_shares[m]['wind_TWh'] for m in range(1,13)]):.0f}"
          f"+{np.mean([real_shares[m]['solar_TWh'] for m in range(1,13)]):.0f} TWh")

    # FD 月表 vs 真实份额
    cfgd = cn_monthly_table(SHANXI_GEN_2023, thermal_efs=SHANXI_THERMAL_EFS,
                            lat=37.87, ann_windcf=0.25, ann_csi=0.55)
    tab = cfgd["table"]
    share_rows = []
    for m in range(1, 13):
        share_rows.append({
            "month": m,
            "fd_wind_share": float(tab[m - 1, 2 + 6]),
            "real_wind_share": real_shares[m]["wind"],
            "fd_solar_share": float(tab[m - 1, 2 + 5]),
            "real_solar_share": real_shares[m]["solar"],
        })
    dfsh = pd.DataFrame(share_rows)
    print("\n[FD-18 表 (2023 口径, 全口径发电估计) vs ESPS 真实 (2024 统调市场口径)]")
    print("  注: ESPS 份额分母为统调市场平衡 (PDL+TLP, 不含自备/地调),")
    print("  系统性偏高; 且 2024 风光装机同比大增。两者差 = 口径差+年份差+估计误差。")
    print(dfsh.round(3).to_string(index=False))

    # 模型形状
    model_monthly, model_daily = model_shapes("shanxi", day_step, device)

    # 月度对照表
    rows = []
    for m in range(1, 13):
        r, mo = real_monthly[m], model_monthly[m]
        rm, mm = shape_metrics(r), shape_metrics(mo)
        rows.append({
            "month": m, "shape_mae_permille": shape_mae(r, mo) * 1000,
            "shape_mae_pct": shape_mae(r, mo) * 100,
            "real_peak_h": rm["peak_h"], "real_trough_h": rm["trough_h"],
            "real_amp_pct": rm["amp"] * 100,
            "model_peak_h": mm["peak_h"], "model_trough_h": mm["trough_h"],
            "model_amp_pct": mm["amp"] * 100,
        })
    dfr = pd.DataFrame(rows)
    print("\n[山西 月均形状对照 (日均=1 口径)]")
    print(dfr.round(2).to_string(index=False))
    ann_mae = float(dfr["shape_mae_permille"].mean())
    print(f"→ 年均形状 MAE: {ann_mae:.1f} ‰ ({ann_mae/10:.2f} % of daily mean)")

    # 逐日 MAE (含天气特异性); 真实日形状缺时点处 NaN → 逐对 nan 掩码
    both = sorted(set(real_days) & set(model_daily))
    day_maes = []
    for d in both:
        a = np.asarray(real_days[d], float)
        b = np.asarray(model_daily[d], float)
        ok = np.isfinite(a) & np.isfinite(b)
        day_maes.append(np.abs(a[ok] - b[ok]).mean())
    day_maes = np.array(day_maes)
    print(f"→ 逐日形状 MAE ({len(both)} 天): mean {day_maes.mean()*1000:.1f}‰ "
          f"| median {np.median(day_maes)*1000:.1f}‰ "
          f"| p90 {np.percentile(day_maes, 90)*1000:.1f}‰ "
          f"| max {day_maes.max()*1000:.1f}‰")

    # 保存
    os.makedirs(OUT_DIR, exist_ok=True)
    dfr.to_csv(os.path.join(OUT_DIR, "shanxi_shape_comparison.csv"),
               index=False)
    dfsh.to_csv(os.path.join(OUT_DIR, "shanxi_fd_vs_real_shares.csv"),
                index=False)
    shape_mat = []
    for m in range(1, 13):
        shape_mat.append(
            {"month": m, "source": "real",
             **{f"h{h:02d}": real_monthly[m][h] for h in range(24)}})
        shape_mat.append(
            {"month": m, "source": "model",
             **{f"h{h:02d}": model_monthly[m][h] for h in range(24)}})
    pd.DataFrame(shape_mat).to_csv(
        os.path.join(OUT_DIR, "shanxi_monthly_shapes_24h.csv"), index=False)

    # ---- 图 1: 季节面板 真实 vs 模型 ----
    plt = setup_plot()
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5), sharex=True,
                             sharey=True)
    hours = np.arange(24)
    for ax, (sname, months) in zip(axes.flat, SEASONS):
        r = np.mean([real_monthly[m] for m in months], axis=0)
        mo = np.mean([model_monthly[m] for m in months], axis=0)
        ax.plot(hours, r, "o-", color="#d62728", lw=2, ms=3.5,
                label="真实形状 (ESPS 2024 运行值)")
        ax.plot(hours, mo, "s-", color="#1f77b4", lw=1.8, ms=3,
                label="模型形状层 (零遥测, ERA5)")
        ax.axhline(1.0, color="gray", lw=0.6, ls=":")
        ax.set_title(f"{sname} — 形状 MAE {shape_mae(r, mo)*100:.1f}%",
                     fontsize=11)
        ax.set_xticks(range(0, 24, 2))
        ax.grid(alpha=0.25)
    axes.flat[0].set_ylabel("归一 CIF (日均=1)")
    axes.flat[2].set_ylabel("归一 CIF (日均=1)")
    axes.flat[2].set_xlabel("本地时刻 (h)")
    axes.flat[3].set_xlabel("本地时刻 (h)")
    axes.flat[0].legend(fontsize=9, loc="best")
    fig.suptitle("山西 2024: 真实日内 CIF 形状 vs 零遥测模型形状层 "
                 "(日均=1 口径)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p1 = os.path.join(OUT_DIR, "shanxi_real_vs_model.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)

    # ---- 图 2: 真实 duck + FD/真实份额 ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    for sname, months in SEASONS:
        d = np.mean([duck[m] for m in months], axis=0)
        ax1.plot(hours, d * 100, "o-", ms=3, label=sname)
    ax1.set_xlabel("本地时刻 (h)")
    ax1.set_ylabel("新能源出力份额 (%)")
    ax1.set_title("山西 2024 真实 duck (ESPS: NEL/(负荷+外送))")
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(alpha=0.25)
    ax1.legend(fontsize=9)
    ax2.plot(dfsh["month"], dfsh["fd_wind_share"] * 100, "s--",
             color="#1f77b4", label="FD 表 风份额 (2023 口径)")
    ax2.plot(dfsh["month"], dfsh["real_wind_share"] * 100, "s-",
             color="#1f77b4", alpha=0.55, label="真实 风份额 (2024)")
    ax2.plot(dfsh["month"], dfsh["fd_solar_share"] * 100, "^--",
             color="#ff7f0e", label="FD 表 光份额 (2023 口径)")
    ax2.plot(dfsh["month"], dfsh["real_solar_share"] * 100, "^-",
             color="#ff7f0e", alpha=0.55, label="真实 光份额 (2024)")
    ax2.set_xlabel("月份")
    ax2.set_ylabel("发电份额 (%)")
    ax2.set_title("FD-18 月表份额 vs ESPS 真实月度份额")
    ax2.set_xticks(range(1, 13))
    ax2.grid(alpha=0.25)
    ax2.legend(fontsize=8.5)
    fig.tight_layout()
    p2 = os.path.join(OUT_DIR, "shanxi_ren_duck_and_shares.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)

    print(f"[saved] {p1}\n[saved] {p2}")
    return {"annual_shape_mae_permille": ann_mae,
            "day_mae_mean_permille": float(day_maes.mean() * 1000),
            "n_days": len(both)}


# ---------------------------------------------------------------------------
# 上海: NDRC 真实负荷形状 + 受电结构 → CIF 形状包络 vs 模型
# ---------------------------------------------------------------------------

def _ndrc_hourly(csv_path, drop_t0=False):
    df_ = pd.read_csv(csv_path)
    parts = df_["tp"].str.split(":")
    hf = parts.str[0].astype(float) + parts.str[1].astype(float) / 60.0
    L = np.zeros(24)
    cnt = np.zeros(24)
    for t, v in zip(hf, df_["load_mw"]):
        if t >= 24.0:
            continue
        if drop_t0 and t == 0.0:
            continue  # 轴线像素污染点
        h = int(t)
        L[h] += v
        cnt[h] += 1
    return L / np.maximum(cnt, 1.0)


def run_shanghai(day_step, device):
    print("=" * 72)
    print("上海: NDRC 官方典型负荷曲线 + 受电结构 → CIF 形状包络 vs 模型")
    print("=" * 72)
    Lw = _ndrc_hourly(os.path.join(NDRC_DIR, "shanghai_workday_96pt.csv"))
    Lh = _ndrc_hourly(os.path.join(NDRC_DIR, "shanghai_holiday_96pt.csv"),
                      drop_t0=True)
    print(f"[ndrc] 工作日负荷: min {Lw.min():.0f} / max {Lw.max():.0f} MW "
          f"| 峰 {int(np.argmax(Lw))}:00 / 谷 {int(np.argmin(Lw))}:00 "
          f"| 负荷率 {Lw.mean()/Lw.max()*100:.0f}%")
    print(f"[ndrc] 节假日负荷: min {Lh.min():.0f} / max {Lh.max():.0f} MW "
          f"| 峰 {int(np.argmax(Lh))}:00 / 谷 {int(np.argmin(Lh))}:00")

    # 本地火电结构与电量 (2023 官方口径, 与 cif_5min_daily 常量同源)
    from cif_5min_daily import (SHANGHAI_GEN_2023, SHANGHAI_THERMAL_EFS,
                                SHANGHAI_IMPORTS_2023)
    sc = sum(SHANGHAI_GEN_2023["coal"])
    sg = sum(SHANGHAI_GEN_2023["gas"])
    sp = sum(SHANGHAI_GEN_2023["petroleum"])
    ssolar = sum(SHANGHAI_GEN_2023["solar"])
    swind = sum(SHANGHAI_GEN_2023["wind"])
    ef_coal, ef_gas, ef_oil = (SHANGHAI_THERMAL_EFS["coal"],
                               SHANGHAI_THERMAL_EFS["gas"],
                               SHANGHAI_THERMAL_EFS["petroleum"])
    ef_th = (ef_coal * sc + ef_gas * sg + ef_oil * sp) / (sc + sg + sp)
    ef_imp = float(np.mean([
        sum(SHANGHAI_FLOW_SHARES[k][m] * SHANGHAI_SENDER_CIF[k][m]
            for k in SHANGHAI_SENDER_CIF)
        for m in range(12)]))
    print(f"[ef] 本地火电 EF = {ef_th:.1f} g/kWh (煤{sc} 气{sg} 油{sp:.1f} 亿kWh)"
          f" | 受电 EF 年均 = {ef_imp:.1f} g/kWh (月度流量加权)")

    # 本地光伏日内出力形状 (astro 物理模型, 归一日均=1) — 鸭子效应通道:
    # 上海无公开逐时光伏出力, 用天文形状 (晴天物理上限) × 阴天折扣
    # (0.6, 华东年均云量) 近似; 仅用于形状层, 不涉及电量。
    from transcif.physics.astro import astro_features
    hours_utc = pd.date_range("2024-06-01 16:00", periods=24, freq="h",
                              tz="UTC")
    astro = astro_features(hours_utc, 31.23, 121.47)
    csi_shape = np.clip(astro[:, 1], 0.0, None)
    csi_shape = csi_shape / max(csi_shape.sum(), 1e-9) * 24.0 \
        if csi_shape.sum() > 0 else np.ones(24)
    csi_shape *= 0.6  # 阴天/多云折扣 (形状层近似)

    # CIF 形状包络 (工作日):
    #   上界 A: 受电平坦 I=0.451×mean(L) (HVDC 计划曲线) + 本地光伏鸭子
    #     → 午间本地低碳出力顶替火电 → CIF 双谷加深
    #   下界 B: 受电完全跟随负荷 → CIF 恒平
    solar_share_ann = ssolar / (sc + sg + sp + ssolar + swind)  # 0.5%
    I_flat = 0.451 * Lw.mean()
    gen_local = Lw - I_flat                       # 本地火电+风光出力
    pv = solar_share_ann * gen_local.mean() * csi_shape   # 光伏出力 MW
    cif_A = ef_th - (ef_th - ef_imp) * I_flat / Lw \
        - ef_th * pv / np.maximum(Lw, 1.0)        # 光伏压低 CIF
    shape_A = cif_A / cif_A.mean()
    mA = shape_metrics(shape_A)
    print(f"[infer] 上界 A (受电平坦+光伏鸭子): 峰 {mA['peak_h']}:00 / 谷 "
          f"{mA['trough_h']}:00 / 峰谷幅 {mA['amp']*100:.1f}% "
          f"(CIF {cif_A.min():.0f}-{cif_A.max():.0f} g/kWh)")
    print(f"[infer] 光伏鸭子贡献: 本地光伏份额 {solar_share_ann*100:.2f}% "
          f"(2023 规上真实月表 {ssolar:.2f} 亿 kWh, NBS 月度链条)")
    print("[infer] 受电跟随下界 B: 恒平 (幅 0%) — 真实形状幅值 ∈ [0, A]")

    # 模型形状
    model_monthly, _ = model_shapes("shanghai", day_step, device)
    model_annual = np.mean([model_monthly[m] for m in range(1, 13)], axis=0)
    mModel = shape_metrics(model_annual)
    mae_A = shape_mae(shape_A, model_annual)
    print(f"[model] 上海年均形状: 峰 {mModel['peak_h']}:00 / 谷 "
          f"{mModel['trough_h']}:00 / 峰谷幅 {mModel['amp']*100:.1f}%")
    print(f"[vs]   模型 vs 上界 A 形状 MAE: {mae_A*1000:.1f}‰ "
          f"({mae_A*100:.2f}%) | 模型幅值 {mModel['amp']*100:.1f}% "
          f"∈ [0, {mA['amp']*100:.1f}]% 包络: "
          f"{'是' if mModel['amp'] <= mA['amp'] + 0.01 else '否 (越界!)'}")

    # 保存 + 图
    os.makedirs(OUT_DIR, exist_ok=True)
    pd.DataFrame({
        "hour": range(24),
        "load_workday_mw": Lw.round(0),
        "load_workday_norm": (Lw / Lw.mean()).round(4),
        "load_holiday_norm": (Lh / Lh.mean()).round(4),
        "cif_inferred_flat_import": shape_A.round(4),
        "model_annual_shape": model_annual.round(4),
    }).to_csv(os.path.join(OUT_DIR, "shanghai_shape_comparison.csv"),
              index=False)
    mon_rows = []
    for m in range(1, 13):
        mm = shape_metrics(model_monthly[m])
        mon_rows.append({"month": m,
                         "model_peak_h": mm["peak_h"],
                         "model_trough_h": mm["trough_h"],
                         "model_amp_pct": mm["amp"] * 100})
    pd.DataFrame(mon_rows).to_csv(
        os.path.join(OUT_DIR, "shanghai_model_monthly.csv"), index=False)

    plt = setup_plot()
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    hours = np.arange(24)
    ax.plot(hours, shape_A, "o-", color="#d62728", lw=2, ms=3.5,
            label="推断 CIF 形状上界 A (NDRC 负荷+受电平坦+光伏鸭子)")
    ax.fill_between(hours, 1.0, shape_A, color="#d62728", alpha=0.10,
                    label="真实 CIF 形状包络 [下界 B=恒平, 上界 A]")
    ax.plot(hours, model_annual, "s-", color="#1f77b4", lw=1.8, ms=3,
            label="模型形状层 (零遥测, ERA5, 2024 年均)")
    ax.axhline(1.0, color="gray", lw=0.6, ls=":")
    ax2 = ax.twinx()
    ax2.plot(hours, Lw / Lw.mean(), ":", color="#2ca02c", lw=1.4,
             label="NDRC 工作日真实负荷 (右轴, 归一)")
    ax2.set_ylabel("归一负荷 (日均=1)", color="#2ca02c")
    ax.set_xlabel("本地时刻 (h)")
    ax.set_ylabel("归一 CIF (日均=1)")
    ax.set_xticks(range(0, 24, 2))
    ax.grid(alpha=0.25)
    ax.set_title("上海: 真实负荷 (NDRC 官方) + 受电结构推断的 CIF 形状包络 "
                 "vs 零遥测模型形状层")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="best")
    fig.tight_layout()
    p = os.path.join(OUT_DIR, "shanghai_shape_envelope.png")
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[saved] {p}")
    return {"model_vs_A_mae_permille": mae_A * 1000,
            "model_amp_pct": mModel["amp"] * 100,
            "bound_A_amp_pct": mA["amp"] * 100}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--province", default="both",
                    choices=["shanxi", "shanghai", "both"])
    ap.add_argument("--day-step", type=int, default=1,
                    help="模型侧逐日重跑的采样步长 (1=全年每天)")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args()

    results = {}
    if args.province in ("shanxi", "both"):
        results["shanxi"] = run_shanxi(args.day_step, args.device)
    if args.province in ("shanghai", "both"):
        results["shanghai"] = run_shanghai(args.day_step, args.device)

    print("\n" + "=" * 72)
    print("SUMMARY — 真实形状 vs 模型形状层")
    print("=" * 72)
    for prov, r in results.items():
        print(f"  {prov}: {r}")


if __name__ == "__main__":
    main()
