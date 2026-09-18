"""检验用户判断「日均值 MAE 超标是模型设计问题」。

两个被当前设计浪费的合法事前信息源:
  A. 历史 CM 日度序列的系统性模式 (周效应 / 月内位置 / 年际重复)
     — 训练只用目标年之前, 严格事前
  B. 年内 45 天滞后的 CM 真值 (部署时 CarbonMonitor 滞后可得)
     — 月水平漂移校准 + 日尺度近期信息

对照基线: 当前设计 (日均值锁死月锚) 的日均值 MAE。
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/Users/cyyc0310/code/tran/src")

import numpy as np
import pandas as pd

from cif_5min_daily import daily_truth, split_cum, days_in, cum_sx, cum_sh

LAG_DAYS = 45  # CM 部署可得滞后


def build_daily(prov_cm, cum_map, years):
    rows = []
    for y in years:
        truth = daily_truth(prov_cm, y)
        truth["date"] = pd.to_datetime(truth["date"]).dt.date
        g = split_cum(cum_map, y)
        tl = dict(zip(truth["date"], truth["value"]))
        for date in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D"):
            d = date.date()
            m = d.month
            if d not in tl:
                continue
            denom = g[m - 1] * 1e8 / days_in(y, m)
            rows.append({"date": d, "year": y, "month": m,
                         "dom": d.day, "doy": date.dayofyear,
                         "dow": date.dayofweek,
                         "true_d": float(tl[d]) * 1e12 / denom})
    df = pd.DataFrame(rows)
    df["true_m"] = df.groupby(["year", "month"])["true_d"].transform("mean")
    df["resid"] = df["true_d"] - df["true_m"]  # 月内残差 (待解释波动)
    return df


def hist_pattern_fit(train_df):
    """历史跨年模式: dow 均值 + 月内 5 日 bin 均值 (加法模型)."""
    dow_m = train_df.groupby("dow")["resid"].mean()
    dom_bin = (train_df["dom"] - 1) // 5
    train_df = train_df.assign(dom_bin=dom_bin)
    bin_m = train_df.groupby("dom_bin")["resid"].mean()
    return dow_m, bin_m


def hist_pattern_pred(dow_m, bin_m, row):
    d = dow_m.get(row["dow"], 0.0)
    b = bin_m.get((row["dom"] - 1) // 5, 0.0)
    return d + b


def run(prov, prov_cm, cum_map, years_all, target_year):
    df_all = build_daily(prov_cm, cum_map, years_all)
    train = df_all[df_all["year"] < target_year].copy()
    test = df_all[df_all["year"] == target_year].copy()
    if train.empty or test.empty:
        print(f"[{prov} {target_year}] data missing, skip")
        return

    # ---- 基线: 当前设计 (日均=月锚) ----
    # 月锚用「真值月均」代表水平层上限口径 (与上轮泄漏上界一致),
    # 这样测的纯是「日尺度结构」的增益, 不混入月水平误差
    base_pred = test["true_m"].values
    true = test["true_d"].values
    mae_base = float(np.abs(base_pred - true).mean())

    # ---- A. 历史模式 (dow + dom_bin, 严格只用目标年之前) ----
    dow_m, bin_m = hist_pattern_fit(train)
    pat = np.array([hist_pattern_pred(dow_m, bin_m, r)
                    for _, r in test.iterrows()])
    mae_hist = float(np.abs(base_pred + pat - true).mean())
    # 年际重复: 同 DOY 残差均值 (训练年)
    yoy = train.groupby("doy")["resid"].mean()
    pat_yoy = test["doy"].map(yoy).fillna(0.0).values
    mae_yoy = float(np.abs(base_pred + pat_yoy - true).mean())
    mae_hist_combo = float(np.abs(base_pred + 0.5 * pat + 0.5 * pat_yoy - true).mean())

    # ---- A2. 日内残差自相关 (目标年内 lag>=45 天的 CM 真值, 部署合法) ----
    # 检验: 月内残差是否存在可用的短程结构 → 用 45 天前同省残差水平修正
    lag_resid = []
    for i, r in test.reset_index(drop=True).iterrows():
        d_lag = pd.Timestamp(r["date"]) - pd.Timedelta(days=LAG_DAYS)
        d_lag = d_lag.date()
        sub = df_all[df_all["date"] == d_lag]
        lag_resid.append(float(sub["resid"].iloc[0]) if len(sub) else np.nan)
    lag_resid = np.array(lag_resid)
    ok = ~np.isnan(lag_resid)
    # 45 天前残差对当日残差的相关 (信息存在性检验)
    if ok.sum() > 30:
        corr = float(np.corrcoef(lag_resid[ok], (true - base_pred)[ok])[0, 1])
    else:
        corr = np.nan

    # ---- B. 年内 45 天滞后水平校准 (月锚乘近期漂移因子) ----
    # 部署叙事: 预测月 m 时, 目标年内 ≤ m-2 的真值月已知 (CM 滞后)
    # 漂移因子 = 近期真值 / 近期「模型预测」(同月锚基线代表模型水平),
    # 加权: 越近的月权重越大 (半衰期 2 月)
    true_monthly = df_all.groupby(["year", "month"])["true_d"].mean()
    # 模型历史月锚: 各训练年该月真值均值 (代表"模型预测"的 climatology)
    clim_monthly = train.groupby("month")["true_d"].mean()
    # 历史漂移方差 (用于 shrinkage: 漂移不稳定时缩回 1)
    hist_drifts = []
    for y in sorted(train["year"].unique()):
        for m in range(3, 13):
            if (y, m) in true_monthly.index:
                # 该月预测 = 全训练年 climatology (不含 y)
                excl = train[(train["year"] != y) & (train["month"] == m)]["true_d"]
                if len(excl) >= 30:
                    hist_drifts.append(true_monthly[(y, m)] / excl.mean())
    sd_drift = float(np.std(hist_drifts)) if len(hist_drifts) >= 6 else 0.2
    sd_drift = max(sd_drift, 0.02)

    pred_m_adj2 = []
    for i, r in test.reset_index(drop=True).iterrows():
        y, m = r["year"], r["month"]
        past = [(y, mm) for mm in range(1, m - 1)]
        past = [pm for pm in past if pm in true_monthly.index]
        if len(past) >= 2:
            w = np.array([0.5 ** ((m - 1 - pm[1]) / 2.0) for pm in past])
            obs = float(np.average([true_monthly[pm] for pm in past], weights=w))
            exp = float(np.average([clim_monthly.get(pm[1], np.nan) for pm in past], weights=w))
            drift_raw = obs / exp if exp and np.isfinite(exp) else 1.0
            # shrinkage: 信号弱时缩回 1 (James-Stein 风格)
            drift = 1.0 + (drift_raw - 1.0) * 0.5
            drift = float(np.clip(drift, 0.7, 1.3))
        else:
            drift = 1.0
        pred_m_adj2.append(base_pred[i] * drift)
    mae_drift2 = float(np.abs(np.array(pred_m_adj2) - true).mean())

    # ---- 汇总 ----
    print(f"\n[{prov} {target_year}] n={len(test)}  (真值月均作锚基线 = 泄漏上界口径)")
    print(f"  基线 (日均=月锚)            : {mae_base:5.1f}")
    print(f"  +A dow/dom 历史模式         : {mae_hist:5.1f}  (增益 {mae_base - mae_hist:+.1f})")
    print(f"  +A 年际同DOY 模式           : {mae_yoy:5.1f}  (增益 {mae_base - mae_yoy:+.1f})")
    print(f"  +A 组合 (0.5/0.5)          : {mae_hist_combo:5.1f}  (增益 {mae_base - mae_hist_combo:+.1f})")
    # dow 效应显著性 (训练集内): 组内 t 检验
    dows = train["dow"].values
    resids = train["resid"].values
    dm = train.groupby("dow")["resid"].mean()
    if len(dm) == 7:
        se = resids.std() / np.sqrt(len(train) / 7)
        t_stat = float(dm.abs().max() / se)
        print(f"  dow 效应: 训练集周一~周日 {np.round(dm.values, 1)}")
        print(f"  dow 最大幅值 t 值          : {t_stat:.1f}  (>2 ≈ 显著)")
    print(f"  A2 45天前残差↔当日残差相关  : r={corr:.3f}" if np.isfinite(corr) else "  A2: 数据不足")
    print(f"  +B 年内45天滞后水平校准     : {mae_drift2:5.1f}  (增益 {mae_base - mae_drift2:+.1f})")
    return {"base": mae_base, "hist": mae_hist, "yoy": mae_yoy,
            "combo": mae_hist_combo, "drift": mae_drift2, "corr_lag45": corr}


if __name__ == "__main__":
    run("shanxi", "Shanxi", cum_sx, range(2019, 2025), 2023)
    run("shanxi", "Shanxi", cum_sx, range(2019, 2025), 2024)
    run("shanghai", "Shanghai", cum_sh, range(2020, 2025), 2023)
    run("shanghai", "Shanghai", cum_sh, range(2020, 2025), 2024)
