"""测真实天气的可预测性上限: CM 日际波动 vs ERA5 真实天气的相关性。

回答: 日均值误差中那 15-20 g/kWh 的「日际波动」有多少是真实天气
驱动的 (可预测), 有多少是 CM 方法论噪声 (不可预测)。
方法: 用 CM 日真值偏离月均的残差 (true_d - true_m), 对 ERA5 日天气
特征 (日均温/风/辐射 及其相对气候态的距平) 回归, 看 R² 与残差。
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, "/Users/cyyc0310/code/tran/src")

import numpy as np
import pandas as pd

from cif_5min_daily import daily_truth, split_cum, days_in, cum_sx, cum_sh


def load_era5(prov):
    path = f"/tmp/era5_{'shanxi' if prov=='shanxi' else 'shanghai'}.json"
    d = json.load(open(path))
    df = pd.DataFrame(d["hourly"])
    df["ts"] = pd.to_datetime(df["time"])
    df["date"] = df["ts"].dt.date
    # 日聚合
    g = df.groupby("date").agg(
        temp=("temperature_2m", "mean"),
        wind=("wind_speed_10m", "mean"),
        rad=("wind_speed_10m", lambda s: 0),  # placeholder, 用 rad 单独算
        rad_mean=("shortwave_radiation", "mean"),
        wind_max=("wind_speed_10m", "max"),
    )
    return g


def climate_anomaly(g):
    """相对逐月气候态 (2019-2024 均值不可得, 用当月均值) 的距平。"""
    g = g.copy()
    g["month"] = pd.to_datetime(g.index.astype(str)).month
    # 用当月均值作气候态 (年内距平)
    for col in ["temp", "wind", "rad_mean"]:
        clim = g.groupby("month")[col].transform("mean")
        g[f"{col}_anom"] = g[col] - clim
    return g


def regress(prov, year):
    if prov == "shanxi":
        cum, cm = cum_sx, "Shanxi"
    else:
        cum, cm = cum_sh, "Shanghai"
    truth = daily_truth(cm, year)
    truth["date"] = pd.to_datetime(truth["date"]).dt.date
    truth_lookup = dict(zip(truth["date"], truth["value"]))
    g = split_cum(cum, year)
    rows = []
    for date in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
        d = date.date()
        m = d.month
        if d not in truth_lookup:
            continue
        denom = g[m - 1] * 1e8 / days_in(year, m)
        rows.append({"date": d, "true_d": float(truth_lookup[d]) * 1e12 / denom,
                     "month": m})
    tdf = pd.DataFrame(rows)
    tdf["true_m"] = tdf.groupby("month")["true_d"].transform("mean")
    tdf["resid"] = tdf["true_d"] - tdf["true_m"]  # 待解释的日际波动

    wx = climate_anomaly(load_era5(prov))
    wx = wx[~wx.index.duplicated(keep="first")]
    mdf = tdf.merge(wx, left_on="date", right_index=True, how="inner")
    mdf = mdf.dropna()
    feats = ["temp_anom", "wind_anom", "rad_mean_anom", "wind_max"]
    X = mdf[feats].values
    y = mdf["resid"].values
    # 最小二乘 (含截距)
    Xc = np.column_stack([np.ones(len(mdf))] + [X[:, i] for i in range(X.shape[1])])
    beta, *_ = np.linalg.lstsq(Xc, y, rcond=None)
    pred = Xc @ beta
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot
    print(f"\n[{prov} {year}] CM 日际波动 vs ERA5 天气距平")
    print(f"  n={len(mdf)}  R²={r2:.3f}")
    for f, b in zip(["const"] + feats, beta):
        print(f"    {f:12s} β={b:+.2f}")
    print(f"  波动 MAE (未解释): {np.abs(y).mean():.1f} g/kWh")
    print(f"  残差 MAE (天气解释后): {np.abs(y - pred).mean():.1f} g/kWh")
    print(f"  → 天气可解释 {np.abs(y).mean() - np.abs(y - pred).mean():+.1f} g/kWh")
    return mdf, beta


if __name__ == "__main__":
    regress(sys.argv[1], int(sys.argv[2]))
