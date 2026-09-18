"""误差分解: 真逐日日均值 MAE 拆成 月水平误差 + CM 日际波动 两个分量。

结论要点: 超额 MAE 的构成是「月水平误差」还是「模型没有对真实
日际排放波动建模」。分解式:
    err_d = (pred_d - month_pred) + (month_pred - month_true) + (month_true - true_d)
          = [形状/日际项] + [月水平项] + [CM 日际波动真值项]
MAE(err) 中月水平项是可修的, 日际波动项是信息上限 (无真值天气/负荷
信息时无法预测的部分)。
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
sys.path.insert(0, "/Users/cyyc0310/code/tran/src")

import numpy as np
import pandas as pd

from cif_5min_daily import (
    build_forecaster, daily_truth, split_cum, days_in, cum_sx, cum_sh)


def decompose(prov_name, year, epochs=60):
    predict_day, meta = build_forecaster(prov_name, epochs=epochs, device="cpu")
    prov = meta["province"]
    cm_name = meta["cm_name"]
    truth_daily = daily_truth(cm_name, year)
    truth_daily["date"] = pd.to_datetime(truth_daily["date"]).dt.date
    truth_lookup = dict(zip(truth_daily["date"], truth_daily["value"]))
    cum_map = cum_sx if prov == "shanxi" else cum_sh
    g = split_cum(cum_map, year)

    rows = []
    for date in pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"):
        d = date.date()
        m = d.month
        try:
            cif_5min, hourly, pred_month = predict_day(d)
        except Exception:
            continue
        if d not in truth_lookup:
            continue
        denom = g[m - 1] * 1e8 / days_in(year, m)
        true_d = float(truth_lookup[d]) * 1e12 / denom
        rows.append({
            "date": d, "month": m,
            "pred_d": float(cif_5min.mean()),
            "true_d": true_d,
            "pred_m": float(pred_month[m - 1]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"[{prov} {year}] no rows")
        return
    # 真值月均 (从日真值聚合, 与月聚合层同口径)
    df["true_m"] = df.groupby("month")["true_d"].transform("mean")
    df["pred_m_from_d"] = df.groupby("month")["pred_d"].transform("mean")

    err = df["pred_d"] - df["true_d"]
    mae_total = float(np.abs(err).mean())
    # 分解: err = (pred_d - pred_m) + (pred_m - true_m) + (true_m - true_d)
    comp_shape = df["pred_d"] - df["pred_m"]
    comp_level = df["pred_m"] - df["true_m"]
    comp_fluct = df["true_m"] - df["true_d"]   # CM 日际波动 (未建模)

    print(f"\n[{prov} {year}] n_days={len(df)}")
    print(f"  总日均值 MAE      : {mae_total:.1f} g/kWh")
    print(f"  分量均值 (代数和=总误差):")
    print(f"    月水平项 (可修)  : {comp_level.mean():+.1f}  MAE {np.abs(comp_level).mean():.1f}")
    print(f"    形状/日际项      : {comp_shape.mean():+.1f}  MAE {np.abs(comp_shape).mean():.1f}")
    print(f"    CM日际波动项     : {comp_fluct.mean():+.1f}  MAE {np.abs(comp_fluct).mean():.1f}")
    # 若模型完美预测月水平 (pred_m = true_m), 剩余 MAE 是多少?
    resid = comp_shape - comp_fluct
    print(f"  → 若月水平完美: 剩余日均值 MAE = {np.abs(resid).mean():.1f} g/kWh")
    # 上限口径: 用真值月均作 pred_m (泄漏上界, 仅诊断)
    print(f"  → 3-12月单月分解 (排除 1-2 月春节错位):")
    mask = df["month"] >= 3
    err_312 = df.loc[mask, "pred_d"] - df.loc[mask, "true_d"]
    print(f"    3-12月总 MAE     : {np.abs(err_312).mean():.1f}")
    lvl312 = (df.loc[mask, "pred_m"] - df.loc[mask, "true_m"])
    print(f"    3-12月月水平 MAE : {np.abs(lvl312).mean():.1f}")
    shp312 = (df.loc[mask, "pred_d"] - df.loc[mask, "pred_m"])
    print(f"    3-12月形状项 MAE : {np.abs(shp312).mean():.1f}")
    flu312 = (df.loc[mask, "true_m"] - df.loc[mask, "true_d"])
    print(f"    3-12月日际波动 MAE: {np.abs(flu312).mean():.1f}")
    df.to_csv(f"results_5min/decompose_{prov}_{year}.csv", index=False)
    print(f"  saved: results_5min/decompose_{prov}_{year}.csv")


if __name__ == "__main__":
    decompose(sys.argv[1], int(sys.argv[2]))
