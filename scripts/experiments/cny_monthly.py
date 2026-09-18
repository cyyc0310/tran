"""春节感知月度模型 (CNY-aware monthly model)
核心: 1-2月作为整体块预测, 块内拆分比由春节日期决定
训练2019-2022 → 评估2023; 训练2019-2023 → 评估2024
"""
import pandas as pd
import numpy as np

cum = {
    2019: [517.1, 772.5, 1024.8, 1287.6, 1560.0, 1862.6, 2151.7, 2402.7, 2647.2, 2922.6, 3238.0],
    2020: [494.0, 765.6, 1017.7, 1272.7, 1540.6, 1842.1, 2137.4, 2409.9, 2685.8, 3024.5, 3366.9],
    2021: [611.5, 927.8, 1199.6, 1466.4, 1782.9, 2117.5, 2456.8, 2747.6, 3047.7, 3370.5, 3734.4],
    2022: [663.7, 1009.2, 1297.4, 1598.5, 1939.2, 2321.6, 2718.0, 3062.5, 3391.7, 3736.5, 4153.3],
    2023: [727.1, 1102.9, 1423.5, 1739.8, 2097.1, 2500.0, 2898.7, 3229.8, 3559.0, 3930.9, 4376.1],
    2024: [782.5, 1144.0, 1457.6, 1771.0, 2124.8, 2515.1, 2898.7, 3233.0, 3574.1, 3954.2, 4386.2],
}


def monthly_gen(y):
    c = cum[y]
    feb_tot = c[0]
    d1, d2 = 31, (29 if y % 4 == 0 else 28)
    jan = feb_tot * d1 / (d1 + d2)
    feb = feb_tot - jan
    out = [jan, feb]
    for i in range(1, 11):
        out.append(c[i] - c[i - 1])
    return out


df = pd.read_csv('data_2023/carbonmonitor_cn/carbonmonitor_china_provinces.csv')
df = df[(df['sector'] == 'Power') & (df['state'] == 'Shanxi')].copy()
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
df = df[df['year'].between(2019, 2024)]
days_in = lambda y, m: [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
mo_emis = df.groupby(['year', 'month'])['value'].mean()

true_cif = {}
for y in range(2019, 2025):
    g = monthly_gen(y)
    true_cif[y] = [mo_emis[(y, m)] * days_in(y, m) * 1e12 / (g[m - 1] * 1e8) for m in range(1, 13)]

piv = pd.DataFrame(true_cif, index=range(1, 13))

# 春节日期(day of year)
cny_doy = {2019: 36, 2020: 25, 2021: 43, 2022: 32, 2023: 22, 2024: 41}


def fit_ratio_model(train_years):
    """线性拟合: Feb/Jan CIF比 ~ CNY doy"""
    xs, ys = [], []
    for y in train_years:
        t = true_cif[y]
        xs.append(cny_doy[y])
        ys.append(t[1] / t[0])
    xs, ys = np.array(xs), np.array(ys)
    A = np.vstack([np.ones(len(xs)), xs]).T
    coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
    return coef


def predict_year(target, train_years, lvl_model, lvl_last):
    coef = fit_ratio_model(train_years)
    ratio = coef[0] + coef[1] * cny_doy[target]
    ratio = np.clip(ratio, 0.7, 1.4)
    lvl = 0.5 * lvl_model + 0.5 * lvl_last
    cm = piv[train_years].mean(axis=1)
    s = (cm / cm.mean()).values
    pred = s * lvl
    # 1-2月块: 保持块均值, 用春节比值拆分
    block = (pred[0] + pred[1]) / 2
    pred[1] = block * 2 * ratio / (1 + ratio)  # Feb
    pred[0] = block * 2 / (1 + ratio)  # Jan
    return pred, ratio


# 2023: lvl_model=618.9(模型预测), lvl_last=653(2022年均)
pred23, r23 = predict_year(2023, [2019, 2020, 2021, 2022], 618.9, np.mean(true_cif[2022]))
err23 = pred23 - np.array(true_cif[2023])
print('=== 2023: 春节感知模型 (训练19-22) ===')
print(f'拟合的2023 Feb/Jan比值: {r23:.3f} (真值: {true_cif[2023][1]/true_cif[2023][0]:.3f})')
for m in range(1, 13):
    print(f'{m:2d} | 预测{pred23[m-1]:.0f} | 真值{true_cif[2023][m-1]:.0f} | {err23[m-1]:+.0f}')
print('月均 LEVEL MAE:', round(np.abs(err23).mean(), 1))
sp = pd.Series(pred23).corr(pd.Series(true_cif[2023]), method='spearman')
print('月度 Spearman:', round(sp, 3))

# 2024: lvl_model=618.9? 模型2024预测应基于当时可得信息; 简化: 用2023模型年均同方法
# 模型年均2024 = 618.9 * (趋势)? 直接用之前的口径: 模型年均仍取618.9的水平外推
pred24, r24 = predict_year(2024, [2019, 2020, 2021, 2022, 2023], 618.9, np.mean(true_cif[2023]))
err24 = pred24 - np.array(true_cif[2024])
print()
print('=== 2024: 春节感知模型 (训练19-23) ===')
print(f'拟合的2024 Feb/Jan比值: {r24:.3f} (真值: {true_cif[2024][1]/true_cif[2024][0]:.3f})')
for m in range(1, 13):
    print(f'{m:2d} | 预测{pred24[m-1]:.0f} | 真值{true_cif[2024][m-1]:.0f} | {err24[m-1]:+.0f}')
print('月均 LEVEL MAE:', round(np.abs(err24).mean(), 1))
sp = pd.Series(pred24).corr(pd.Series(true_cif[2024]), method='spearman')
print('月度 Spearman:', round(sp, 3))

# 水平敏感性: 若模型年均不准
print()
print('=== 水平敏感性(2023, 春节感知形态) ===')
for lm in [600, 618.9, 640, 660]:
    p, _ = predict_year(2023, [2019, 2020, 2021, 2022], lm, np.mean(true_cif[2022]))
    print(f'模型年均={lm}: MAE={np.abs(p - np.array(true_cif[2023])).mean():.1f}')
