"""全部年份(2019-2024)真实月度发电量重算真值CIF + 复测当前最优方案
1-2月合并公布 → 按天数比例拆分
"""
import pandas as pd
import numpy as np

# 累计发电量(亿kWh) 2月累计=1+2月合计
cum = {
    2019: [517.1, 772.5, 1024.8, 1287.6, 1560.0, 1862.6, 2151.7, 2402.7, 2647.2, 2922.6, 3238.0],
    2020: [494.0, 765.6, 1017.7, 1272.7, 1540.6, 1842.1, 2137.4, 2409.9, 2685.8, 3024.5, 3366.9],
    2021: [611.5, 927.8, 1199.6, 1466.4, 1782.9, 2117.5, 2456.8, 2747.6, 3047.7, 3370.5, 3734.4],
    2022: [663.7, 1009.2, 1297.4, 1598.5, 1939.2, 2321.6, 2718.0, 3062.5, 3391.7, 3736.5, 4153.3],
    2023: [727.1, 1102.9, 1423.5, 1739.8, 2097.1, 2500.0, 2898.7, 3229.8, 3559.0, 3930.9, 4376.1],
    2024: [782.5, 1144.0, 1457.6, 1771.0, 2124.8, 2515.1, 2898.7, 3233.0, 3574.1, 3954.2, 4386.2],
}  # 索引0=2月末累计, 1=3月末, ..., 10=12月末


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
    row = []
    for m in range(1, 13):
        emis_m = mo_emis[(y, m)] * days_in(y, m)  # Mt CO2 (value单位Mt/day)
        # gCO2/kWh = Mt*1e12 g / (亿kWh*1e8 kWh)
        row.append(emis_m * 1e12 / (g[m - 1] * 1e8))
    true_cif[y] = row

print('=== 真实月度CIF真值(月排放/月发电, 1-2月按天数拆分) ===')
print('年\\月 ' + ''.join(f'{m:6d}' for m in range(1, 13)) + '   年均')
for y in range(2019, 2025):
    print(f'{y} ' + ''.join(f'{v:6.0f}' for v in true_cif[y]) + f' {np.mean(true_cif[y]):6.0f}')

# 复测: clim形态(2019-2022) x 融合水平 vs 2023真实口径
piv = pd.DataFrame(true_cif, index=range(1, 13))
clim = piv[[2019, 2020, 2021, 2022]].mean(axis=1)
s = (clim / clim.mean()).values
# 水平: 0.5*模型年均 + 0.5*上年(2022)披露年均
lvl22 = np.mean(true_cif[2022])
lvl_model = 618.9
lvl = 0.5 * lvl_model + 0.5 * lvl22
pred23 = s * lvl
err23 = pred23 - np.array(true_cif[2023])
print()
print('=== 复测2023: clim(19-22)形态 x 融合水平 ===')
for m in range(1, 13):
    print(f'{m:2d} | 预测{pred23[m-1]:.0f} | 真值{true_cif[2023][m-1]:.0f} | {err23[m-1]:+.0f}')
print('月均 LEVEL MAE:', round(np.abs(err23).mean(), 1))

# 2024
lvl23 = np.mean(true_cif[2023])
lvl24 = 0.5 * lvl_model + 0.5 * lvl23
clim5 = piv[[2019, 2020, 2021, 2022, 2023]].mean(axis=1)
s5 = (clim5 / clim5.mean()).values
pred24 = s5 * lvl24
err24 = pred24 - np.array(true_cif[2024])
print()
print('=== 复测2024: clim(19-23)形态 x 融合水平 ===')
for m in range(1, 13):
    print(f'{m:2d} | 预测{pred24[m-1]:.0f} | 真值{true_cif[2024][m-1]:.0f} | {err24[m-1]:+.0f}')
print('月均 LEVEL MAE:', round(np.abs(err24).mean(), 1))

# LOYO 形态上限
print()
print('=== LOYO(水平完美)形态上限 ===')
for yy in range(2019, 2025):
    tr = [y for y in range(2019, 2025) if y != yy]
    cm = piv[tr].mean(axis=1)
    sm = (cm / cm.mean()).values * np.mean(true_cif[yy])
    print(f'{yy}: 形态MAE = {np.abs(sm - np.array(true_cif[yy])).mean():.1f}')
