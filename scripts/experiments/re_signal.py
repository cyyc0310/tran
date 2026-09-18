"""新能源出力信号实验：风速+辐照 → 风光容量系数 → CIF形态
理论: 山西CIF ≈ EF_coal * (1 - 风光出力占比), 风光出力可由再分析天气事前估计
"""
import pandas as pd
import numpy as np

df = pd.read_csv('data_2023/carbonmonitor_cn/carbonmonitor_china_provinces.csv')
df = df[(df['sector'] == 'Power') & (df['state'] == 'Shanxi')].copy()
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
gens = {2019: 3867, 2020: 4051, 2021: 4178, 2022: 4178, 2023: 4376.1, 2024: 4516.11}
df = df[df['year'].isin(gens)]
df['cif'] = df.apply(lambda r: r['value'] * 1e12 / (gens[r['year']] * 1e8 / 365), axis=1)

w = pd.read_csv('data_2023/carbonmonitor_cn/taiyuan_daily_wind_rad.csv')
w['date'] = pd.to_datetime(w['time'])
df = df.merge(w[['date', 'wind_speed_10m_max', 'shortwave_radiation_sum']], on='date', how='left')
df = df.rename(columns={'wind_speed_10m_max': 'wsm', 'shortwave_radiation_sum': 'rad'})

# 风电容量系数代理: P ~ v^3 (10m风速→轮毂高度近似)
df['wind_pow'] = df['wsm'] ** 3
# 光伏容量系数代理: 辐照/晴空辐照(归一)
df['pv_cf'] = df['rad'] / df.groupby(df['date'].dt.month)['rad'].transform('mean')

# 装机数据(万千瓦): 年末值 → 当年平均装机
# 2019: 风1139/光825, 2020: 风1416/光1316, 2021: 风2123/光1635, 2022: 风2491/光1920(?), 2023: 风2491/光2810(?), 2024: 风2616.5/光3476.8
# 用可得的年末装机线性插值日度装机
inst = {
    2018: (1000.0, 700.0),  # 2018年末近似(线性外推基线)
    2019: (1139.2, 825.0), 2020: (1416.1, 1316.0), 2021: (2123.0, 1635.0),
    2022: (2491.0, 1920.0), 2023: (2491.0, 2810.0), 2024: (2616.5, 3476.8),
}

def daily_inst(row):
    y = row['year']
    w0, s0 = inst[y - 1]
    w1, s1 = inst[y]
    doy = row['date'].timetuple().tm_yday
    days = 366 if (y % 4 == 0) else 365
    frac = doy / days
    return pd.Series({'inst_w': w0 + (w1 - w0) * frac, 'inst_s': s0 + (s1 - s0) * frac})

df[['inst_w', 'inst_s']] = df.apply(daily_inst, axis=1)
# 总装机(新能源+水电, 不含常规火电) → 新能源"潜在出力占比"
df['wind_gen'] = df['inst_w'] * df['wind_pow'] / 10000  # 万kW*v3 → 任意单位
df['pv_gen'] = df['inst_s'] * df['pv_cf'] / 10000

# 风电v3代理需归一: 用各年风电v3均值≈容量系数0.25的假设校准
# 简化: 直接用 (inst_w*wind_pow_norm + inst_s*pv_cf) / 总装机
wp_norm = df['wind_pow'] / df['wind_pow'].mean()
df['re_share_proxy'] = (df['inst_w'] * wp_norm * 0.3 + df['inst_s'] * df['pv_cf'] * 0.15) / (df['inst_w'] + df['inst_s'] + 8100)  # 火电+水电基荷8100万kW

# 月度聚合
mo = df.groupby(['year', 'month']).agg(
    cif=('cif', 'mean'), re_share=('re_share_proxy', 'mean')).reset_index()

# 理论关系: 1 - CIF/EF = re_share → 检验相关性
piv_cif = mo.pivot_table(index='month', columns='year', values='cif')
piv_re = mo.pivot_table(index='month', columns='year', values='re_share')
print('各年: corr(CIF, re_share_proxy) | CIF年均 | re_share年均')
for y in range(2019, 2025):
    c = piv_cif[y].values
    r = piv_re[y].values
    print(f'{y}: corr={np.corrcoef(c, r)[0,1]:+.3f} | CIF={c.mean():.0f} | RE={r.mean():.3f}')

# 全期月度面板回归: cif ~ re_share
X = mo['re_share'].values
yv = mo['cif'].values
from numpy.polynomial import polynomial as P
A = np.vstack([np.ones(len(X)), X]).T
coef, *_ = np.linalg.lstsq(A, yv, rcond=None)
pred = A @ coef
print()
print(f'面板回归: CIF = {coef[0]:.1f} + {coef[1]:.1f} * re_share, 月度MAE = {np.abs(pred-yv).mean():.1f}, R2={1-np.sum((pred-yv)**2)/np.sum((yv-yv.mean())**2):.3f}')

# 2023/2024单独
for yy in [2023, 2024]:
    mm = mo[mo['year'] == yy]
    pr = coef[0] + coef[1] * mm['re_share'].values
    print(f'{yy} 用面板回归: 月度MAE = {np.abs(pr - mm["cif"].values).mean():.1f}')

# 信号增量: 控制月历后 re_share 的偏相关
mo['doy_sin'] = np.sin(2*np.pi*mo['month']/12); mo['doy_cos'] = np.cos(2*np.pi*mo['month']/12)
# 残差化
for col in ['cif', 're_share']:
    Xc = np.vstack([np.ones(len(mo)), mo['doy_sin'], mo['doy_cos']]).T
    c2, *_ = np.linalg.lstsq(Xc, mo[col].values, rcond=None)
    mo[f'{col}_res'] = mo[col].values - Xc @ c2
print()
print('控制月历后: corr(cif_res, re_share_res) =', round(np.corrcoef(mo['cif_res'], mo['re_share_res'])[0,1], 3))
