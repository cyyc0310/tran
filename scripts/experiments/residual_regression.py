"""残差回归：温度HDD + 春节对齐日历 → 逐日CIF残差模型（山西）
目标：把 LEVEL MAE 从 35 压到 <20
"""
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

df = pd.read_csv('data_2023/carbonmonitor_cn/carbonmonitor_china_provinces.csv')
df = df[(df['sector'] == 'Power') & (df['state'] == 'Shanxi')].copy()
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['year'] = df['date'].dt.year
gens = {2019: 3867, 2020: 4051, 2021: 4178, 2022: 4178, 2023: 4376.1}
df = df[df['year'].isin(gens)]
df['cif'] = df.apply(lambda r: r['value'] * 1e12 / (gens[r['year']] * 1e8 / 365), axis=1)

t = pd.read_csv('data_2023/carbonmonitor_cn/taiyuan_daily_temp.csv')
t['date'] = pd.to_datetime(t['date'])
df = df.merge(t, on='date', how='left')

sf = {
    2019: pd.Timestamp('2019-02-05'),
    2020: pd.Timestamp('2020-01-25'),
    2021: pd.Timestamp('2021-02-12'),
    2022: pd.Timestamp('2022-02-01'),
    2023: pd.Timestamp('2023-01-22'),
}

# 特征：HDD(18°C基准)、HDD滞后、春节窗、DOY正弦对
df['hdd'] = np.clip(18 - df['t2m'], 0, None)
df['hdd_lag3'] = df['hdd'].rolling(3, min_periods=1).mean()
df['hdd_lag7'] = df['hdd'].rolling(7, min_periods=1).mean()
df['sf_offset'] = df.apply(lambda r: (r['date'] - sf[r['year']]).days, axis=1)
# 春节窗哑变量(节前7~节后20)
df['sf_win'] = ((df['sf_offset'] >= -7) & (df['sf_offset'] <= 20)).astype(int)
df['sf_dow'] = df['date'].dt.dayofweek
df['doy'] = df['date'].dt.dayofyear
df['sin1'] = np.sin(2 * np.pi * df['doy'] / 365.25)
df['cos1'] = np.cos(2 * np.pi * df['doy'] / 365.25)
df['sin2'] = np.sin(4 * np.pi * df['doy'] / 365.25)
df['cos2'] = np.cos(4 * np.pi * df['doy'] / 365.25)

FEATS = ['hdd', 'hdd_lag3', 'hdd_lag7', 'sf_win', 'sin1', 'cos1', 'sin2', 'cos2']
X = df[FEATS].values
y = df['cif'].values

# 逐年留一交叉验证（LOYO）：训练2019-2022中四年以外的三年，测2023重点看
# 更直接：训练2019-2022全部，测2023（这是部署场景）
tr = df['year'] <= 2022
te = df['year'] == 2023
m = Ridge(alpha=10.0).fit(X[tr], y[tr])
pred23 = m.predict(X[te])
true23 = y[te]
err = pred23 - true23
print('=== 残差回归直接预测2023（训练2019-2022）===')
print('日级 MAE:', round(np.abs(err).mean(), 1))
# 月度聚合
dte = df[te].copy(); dte['pred'] = pred23
mo = dte.groupby(dte['date'].dt.month).agg(true=('cif', 'mean'), pred=('pred', 'mean'))
mo['err'] = mo['pred'] - mo['true']
print('月度 | 预测 | 真值 | 误差')
for m_, r in mo.iterrows():
    print(f"{m_:2d} | {r['pred']:.1f} | {r['true']:.1f} | {r['err']:+.1f}")
mae_m = mo['err'].abs().mean()
print('月均 MAE (LEVEL):', round(mae_m, 1))
sp = pd.Series(pred23).corr(dte.reset_index(drop=True)['cif'], method='spearman')
print('日级 Spearman vs 真值:', round(sp, 3))

# 系数解读
print()
print('系数:', dict(zip(FEATS, np.round(m.coef_, 2))))

# 对照: 纯月历climatology + 融合水平 (当前最优35.3)
piv = df.pivot_table(index='month', columns='year', values='cif', aggfunc='mean')
clim = piv[[2019, 2020, 2021, 2022]].mean(axis=1)
clim_shape = (clim / clim.mean()).values
fuse = 0.5 * 618.9 + 0.5 * 649.3
base_pred = clim_shape * fuse
base_err = base_pred - piv[2023].values
print()
print('对照-当前最优(月历形态x融合水平): 月均MAE =', round(np.abs(base_err).mean(), 1))

# 混合方案: 回归残差修正 climatology 基线
# 基线 = clim形态 x 2023水平=真值年均(理想化) → 看形态上限
print()
print('各月真值 vs 4年clim形态(理想水平):')
ideal = clim_shape * piv[2023].mean()
print('形态上限MAE(水平完美):', round(np.abs(ideal.values - piv[2023].values).mean(), 1))
