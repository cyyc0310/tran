"""两段式模型：归一化形态回归 × 融合水平（山西2023）
- 水平: 0.5*模型年均(618.9) + 0.5*2022披露年均(649.3) = 634.1  [已验证误差-7.8]
- 形态: 训练2019-2022归一化CIF(cif/年均) ~ HDD距平 + 春节对齐 + DOY谐波
目标: 月度 LEVEL MAE < 20
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

df['month'] = df['date'].dt.month
df['doy'] = df['date'].dt.dayofyear
df['hdd'] = np.clip(18 - df['t2m'], 0, None)

# HDD逐DOY气候态(2019-2022平均)
hdd_clim = df[df['year'] <= 2022].groupby('doy')['hdd'].mean()
df['hdd_anom'] = df['doy'].map(hdd_clim).fillna(df['hdd']) - df['hdd']  # 正=比常年暖, 负=比常年冷
df['hdd_anom7'] = df['hdd_anom'].rolling(7, min_periods=1).mean()

# 春节对齐偏移
df['sf_off'] = df.apply(lambda r: (r['date'] - sf[r['year']]).days, axis=1)

# DOY谐波
for k in [1, 2, 3]:
    df[f'sin{k}'] = np.sin(2 * np.pi * k * df['doy'] / 365.25)
    df[f'cos{k}'] = np.cos(2 * np.pi * k * df['doy'] / 365.25)
# SF窗哑变量(节前3天~节后25天, 分段)
df['sf_pre'] = ((df['sf_off'] >= -3) & (df['sf_off'] < 0)).astype(int)
df['sf_week1'] = ((df['sf_off'] >= 0) & (df['sf_off'] < 7)).astype(int)
df['sf_week2'] = ((df['sf_off'] >= 7) & (df['sf_off'] < 15)).astype(int)
df['sf_week3'] = ((df['sf_off'] >= 15) & (df['sf_off'] < 25)).astype(int)

# 归一化目标
ymean = df.groupby('year')['cif'].transform('mean')
df['cif_n'] = df['cif'] / ymean

FEATS = ['hdd_anom', 'hdd_anom7', 'sf_pre', 'sf_week1', 'sf_week2', 'sf_week3',
         'sin1', 'cos1', 'sin2', 'cos2', 'sin3', 'cos3']
X = df[FEATS].values
y = df['cif_n'].values

tr = (df['year'] <= 2022).values
te = (df['year'] == 2023).values
m = Ridge(alpha=5.0).fit(X[tr], y[tr])

LVL = 0.5 * 618.9 + 0.5 * 649.3  # 634.1
pred23 = m.predict(X[te]) * LVL
true23 = df.loc[te, 'cif'].values
dte = df[te].copy()
dte['pred'] = pred23

mo = dte.groupby('month').agg(true=('cif', 'mean'), pred=('pred', 'mean'))
mo['err'] = mo['pred'] - mo['true']
print('=== 两段式: 形态回归 x 融合水平 ===')
print('月度 | 预测 | 真值 | 误差')
for m_, r in mo.iterrows():
    print(f"{m_:2d} | {r['pred']:.1f} | {r['true']:.1f} | {r['err']:+.1f}")
print('月均 LEVEL MAE:', round(mo['err'].abs().mean(), 1))
print('日级 MAE:', round(np.abs(pred23 - true23).mean(), 1))
print('日级 Spearman:', round(pd.Series(pred23).corr(pd.Series(true23), method='spearman'), 3))

print()
print('系数:', dict(zip(FEATS, np.round(m.coef_, 4))))

# 对照: 当前最优 35.3
piv = df.pivot_table(index='month', columns='year', values='cif', aggfunc='mean')
clim = piv[[2019, 2020, 2021, 2022]].mean(axis=1)
clim_shape = (clim / clim.mean()).values
base = clim_shape * LVL
print()
print('对照-当前最优:', round(np.abs(base - piv[2023].values).mean(), 1))

# 上限: 若形态用"各训练年单独"对2023的表现
print()
print('--- 灵敏度: 只用HDD距平(无春节特征) ---')
F2 = ['hdd_anom', 'hdd_anom7', 'sin1', 'cos1', 'sin2', 'cos2', 'sin3', 'cos3']
m2 = Ridge(alpha=5.0).fit(df[F2].values[tr], y[tr])
p2 = m2.predict(df[F2].values[te]) * LVL
d2 = dte.copy(); d2['pred'] = p2
mo2 = d2.groupby('month').agg(true=('cif', 'mean'), pred=('pred', 'mean'))
print('月均 LEVEL MAE:', round((mo2['pred'] - mo2['true']).abs().mean(), 1))

print()
print('--- 灵敏度: 只用春节特征(无HDD) ---')
F3 = ['sf_pre', 'sf_week1', 'sf_week2', 'sf_week3', 'sin1', 'cos1', 'sin2', 'cos2', 'sin3', 'cos3']
m3 = Ridge(alpha=5.0).fit(df[F3].values[tr], y[tr])
p3 = m3.predict(df[F3].values[te]) * LVL
d3 = dte.copy(); d3['pred'] = p3
mo3 = d3.groupby('month').agg(true=('cif', 'mean'), pred=('pred', 'mean'))
print('月均 LEVEL MAE:', round((mo3['pred'] - mo3['true']).abs().mean(), 1))
