"""春节对齐气候态 (CNY-aligned climatology)
思路: 把每年日历按"距春节天数"重排, 在春节轴上构建日度归一化CIF气候态
再映射回目标年月历 → 预测月度CIF
检验: 2019-2022训练 → 2023/2024评估, 目标 LEVEL MAE < 20
"""
import pandas as pd
import numpy as np

sf = {
    2019: pd.Timestamp('2019-02-05'), 2020: pd.Timestamp('2020-01-25'),
    2021: pd.Timestamp('2021-02-12'), 2022: pd.Timestamp('2022-02-01'),
    2023: pd.Timestamp('2023-01-22'), 2024: pd.Timestamp('2024-02-10'),
}

df = pd.read_csv('data_2023/carbonmonitor_cn/carbonmonitor_china_provinces.csv')
df = df[(df['sector'] == 'Power') & (df['state'] == 'Shanxi')].copy()
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
gens = {2019: 3867, 2020: 4051, 2021: 4178, 2022: 4178, 2023: 4376.1, 2024: 4516.11}
df = df[df['year'].isin(gens)]
df['cif'] = df.apply(lambda r: r['value'] * 1e12 / (gens[r['year']] * 1e8 / 365), axis=1)
df['cif_n'] = df['cif'] / df.groupby('year')['cif'].transform('mean')
df['sf_off'] = df.apply(lambda r: (r['date'] - sf[r['year']]).days, axis=1)


def cny_clim(train_years, window=9):
    """春节轴上的平滑气候态: sf_off → 期望cif_n"""
    d = df[df['year'].isin(train_years)]
    g = d.groupby('sf_off')['cif_n'].agg(['mean', 'count'])
    # 平滑(移动平均)
    g = g.reindex(range(-70, 330), fill_value=np.nan)
    g['mean'] = g['mean'].rolling(window, center=True, min_periods=3).mean()
    # 覆盖不到的区域(远端)用季节DOY气候态填充
    return g['mean']


def doy_clim(train_years, window=15):
    d = df[df['year'].isin(train_years)]
    d = d.copy()
    d['doy'] = d['date'].dt.dayofyear
    g = d.groupby('doy')['cif_n'].agg(['mean'])
    g = g.reindex(range(1, 367), fill_value=np.nan)
    g['mean'] = g['mean'].rolling(window, center=True, min_periods=5).mean()
    g.loc[366] = g.loc[365]
    return g['mean']


def predict(target_year, train_years, lvl):
    gc = cny_clim(train_years)
    gd = doy_clim(train_years)
    d = df[df['year'] == target_year].copy()
    # 春节窗口(-70~+100)用CNY轴, 其余用DOY轴, 边界线性混合
    preds = []
    for _, r in d.iterrows():
        off = r['sf_off']
        vc = gc.get(off, np.nan)
        doy = r['date'].dayofyear
        vd = gd.get(doy, np.nan)
        # 混合权重: CNY轴在±60天内权重1, 之外线性衰减到0(至±100)
        w = 1.0 if abs(off) <= 60 else max(0.0, 1 - (abs(off) - 60) / 40)
        if np.isnan(vc):
            w = 0.0
        if np.isnan(vd):
            w = 1.0
            vd = vc
        v = w * vc + (1 - w) * vd
        preds.append(v)
    d['pred_n'] = preds
    d['pred'] = d['pred_n'] * lvl
    mo = d.groupby('month').agg(true=('cif', 'mean'), pred=('pred', 'mean'))
    return mo, d


print('=== 2023 (训练2019-2022, 水平=融合634.1) ===')
mo23, _ = predict(2023, [2019, 2020, 2021, 2022], 634.1)
for m, r in mo23.iterrows():
    print(f"{m:2d} | 预测{r['pred']:.1f} | 真值{r['true']:.1f} | {r['pred']-r['true']:+.1f}")
print('月均 LEVEL MAE:', round((mo23['pred'] - mo23['true']).abs().mean(), 1))

print()
print('=== 2024 (训练2019-2023, 水平=0.5*618.9+0.5*641.9=630.4) ===')
mo24, _ = predict(2024, [2019, 2020, 2021, 2022, 2023], 630.4)
for m, r in mo24.iterrows():
    print(f"{m:2d} | 预测{r['pred']:.1f} | 真值{r['true']:.1f} | {r['pred']-r['true']:+.1f}")
print('月均 LEVEL MAE:', round((mo24['pred'] - mo24['true']).abs().mean(), 1))

print()
print('=== LOYO 交叉验证: 每年用其余年训练, 水平=真值年均(隔离形态误差) ===')
for yy in range(2019, 2025):
    tr = [y for y in range(2019, 2025) if y != yy]
    d = df[df['year'] == yy].copy()
    gc = cny_clim(tr)
    gd = doy_clim(tr)
    preds = []
    for _, r in d.iterrows():
        off = r['sf_off']
        vc = gc.get(off, np.nan)
        vd = gd.get(r['date'].dayofyear, np.nan)
        w = 1.0 if abs(off) <= 60 else max(0.0, 1 - (abs(off) - 60) / 40)
        if np.isnan(vc):
            w = 0.0
        if np.isnan(vd):
            w = 1.0
            vd = vc
        preds.append((w * vc + (1 - w) * vd) * d['cif'].mean())
    d['pred'] = preds
    mo = d.groupby('month').agg(true=('cif', 'mean'), pred=('pred', 'mean'))
    print(f'{yy}: 形态MAE(水平完美) = {round((mo["pred"]-mo["true"]).abs().mean(),1)}')
