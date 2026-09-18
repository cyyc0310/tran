"""春节错位对山西CIF形态的影响分析"""
import pandas as pd
import numpy as np

df = pd.read_csv('data_2023/carbonmonitor_cn/carbonmonitor_china_provinces.csv')
df = df[(df['sector'] == 'Power') & (df['state'] == 'Shanxi')].copy()
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month
gens = {2019: 3867, 2020: 4051, 2021: 4178, 2022: 4178, 2023: 4376.1}
df = df[df['year'].isin(gens)]
df['cif'] = df.apply(lambda r: r['value'] * 1e12 / (gens[r['year']] * 1e8 / 365), axis=1)

sf = {
    2019: pd.Timestamp('2019-02-05'),
    2020: pd.Timestamp('2020-01-25'),
    2021: pd.Timestamp('2021-02-12'),
    2022: pd.Timestamp('2022-02-01'),
    2023: pd.Timestamp('2023-01-22'),
}

print('年份 | 春节前20天 | 春节周 | 节后8-40天 | 全年日均')
for y in range(2019, 2024):
    d = df[df['year'] == y].set_index('date')['cif']
    base = sf[y]
    pre = d[base - pd.Timedelta(days=20): base - pd.Timedelta(days=1)].mean()
    wk = d[base: base + pd.Timedelta(days=7)].mean()
    post = d[base + pd.Timedelta(days=8): base + pd.Timedelta(days=40)].mean()
    print(f'{y} | {pre:.0f} | {wk:.0f} | {post:.0f} | {d.mean():.0f}')

print()
print('各年1/2/3月均值:')
for y in range(2019, 2024):
    jan = df[(df['year'] == y) & (df['month'] == 1)]['cif'].mean()
    feb = df[(df['year'] == y) & (df['month'] == 2)]['cif'].mean()
    mar = df[(df['year'] == y) & (df['month'] == 3)]['cif'].mean()
    print(f'{y}: 1月={jan:.0f} 2月={feb:.0f} 3月={mar:.0f}')

# 以春节对齐的60天窗口形态(归一化)
print()
print('以春节对齐的形态(-20~+40天, 按各年日均归一):')
d23 = df[df['year'] == 2023].set_index('date')['cif']
base23 = sf[2023]
days = pd.date_range(base23 - pd.Timedelta(days=20), base23 + pd.Timedelta(days=40))
v23 = np.array([d23.get(x, np.nan) for x in days])
v23n = v23 / np.nanmean(v23)
for y in range(2019, 2023):
    d = df[df['year'] == y].set_index('date')['cif']
    base = sf[y]
    v = np.array([d.get(x, np.nan) for x in pd.date_range(base - pd.Timedelta(days=20), base + pd.Timedelta(days=40))])
    vn = v / np.nanmean(v)
    m = ~np.isnan(vn) & ~np.isnan(v23n)
    print(f'{y} vs 2023春节窗口形态: corr={np.corrcoef(vn[m], v23n[m])[0,1]:.3f} | MAE={np.abs(vn[m]-v23n[m]).mean()*np.nanmean(v23):.1f}')
