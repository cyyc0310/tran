"""信息论上限检验：历史真值跨年同月可预测性
若连"用去年同月真值"都做不到 MAE<20，则月度数据的信息上限不支持20
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

piv = df.pivot_table(index='month', columns='year', values='cif', aggfunc='mean')

print('=== Naive: 用上一年同月真值直接预测 (信息上限测试) ===')
for yy in range(2020, 2025):
    err = piv[yy].values - piv[yy - 1].values
    print(f'{yy} (naive上年同月): MAE = {np.abs(err).mean():.1f}')

print()
print('=== Naive: 用前两年平均同月真值 ===')
for yy in range(2021, 2025):
    avg = (piv[yy-1] + piv[yy-2]) / 2
    err = piv[yy].values - avg.values
    print(f'{yy} (naive前两年均值): MAE = {np.abs(err).mean():.1f}')

print()
print('=== Naive: 用全部历史同月均值 ===')
for yy in range(2020, 2025):
    hist = piv[[y for y in range(2019, yy)]].mean(axis=1)
    err = piv[yy].values - hist.values
    print(f'{yy} (naive全历史均值): MAE = {np.abs(err).mean():.1f}')

print()
print('=== 上限参考: 乐观情形(水平+形态都naive上年) vs 各月离散 ===')
# 每月跨年std
for m in range(1, 13):
    vals = piv.loc[m].values
    print(f'{m:2d}月: 跨年std={vals.std(ddof=1):.1f} | 范围={vals.min():.0f}~{vals.max():.0f}')
print()
print('全部月份跨年std平均 =', round(piv.std(axis=1, ddof=1).mean(), 1))
