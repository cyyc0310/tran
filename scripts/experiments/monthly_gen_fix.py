"""用真实月度发电量重算山西真值CIF并复测全部方法
关键口径修正: 之前用均匀分母(年/365), 2月+28%异常部分是分母假象
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
# 真值月度排放均值(kt CO2/day)
mo_emis = df.groupby(['year', 'month'])['value'].mean().reset_index()

# 山西规上工业月度发电量(亿kWh) - 国家统计局/中商产业研究院
# 2023: [1月375.8, 2月351.3, 3月373.9, 4月319.8, 5月316.4, 6月357.2, 7月402.6, 8月396.9, 9月330.7, 10月324.7, 11月367.5, 12月443.3]
# 2024: [1月380.1?, 2月363?, 3月363.1, 4月313.5, 5月313.8, 6月353.8, 7月390, 8月381.6, 9月328, 10月326, 11月374.2, 12月430.7]
gen23 = [375.8, 351.3, 373.9, 319.8, 316.4, 357.2, 402.6, 396.9, 330.7, 324.7, 367.5, 443.3]
# 2024年1-2月合并782.5-764.4? 中商: 2024年02月累计782.5, 2024年01月累计380.1? 用差分
# 2024累计: 1月 380.1, 2月 782.5, 3月 1144, 4月 1457.6, 5月 1771, 6月 2124.8, 7月 2515.1, 8月 2898.7, 9月 3233, 10月 3574.1, 11月 3954.2, 12月 4386.2
cum24 = [380.1, 782.5, 1144.0, 1457.6, 1771.0, 2124.8, 2515.1, 2898.7, 3233.0, 3574.1, 3954.2, 4386.2]
gen24 = [cum24[0]] + [cum24[i] - cum24[i-1] for i in range(1, 12)]
# 2023累计: 1月 375.8?, 2月 727.1, 3月 1102.9, ..., 12月 4376.1
cum23 = [375.8, 727.1, 1102.9, 1423.5, 1739.8, 2097.1, 2500.0, 2898.7, 3229.8, 3559.0, 3930.9, 4376.1]
gen23b = [cum23[0]] + [cum23[i] - cum23[i-1] for i in range(1, 12)]
print('2023 差分核对:', [round(x, 1) for x in gen23b])
print('2024 差分:', [round(x, 1) for x in gen24])

days = {1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30, 7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31}
# 2024是闰年
days24 = days.copy(); days24[2] = 29

for label, y, g in [('2023', 2023, gen23b), ('2024', 2024, gen24)]:
    dd = days24 if y == 2024 else days
    sub = mo_emis[mo_emis['year'] == y].sort_values('month')
    cif = [sub[sub['month'] == m]['value'].values[0] * 1e3 * dd[m] / (g[m-1] * 1e8 / 1e5) for m in range(1, 13)]
    # kt CO2/day * days = kt CO2/month; g/kWh = kt*1e9 g / (亿kWh*1e8 Wh/1000)  → kt*1e9/(亿*1e5 kWh)
    cif = [sub[sub['month'] == m]['value'].values[0] * dd[m] * 1e9 / (g[m-1] * 1e5) for m in range(1, 13)]
    print()
    print(f'{label} 真实月度CIF (月排放/月发电):')
    for m in range(1, 13):
        print(f'{m:2d}月: {cif[m-1]:.0f}')
    print('年均:', round(np.mean(cif), 1))
    globals()[f'cif_{y}'] = cif
