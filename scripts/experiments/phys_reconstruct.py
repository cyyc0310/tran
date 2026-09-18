"""分月物理重构：CIF = EF_coal * coal_share + EF_gas*gas_share + 水电/核电稀释
全零遥测公开输入: 装机(公开) + 再分析天气(公开) + EF(公开)
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
t = pd.read_csv('data_2023/carbonmonitor_cn/taiyuan_daily_temp.csv')
t['date'] = pd.to_datetime(t['date'])
df = df.merge(w[['date', 'wind_speed_10m_max', 'shortwave_radiation_sum']], on='date', how='left')
df = df.merge(t[['date', 't2m']], on='date', how='left')
df = df.rename(columns={'wind_speed_10m_max': 'wsm', 'shortwave_radiation_sum': 'rad'})

inst = {
    2018: (1000.0, 700.0),
    2019: (1139.2, 825.0), 2020: (1416.1, 1316.0), 2021: (2123.0, 1635.0),
    2022: (2491.0, 1920.0), 2023: (2491.0, 2810.0), 2024: (2616.5, 3476.8),
}
hydro_coal = {  # (水电, 煤电)万kW 年末装机
    2018: (180.0, 7200.0), 2019: (180.0, 7380.0), 2020: (225.0, 7570.0),
    2021: (225.0, 7700.0), 2022: (225.0, 7900.0), 2023: (225.0, 8090.0), 2024: (225.6, 8197.7),
}

def daily_inst(row):
    y = row['year']
    w0, s0 = inst[y - 1]
    w1, s1 = inst[y]
    h0, c0 = hydro_coal[y - 1]
    h1, c1 = hydro_coal[y]
    doy = row['date'].timetuple().tm_yday
    days = 366 if (y % 4 == 0) else 365
    frac = doy / days
    return pd.Series({
        'inst_w': w0 + (w1 - w0) * frac, 'inst_s': s0 + (s1 - s0) * frac,
        'inst_h': h0 + (h1 - h0) * frac, 'inst_c': c0 + (c1 - c0) * frac})

df[['inst_w', 'inst_s', 'inst_h', 'inst_c']] = df.apply(daily_inst, axis=1)

# 物理重构(日度):
# 风电出力 = inst_w * CF_w(v), CF_w(v) = clip((v/v_rat)^3, 0, 1), v_rat≈9m/s, cutin 3
v = df['wsm'].astype(float)
cf_w = np.clip(((v - 3) / (9 - 3)) ** 3, 0, 1) * 0.9 + 0.05
# 光伏出力 = inst_s * CF_s(rad), CF_s = rad/日晴空峰值(≈25 MJ/m2) * 0.85
cf_s = np.clip(df['rad'] / 25.0 * 0.85, 0, 0.35)
# 水电: 山西水电小, 按年均CF 0.4
gen_w = df['inst_w'] * cf_w
gen_s = df['inst_s'] * cf_s
gen_h = df['inst_h'] * 0.4
# 煤电 = 总发电 - 风光水 (山西总发电月度公开, 但日度不可得 → 用装机负荷率回归)
# 部署口径: 总发电量月度事后可得但事前不可得 → 用煤电装机 * 负荷率(待估)
# 这里先检验信息上限: 用真值总发电量(作弊上限)
total_gen = gens  # 日度近似 uniform
df['gen_total'] = df['year'].map(gens) * 1e8 / 365 / 24  # kW(日均匀)
gen_re = (gen_w + gen_s + gen_h) / 10  # 万kW→kW? 单位对齐: 万kW = 1e4 kW
gen_w_kw = df['inst_w'] * 1e4 * cf_w
gen_s_kw = df['inst_s'] * 1e4 * cf_s
gen_h_kw = df['inst_h'] * 1e4 * 0.4
re_frac = (gen_w_kw + gen_s_kw + gen_h_kw) / df['gen_total']

EF_coal = 0.875  # gCO2/kWh? 山西煤电EF≈758 g/kWh(CM真值隐含) → 用公开指南0.875? 先用758
for EF in [758, 850]:
    cif_phys = EF * (1 - re_frac)
    break
df['cif_phys'] = 758 * (1 - re_frac)
mo = df.groupby(['year', 'month']).agg(true=('cif', 'mean'), phys=('cif_phys', 'mean')).reset_index()
for yy in [2023, 2024]:
    mm = mo[mo['year'] == yy]
    print(f'{yy} 物理重构(真值总量上限): 月度MAE = {np.abs(mm["phys"] - mm["true"]).mean():.1f} | 相关 = {np.corrcoef(mm["phys"], mm["true"])[0,1]:.3f}')
    for _, r in mm.iterrows():
        print(f"  {int(r['month']):2d}月: 重构{r['phys']:.0f} vs 真值{r['true']:.0f} ({r['phys']-r['true']:+.0f})")
# 检查重构形态与真值形态
piv_p = mo.pivot_table(index='month', columns='year', values='phys')
piv_t = mo.pivot_table(index='month', columns='year', values='true')
for yy in [2023, 2024]:
    sp = piv_p[yy] / piv_p[yy].mean()
    st = piv_t[yy] / piv_t[yy].mean()
    print(f'{yy} 形态相关: {np.corrcoef(sp, st)[0,1]:.3f}, 形态MAE = {np.abs(sp*st.mean()-st).mean()*100:.1f} (相对%)')
