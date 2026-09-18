"""上海 MAE<20 验证: 真实月度分母(规上口径) + 春节感知月度模型
真值: CarbonMonitor Power 排放 / 规上工业月度发电量 (发电侧主口径)
模型: climatology 形态 × 融合水平 + Feb/Jan 块拆分 (与山西获胜方法一致)
输出 2023/2024 MAE / Spearman, 2021 异常年诊断, 敏感性分析
"""
import pandas as pd
import numpy as np

# ============ 上海规上月度累计发电量 (亿kWh, 索引0=2月末 ... 10=12月末) ============
cum_sh = {
    2020: [143.1, 203.7, 262.1, 323.2, 387.5, 450.5, 544.9, 604.1, 651.5, 715.9, 819.0],
    2021: [167.0, 242.9, 312.6, 382.3, 463.3, 552.4, 646.7, 717.9, 784.1, 857.6, 956.8],
    2022: [177.1, 250.3, 292.0, 333.4, 395.5, 499.1, 611.2, 686.7, 747.4, 810.2, 901.2],
    2023: [145.5, 228.9, 300.2, 367.0, 448.5, 558.1, 655.8, 725.6, 790.9, 859.5, 954.9],
    2024: [180.8, 268.5, 337.6, 401.6, 458.8, 562.9, 681.0, 781.7, 849.6, 921.9, 1017.1],
}
cny_doy = {2019: 36, 2020: 25, 2021: 43, 2022: 32, 2023: 22, 2024: 41}

# ============ CarbonMonitor 上海 Power 排放 ============
cm_df = pd.read_csv('data_2023/carbonmonitor_cn/carbonmonitor_china_provinces.csv')
d = cm_df[(cm_df['sector'] == 'Power') & (cm_df['state'] == 'Shanghai')].copy()
d['date'] = pd.to_datetime(d['date'], format='%d/%m/%Y')
d['year'] = d['date'].dt.year
d['month'] = d['date'].dt.month
d = d[d['year'].between(2019, 2024)]
mo_emis = d.groupby(['year', 'month'])['value'].mean()  # Mt/day
days_in = lambda y, m: [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]


def monthly_gen(cum_map, y):
    c = cum_map[y]
    d1, d2 = 31, (29 if y % 4 == 0 else 28)
    jan = c[0] * d1 / (d1 + d2)
    out = [jan, c[0] - jan]
    for i in range(1, 11):
        out.append(c[i] - c[i - 1])
    return out


# ============ 真值月度 CIF (发电侧, 规上分母) ============
true_sh = {}
for y in range(2020, 2025):
    g = monthly_gen(cum_sh, y)
    true_sh[y] = [mo_emis[(y, m)] * days_in(y, m) * 1e12 / (g[m - 1] * 1e8) for m in range(1, 13)]

print('上海真值月度 CIF (g/kWh, 发电侧/规上分母):')
for y in range(2020, 2025):
    row = ' '.join(f'{v:.0f}' for v in true_sh[y])
    print(f'  {y}: {row}  (年均 {np.mean(true_sh[y]):.0f})')

print('\nFeb/Jan 真值比值 vs 春节DOY:')
for y in range(2020, 2025):
    r = true_sh[y][1] / true_sh[y][0]
    print(f'  {y}: ratio={r:.3f}  CNY_doy={cny_doy[y]}')

# ============ 2021 异常诊断 ============
print('\n2021 异常诊断 (vs 2020/2022 形态):')
for m in range(12):
    ref = (true_sh[2020][m] + true_sh[2022][m]) / 2
    print(f'  {m+1}月: 2021={true_sh[2021][m]:.0f}  2020/22均值={ref:.0f}  偏差={true_sh[2021][m]-ref:+.0f}')

# ============ 模型 ============
# 发电侧模型年均: SHANGHAI_GEN_2023 (KPI 全口径结构) + CM-implied EF
SH_COAL = [56, 52, 54, 50, 48, 46, 47, 46, 49, 52, 55, 58]
SH_GAS = [31, 29, 30, 28, 27, 29, 32, 32, 29, 27, 29, 31]
SH_PETRO = [0.4] * 12
SH_SOLAR = [0.25, 0.3, 0.4, 0.5, 0.55, 0.57, 0.55, 0.5, 0.4, 0.35, 0.27, 0.23]
SH_WIND = [1.9, 2.0, 2.2, 2.3, 2.1, 1.9, 1.7, 1.7, 1.9, 2.1, 2.1, 2.0]
EF = {'coal': 448.0, 'gas': 490.0, 'petroleum': 720.0}
num = sum(c * EF['coal'] for c in SH_COAL) + sum(g * EF['gas'] for g in SH_GAS) \
      + sum(p * EF['petroleum'] for p in SH_PETRO)
den = sum(SH_COAL) + sum(SH_GAS) + sum(SH_PETRO) + sum(SH_SOLAR) + sum(SH_WIND)
MODEL_LVL_SH = num / den
print(f'\n发电侧模型年均 (KPI结构×CM-implied EF): {MODEL_LVL_SH:.1f} g/kWh')

piv = pd.DataFrame(true_sh, index=range(1, 13))


def fit_ratio(train_years, tc):
    xs = np.array([cny_doy[y] for y in train_years])
    ys = np.array([tc[y][1] / tc[y][0] for y in train_years])
    A = np.vstack([np.ones(len(xs)), xs]).T
    return np.linalg.lstsq(A, ys, rcond=None)[0]


def predict(target, train_years, tc, lvl_model, lvl_last, use_cny=True):
    coef = fit_ratio(train_years, tc)
    ratio = np.clip(coef[0] + coef[1] * cny_doy[target], 0.7, 1.4)
    lvl = 0.5 * lvl_model + 0.5 * lvl_last
    cm = piv[[c for c in train_years if c in piv.columns]].mean(axis=1)
    s = (cm / cm.mean()).values
    pred = s * lvl
    if use_cny:
        block = (pred[0] + pred[1]) / 2
        pred[1] = block * 2 * ratio / (1 + ratio)
        pred[0] = block * 2 / (1 + ratio)
    return pred


def report(name, pred, truth):
    err = np.array(pred) - np.array(truth)
    mae = np.abs(err).mean()
    sp = pd.Series(pred).corr(pd.Series(truth), method='spearman')
    print(f'  {name}: LEVEL MAE = {mae:.1f} | Spearman = {sp:.3f} | 年均偏差 = {np.mean(pred)-np.mean(truth):+.1f} | 拟合比值 = {pred[1]/pred[0]:.3f} vs 真值 {truth[1]/truth[0]:.3f}')
    return mae, sp


print('\n================ 上海 2023 (训练 2020-2022, 含2021) ================')
p = predict(2023, [2020, 2021, 2022], true_sh, MODEL_LVL_SH, np.mean(true_sh[2022]))
report('2023 春节感知', p, true_sh[2023])
p_nc = predict(2023, [2020, 2021, 2022], true_sh, MODEL_LVL_SH, np.mean(true_sh[2022]), use_cny=False)
report('2023 无拆分对照', p_nc, true_sh[2023])

print('\n================ 上海 2023 (训练剔除2021) ================')
p2 = predict(2023, [2020, 2022], true_sh, MODEL_LVL_SH, np.mean(true_sh[2022]))
report('2023 春节感知', p2, true_sh[2023])

print('\n================ 上海 2024 (训练 2020-2023, 含2021) ================')
p = predict(2024, [2020, 2021, 2022, 2023], true_sh, MODEL_LVL_SH, np.mean(true_sh[2023]))
report('2024 春节感知', p, true_sh[2024])

print('\n================ 上海 2024 (训练剔除2021) ================')
p2 = predict(2024, [2020, 2022, 2023], true_sh, MODEL_LVL_SH, np.mean(true_sh[2023]))
report('2024 春节感知', p2, true_sh[2024])

# ============ 最优配置(含2021)的敏感性 ============
print('\n================ 水平敏感性 (含2021训练) ================')
for lm in [MODEL_LVL_SH - 20, MODEL_LVL_SH, MODEL_LVL_SH + 20]:
    p = predict(2023, [2020, 2021, 2022], true_sh, lm, np.mean(true_sh[2022]))
    m1 = np.abs(np.array(p) - np.array(true_sh[2023])).mean()
    p = predict(2024, [2020, 2021, 2022, 2023], true_sh, lm, np.mean(true_sh[2023]))
    m2 = np.abs(np.array(p) - np.array(true_sh[2024])).mean()
    print(f'  模型年均={lm:.0f}: 2023 MAE={m1:.1f} | 2024 MAE={m2:.1f}')

# ============ 逐月误差明细 (最优配置) ============
print('\n================ 逐月明细 (含2021训练) ================')
p23 = predict(2023, [2020, 2021, 2022], true_sh, MODEL_LVL_SH, np.mean(true_sh[2022]))
p24 = predict(2024, [2020, 2021, 2022, 2023], true_sh, MODEL_LVL_SH, np.mean(true_sh[2023]))
for m in range(12):
    print(f'  {m+1}月: 2023 pred={p23[m]:.0f} true={true_sh[2023][m]:.0f} err={p23[m]-true_sh[2023][m]:+.0f} | 2024 pred={p24[m]:.0f} true={true_sh[2024][m]:.0f} err={p24[m]-true_sh[2024][m]:+.0f}')
