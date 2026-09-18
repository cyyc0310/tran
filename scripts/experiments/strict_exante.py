"""严格事前原则复检 (ex-ante strict audit)

背景: 获胜评估中 lvl_model 的热力 EF 原型阶段由 CM 真值反推
  (山西 coal 760 = 2023 CM-implied; 上海 coal 448 = 2023 CM-implied)
  部署叙事中应为「上年官方披露 EF」。2024 评估天然干净 (2023=上年),
  但 2023 评估存在同年反推。本脚本用上年 (2022) implied EF 重建
  lvl_model, 重算 2023 MAE, 检验 <20 结论在严格事前口径下是否成立。
"""
import pandas as pd
import numpy as np

cm = pd.read_csv('data_2023/carbonmonitor_cn/carbonmonitor_china_provinces.csv')
cm['date'] = pd.to_datetime(cm['date'], format='%d/%m/%Y')
cm['year'] = cm['date'].dt.year
cm['month'] = cm['date'].dt.month


def emis_mt(prov, year):
    d = cm[(cm.sector == 'Power') & (cm.state == prov) & (cm.year == year)]
    return float(d['value'].sum())


def mo_emis(prov, y, m):
    d = cm[(cm.sector == 'Power') & (cm.state == prov) & (cm.year == y) & (cm.month == m)]
    return float(d['value'].mean())


days_in = lambda y, m: [31, 29 if y % 4 == 0 else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1]
cny_doy = {2019: 36, 2020: 25, 2021: 43, 2022: 32, 2023: 22, 2024: 41}


def monthly_gen(c):
    d1, d2 = 31, (29 if 2024 % 4 == 0 else 28)
    return c  # placeholder


def split_cum(cum_map, y):
    c = cum_map[y]
    d1, d2 = 31, (29 if y % 4 == 0 else 28)
    jan = c[0] * d1 / (d1 + d2)
    out = [jan, c[0] - jan]
    for i in range(1, 11):
        out.append(c[i] - c[i - 1])
    return out


# ============ 真值 (与获胜评估完全一致) ============
cum_sx = {
    2019: [517.1, 772.5, 1024.8, 1287.6, 1560.0, 1862.6, 2151.7, 2402.7, 2647.2, 2922.6, 3238.0],
    2020: [494.0, 765.6, 1017.7, 1272.7, 1540.6, 1842.1, 2137.4, 2409.9, 2685.8, 3024.5, 3366.9],
    2021: [611.5, 927.8, 1199.6, 1466.4, 1782.9, 2117.5, 2456.8, 2747.6, 3047.7, 3370.5, 3734.4],
    2022: [663.7, 1009.2, 1297.4, 1598.5, 1939.2, 2321.6, 2718.0, 3062.5, 3391.7, 3736.5, 4153.3],
    2023: [727.1, 1102.9, 1423.5, 1739.8, 2097.1, 2500.0, 2898.7, 3229.8, 3559.0, 3930.9, 4376.1],
    2024: [782.5, 1144.0, 1457.6, 1771.0, 2124.8, 2515.1, 2898.7, 3233.0, 3574.1, 3954.2, 4386.2],
}
cum_sh = {
    2020: [143.1, 203.7, 262.1, 323.2, 387.5, 450.5, 544.9, 604.1, 651.5, 715.9, 819.0],
    2021: [167.0, 242.9, 312.6, 382.3, 463.3, 552.4, 646.7, 717.9, 784.1, 857.6, 956.8],
    2022: [177.1, 250.3, 292.0, 333.4, 395.5, 499.1, 611.2, 686.7, 747.4, 810.2, 901.2],
    2023: [145.5, 228.9, 300.2, 367.0, 448.5, 558.1, 655.8, 725.6, 790.9, 859.5, 954.9],
    2024: [180.8, 268.5, 337.6, 401.6, 458.8, 562.9, 681.0, 781.7, 849.6, 921.9, 1017.1],
}


def build_truth(prov, cum_map, years):
    tc = {}
    for y in years:
        g = split_cum(cum_map, y)
        tc[y] = [mo_emis(prov, y, m) * days_in(y, m) * 1e12 / (g[m - 1] * 1e8) for m in range(1, 13)]
    return tc


def predict(target, train_years, tc, lvl_model, lvl_last):
    xs = np.array([cny_doy[y] for y in train_years])
    ys = np.array([tc[y][1] / tc[y][0] for y in train_years])
    A = np.vstack([np.ones(len(xs)), xs]).T
    coef = np.linalg.lstsq(A, ys, rcond=None)[0]
    ratio = np.clip(coef[0] + coef[1] * cny_doy[target], 0.7, 1.4)
    lvl = 0.5 * lvl_model + 0.5 * lvl_last
    piv = pd.DataFrame(tc, index=range(1, 13))
    cm_ = piv[[c for c in train_years if c in piv.columns]].mean(axis=1)
    s = (cm_ / cm_.mean()).values
    pred = s * lvl
    block = (pred[0] + pred[1]) / 2
    pred[1] = block * 2 * ratio / (1 + ratio)
    pred[0] = block * 2 / (1 + ratio)
    return pred


def mae(pred, truth):
    return float(np.abs(np.array(pred) - np.array(truth)).mean())


print('=' * 68)
print('严格事前复检: lvl_model 的 EF 换成上年 (2022) implied 口径')
print('=' * 68)

# ============ 山西 2023 ============
true_sx = build_truth('Shanxi', cum_sx, range(2019, 2025))
e23, e22 = emis_mt('Shanxi', 2023), emis_mt('Shanxi', 2022)
thermal23 = 3704.1  # 亿kWh (公布)
print(f'\n山西: CM 排放 2023={e23:.1f} Mt / 2022={e22:.1f} Mt')
print(f'  EF23 (760 原型值) = {e23*1e6/(thermal23*1e8)*1000:.1f} g/kWh (thermal {thermal23:.0f}亿)')
lvl_orig = 618.9
print('\n--- 山西 2023: 严格版 (2022-implied EF, 热力量括幅) ---')
for thermal22 in [3480, 3542, 3600]:
    ef22 = e22 * 1e6 / (thermal22 * 1e8) * 1000
    lvl_strict = lvl_orig * ef22 / (e23 * 1e6 / (thermal23 * 1e8) * 1000)
    fused = 0.5 * lvl_strict + 0.5 * np.mean(true_sx[2022])
    p = predict(2023, [2019, 2020, 2021, 2022], true_sx, lvl_strict, np.mean(true_sx[2022]))
    print(f'  thermal22={thermal22}亿 → EF22={ef22:.0f} → lvl_strict={lvl_strict:.0f}'
          f' (fused {fused:.0f}) → MAE={mae(p, true_sx[2023]):.1f}')
p = predict(2023, [2019, 2020, 2021, 2022], true_sx, lvl_orig, np.mean(true_sx[2022]))
print(f'  [原型对照 lvl=618.9 (2023-implied) → MAE={mae(p, true_sx[2023]):.1f}]')

# 2024 披露时滞括幅: 年初严格版只有 2022 年报 (lvl_last=2022 而非 2023)
p = predict(2024, [2019, 2020, 2021, 2022, 2023], true_sx, lvl_orig, np.mean(true_sx[2023]))
print(f'  山西 2024 (上年披露, 原配置) MAE={mae(p, true_sx[2024]):.1f}')
p = predict(2024, [2019, 2020, 2021, 2022, 2023], true_sx, lvl_orig, np.mean(true_sx[2022]))
print(f'  山西 2024 (披露时滞严格版 lvl_last=2022年报) MAE={mae(p, true_sx[2024]):.1f}')

# ============ 上海 2023 ============
true_sh = build_truth('Shanghai', cum_sh, range(2020, 2025))
esh23, esh22 = emis_mt('Shanghai', 2023), emis_mt('Shanghai', 2022)
gen_sh23, gen_sh22 = 1015.0, 901.2  # KPI/规上口径 亿kWh
avg23 = esh23 * 1e6 / (gen_sh23 * 1e8) * 1000
avg22 = esh22 * 1e6 / (gen_sh22 * 1e8) * 1000
print(f'\n上海: CM 排放 2023={esh23:.2f} Mt / 2022={esh22:.2f} Mt')
print(f'  全电源平均 implied EF: 2023={avg23:.1f} (coal 448 原型锚) / 2022={avg22:.1f} g/kWh')
lvl_orig_sh = 451.3
SH_COAL, SH_GAS, SH_PETRO = 613.0, 346.0, 4.8
SH_SOLAR, SH_WIND = 4.87, 23.9
DEN = SH_COAL + SH_GAS + SH_PETRO + SH_SOLAR + SH_WIND
print('\n--- 上海 2023: 严格版 (2022-implied) ---')
# 变体A: 仅重推 coal EF (gas EF 年际稳定)
coal_a = 448.0 * avg22 / avg23
lvl_a = (SH_COAL * coal_a + SH_GAS * 490.0 + SH_PETRO * 720.0) / DEN
# 变体B: 全热力 EF 等比缩放
lvl_b = lvl_orig_sh * avg22 / avg23
# 变体C: 无结构模型, 纯上年披露 (lvl_model = lvl_last)
for name, lvl in [('A 仅重推coal', lvl_a), ('B 全热力缩放', lvl_b), ('C 纯上年披露', float(np.mean(true_sh[2022])))]:
    p = predict(2023, [2020, 2021, 2022], true_sh, lvl, np.mean(true_sh[2022]))
    print(f'  变体{name}: lvl_model={lvl:.0f} → fused={0.5*lvl+0.5*np.mean(true_sh[2022]):.0f}'
          f' → MAE={mae(p, true_sh[2023]):.1f}')
p = predict(2023, [2020, 2021, 2022], true_sh, lvl_orig_sh, np.mean(true_sh[2022]))
print(f'  [原型对照 lvl=451.3 (2023-implied) → MAE={mae(p, true_sh[2023]):.1f}]')

# 上海 2024: 原配置 lvl=451.3 属上年(2023)信息, 天然合法; 披露时滞括幅
p = predict(2024, [2020, 2021, 2022, 2023], true_sh, lvl_orig_sh, np.mean(true_sh[2023]))
print(f'  上海 2024 (原配置, 全部上年信息) MAE={mae(p, true_sh[2024]):.1f}')
p = predict(2024, [2020, 2021, 2022, 2023], true_sh, lvl_orig_sh, np.mean(true_sh[2022]))
print(f'  上海 2024 (披露时滞严格版 lvl_last=2022年报) MAE={mae(p, true_sh[2024]):.1f}')

print('\n' + '=' * 68)
print('判定: 严格事前口径下各格 MAE 是否全部 < 20')
