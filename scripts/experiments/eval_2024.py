"""检查2024年山西真值CIF形态，并测试 2019-2023训练→2024评估"""
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

# 2024形态 vs 各训练年形态
a24 = piv[2024].values
s24 = a24 / a24.mean()
print('各年形态 vs 2024 (corr | 同水平形态MAE):')
for y in range(2019, 2024):
    ay = piv[y].values
    sy = ay / ay.mean()
    print(f'{y}: corr={np.corrcoef(sy, s24)[0,1]:.3f} | MAE={np.abs(sy*a24.mean()-a24).mean():.1f}')

# 5年clim形态
clim5 = piv[[2019, 2020, 2021, 2022, 2023]].mean(axis=1)
s5 = (clim5 / clim5.mean()).values
print()
print(f'5年clim形态 vs 2024: corr={np.corrcoef(s5, s24)[0,1]:.3f} | MAE(同水平)={np.abs(s5*a24.mean()-a24).mean():.1f}')
print('2024年均真值:', round(a24.mean(), 1))

# 关键: 各月真值/clim5形态 的偏离
print()
print('2024各月: 真值 | clim5形态x2024年均 | 偏差%')
pred_ideal = s5 * a24.mean()
for m in range(12):
    print(f'{m+1:2d} | {a24[m]:.0f} | {pred_ideal[m]:.0f} | {(a24[m]/pred_ideal[m]-1)*100:+.1f}%')

# 预测2024: 水平= 0.5*模型年均 + 0.5*2023披露年均
# 2023披露年均 = 618.9*4376.1/4178? 不对, 618.9是模型2023年均(基于2022披露), 2023真值年均=641.9
# 部署时2023披露可得 → 水平锚 = 2023年均(641.9) 与模型年均融合
lvl_model = 618.9  # 模型2023预测年均
lvl_2023 = 641.9   # 2023披露年均(部署时已知)
fuse = 0.5 * lvl_model * (gens[2024]/gens[2023]) + 0.5 * lvl_2023 * (gens[2024]/gens[2023])
# 简化: 直接用0.5*(618.9+641.9)按2024发电量调整其实不影响CIF水平(排放/发电同步)
# CIF水平预测= 0.5*模型预测年均 + 0.5*上年披露年均(假设EF不变)
fuse_lvl = 0.5 * 618.9 + 0.5 * 641.9
pred24 = s5 * fuse_lvl
err = pred24 - a24
print()
print(f'=== 预测2024: 5年clim形态 x 融合水平({fuse_lvl:.1f}) ===')
for m in range(12):
    print(f'{m+1:2d} | 预测{pred24[m]:.0f} | 真值{a24[m]:.0f} | {err[m]:+.1f}')
print('月均 LEVEL MAE:', round(np.abs(err).mean(), 1))

# 也试只用2019-2023中形态最接近的(2021)
s21 = (piv[2021]/piv[2021].mean()).values
pred21 = s21 * fuse_lvl
print('单用2021形态: MAE =', round(np.abs(pred21 - a24).mean(), 1))

# 水平误差分解
print()
print('2024水平误差:', round(fuse_lvl - a24.mean(), 1))
perfect = s5 * a24.mean()
print('形态上限(水平完美):', round(np.abs(perfect - a24).mean(), 1))
