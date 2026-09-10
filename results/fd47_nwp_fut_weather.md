# FD-47 部署真实 fut_weather：GFS/ICON 业务预报 vs ERA5 代理

同一模型（28 源训练，seed 0，ep900 官方口径），仅目标区推理时的未来 24h 天气来源不同：

- era5：ERA5 再分析代理（FD-41 基线臂）
- gfs / icon / ensemble：Open-Meteo 业务预报归档（gfs_seamless / icon_seamless / 双模型均值）

gust/pressure 未来通道保留 ERA5 代理（归档无此二列，次要特征，文档化近似）；regime24/tend6 用 ERA5 历史 + NWP 未来的复合轴重算；336h 历史在所有臂中恒为 ERA5。

## I_cfg 层（零遥测，论文核心口径）

| target | has_real_weather | mae_era5_icfg | mae_gfs_icfg | mae_icon_icfg | mae_ensemble_icfg |
|---|---|---|---|---|---|
| NSW1 | 1 | 66.75 | 86.28 | 86.75 | 86.11 |
| QLD1 | 1 | 28.03 | 28.49 | 29.13 | 28.26 |
| SA1 | 1 | 86.48 | 93.55 | 103.49 | 92.79 |
| UK_01_North_Scotland | 1 | 48.18 | 77.50 | 66.49 | 71.63 |
| UK_02_South_Scotland | 1 | 16.33 | 23.55 | 19.71 | 21.40 |
| UK_03_North_West_England | 1 | 26.43 | 24.22 | 24.04 | 23.98 |
| UK_05_Yorkshire | 1 | 53.55 | 68.12 | 67.85 | 67.36 |
| UK_06_North_Wales_Merseyside | 1 | 44.84 | 47.88 | 47.20 | 46.48 |
| UK_07_South_Wales | 1 | 66.13 | 74.73 | 73.80 | 73.70 |
| UK_08_West_Midlands | 1 | 73.95 | 82.27 | 82.94 | 81.84 |
| UK_09_East_Midlands | 1 | 79.07 | 78.86 | 79.33 | 77.93 |
| UK_10_East_England | 1 | 33.34 | 41.39 | 37.89 | 37.94 |
| UK_11_South_West_England | 1 | 53.40 | 61.76 | 61.01 | 61.08 |
| UK_12_South_England | 1 | 54.20 | 62.61 | 64.17 | 62.78 |
| UK_13_London | 1 | 38.45 | 46.12 | 50.12 | 47.42 |
| UK_14_South_East_England | 1 | 44.99 | 46.84 | 45.88 | 45.76 |
| UK_15_England | 1 | 35.76 | 38.10 | 39.22 | 37.82 |
| UK_16_Scotland | 1 | 22.92 | 43.39 | 27.69 | 34.05 |
| UK_17_Wales | 1 | 54.90 | 72.43 | 70.18 | 69.51 |
| UK_18_GB | 1 | 35.21 | 37.58 | 38.41 | 36.92 |
| US_BPAT | 1 | 20.97 | 40.92 | 42.10 | 41.34 |
| US_CISO | 1 | 52.07 | 55.42 | 53.56 | 54.37 |
| US_ERCO | 1 | 38.17 | 47.08 | 47.04 | 45.25 |
| US_FPL | 1 | 24.02 | 23.18 | 22.29 | 22.52 |
| US_ISNE | 1 | 24.65 | 24.50 | 24.75 | 24.56 |
| US_MISO | 1 | 33.48 | 37.97 | 36.91 | 36.27 |
| US_NYIS | 1 | 21.74 | 24.18 | 23.87 | 24.07 |
| US_PJM | 1 | 19.05 | 19.19 | 18.77 | 18.89 |
| VIC1 | 1 | 78.26 | 93.31 | 90.03 | 89.95 |

## 汇总（窗口级配对 ΔMAE，正值 = NWP 臂更差）

era5-real 家族 = 原本有 ERA5 天气的区（配对差 = 预报技巧损失）；no-era5 fallback 家族 = 原本天气全零的区（差值 = 获得公开业务预报基础设施的收益）

| arm | family | pooled ΔMAE (icfg) | better | worse |
|---|---|---|---|---|
| gfs | era5-real | +7.80 | 4 | 25 |
| icon | era5-real | +6.88 | 3 | 26 |
| ensemble | era5-real | +6.44 | 5 | 24 |

## 覆盖率异常区（join 覆盖 < 99%）

无
