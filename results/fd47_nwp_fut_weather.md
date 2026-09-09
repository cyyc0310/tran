# FD-47 部署真实 fut_weather：GFS/ICON 业务预报 vs ERA5 代理

同一模型（28 源训练，seed 0，ep900 官方口径），仅目标区推理时的未来 24h 天气来源不同：

- era5：ERA5 再分析代理（FD-41 基线臂）
- gfs / icon / ensemble：Open-Meteo 业务预报归档（gfs_seamless / icon_seamless / 双模型均值）

gust/pressure 未来通道保留 ERA5 代理（归档无此二列，次要特征，文档化近似）；regime24/tend6 用 ERA5 历史 + NWP 未来的复合轴重算；336h 历史在所有臂中恒为 ERA5。

## I_cfg 层（零遥测，论文核心口径）

| target | has_real_weather | mae_era5_icfg | mae_gfs_icfg | mae_icon_icfg | mae_ensemble_icfg |
|---|---|---|---|---|---|
| NSW1 | 1 | 56.45 | 61.02 | 58.24 | 58.29 |
| QLD1 | 1 | 36.33 | 35.75 | 36.71 | 36.06 |
| SA1 | 1 | 71.04 | 65.32 | 71.84 | 68.33 |
| UK_01_North_Scotland | 0 | 51.94 | 49.08 | 47.36 | 48.36 |
| UK_02_South_Scotland | 0 | 24.54 | 19.84 | 19.32 | 19.47 |
| UK_03_North_West_England | 0 | 37.89 | 29.09 | 29.56 | 29.25 |
| UK_05_Yorkshire | 0 | 66.14 | 34.77 | 34.07 | 33.54 |
| UK_06_North_Wales_Merseyside | 0 | 67.29 | 54.98 | 53.40 | 53.97 |
| UK_07_South_Wales | 0 | 106.55 | 86.78 | 85.72 | 86.00 |
| UK_08_West_Midlands | 0 | 112.86 | 88.50 | 89.26 | 88.47 |
| UK_09_East_Midlands | 1 | 100.52 | 97.38 | 98.42 | 97.39 |
| UK_10_East_England | 1 | 41.63 | 40.80 | 39.53 | 39.57 |
| UK_11_South_West_England | 1 | 74.35 | 78.85 | 77.28 | 77.91 |
| UK_12_South_England | 1 | 68.45 | 67.25 | 69.12 | 67.79 |
| UK_13_London | 1 | 49.34 | 49.52 | 52.22 | 50.48 |
| UK_14_South_East_England | 1 | 62.09 | 59.90 | 60.13 | 59.91 |
| UK_15_England | 1 | 46.14 | 43.89 | 45.20 | 43.96 |
| UK_16_Scotland | 1 | 38.71 | 41.90 | 39.18 | 40.42 |
| UK_17_Wales | 1 | 75.00 | 73.75 | 68.36 | 70.36 |
| UK_18_GB | 1 | 43.03 | 42.54 | 43.95 | 42.75 |
| US_BPAT | 0 | 14.29 | 17.71 | 18.41 | 18.08 |
| US_CISO | 1 | 60.01 | 61.11 | 60.68 | 60.63 |
| US_ERCO | 1 | 60.42 | 48.17 | 47.39 | 45.51 |
| US_FPL | 0 | 20.54 | 18.20 | 17.90 | 17.85 |
| US_ISNE | 1 | 25.66 | 25.73 | 26.05 | 25.78 |
| US_MISO | 1 | 43.49 | 34.57 | 35.33 | 34.12 |
| US_NYIS | 1 | 16.29 | 18.29 | 18.11 | 18.13 |
| US_PJM | 1 | 22.33 | 22.93 | 22.18 | 22.36 |
| VIC1 | 1 | 132.08 | 137.89 | 132.97 | 134.85 |

## 汇总（窗口级配对 ΔMAE，正值 = NWP 臂更差）

era5-real 家族 = 原本有 ERA5 天气的区（配对差 = 预报技巧损失）；no-era5 fallback 家族 = 原本天气全零的区（差值 = 获得公开业务预报基础设施的收益）

| arm | family | pooled ΔMAE (icfg) | better | worse |
|---|---|---|---|---|
| gfs | era5-real | -0.84 | 11 | 9 |
| gfs | no-era5 fallback | -11.43 | 8 | 1 |
| icon | era5-real | -1.03 | 8 | 12 |
| icon | no-era5 fallback | -11.87 | 8 | 1 |
| ensemble | era5-real | -1.44 | 11 | 9 |
| ensemble | no-era5 fallback | -11.87 | 8 | 1 |

## 覆盖率异常区（join 覆盖 < 99%）

无
