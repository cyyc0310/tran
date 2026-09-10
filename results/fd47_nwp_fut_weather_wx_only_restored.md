# FD-47 部署真实 fut_weather：GFS/ICON 业务预报 vs ERA5 代理

同一模型（28 源训练，seed 0，ep900 官方口径），仅目标区推理时的未来 24h 天气来源不同：

- era5：ERA5 再分析代理（FD-41 基线臂）
- gfs / icon / ensemble：Open-Meteo 业务预报归档（gfs_seamless / icon_seamless / 双模型均值）

gust/pressure 未来通道保留 ERA5 代理（归档无此二列，次要特征，文档化近似）；regime24/tend6 用 ERA5 历史 + NWP 未来的复合轴重算；336h 历史在所有臂中恒为 ERA5。

## I_cfg 层（零遥测，论文核心口径）

| target | has_real_weather | mae_era5_icfg | mae_gfs_icfg | mae_icon_icfg | mae_ensemble_icfg |
|---|---|---|---|---|---|
| NSW1 | 1 | 56.47 | 71.70 | 70.64 | 69.78 |
| QLD1 | 1 | 34.10 | 31.83 | 32.76 | 32.09 |
| SA1 | 1 | 82.29 | 73.68 | 89.44 | 80.57 |
| UK_01_North_Scotland | 1 | 41.29 | 55.16 | 48.97 | 51.76 |
| UK_02_South_Scotland | 1 | 15.23 | 21.78 | 18.00 | 19.46 |
| UK_03_North_West_England | 1 | 21.79 | 21.24 | 21.77 | 21.31 |
| UK_05_Yorkshire | 1 | 38.16 | 56.56 | 53.33 | 54.72 |
| UK_06_North_Wales_Merseyside | 1 | 39.25 | 44.37 | 41.71 | 42.45 |
| UK_07_South_Wales | 1 | 72.57 | 77.41 | 75.17 | 75.97 |
| UK_08_West_Midlands | 1 | 67.60 | 75.14 | 75.80 | 74.86 |
| UK_09_East_Midlands | 1 | 83.64 | 78.83 | 78.11 | 77.57 |
| UK_10_East_England | 1 | 33.95 | 52.20 | 45.31 | 48.34 |
| UK_11_South_West_England | 1 | 70.96 | 72.88 | 70.07 | 70.99 |
| UK_12_South_England | 1 | 53.50 | 61.83 | 62.07 | 61.37 |
| UK_13_London | 1 | 46.01 | 53.61 | 59.80 | 56.54 |
| UK_14_South_East_England | 1 | 49.67 | 48.90 | 49.40 | 48.99 |
| UK_15_England | 1 | 29.63 | 35.27 | 35.17 | 34.41 |
| UK_16_Scotland | 1 | 23.48 | 59.18 | 37.44 | 46.88 |
| UK_17_Wales | 1 | 55.42 | 71.46 | 60.25 | 64.63 |
| UK_18_GB | 1 | 27.88 | 34.42 | 36.54 | 34.60 |
| US_BPAT | 1 | 34.24 | 61.61 | 63.85 | 62.73 |
| US_CISO | 1 | 58.21 | 59.70 | 57.67 | 58.26 |
| US_ERCO | 1 | 39.44 | 46.17 | 45.79 | 44.42 |
| US_FPL | 1 | 19.05 | 19.05 | 18.63 | 18.67 |
| US_ISNE | 1 | 24.94 | 24.22 | 24.44 | 24.26 |
| US_MISO | 1 | 28.66 | 33.51 | 33.78 | 32.45 |
| US_NYIS | 1 | 17.02 | 15.45 | 15.78 | 15.62 |
| US_PJM | 1 | 23.04 | 23.54 | 22.79 | 22.98 |
| VIC1 | 1 | 126.70 | 142.19 | 139.38 | 140.16 |

## 汇总（窗口级配对 ΔMAE，正值 = NWP 臂更差）

era5-real 家族 = 原本有 ERA5 天气的区（配对差 = 预报技巧损失）；no-era5 fallback 家族 = 原本天气全零的区（差值 = 获得公开业务预报基础设施的收益）

| arm | family | pooled ΔMAE (icfg) | better | worse |
|---|---|---|---|---|
| gfs | era5-real | +7.19 | 7 | 21 |
| icon | era5-real | +5.86 | 10 | 19 |
| ensemble | era5-real | +5.95 | 9 | 20 |

## 覆盖率异常区（join 覆盖 < 99%）

无
