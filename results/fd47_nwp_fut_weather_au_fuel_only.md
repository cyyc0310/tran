# FD-47 部署真实 fut_weather：GFS/ICON 业务预报 vs ERA5 代理

同一模型（28 源训练，seed 0，ep900 官方口径），仅目标区推理时的未来 24h 天气来源不同：

- era5：ERA5 再分析代理（FD-41 基线臂）
- gfs / icon / ensemble：Open-Meteo 业务预报归档（gfs_seamless / icon_seamless / 双模型均值）

gust/pressure 未来通道保留 ERA5 代理（归档无此二列，次要特征，文档化近似）；regime24/tend6 用 ERA5 历史 + NWP 未来的复合轴重算；336h 历史在所有臂中恒为 ERA5。

## I_cfg 层（零遥测，论文核心口径）

| target | has_real_weather | mae_era5_icfg | mae_gfs_icfg | mae_icon_icfg | mae_ensemble_icfg |
|---|---|---|---|---|---|
| NSW1 | 1 | 72.12 | 89.82 | 91.01 | 90.10 |
| QLD1 | 1 | 31.15 | 31.13 | 31.35 | 30.56 |
| SA1 | 1 | 99.01 | 97.49 | 111.55 | 98.53 |
| UK_01_North_Scotland | 1 | 42.26 | 54.47 | 49.41 | 51.84 |
| UK_02_South_Scotland | 1 | 15.60 | 19.25 | 16.30 | 17.49 |
| UK_03_North_West_England | 1 | 20.79 | 21.32 | 22.19 | 21.44 |
| UK_05_Yorkshire | 1 | 38.62 | 58.06 | 54.49 | 55.90 |
| UK_06_North_Wales_Merseyside | 1 | 41.75 | 46.38 | 43.56 | 44.50 |
| UK_07_South_Wales | 1 | 62.10 | 68.39 | 65.46 | 66.52 |
| UK_08_West_Midlands | 1 | 64.94 | 73.48 | 74.61 | 73.44 |
| UK_09_East_Midlands | 1 | 83.95 | 77.61 | 77.71 | 77.06 |
| UK_10_East_England | 1 | 34.81 | 54.38 | 46.60 | 49.97 |
| UK_11_South_West_England | 1 | 73.51 | 77.12 | 73.27 | 74.67 |
| UK_12_South_England | 1 | 51.38 | 58.84 | 59.37 | 58.43 |
| UK_13_London | 1 | 47.80 | 53.81 | 60.96 | 57.00 |
| UK_14_South_East_England | 1 | 47.54 | 45.79 | 45.49 | 45.40 |
| UK_15_England | 1 | 30.54 | 35.47 | 35.50 | 34.62 |
| UK_16_Scotland | 1 | 24.07 | 63.85 | 41.71 | 51.64 |
| UK_17_Wales | 1 | 52.86 | 70.20 | 58.70 | 63.03 |
| UK_18_GB | 1 | 28.29 | 34.88 | 36.84 | 35.06 |
| US_BPAT | 1 | 34.85 | 60.69 | 61.69 | 60.78 |
| US_CISO | 1 | 75.16 | 75.55 | 74.87 | 75.17 |
| US_ERCO | 1 | 39.72 | 46.98 | 45.47 | 44.33 |
| US_FPL | 1 | 22.29 | 22.11 | 21.90 | 21.83 |
| US_ISNE | 1 | 28.31 | 27.65 | 27.93 | 27.72 |
| US_MISO | 1 | 31.21 | 35.51 | 33.29 | 33.35 |
| US_NYIS | 1 | 20.73 | 24.30 | 23.72 | 24.05 |
| US_PJM | 1 | 19.89 | 20.05 | 19.41 | 19.62 |
| VIC1 | 1 | 77.11 | 91.50 | 89.09 | 88.25 |

## 汇总（窗口级配对 ΔMAE，正值 = NWP 臂更差）

era5-real 家族 = 原本有 ERA5 天气的区（配对差 = 预报技巧损失）；no-era5 fallback 家族 = 原本天气全零的区（差值 = 获得公开业务预报基础设施的收益）

| arm | family | pooled ΔMAE (icfg) | better | worse |
|---|---|---|---|---|
| gfs | era5-real | +7.71 | 6 | 23 |
| icon | era5-real | +6.25 | 7 | 22 |
| ensemble | era5-real | +6.21 | 7 | 21 |

## 覆盖率异常区（join 覆盖 < 99%）

无
