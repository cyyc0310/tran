# FD-45 Dunkelflaute 分桶评估（29 区, seed 0, ep900 官方口径）

事件定义（零遥测可复现，纯 ERA5）：
- wind_lull：日均 100m 风速 < 该区全年 20 分位
- solar_dim：日均短波 < 该区全年 20 分位
- dunkelflaute：两者同时成立（复合事件）；暗冬区退化为 wind-only

## I_cfg 层分桶（compound 口径）

| target | n_event_days | mae_event | mae_regular | ratio | p_perm |
|---|---|---|---|---|---|
| QLD1 | 2.0 | 24.83 | 28.12 | 0.883 | 0.5736 |
| UK_01_North_Scotland | 7.0 | 87.09 | 43.99 | 1.98 | 0.0037 |
| UK_02_South_Scotland | 3.0 | 28.66 | 15.8 | 1.815 | 0.0523 |
| UK_03_North_West_England | 3.0 | 62.71 | 24.85 | 2.523 | 0.0051 |
| UK_05_Yorkshire | 3.0 | 53.56 | 53.55 | 1.0 | 0.9983 |
| UK_06_North_Wales_Merseyside | 3.0 | 84.1 | 43.13 | 1.95 | 0.0533 |
| UK_07_South_Wales | 3.0 | 49.68 | 66.84 | 0.743 | 0.185 |
| UK_08_West_Midlands | 3.0 | 131.91 | 71.43 | 1.847 | 0.0163 |
| UK_09_East_Midlands | 3.0 | 90.88 | 78.55 | 1.157 | 0.6134 |
| UK_10_East_England | 4.0 | 43.16 | 32.76 | 1.317 | 0.187 |
| UK_11_South_West_England | 3.0 | 47.1 | 53.67 | 0.878 | 0.6203 |
| UK_12_South_England | 3.0 | 76.59 | 53.23 | 1.439 | 0.105 |
| UK_13_London | 3.0 | 34.94 | 38.6 | 0.905 | 0.7327 |
| UK_14_South_East_England | 3.0 | 31.45 | 45.58 | 0.69 | 0.1695 |
| UK_15_England | 3.0 | 47.85 | 35.24 | 1.358 | 0.2017 |
| UK_16_Scotland | 7.0 | 66.99 | 18.17 | 3.686 | 0.0001 |
| UK_17_Wales | 3.0 | 56.95 | 54.81 | 1.039 | 0.8528 |
| UK_18_GB | 3.0 | 43.49 | 34.85 | 1.248 | 0.3988 |
| US_CISO | 17.0 | 48.17 | 53.25 | 0.905 | 0.2174 |
| US_PJM | 5.0 | 22.22 | 18.82 | 1.181 | 0.3499 |
| US_MISO | 3.0 | 19.72 | 34.07 | 0.579 | 0.1743 |
| US_ERCO | 6.0 | 48.25 | 37.27 | 1.295 | 0.0719 |
| US_ISNE | 5.0 | 46.24 | 23.06 | 2.005 | 0.0009 |
| US_NYIS | 3.0 | 15.08 | 22.03 | 0.685 | 0.1326 |
| US_FPL | 2.0 | 25.11 | 23.99 | 1.047 | 0.8345 |
| US_BPAT | 4.0 | 34.97 | 20.16 | 1.735 | 0.005 |

复合事件为 0 的区域：NSW1, VIC1, SA1

## 全部明细（含 i0/persistence/config_constant 与 wind_lull/solar_dim 口径）见 dunkelflaute_buckets.json
