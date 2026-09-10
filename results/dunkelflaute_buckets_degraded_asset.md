# FD-45 Dunkelflaute 分桶评估（29 区, seed 0, ep900 官方口径）

事件定义（零遥测可复现，纯 ERA5）：
- wind_lull：日均 100m 风速 < 该区全年 20 分位
- solar_dim：日均短波 < 该区全年 20 分位
- dunkelflaute：两者同时成立（复合事件）；暗冬区退化为 wind-only

## I_cfg 层分桶（compound 口径）

| target | n_event_days | mae_event | mae_regular | ratio | p_perm |
|---|---|---|---|---|---|
| QLD1 | 3.0 | 40.63 | 36.14 | 1.124 | 0.4566 |
| UK_09_East_Midlands | 3.0 | 162.16 | 97.84 | 1.657 | 0.033 |
| UK_10_East_England | 3.0 | 60.62 | 40.8 | 1.486 | 0.052 |
| UK_11_South_West_England | 3.0 | 81.43 | 74.05 | 1.1 | 0.5654 |
| UK_12_South_England | 3.0 | 95.54 | 67.27 | 1.42 | 0.1186 |
| UK_13_London | 1.0 | 17.96 | 49.79 | 0.361 | 0.1131 |
| UK_15_England | 3.0 | 93.7 | 44.07 | 2.126 | 0.0018 |
| UK_16_Scotland | 5.0 | 101.03 | 34.06 | 2.966 | 0.0007 |
| UK_17_Wales | 3.0 | 75.36 | 74.98 | 1.005 | 0.9805 |
| UK_18_GB | 4.0 | 74.44 | 41.18 | 1.808 | 0.0168 |
| US_CISO | 13.0 | 65.82 | 58.75 | 1.12 | 0.1888 |
| US_PJM | 5.0 | 23.31 | 22.25 | 1.047 | 0.8551 |
| US_MISO | 3.0 | 31.81 | 43.99 | 0.723 | 0.3067 |
| US_ERCO | 5.0 | 48.4 | 61.3 | 0.79 | 0.1833 |
| US_ISNE | 5.0 | 34.62 | 25.0 | 1.385 | 0.0887 |
| US_NYIS | 3.0 | 11.46 | 16.5 | 0.694 | 0.3034 |

复合事件为 0 的区域：NSW1, VIC1, SA1, UK_14_South_East_England

## 全部明细（含 i0/persistence/config_constant 与 wind_lull/solar_dim 口径）见 dunkelflaute_buckets.json
