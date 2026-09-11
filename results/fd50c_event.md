# FD-50c 事件日条件化变换（wind-lull origin 日，零遥测标签）

oracle：a_reg/a_evt/b_evt 用目标自身标签拟合（作弊上界）；
dep_g/dep_f：donor 中位（全局/家族 LOO）代入，零目标标签

## 汇总（29 区 pooled）

- 基线：43.98
- dep 全局 donor：42.70 (-1.28)
- dep 家族 donor：42.95 (-1.02)
- oracle 分日幅度：41.45 (-2.53)
- oracle 分日幅度+事件偏移：39.90 (-4.08)

## 事件日方向一致性（b_evt 与 bias_evt 符号）

- SA1: base 86.5 (evt 101.9/reg 84.8, n_evt 7), a_reg 0.34 a_evt 0.5800000000000003, b_evt -60.3, bias evt 61.1 / reg 50.4, dep_g 80.9 dep_f 77.2 orc_ab 74.3
- UK_09_East_Midlands: base 79.1 (evt 92.7/reg 78.0, n_evt 5), a_reg 1.10 a_evt 1.7200000000000013, b_evt 91.6, bias evt -89.4 / reg -46.8, dep_g 79.4 dep_f 77.4 orc_ab 75.7
- VIC1: base 78.3 (evt 107.2/reg 73.1, n_evt 11), a_reg 0.52 a_evt 0.3, b_evt 20.3, bias evt -24.2 / reg 11.2, dep_g 75.6 dep_f 77.0 orc_ab 75.0
- UK_08_West_Midlands: base 74.0 (evt 129.1/reg 69.8, n_evt 5), a_reg 1.30 a_evt 1.9400000000000015, b_evt 106.5, bias evt -125.1 / reg -41.3, dep_g 75.5 dep_f 73.1 orc_ab 67.4
- NSW1: base 66.8 (evt 78.5/reg 64.0, n_evt 14), a_reg 0.52 a_evt 0.7400000000000004, b_evt -51.5, bias evt 72.6 / reg 29.8, dep_g 67.5 dep_f 63.6 orc_ab 58.9
- UK_07_South_Wales: base 66.1 (evt 52.2/reg 67.2, n_evt 5), a_reg 1.74 a_evt 1.5400000000000011, b_evt 38.1, bias evt -33.2 / reg 4.3, dep_g 68.0 dep_f 66.2 orc_ab 62.0
- UK_17_Wales: base 54.9 (evt 59.0/reg 54.6, n_evt 5), a_reg 0.90 a_evt 1.260000000000001, b_evt 40.7, bias evt -38.4 / reg -4.7, dep_g 54.5 dep_f 53.5 orc_ab 53.5
- UK_12_South_England: base 54.2 (evt 77.2/reg 52.5, n_evt 5), a_reg 0.98 a_evt 1.7400000000000013, b_evt 67.1, bias evt -73.8 / reg -16.2, dep_g 53.7 dep_f 52.4 orc_ab 51.8
- UK_05_Yorkshire: base 53.5 (evt 55.1/reg 53.4, n_evt 5), a_reg 0.36 a_evt 0.48000000000000015, b_evt -32.3, bias evt 28.1 / reg 9.9, dep_g 41.0 dep_f 46.2 orc_ab 32.6
- UK_11_South_West_England: base 53.4 (evt 45.2/reg 54.0, n_evt 5), a_reg 0.72 a_evt 0.4000000000000001, b_evt 41.7, bias evt -43.1 / reg -5.6, dep_g 51.2 dep_f 51.9 orc_ab 50.2
- US_CISO: base 52.1 (evt 54.7/reg 50.0, n_evt 32), a_reg 0.60 a_evt 0.6200000000000003, b_evt 40.8, bias evt -36.4 / reg -28.7, dep_g 44.3 dep_f 51.2 orc_ab 40.1
- UK_01_North_Scotland: base 48.2 (evt 100.7/reg 36.6, n_evt 13), a_reg 0.56 a_evt 0.9000000000000006, b_evt 24.3, bias evt 2.2 / reg 15.4, dep_g 47.5 dep_f 48.6 orc_ab 47.5
- UK_14_South_East_England: base 45.0 (evt 38.0/reg 45.5, n_evt 5), a_reg 0.80 a_evt 0.5400000000000003, b_evt 19.5, bias evt -15.9 / reg 8.4, dep_g 44.8 dep_f 45.5 orc_ab 44.7
- UK_06_North_Wales_Merseyside: base 44.8 (evt 86.8/reg 41.7, n_evt 5), a_reg 0.70 a_evt 1.1200000000000008, b_evt 80.6, bias evt -79.7 / reg -33.8, dep_g 43.7 dep_f 42.9 orc_ab 42.4