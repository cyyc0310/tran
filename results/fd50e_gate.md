# FD-50e 门控延迟自校准（部署配方）

late 半段 pooled 基线：43.58
R5 无门控参照：41.59 (-1.99)

## tau 扫描（pooled late MAE）

- tau_a=0.00/family/tau_b=0.02: 41.64 (-1.94)
- tau_a=0.00/family/tau_b=0.05: 41.65 (-1.93)
- tau_a=0.00/family/tau_b=0.99: 41.59 (-1.99)
- tau_a=0.00/identity/tau_b=0.02: 41.64 (-1.94)
- tau_a=0.00/identity/tau_b=0.05: 41.65 (-1.93)
- tau_a=0.00/identity/tau_b=0.99: 41.59 (-1.99)
- tau_a=0.02/family/tau_b=0.02: 41.76 (-1.82)
- tau_a=0.02/family/tau_b=0.05: 41.77 (-1.81)
- tau_a=0.02/family/tau_b=0.99: 41.70 (-1.88)
- tau_a=0.02/identity/tau_b=0.02: 41.88 (-1.70)
- tau_a=0.02/identity/tau_b=0.05: 41.89 (-1.69)
- tau_a=0.02/identity/tau_b=0.99: 41.85 (-1.73)
- tau_a=0.05/family/tau_b=0.02: 41.78 (-1.80)
- tau_a=0.05/family/tau_b=0.05: 41.79 (-1.79)
- tau_a=0.05/family/tau_b=0.99: 41.54 (-2.04)
- tau_a=0.05/identity/tau_b=0.02: 41.96 (-1.62)
- tau_a=0.05/identity/tau_b=0.05: 41.97 (-1.61)
- tau_a=0.05/identity/tau_b=0.99: 41.75 (-1.83)
- tau_a=0.08/family/tau_b=0.02: 41.78 (-1.80)
- tau_a=0.08/family/tau_b=0.05: 41.79 (-1.79)
- tau_a=0.08/family/tau_b=0.99: 41.54 (-2.04)
- tau_a=0.08/identity/tau_b=0.02: 42.17 (-1.41)
- tau_a=0.08/identity/tau_b=0.05: 42.18 (-1.40)
- tau_a=0.08/identity/tau_b=0.99: 41.96 (-1.62)
- tau_a=0.12/family/tau_b=0.02: 42.22 (-1.36)
- tau_a=0.12/family/tau_b=0.05: 42.23 (-1.35)
- tau_a=0.12/family/tau_b=0.99: 41.96 (-1.62)
- tau_a=0.12/identity/tau_b=0.02: 42.71 (-0.87)
- tau_a=0.12/identity/tau_b=0.05: 42.72 (-0.86)
- tau_a=0.12/identity/tau_b=0.99: 42.42 (-1.16)

## 最优配置：tau_a=0.05/family/tau_b=0.99
- pooled：41.54 (-2.04)

## 右尾区明细（最优配置 vs 无门控 R5）

| target | late基线 | R5 | gated | Δ(gated−R5) |
|---|---|---|---|---|
| SA1 | 87.2 | 76.9 | 76.9 | +0.0 |
| VIC1 | 74.7 | 71.9 | 71.9 | +0.0 |
| UK_07_South_Wales | 71.7 | 68.8 | 72.8 | +4.0 |
| UK_08_West_Midlands | 66.0 | 68.1 | 66.8 | -1.3 |
| UK_12_South_England | 60.8 | 60.9 | 61.2 | +0.2 |
| UK_09_East_Midlands | 60.8 | 68.3 | 60.6 | -7.7 |
| UK_11_South_West_England | 60.4 | 59.3 | 59.5 | +0.2 |
| UK_17_Wales | 56.8 | 57.1 | 57.1 | +0.0 |
| UK_14_South_East_England | 52.0 | 51.9 | 51.9 | -0.0 |
| UK_05_Yorkshire | 51.4 | 32.0 | 32.0 | +0.0 |
| NSW1 | 50.4 | 45.7 | 45.7 | +0.0 |
| UK_06_North_Wales_Merseyside | 50.4 | 50.2 | 50.1 | -0.1 |
| US_CISO | 48.1 | 43.3 | 43.3 | +0.0 |
| UK_01_North_Scotland | 46.7 | 45.8 | 45.8 | +0.0 |
| UK_13_London | 42.4 | 42.3 | 42.3 | -0.0 |