# Phase FD-6 Verdict — 形态损失(I_cfg 形态提升)

日期:2026-08-15 · 快速协议:8 难区域 × 2 seeds(`fuel_decomp_eval_shape.json` vs FD-5 `fuel_decomp_eval_anchor.json`);全量协议:29×5=145 对(`fuel_decomp_eval_full_shape.json` vs `fuel_decomp_eval_full.json`)

## 一句话结论

**PASS**:形态损失(逐窗去均值 CIF MAE,λ=0.5)在两个协议上一致改善 I_cfg 形态——**全量 Spearman 0.280→0.303(win 56%,p=0.049),52% 区域 >0.3、79% >0.1**;MAE 持平(65.67→65.79,vs 年度常数 70.90 / 月度 oracle 64.34);I_+ 46.99(vs persistence win 99%,p=1.6e-25)。λ_shape=0.5 采纳为默认,leaderboard I_cfg 条目已更新为该配置。

## 全量数字(145 对)

| 层级 | MAE | diurnal | Spearman |
|---|---:|---:|---:|
| **I_cfg** | 65.79(年度常数 70.90,oracle 64.34) | 34.32 | **0.303**(λ=0 时 0.280,p=0.049) |
| I_0 | 55.22 | 35.40 | 0.273 |
| I_+ | 46.99(legacy ZS+ 46.80,equalizer 平局) | 28.70 | 0.386 |

快速协议(16 对):I_cfg Spearman 0.230→0.239(win 69%,p=0.04);**I_+ 36.81→36.59(win 88%,p<0.01,该集最佳)**;UK_08 Spearman 0.292→0.384。

## 机制

FD-4 证明 I_cfg 水平已近 oracle(65.67 vs 64.34),剩余空间在形态。总损失此前由水平主导(CIF MAE);形态项(λ=0.5)让梯度预算显式流向日内轨迹——Spearman/碳感知调度实际消费的量。与锚定机制正交:锚定管水平,形态项管形状,互不侵蚀。

## 数字(16 对,配对 Wilcoxon vs FD-5)

| 层级 | MAE | diurnal | Spearman |
|---|---|---|---|
| I_cfg | 71.01→71.76(win 69%,p=0.38) | 50.98→**49.51**(p=0.10) | 0.230→**0.239**(win 69%,**p=0.04**) |
| I_0 | 45.13→45.68(win 75%,p=0.10) | 37.76→38.88(p=0.09) | 0.292→0.297(p=0.30) |
| I_+ | 36.81→**36.59**(win 88%,**p<0.01**) | 30.81→30.66 | 0.412→0.413 |

逐区域:UK_08 MAE −2.7 且 Spearman 0.292→**0.384**;PJM −1.3;QLD1/UK_02/BPAT 持平;小幅回归(NSW1 −1.8、SA1 −2.4、VIC1 −1.0)均在噪声量级。win 69-75% 但中位数反向 = 少数对的小幅回归主导中位数,多数对改善。

## 决策

1. `LAMBDA_SHAPE = 0.5` 为默认(`--lambda-shape 0` 复现 FD-5 目标函数)
2. 全量 145 对复跑中(`fuel_decomp_eval_full_shape.json`)——完成后 I_cfg 条目以最优配置进入 leaderboard
3. 后续 I_cfg 形态杠杆(若继续):季节性 baseload 显式建模(水利月际)、solar 晴空指数调制增强(当前 0.4·tanh 界)
