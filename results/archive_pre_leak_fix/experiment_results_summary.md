# 实验结果汇总（截至 2026-08-12）

> Phase 8 全可微联合训练 pipeline 完成后的最新结果整理。
> 源代码 commit：`25fbd9b feat(phase8): joint training pipeline`

## 一、Headline

**Joint-trained 模型在完整 LORO 上 median MAE = 40.53，目标 < 41 达成。**

- 评测：29 区域 × 5 seed = 145 对
- 总耗时：6474s ≈ 1.8 小时（mean 44.7s/pair，远低于 30 min DoD）
- 协议：每对前 12 个 test origin 训练 + 后 12 个 disjoint origin 评估

## 二、方法对比（统一 LORO 协议）

| 方法 | Median MAE | Mean | Std | 备注 |
|------|-----------:|-----:|----:|------|
| Persistence | 51.52 | 50.49 | 24.86 | 基线 |
| Best single direction (causal) | 50.60 | 51.94 | 22.24 | 单方向最强 |
| BasisMix (no ZS+) | 54.65 | 55.20 | 21.87 | 5 方向等权融合 |
| BasisMix+ (ZS+) | 46.89 | 45.01 | 22.62 | Phase 5 之前的 SOTA |
| PatchTST-supervised (外部) | 41.47 | — | — | 监督基线 |
| **Joint-trained (Phase 8)** | **40.53** | **42.67** | **23.44** | 本轮结果 |

源文件：
- `results/fused_five_full_summary.json`（前 4 行）
- `results/joint_train_full_summary.json`（最后一行）

### Win rate（145 对）

- 击败 BasisMix+ 46.89：**69.0%**（100/145）
- 击败目标 41.0：**55.2%**（80/145）
- 击败 PatchTST-supervised 41.47：**55.2%**（80/145）

## 三、MAE 分布（145 对）

| 分位 | p10 | p25 | median | p75 | p90 | max |
|------|----:|----:|-------:|----:|----:|----:|
| MAE  | 14.6 | 27.8 | 40.5 | 49.8 | 81.3 | 100.1 |

## 四、最佳 / 最差区域

### Top 5（median MAE，n=5 seed/region）

| 区域 | median MAE |
|------|----------:|
| US_BPAT | 9.36 |
| US_PJM | 13.01 |
| US_FPL | 13.04 |
| UK_02_South_Scotland | 15.42 |
| US_ISNE | 15.70 |

### Bottom 5（median MAE）

| 区域 | median MAE | 诊断 |
|------|----------:|------|
| UK_07_South_Wales | 67.39 | persistence 本身就很高 |
| UK_17_Wales | 69.73 | 同上 |
| VIC1 | 89.63 | 高排放强度 + donor pool 不匹配 |
| UK_08_West_Midlands | 96.45 | 同上 |
| UK_09_East_Midlands | 99.68 | 同上 |

**模式**：最差 5 个区域都是 persistence 基线本身已经很大的电网（>60 gCO2/kWh），donor pool 转移效果差——是 LORO 协议的已知短板，不是 joint training 能独立解决的。

## 五、5 方向贡献（directions_eval_summary.json，n=29）

| 方向 | Median MAE (ZS) | Median MAE (+ZS+) | +ZS+/ZS ratio |
|------|---------------:|------------------:|--------------:|
| causal | 41.18 | 45.96 | 1.12（ZS+ 反而拖累） |
| phys_irm | 44.01 | 45.83 | 1.04 |
| rag | 50.33 | 46.83 | 0.93 |
| icl | 49.94 | 47.07 | 0.94 |
| hier | 71.15 | 46.58 | 0.66（ZS+ 拉平劣势） |

**观察**：ZS+ 标定对所有方向都有"拉平"作用——causal 这种本就强的方向被略微拉差，hier 这种本就弱的方向被大幅拉好。Joint training 是打破这层"标定天花板"的关键。

## 六、ZS+ 分支消融（zs_plus_ablation.json，4 个区域）

5 个 ZS+ branch menu：

| Branch | 含义 |
|--------|------|
| `DEFAULT` | 模型 delta + persistence lag（默认） |
| `MODEL_DELTA` | 仅模型 delta，无 lag 兜底 |
| `MODEL_RAW` | 仅模型 raw output，无 lag |
| `MODEL_BOTH` | 模型 delta + raw，无 lag |
| `LAG_ONLY` | 仅 persistence lag |

QLD1 / NSW1 / VIC1 / SA1 上一致结果：**`DEFAULT` 在所有方向上最优**。`MODEL_RAW` 方差极大（QLD1 上 rag 56 vs hier 134），证实了 Phase 5 的诊断——原 ZS+ = "增强版 persistence"。

## 七、Caveats（写论文时必须坦白）

1. **Supervised calibration，非严格 zero-shot。** Stage 1/2 用了前 12 个 test origin 的 CIF label 做标定微调。对 BasisMix+（纯 zero-shot）的对比不是 apples-to-apples。诚实表述为："joint calibration training with held-out protocol"。
2. **0.47 的余量很薄。** 改 seed selection 或 train/eval split 可能落到 41.1 或 40.9。属于"边缘胜出"，不是决定性碾压。
3. **高方差。** Std 23.44，p90 高达 81.3。最差 5 个区域 67-100 MAE。
4. **Stage 2 correction 贡献有限。** Stage 1 → Stage 2 median MAE 下降约 1 点；主要 lift 来自 Stage 1 的 soft attention + adversarial-persistence loss。

## 八、运行配置

| 项 | 值 |
|----|-----|
| HORIZON | 24 |
| SEQ_LEN | 336 |
| TRAIN_FRACTION | 0.8 |
| TEST_STRIDE | 24 |
| n_origins (train) | 12 |
| n_origins (eval) | 12 |
| n_steps (Stage 1) | 30 |
| n_steps (Stage 2) | 30 |
| lr (Stage 1) | 5e-2 |
| lr (Stage 2) | 1e-2 |
| margin (adversarial) | 0.10 |
| adv_loss_weight | 0.5 |

### adv_loss_weight = 0.5 的取舍

| 值 | 行为 |
|----|------|
| 0.0 | 纯 MAE，模型会塌缩到 persistence baseline（ZS+ branch 1-4 本就是 persistence） |
| 0.5 | 半权重，QLD1 sanity 实证选定（两个 stage 单调下降） |
| 1.0 | 反 persistence 压力 = MAE 压力，在某些区域过度推离合理 lag baseline |

## 九、产物索引

### 代码
- `scripts/experiments/run_joint_train.py` — 2-stage pipeline（Stage 1 + Stage 2 + held-out eval）
- `scripts/experiments/run_joint_train_sanity.py` — QLD1 sanity driver
- `scripts/experiments/run_joint_train_full.py` — 29 × 5 LORO driver
- `scripts/experiments/run_zs_plus_ablation.py` — ZS+ branch menu 诊断
- `src/transcif/calibration/differentiable_zs_plus.py` — 可微 ZS+ 模块
- `src/transcif/training/adversarial_loss.py` — adversarial-persistence loss

### 测试
17/17 green
- `tests/test_differentiable_zs_plus.py`（5）
- `tests/test_adversarial_loss.py`（6）
- `tests/test_joint_train.py`（6，含 held-out eval 覆盖）

### 结果文件
- `results/joint_train_full.json`（145 行 per-pair 结果）
- `results/joint_train_full_summary.json`（聚合统计）
- `results/joint_train_verdict.md`（Task 8.7 最终判定）
- `results/joint_train_gate.md`（Task 8.5 Go/No-Go）
- `results/joint_train_sanity.md`（Task 8.4 QLD1 sanity）
- `results/fused_five_full.json` / `_summary.json`（前 Phase SOTA）
- `results/directions_eval_combined.json` / `_summary.json`（5 方向单独评估）
- `results/zs_plus_ablation.json`（ZS+ branch menu）

### Git 历史
- `25fbd9b feat(phase8): joint training pipeline — median MAE 40.53 on full LORO`
- `5f924de perf(fused_five_full): pass small donor pool to direction trainers`

## 十、下一步候选

| 项 | 价值 | 工作量 |
|----|------|-------|
| 论文 Phase 6/7：写 results + ablation 章节 | 高 | 中 |
| Stage 2 真正 unfreeze 方向模型 output layer（需 torch-native 重写 5 个 predictor） | 中（估计再降 0.5-1 MAE） | 高 |
| 扩展 donor pool（top-k by similarity 而非前 3 个） | 中（最差区域可能改善） | 低 |
| 多种 train/eval split 验证 0.47 margin 稳健性 | 高（论文 robustness 必需） | 低 |
| PatchTST-supervised 公平复现（同 donor pool + 同 split） | 高（apples-to-apples 对比） | 中 |
