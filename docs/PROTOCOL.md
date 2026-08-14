# TransCIF 评测协议与信息集定义

> 本文档明确每种实验协议的**信息集边界**（哪些数据在训练/校准/推理时可见），以及
> leak-fix 后 `mean_rs` / `config` 的计算口径。所有 benchmark 结果必须标注所用的协议。

---

## 一、信息集定义

TransCIF 的评测严格区分三种信息集。每个区域的全年时间序列（8760 小时）被按时间轴
切分为 **训练段（前 80%）** 和 **测试段（后 20%）**，切分点为
`split = int(n_hours * TRAIN_FRACTION)`，`TRAIN_FRACTION = 0.8`。

| 信息集 | 可见数据 | 用途 |
|---|---|---|
| **Training (源域)** | 所有源区域的训练段 (rs, cif) | 训练 ZS 模型 / direction 模型 |
| **Live input (目标域)** | 目标域的实时 rs 序列（测试段输入窗口）+ 标量配置 `(mean_rs, ef_nr)` | ZS / ZS+ 推理的模型输入 |
| **Calibration (目标域)** | 目标域测试段的 CIF 观测流 | 仅 ZS+ / Joint 协议可用 |

### 1.1 标量配置的计算口径（leak-fix 后）

`mean_rs` 和 `config` 向量 **只用训练段** 计算，不接触测试段：

```python
split = int(len(rs) * TRAIN_FRACTION)
train_mean_rs = float(rs[:split].mean())
config = np.array([train_mean_rs, ef_nr / 1000.0], dtype=np.float32)
```

- **AU / US 区域**：`ef_nr` 是硬编码物理常数（`config/constants.py`），不涉及泄漏。
- **UK 区域**：`ef_nr` 由 `discover_uk_regions` 从**训练段**的 `CIF/(1-rs)` 中位数反估
  （`loaders.py:discover_uk_regions`），修复前用的是全年数据。

此修复确保 ZS 模型的配置距离加权
`w = config_weight(src_mean_rs, tgt_mean_rs) = 1/(|Δmean_rs| + 0.05)`
不泄漏目标域测试期的可再生占比分布。

---

## 二、四种评测协议

### 2.1 ZS — 纯零样本 (Pure Zero-Shot)

| 项 | 值 |
|---|---|
| 目标域 CIF 标签用量 | **0 小时** |
| 配置来源 | 训练段统计量 |
| 入口脚本 | `scripts/benchmark/run_unified_eval.py` |
| 核心函数 | `base_zs.evaluate_target` → `train_zero_shot` |
| 结果文件 | `results/unified_eval_full.json` |

源域训练 → 目标域推理时只用 `(config, rs_input)`。配置距离加权用
`config_weight()`。这是最严格的协议，目标域完全不可见 CIF 标签。

### 2.2 ZS+ — 测试时校准 (Test-Time Calibration)

| 项 | 值 |
|---|---|
| 目标域 CIF 标签用量 | **0 小时**（用测试段输入窗口的 CIF 历史做 branch fusion，无未来标签） |
| 配置来源 | 训练段统计量 |
| 入口脚本 | 同 ZS，`evaluate_target` 的 `zs_plus` 路径 |
| 核心函数 | `calibration.zs_plus.zs_plus_predict` |

ZS+ 在 ZS 基础上，用目标域测试段的**已观测 CIF 历史**（非未来标签）做多分支融合
（model delta + persistence lag），属于无监督的测试时自适应。这是严格无监督的。

### 2.3 Joint — 最小校准 (Minimal-Calibration, Phase 8)

| 项 | 值 |
|---|---|
| 目标域 CIF 标签用量 | **288 小时**（前 12 个 test origin × 24h horizon） |
| 配置来源 | 训练段统计量 |
| 入口脚本 | `scripts/experiments/run_joint_train_full.py` |
| 核心函数 | `run_joint_train`（Stage 1 + Stage 2） |
| 评估 | 后 12 个 **disjoint** test origin（时间上不重叠） |

这是**有监督校准**协议：用目标域前 12 天的 CIF 标签微调 ZS+ 的 attention 权重和
per-direction correction，在后 12 天 disjoint 窗口上评估。**不是纯零样本**，
论文中必须标注为 "joint calibration training with held-out protocol"。

### 2.4 Supervised — 监督基线

| 项 | 值 |
|---|---|
| 目标域 CIF 标签用量 | 训练段全部（~7008 小时） |
| 入口脚本 | `scripts/experiments/run_phase1_complete.py`, `base_zs.evaluate_target` 的 PatchTST 路径 |
| 用途 | 提供 ZS/Joint 的性能上限参照 |

---

## 三、Persistence 基线的等价性

代码库中 persistence 基线有两种写法，它们物理等价（由 Theorem 1 恒等式保证）：

1. **CIF-lag**（`base_zs.py:171`）：`persist_pred = x_cif_test[:, -HORIZON:]`
   — 直接取真实 CIF 历史的最后 24 小时。

2. **rs→CIF 合成**（`run_fused_five_full.py:241`）：
   `pred = last_window_rs * ef_r + (1 - last_window_rs) * ef_nr`
   — 取 rs 历史最后 24 小时，经物理层重建 CIF。

两者数值差异 < 0.05 gCO2/kWh（Theorem 1 验证误差）。保留两种写法因为：
- CIF-lag 在 `evaluate_target` 中更直接（已有 `x_cif_test` 缓冲）
- rs→CIF 合成在 direction eval 中更自然（只有 rs 输入）

**不强行统一**，以避免破坏历史 benchmark 数字的精确可复现性。新脚本可任选其一。

---

## 四、Test Origin 语义

- **test origin** = 一个测试窗口的起点（绝对时间索引）
- 测试段约 1752 小时，`stride=TEST_STRIDE=24` → 约 58 个 origin（每天一个）
- **Joint 协议的 12+12 切分**（`run_joint_train_full.split_origins`）：
  - `train_origins = all_origins[:12]`（最早的 12 天）
  - `eval_origins = all_origins[12:24]`（接下来的 12 天，disjoint）

---

## 五、协议标注规范

所有结果文件和论文表格必须标注协议：

| 标注 | 协议 | 含义 |
|---|---|---|
| `ZS` | 纯零样本 | 目标域 0 CIF 标签 |
| `ZS+` | 测试时校准 | 目标域 0 未来 CIF 标签（用历史做 branch fusion） |
| `Joint` | 最小校准 | 目标域 288h CIF 标签，disjoint eval |
| `Sup` | 监督 | 目标域训练段全部 CIF |

当前 headline 数字对照（leak-fix 前的存档值，重跑后会更新）：

| 方法 | 协议 | Median MAE | 备注 |
|---|---|---:|---|
| Persistence | — | 51.52 | 基线 |
| BasisMix+ | ZS+ | 46.89 | 纯零样本 SOTA |
| PatchTST | Sup | 41.47 | 监督上限 |
| Joint-trained | Joint | 40.53 | 最小校准，0.47 余量 |
