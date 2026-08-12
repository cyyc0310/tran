# 电力碳强度数据：Regime 分析与分层报告

> 基于 29 个 benchmark 区域（AU 4 + UK 17 + US 8）2023 年逐小时数据的系统性诊断。
> 所有统计量用训练段（前 80%）计算，避免测试期泄漏。
> 生成脚本：`scripts/experiments/diagnose_regime_split.py`

---

## 一、MAE 的三层难度结构

按 persistence MAE（"用今天预测明天"的理论天花板）分层：

| 难度层 | persistence MAE | 区域数 | Joint MAE 中位数 | 代表区域 |
|---|---:|---:|---:|---|
| **容易** | <30 | 11 | **15.7** | US_BPAT(5.9), US_FPL(9.0), US_PJM(14.2), US_ISNE(16.2) |
| **中等** | 30-60 | 13 | **40.8** | US_CISO(21.2), UK_London(53.3), US_MISO(40.4), US_ERCO(52.7) |
| **病态** | >60 | 7 | **69.7** | VIC1(113), SA1(91), UK_09(77), UK_08(74), UK_17(73) |

**关键观察**：病态层的 joint MAE 中位数 69.7，几乎等于该层的 persistence MAE 中位数。
这说明**即使有联合训练校准，也无法把病态区域拉到中等层的水平**——它们的 MAE 高
是因为信号本身的日变化幅度就大。

---

## 二、ef_nr 不可比性诊断

AU 区域的 ef_nr 远超 donor pool 中的任何其他区域：

| 层 | ef_nr | 区域数 | Joint MAE 中位数 | 区域 |
|---|---:|---:|---:|---|
| **可比** | <600 | 24 | 41.9 | 所有 US + 大部分 UK + SA1 |
| **不可比** | ≥600 | 3 | 40.5* | QLD1(842), NSW1(875), VIC1(1160) |
| 未知 | — | 4 | 28.7 | UK Scotland 系列（高 rs 无法反估） |

\* 不可比层的 joint MAE 看似不高（40.5），但这只是因为这 3 个 AU 区域里 NSW1 和 QLD1
的 persistence 本身较低（26-50），VIC1 才是真正的灾难（89.6）。

**可比子集 benchmark**：剔除不可比的 3 个 AU 区域后，24 个可比区域的 joint MAE
中位数 41.9（vs 全 29 区域 40.53）。这个反直觉的"剔除后变差"是因为被剔除的
QLD1/NSW1 恰好是相对容易的区域。**结论：ef_nr 不可比本身不恶化整体指标，
但 VIC1 是极端 outlier，单独报告更有意义。**

---

## 三、病态区域的根因分解

| 区域 | persist MAE | joint MAE | gap | 根因 |
|---|---:|---:|---:|---|
| **VIC1** | 113.1 | 89.6 | 23.5 | ef_nr=1160（褐煤），donor pool 无近邻 |
| **UK_09** | 77.0 | 99.7 | -22.7 | ramp 频繁，rs 波动大，校准反而过拟合 |
| **UK_08** | 74.3 | 96.5 | -22.2 | 同上 |
| **SA1** | 91.2 | 55.9 | 35.3 | 风电主导，rs 波动极大，但校准有效 |
| **UK_17** | 72.7 | 69.7 | 3.0 | gas+风混合，persistence 接近天花板 |
| **UK_07** | 67.1 | 67.4 | -0.3 | 同上 |

**两类病态**：
1. **VIC1 型（ef_nr outlier）**：唯一根因是配置空间孤立，物理放大系数爆表。
   只有方向 A（褐煤单独建模）能救。
2. **UK Midlands 型（高波动）**：persistence 本身就高，信号噪声比低。
   校准甚至会过拟合（UK_08/09 的 joint 比 persistence 更差）。
   方向 B（天气）和 C（regime 专家）可能帮助有限——这些区域的 CIF 波动
   部分来自调度决策而非天气。

---

## 四、燃料结构 Regime 聚类

基于 6 种燃料份额（煤/气/核/水/光/风）的层次聚类（4 类）：

| 类 | 主导燃料 | 区域数 | mean_rs 范围 | 典型成员 |
|---|---|---:|---|---|
| **类 1** | 气52% + 核21% | 6 (全US) | 0.06-0.43 | FPL, PJM, ISNE, MISO, NYIS, CISO |
| **类 2** | 气45% + 风24% | 13 (全UK中南部) | 0.27-0.58 | London, Midlands, Wales, England 系列 |
| **类 3** | 风43% + 核28% | 6 (UK北部) | 0.53-0.91 | Scotland 系列, N.Wales, N.W.England |
| **类 4** | 水73% | 1 (孤例) | 0.79 | BPAT |

**关键洞察**：
- 类 1（化石+核）全是 US 区域，类 2/3 全是 UK 区域——**司法管辖区与燃料结构高度耦合**
- AU 4 区域没有燃料数据（未接入 NEMED DUID），不参与聚类
- BPAT 自成一类（水电主导），它的 mean_rs=0.79 与 Scotland 接近但燃料完全不同——
  这是伪近邻的典型例子

---

## 五、伪近邻诊断（方向 A 的核心论据）

mean_rs 近邻但燃料结构迥异的区域对——这些是当前 config 距离加权 `1/(|Δmean_rs|+0.05)`
产生负迁移的根源：

| 区域对 | \|Δmean_rs\| | 燃料 L1 距离 | 比值 | 解读 |
|---|---:|---:|---:|---|
| UK_03 vs US_BPAT | 0.007 | 1.342 | **134×** | mean_rs 几乎相同，燃料完全不同 |
| US_FPL vs US_PJM | 0.006 | 0.603 | **60×** | 都~0.06 mean_rs，但 FPL 纯气、PJM 含煤+核 |
| UK_07 vs US_NYIS | 0.005 | 0.802 | 80× | gas+风 vs gas+水 |

**这些区域对在当前加权下被当作"近邻"训练，但实际上燃料结构迥异**——多维燃料 config
（方向 A）能正确区分它们。

---

## 六、论文建议：分层报告协议

基于以上分析，建议 benchmark 结果分三层报告：

1. **全 29 区域**（完整协议，median MAE 40.50）：保持与现有论文数字一致
2. **可比子集 24 区域**（剔除 ef_nr≥600 的 AU 3 区，median MAE 41.94）：
   审稿人质疑 VIC1 时可以引用
3. **按难度分层**：
   - 容易层 11 区（joint MAE 15.7）
   - 中等层 13 区（joint MAE 40.8）
   - 病态层 7 区（joint MAE 69.7）

这种分层报告既诚实（不回避病态区域），又公平（让审稿人看到在可比区域的实际表现）。

---

## 七、方向 A-F 的数据支持总结

| 方向 | 数据支持 | 预期收益区域 | 状态 |
|---|---|---|---|
| **A 燃料 config** | 表 5 伪近邻 | ERCO/MISO, FPL/PJM | ✅ ZS 改善 9/12（-0.7），ZS+ 持平 |
| **B 天气** | 天气-CIF 相关性（CISO 辐射-0.66） | 太阳能/风电主导区 | ✅ 负结果（ZS +1.7，编码器太简单） |
| **C Regime MoE** | 表 4 四类聚类 | 跨 regime 混合训练 | ✅ **ZS 改善 8/12（-4.2）**，ZS+ 持平 |
| **D RevIN** | 季节摆幅 + ef_nr 跨度大 | 高碳强度 VIC1/QLD1 | ✅ 负结果（ZS +15.4，破坏 rs 绝对水平） |
| **E 分层报告** | 表 1-3 难度分层 | 全部 | ✅ 完成（本文档） |

---

## 八、Stage B/C/D POC 完整结果（2026-08-12）

12 区域 × 3 seed LORO，对比 flagship / RevIN / MoE / Weather 四个模型变体。

### 纯 ZS（无校准）— 只有 MoE 和燃料 config 有效

| 变体 | ZS median | Δ vs flagship | 改善区域数 | 诊断 |
|---|---:|---:|---:|---|
| flagship | 44.9 | — | — | 基线 |
| **MoE (C)** | **40.6** | **-4.2** | **8/12** | ✅ 燃料路由有效，QLD1 -12.1 最显著 |
| 燃料 config (A) | 44.2 | -0.7 | 9/12 | ✅ 温和改善，伪近邻缓解 |
| Weather (B) | 46.5 | +1.7 | 0/12 | ✗ 编码器太简单（AvgPool+Linear） |
| RevIN (D) | 60.2 | +15.4 | 0/12 | ✗ 破坏 rs 绝对水平，物理分解失效 |

### ZS+（测试时校准）— 均衡器效应全面确认

| 变体 | ZS+ median | Δ vs flagship |
|---|---:|---:|
| flagship | 26.10 | — |
| MoE (C) | 26.10 | +0.00 |
| 燃料 config (A) | 26.11 | +0.01 |
| Weather (B) | 26.10 | -0.01 |
| RevIN (D) | 26.15 | +0.05 |

**所有四个变体的 ZS+ 几乎完全相同**（最大差异 0.05 MAE）。ZS+ 的 branch fusion
用目标域历史 CIF 做 anchor，无论 ZS 模型好坏，校准都能拉回同一水平。

### 天气方向（B）的负结果解读

天气-CIF 相关性分析显示物理基础存在（CISO 辐射 -0.66，VIC1 风速 -0.42），但
`WeatherAdaptivePersistDLinear` 的天气编码器（AvgPool(24) + Linear）过于简单，
无法捕捉天气→发电的非线性映射（如太阳能的阈值效应、风电的功率曲线立方关系）。
未来可用更强的编码器（如 CNN/attention）或直接预测可再生发电量而非 rs。

### 跨方向统一结论：ZS+ 均衡器效应

| 改进 | 纯 ZS 效果 | ZS+ 效果 |
|---|---|---|
| 多维燃料 config（A） | 9/12 改善，median -0.7 | 无影响（+0.01） |
| 天气侧通道（B） | 0/12 改善，median +1.7 | 无影响（-0.01） |
| Regime MoE（C） | 8/12 改善，median -4.2 | 无影响（+0.00） |
| RevIN（D） | 0/12 改善，median +15.4 | 无影响（+0.05） |

**论文叙事（路线 1）**："config 精度和模型架构决定冷启动（ZS）性能下限，
而校准数据（ZS+）决定性能上限。一旦有校准数据，模型架构的边际价值消失——
这证明框架的竞争力来自物理分解 + config 条件化 + 校准的组合，而非特定架构。"

---

## 九、校准数据量曲线（信息集分层定量验证）

> 12 区域 × 3 seed，sweep n_train origins ∈ {0, 3, 6, 12, 24}（= 0/72/144/288/576 校准小时），
> 12 个 disjoint eval origins。脚本：`scripts/experiments/probe_calibration_curve.py`。

### 核心发现：非单调曲线，144h 是甜点

| 校准小时 | median MAE | 趋势 |
|---:|---:|---|
| 0h (纯 ZS+) | 36.68 | 基线 |
| 72h (3天) | 33.92 | ↓ 改善 |
| **144h (6天)** | **32.67** | **↓ 最优点** |
| 288h (12天) | 38.79 | ↑ 过拟合 |
| 576h (24天) | 38.14 | ↑ 过拟合 |

**0h→144h 改善 4.0 MAE，但 144h→576h 恶化 5.5 MAE（过拟合）。**

### 区域分层的不同响应

| 区域类型 | 代表 | 校准响应 |
|---|---|---|
| **容易区域**（persist<20） | BPAT, PJM, FPL, ISNE | 校准有害或无益——ZS+ 已足够，joint SGD 扰动 |
| **中等区域**（persist 20-60） | CISO, MISO, ERCO | 144h 校准温和改善（-2 到 -7 MAE） |
| **病态区域**（persist>60） | VIC1, UK_09 | 剧烈波动——VIC1 在 72h 恶化到 152，576h 才恢复 |

### 论文意义

这个非单调曲线比"更多数据→更好"更有价值：

1. **ZS+ 的价值确认**：0 校准标签时 ZS+ 已达到 36.68，证明 test-time calibration 的有效性
2. **甜点区间**：6 天（144h）校准数据即可达到最优点，超过后过拟合
3. **病态区域不可救**：VIC1/UK_09 对校准极度敏感，joint training 在这些区域不稳定
4. **信息集分层的方法论**：不同难度的区域需要不同量的校准数据——"一刀切"的 Joint 协议不是最优

图：`figures/calibration_curve.png`

---

## 附录：数据生成命令

```bash
# 生成此报告
PYTHONPATH=src python scripts/experiments/diagnose_regime_split.py

# 燃料聚类原始数据（Stage A）
PYTHONPATH=src python scripts/data/extract_fuel_breakdown.py
PYTHONPATH=src python scripts/data/extract_uk_fuel_breakdown.py

# benchmark 结果
PYTHONPATH=src python scripts/experiments/run_joint_train_full.py
```
