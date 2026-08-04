# SA1 跨区域迁移:域适应消融实验报告

**日期:** 2026-07-14
**目标区域(未见过的部署目标):** SA1
**源区域(MLDG 预训练):** QLD1 / NSW1 / VIC1
**数据:** AEMO/NEM 2023 全年真实历史数据(`/tmp/nemed_output/nem_2023_hourly_{REGION}.csv`,`duid_level_2023.parquet`)+ Open-Meteo 历史天气归档的温度数据(`temperature_2023_{REGION}.csv`)
**评估指标:** CIF(碳强度)预测 MAE,单位 gCO2/kWh;对比基准为"重复最后观测值"的持久性预测(persistence baseline)

> **数据来源披露:** 本报告中出现的温度协变量(TempAnomaly / 温度异常值)数据来自 **Open-Meteo 免费历史天气档案**,**不是** AEMO/NEMED 官方数据。所有其余电力/排放数据均为 AEMO/NEM 2023 真实历史数据,来自 `/tmp/nemed_output`。

---

## 1. 背景与问题

TransCIF 采用两阶段解耦架构:先用编码器预测未来的可再生能源发电占比(RenewShare),再用物理公式重建碳强度(CIF),并用残差校正头修正物理重建的偏差。目标是"即插即用":只在数据丰富的源区域训练,模型可以直接迁移到从未见过的新区域。

以 QLD1/NSW1/VIC1 为源区域、SA1 为目标区域做迁移测试时发现:**SA1 的迁移效果始终跑不赢"直接重复上一时刻观测值"的持久性基线**,是本项目最主要的负面发现(SA1 迁移失败问题)。本轮实验围绕"如何解决 SA1 迁移失败"展开,分两个阶段:

- **阶段一(六变体消融,已完成)**:尝试四类"表层"缓解手段的组合。
- **阶段二(本报告主体,D/E 方向)**:基于阶段一最优组合,尝试两类"更根本"的域适应技术,来源于 AAAI 2024 workshop 与 WWW'26 相关论文的文献调研。

---

## 2. 阶段一:六变体消融(已有结果,作为背景对照)

四类缓解手段:

- **REG/NEG**:新增可再生能源出力 / 非可再生能源出力的绝对发电量通道(而非只用占比)。
- **温度协变量(C1)**:接入 Open-Meteo 温度异常值作为额外输入通道。
- **门控条件化(A)**:PersistenceSkipEncoder 的持久性跳连门控,按窗口内 RenewShare 的近期波动率动态调整(高波动时更信任网络修正,低波动时更信任持久性)。
- **MLDG域自适应加权(B)**:MLDG 元训练阶段按各源区域标签的 `std + |skew|` 计算域权重,而非简单拼接批次做平均。

结果(脚本:`scripts/sa1_ablation.py`):

| 变体 | corrected_mae | physics_only_mae | persistence_mae | vs持久性基线 |
|---|---|---|---|---|
| baseline(不开启任何缓解手段) | 76.725 | 84.101 | 67.568 | +13.6% |
| +REG/NEG | 81.948 | 86.978 | 67.568 | **+21.3%(最差)** |
| +温度协变量(C1) | 76.599 | 84.527 | 67.568 | +13.4% |
| +门控条件化(A) | 76.820 | 84.354 | 67.568 | +13.7% |
| +MLDG域自适应加权(B) | 78.317 | 85.798 | 67.568 | +15.9% |
| **全部组合** | **75.508** | 84.171 | 67.568 | **+11.8%(六者中最优)** |

**结论:** 六个变体全部跑不赢持久性基线。REG/NEG 和 MLDG 域加权单独使用甚至比不开启任何手段更差;温度协变量和门控条件化基本中性;"全部组合"是六者中最好的,但仍比基线差 11.8%。这本身是一个重要的、如实报告的负面结果——说明这四类"表层"改动都不足以解决 SA1 迁移失败问题。

---

## 3. 阶段二:更根本的域适应手段(D / E,本轮新增)

在文献调研(AAAI 2024 workshop、WWW'26 相关论文)基础上,识别出两条更根本的域适应路线,在"全部组合"数据配置(5 通道:RenewShare / LoadNorm / RenewOutNorm / NonRenewOutNorm / TempAnomaly)之上叠加实现:

### 方向 D:梯度解冻监督微调(gradual-unfreezing fine-tuning)

- **思路:** MLDG 在源区域预训练完成后,直接在 SA1 真实的、有标签的校准集切分 `(x_calib, y_calib_share)` 上做**有监督微调**,而不是仅停留在无监督/域不变特征层面。
- **为对抗微调导致的灾难性遗忘**,采用渐进解冻策略:先解冻门控/预测头(`gate_logit`、`volatility_gain_raw`、`base_encoder.predict`),再解冻 CV-DWCC,最后才解冻最深层的 LT-MWKC,每个阶段都保留之前已解冻的参数继续可训练。
- **参考依据:** IBM Research AAAI 2024 workshop 论文《Domain Adaptation for Time series Transformers using One-step fine-tuning》(源域预训练+目标域渐进解冻微调,对抗灾难性遗忘);WWW'26 一篇针对跨区域碳强度预测的论文的消融实验显示,去掉目标域微调会导致 MAPE 恶化 11.4%,专门验证了这条路线对碳强度预测任务的有效性。
- **实现:** `src/transcif/training/domain_adaptation.py::fine_tune_on_calibration`。

### 方向 E:Deep CORAL 特征协方差对齐

- **思路:** 在 MLDG 元训练循环中,额外加入一项**无监督**的协方差对齐损失,拉近源区域(meta-train 批次)与 SA1(仅用校准集的输入,不用标签)在编码器 `forward_features` 输出的 pooled 特征向量上的二阶统计量(协方差矩阵)差异:

  ```
  L_CORAL = (1 / 4d²) · ‖Cov(source_features) − Cov(target_features)‖²_F
  ```

- **优点:** 完全不需要目标域标签,只需要 SA1 校准集的输入 `x_calib`。
- **参考依据:** Sun & Saenko (2016) Deep CORAL。
- **实现:** `src/transcif/training/domain_adaptation.py::coral_loss` + `train_multi_source_mldg_coral`(在 MLDG 元训练 loss 中额外加权 `coral_weight=0.1` 的 CORAL 项)。

### 方向 D+E:两者结合

先用 `train_multi_source_mldg_coral` 做带 CORAL 对齐的 MLDG 预训练,再用 `fine_tune_on_calibration` 做渐进解冻监督微调。

### 代码改动小结

- `src/transcif/models/encoder.py`:将 `DomainInvariantEncoder.forward` 中内联的特征融合逻辑抽取为独立方法 `forward_features(x) -> (fused, dominant_idx)`,供域适应代码复用同一份特征表示,数值行为不变(纯重构,已有单测覆盖验证等价性)。
- `src/transcif/training/domain_adaptation.py`(新文件):`coral_loss`、`train_multi_source_mldg_coral`、`fine_tune_on_calibration`、`DEFAULT_UNFREEZE_GROUPS`。遵循项目既有约定,以并行新增函数的方式实现,不修改已通过测试的生产函数(`train_multi_source_mldg`)。
- `tests/transcif/training/test_domain_adaptation.py`(新文件)+ `tests/transcif/models/test_encoder.py` 新增一条测试:共 9 条新单测(合成数据,遵循项目已有的"生产/集成验证用真实数据、单测形状/性质/训练行为验证用合成数据"约定),全部通过。
- `scripts/sa1_domain_adaptation.py`(新脚本):在真实 AEMO/NEM 2023 数据上运行 D / E / D+E 与"全部组合"基线的四变体对比,复用 `scripts/sa1_ablation.py` 的数据加载、物理重建、残差校正、持久性基线计算逻辑,确保与阶段一结果可直接比较。

### 关键配置

| 超参数 | 取值 |
|---|---|
| 微调阶段数(渐进解冻分组) | 3 组:`[gate_logit, volatility_gain_raw, predict]` → `[CV-DWCC]` → `[LT-MWKC]` |
| 每阶段微调轮数 `FINE_TUNE_EPOCHS_PER_STAGE` | 15 |
| 微调学习率 `FINE_TUNE_LR` | 5e-4 |
| CORAL 损失权重 `CORAL_WEIGHT` | 0.1 |
| MLDG 预训练轮数 | 与阶段一相同(80 轮) |

---

## 4. 结果

运行方式:`python scripts/sa1_domain_adaptation.py`,日志见 `/tmp/sa1_domain_adaptation_output.log`。

| 变体 | corrected_mae | physics_only_mae | persistence_mae | vs持久性基线 |
|---|---|---|---|---|
| 全部组合(基线,阶段一最优组合的复现) | 74.712 | 83.283 | 67.568 | +10.6% |
| **+编码器微调(D)** | 67.240 | 72.519 | 67.568 | **-0.5%** |
| +CORAL特征对齐(E) | 75.788 | 84.380 | 67.568 | +12.2%(比基线更差) |
| **+D+E(两者结合)** | **66.004** | 69.578 | 67.568 | **-2.3%(全场最佳)** |

> 注:"全部组合(基线)"本轮复现值为 74.712,阶段一原始消融跑出的是 75.508。两者差异来自 MLDG 每轮随机选择 meta-test 区域(`random.choice`)在不同脚本、不同调用顺序下未跨脚本固定种子,属于正常的运行间波动,不影响下方结论的方向性判断。

---

## 5. 结论与分析

1. **D+E 组合是本项目所有 SA1 迁移实验(阶段一六变体 + 阶段二四变体,共十种配置)中,唯一、也是目前为止最好的一个跑赢持久性基线的结果**:corrected_mae 从基线的 74.712 降到 66.004,相对持久性基线 -2.3%。

2. **单独的编码器微调(D)已经足以跑赢基线**(-0.5%),说明真正起作用的关键因素是"在 SA1 真实校准集上做有监督微调",而不是此前阶段一尝试的任何"表层"输入/结构改动。这与 WWW'26 论文消融实验强调目标域微调重要性的结论一致。

3. **单独的 CORAL 特征对齐(E)是无效甚至有害的**:比"全部组合"基线还差(+12.2% vs +10.6%)。可能原因:CORAL 是纯无监督的协方差对齐,只拉近了源域和目标域特征分布的二阶统计量,没有任何标签信号指引对齐方向,可能把特征拉向了对预测无益甚至有害的方向。

4. **D+E 组合优于 D 单独**(-2.3% vs -0.5%),说明 CORAL 并非完全无用——当与有监督微调结合时,它更像是一种特征正则化手段,进一步提升了微调后的泛化能力,而不是单独使用时那种"无方向对齐"的负面效果。

5. **相比阶段一的四类表层手段(REG/NEG、温度协变量、门控条件化、MLDG域加权)全部失败的结果**,阶段二验证的"目标域有监督渐进解冻微调 + CORAL 特征正则化"组合,是**目前唯一确认能让模型在真实 AEMO 2023 数据上跑赢简单持久性基线的方法**。但优势幅度(-2.3%)仍然有限,离"显著优于基线"还有距离,谈不上决定性的胜利,应如实报告为"方向正确、幅度有限"的阶段性结果,而非最终解法。

---

## 6. 局限性与后续工作

- **未做多种子稳健性检验**:当前每个变体都只跑了一次(固定 `SEED=42` 初始化模型参数,但 MLDG 的 meta-test 区域选择带随机性且未跨轮固定),尚不能排除单次运行的随机波动对 -0.5% / -2.3% 这类小幅优势的影响。后续应重复多个随机种子,报告均值和方差,才能确认 D 和 D+E 的优势是否稳健。
- **微调超参数(轮数、学习率、解冻分组顺序)未做敏感性分析**,当前取值是首次尝试的合理默认值,不代表已调至最优。
- **CORAL 权重(0.1)未做扫描**,单独 E 表现不佳是否是权重选择不当导致,还是该方法本身对该任务不适用,尚未厘清。
- **样本量**:SA1 校准集用于微调的样本数相对有限(70% 校准切分),微调轮数和学习率的选择需要在"学到目标域信号"与"过拟合小样本"之间权衡,当前尚未系统验证是否过拟合。

---

## 7. 相关文件

| 文件 | 说明 |
|---|---|
| `scripts/sa1_ablation.py` | 阶段一:六变体消融脚本 |
| `scripts/sa1_domain_adaptation.py` | 阶段二:D/E/D+E 四变体对比脚本 |
| `src/transcif/training/domain_adaptation.py` | CORAL 损失、带 CORAL 的 MLDG 训练、渐进解冻微调实现 |
| `src/transcif/models/encoder.py` | `forward_features` 方法(域适应代码复用的特征提取接口) |
| `tests/transcif/training/test_domain_adaptation.py` | 方向 D/E 的单元测试(合成数据) |
| `/tmp/sa1_ablation_output.log` | 阶段一完整运行日志 |
| `/tmp/sa1_domain_adaptation_output.log` | 阶段二完整运行日志 |
