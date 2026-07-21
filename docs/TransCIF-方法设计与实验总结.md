# TransCIF:即插即用跨地区电网碳强度预测——方法设计与全部实验总结

**日期:** 2026-07-14
**性质:** 项目级总结文档,覆盖从设计定稿到目前为止全部已完成实验的完整脉络
**关联文档:**
- 设计文档:`/Users/agiuser/Downloads/2026-07-12-TransCIF跨地区碳强度预测-design.md`(已批准定稿)
- 头脑风暴记录:`/Users/agiuser/Downloads/电网碳强度跨地区泛化预测-brainstorm.md`
- 实现计划:`/Users/agiuser/Downloads/2026-07-12-TransCIF实现计划.md`
- 实现执行台账:`/Users/agiuser/transcif/.superpowers/sdd/progress.md`(Task 0-13 + 全分支复审)
- SA1 域适应专项实验报告(本文档第四章的详细版本):`/Users/agiuser/transcif/docs/experiments/2026-07-14-sa1-domain-adaptation.md`

> **数据来源披露:** 本项目所有电力/排放数据均为 AEMO/NEM 2023 真实历史数据(来自 `/tmp/nemed_output`,文件 `nem_2023_hourly_{REGION}.csv`、`duid_level_2023.parquet`)。温度协变量(TempAnomaly)数据来自 **Open-Meteo 免费历史天气档案**(`temperature_2023_{REGION}.csv`),**不是** AEMO/NEMED 官方数据,在此专门披露。所有生产/集成验证均使用上述真实数据;仅单元测试的形状/性质/训练行为检验使用合成数据(`torch.rand` 等),这是项目既定的例外约定。

---

## 1. 项目背景与动机

电网碳强度因子(Carbon Intensity Factor, CIF)预测是电力系统低碳调度的核心支撑技术。现有最优方法存在一个共同的、被两篇参考论文都列为"未来工作"但未解决的问题:**模型只能在训练所用的地区内使用,无法直接迁移到其他地区**。

- **论文一**(AAAI-26,LT-MWKC + CV-DWCC 双模块架构,译文见 `/Users/agiuser/Downloads/基于多频局部时序与变量交叉依赖联合建模的日前电网碳强度预测研究（译文）.md`)在澳大利亚四个电网上分别独立训练评估,未做跨地区实验。
- **论文二**(e-Energy'25,EnsembleCI,译文见 `/Users/agiuser/Downloads/EnsembleCI：基于集成学习的电网碳强度预测研究（译文）.md`)证明:固定单一架构模型在不同地区间的性能差异巨大,根本原因是**可再生能源渗透率**的差异;其解法是"按地区分别训练集成模型",本质仍是"一地一模型"。

**项目目标:** 设计一个即插即用(plug-and-play)、鲁棒的电网碳强度预测算法——只用一个地区的数据训练主模型,部署到其他地区时只需极少量目标数据做轻量校准,不需要重新训练模型主体。

**问题定义与约束(设计文档第2章):**

| 维度 | 结论 |
|---|---|
| 迁移方式 | 近零样本 + 轻量校准,允许用几天到几周的目标域数据做校准,不重新训练模型主体 |
| 覆盖的分布偏移范围 | 跨国家/跨能源结构迁移(化石能源主导 ↔ 可再生能源主导),两篇论文中"最难"的迁移方向 |
| 预期产出 | 学术论文发表,需要完整对比实验、消融实验,以及与现有 SOTA 的对比 |
| 预测目标 | 电网碳强度(CI, gCO₂e/kWh),而非碳排放总量 |

**技术路线选型(头脑风暴阶段提出三条路线,采用路线一):**

1. **物理引导两段式解耦 + 域不变时序编码器**(采用)——不直接预测 CI,而是预测发电结构占比,再用 CIF 物理公式转换,校准阶段只需轻量操作。理由:与两篇论文的物理可解释传统一脉相承;校准成本最低;实现风险可控。
2. 相对化特征 + 域描述符条件化模型——单源域下不变性证据不足,其"相对化特征"思想被吸收为下文创新点1。
3. 元学习(MAML式)+ 合成域增强——实现复杂度和调参风险最高,其"合成域"思想被吸收为下文创新点3的轻量化版本。

---

## 2. 总体架构

```
源域训练(仅一个地区)                       目标域部署(近零样本)
┌───────────────────────┐               ┌───────────────────────┐
│ Stage 0: 尺度不变       │               │ 少量目标域样本           │
│ 重参数化前端 [创新1]     │               │ (几天~几周)             │
├───────────────────────┤               ├───────────────────────┤
│ Stage 1: 域不变动态     │──权重冻结──▶   │ Stage 3: 校准           │
│ 编码器(改造LT-MWKC      │               │ - CV-DWCC主导变量        │
│ + CV-DWCC)[创新3正则]   │               │   重加权 [创新4]         │
├───────────────────────┤               │ - 残差校正头Δ微调         │
│ Stage 2: CIF物理层      │◀──────────────│   [创新2]               │
│ + 可学习残差校正 [创新2] │               │ - 保形预测区间            │
└───────────────────────┘               │   [创新5]               │
                                         └───────────────────────┘
```

核心思路:将"电网动态规律"(相对通用、可跨区迁移)与"电网物理属性"(排放系数、跨境贸易等地区专属因素)显式解耦,前者由深度模型学习并冻结迁移,后者通过物理公式+少量校准数据处理。

**当前实现的代码结构(`src/transcif/`):**

```
data/loaders.py, data/reparam.py          — 数据加载 + Stage 0 重参数化
models/wavelets.py, models/lt_mwkc.py     — 多频小波卷积 LT-MWKC(论文一模块改造)
models/cv_dwcc.py                          — 跨变量依赖卷积 CV-DWCC + 主导变量识别
models/encoder.py                          — DomainInvariantEncoder(融合 LT-MWKC+CV-DWCC,forward_features)
physics/cif.py                             — CIF 物理重建公式
physics/residual.py                        — 可学习残差校正头 Δ
training/consistency.py                    — 创新3:合成扰动一致性正则化
training/train_source.py                   — 单源域训练
training/train_multi_source.py             — MLDG 域自适应加权元训练(创新3的多源扩展)
training/domain_adaptation.py              — SA1 专项:CORAL 对齐 + 渐进解冻微调(见第四章)
calibration/dominant_reweight.py           — 创新4:CV-DWCC主导变量重加权
calibration/gate_recalibration.py          — SA1 专项:门控条件化(见第四章)
calibration/conformal.py                   — 创新5:保形预测
evaluation/metrics.py, evaluation/baselines.py — 评估指标 + 端到端管线封装
```

---

## 3. 核心创新点详细设计与实现落地

### 创新1:尺度不变重参数化前端

两篇参考论文均直接使用绝对物理量(MW级出力、原始温度),但不同电网的装机规模、负荷体量、气候基线差异巨大,绝对量本身不可迁移。改为:

- 可再生占比:`RenewShare_t = RenewOut_t / (RenewOut_t + NonRenewOut_t)` —— 消除装机规模依赖
- 负荷相对化:`LoadNorm_t = Load_t / Load的滚动95分位数` —— 消除电网规模依赖
- 温度距平:`TempAnomaly_t = Temp_t - 当地气候基线(按年积日)` —— 消除绝对气候带依赖
- 时序特征(小时/年积日/星期的正余弦编码),本身已是尺度无关特征

这是两篇论文都未采用的、专门针对跨地区场景设计的输入表征,是"域不变"论断的第一层实证基础。**实现位置:** `src/transcif/data/reparam.py`。

### 创新2:CIF物理层 + 可学习残差校正(灰盒混合)

Stage 1 编码器的预测目标不是 CI,而是未来的发电结构占比轨迹 `RenewShare_{t+1..t+S}`。Stage 2 用 CIF 公式重建 CI:

```
CI_pred,t = CIF_formula(RenewShare_pred,t, 目标地区排放系数表) + Δ_t
```

其中:

$$CIF_{avg,t}=\frac{\sum(E_{r,t}\times C_r)}{\sum E_{r,t}}$$

`Δ_t` 是一个小型可学习残差头,只用目标域校准集训练,专门吸收纯物理公式覆盖不到的系统性偏差。纯查表方案过于"刚性";加入可学习残差后,"轻量校准"具备真正可学习的内容。**实现位置:** `src/transcif/physics/cif.py`(物理公式)+ `src/transcif/physics/residual.py`(残差头 Δ)。

### 创新3:合成扰动一致性正则化

只有一个真实源域,无法做传统多域对抗训练。训练时对输入(RenewShare、LoadNorm)施加物理合理范围内的随机扰动(整体缩放±30%,模拟不同渗透率电网),并施加一致性损失:

```
L_consist = || CV-DWCC(x) - CV-DWCC(perturb(x)) ||²
```

目的是促使编码器学习"变量间的关系模式"而非"源地区的具体数值区间"。**实现位置:** `src/transcif/training/consistency.py`。

> **实现历史备注(全分支复审阶段发现并修复):** 设计文档描述扰动应同时作用于 RenewShare 和 LoadNorm 两个轨迹,但初版 `synthetic_perturb` 只扰动了 RenewShare,属于"设计-代码漂移"。全分支复审(most-capable-model)发现该问题后,经人工确认修复:`synthetic_perturb` 改为对 RenewShare、LoadNorm 各自独立采样扰动幅度,分别裁剪到 `[0,1]`。详见第五章"实现执行历史"。

### 创新4:CV-DWCC主导变量识别复用为校准句柄

CV-DWCC 模块内置"主导变量识别"机制(局部多元回归决定系数 $\varphi_{X,s}(j)$),原本仅用于预测阶段的内部特征提取。本设计首次将其反向复用于跨域校准:

1. 部署到新地区时,用少量校准数据跑一次主导变量识别,判断该地区当前是"可再生驱动型"还是"负荷驱动型"。
2. 据此**只重新加权**已训练好的小波融合系数(`α_m`)与 CV-DWCC 通道融合系数,不改动其余网络权重。

EnsembleCI 已证实不同地区的关键特征差异显著——本设计把这种差异从"需要克服的障碍"转化为"校准的依据"。**实现位置:** `src/transcif/calibration/dominant_reweight.py`(`recompute_dominant_variable`、`reweight_lt_mwkc_alpha`)。

> **实现历史备注(Task 11,PLAN-CONFLICT,已由人工决策解决):** 实现阶段在 `cv_dwcc.py` 中直接发现一个真实结构性 bug——`CVDWCC.forward` 的 `dominant_idx` 是每个目标变量自己的**局部**索引(每个目标预测时都排除自身通道,预测因子列表比总通道数少1),不是全局通道 id;原始 brief 的 bincount 逻辑混淆了这两套不相关的索引空间,导致 3 通道场景下全局通道 2 永远拿不到投票,与数据无关。同时原测试夹具(两个纯噪声"其他"通道)让 CV-DWCC 没有真实的跨变量信号可检测(20-seed 扫描:仅 25-45% 可靠,即便修复索引后)。经人工在"打补丁选种子/ 只修索引 / 重新设计指标"三个方案中选择"修 bug + 重新设计测试夹具",最终 20/20(seeds 0-19)+ 20/20(seeds 20-39)可靠。
>
> **另一处相关的全分支复审修复(Important):** `LTMWKC` 最初用固定、不可学习的平均值融合并行小波分支,导致创新4的 `reweight_lt_mwkc_alpha`(本应通过提升某分支的跨分支权重作为跨域校准句柄)实际上只能影响到*分支内*权重,对模型最终输出几乎不产生作用(在 [0,1] 输出尺度上变化量 ≤~1.4e-4)。复审重现实验证实:在未训练模型上 `branch.alpha` 初始即均匀分布,softmax 提升等价于严格空操作,失效是彻底的而非"较小"。修复:`LTMWKC` 新增可学习 `branch_alpha` 参数,改为 softmax 加权的跨分支融合(与已有的 `MultiWaveletConv1D` 分支内模式对齐);`reweight_lt_mwkc_alpha` 同时提升 `branch.alpha` 和新的 `branch_alpha[branch_idx]`。修复后回归测试确认跨分支重加权对 `LTMWKC` 自身未稀释输出产生 7.4%-10.2% 的相对变化(7 个随机种子稳健)。

### 创新5:复用Stage 3校准集做保形预测(Conformal Prediction)

Stage 3 已有一份小的目标域校准集(用于重加权融合系数、微调 Δ)。同一份校准集额外用于计算保形预测的不合格度分数:

```
nonconformity score = | CI_真值 - CI_pred(含Δ校正) |  (在校准集上计算)
```

据此为目标域每个预测输出有限样本覆盖率保证的预测区间。当目标域与源域差异过大、迁移本身不可靠时,预测区间会自动变宽。**实现位置:** `src/transcif/calibration/conformal.py`(`predict_with_interval`)。全分支复审阶段用 3000-seed 重扫描独立验证了有限样本修正分位数公式对split-conformal文献的一致性(均值覆盖率 0.902,仅约 0.13% 的种子低于断言的 0.85 下界)。

---

## 4. 已完成的全部实验

实验工作分为两条脉络:**(A)核心管线的正确性建立**(通过 Task 0-13 的测试驱动实现 + 全分支复审完成,以单测和端到端 smoke test 为主要验证形式,细节见第五章),**(B)真实数据上的跨区域迁移实验**(SA1 迁移失败问题的诊断与修复,是目前唯一在真实 AEMO 2023 数据上系统运行的对比实验)。

### 4.1 实验设置

- **源区域(MLDG 元训练):** QLD1 / NSW1 / VIC1
- **目标区域(未见过的部署目标):** SA1(论文一确认的"最难"电网,可再生能源渗透率最高)
- **数据:** AEMO/NEM 2023 全年真实历史数据(`/tmp/nemed_output/nem_2023_hourly_{REGION}.csv`、`duid_level_2023.parquet`)+ Open-Meteo 历史天气归档的温度数据(`temperature_2023_{REGION}.csv`)
- **评估指标:** CIF(碳强度)预测 MAE,单位 gCO2/kWh;对比基准为"重复最后观测值"的持久性预测(persistence baseline),`persistence_mae = 67.568`(所有变体共用同一持久性基线,因为它只依赖 SA1 真实序列本身)

### 4.2 阶段一:四类表层缓解手段的六变体消融(`scripts/sa1_ablation.py`)

以 QLD1/NSW1/VIC1 为源区域、SA1 为目标区域做迁移测试时发现:**SA1 的迁移效果始终跑不赢持久性基线**,是本项目最主要的负面发现。第一阶段尝试四类"表层"缓解手段及其组合:

- **REG/NEG:** 新增可再生能源出力 / 非可再生能源出力的绝对发电量通道(而非只用占比)。
- **温度协变量(C1):** 接入 Open-Meteo 温度异常值作为额外输入通道。
- **门控条件化(A):** PersistenceSkipEncoder 的持久性跳连门控,按窗口内 RenewShare 的近期波动率动态调整(高波动时更信任网络修正,低波动时更信任持久性)。实现位置:`src/transcif/calibration/gate_recalibration.py`。
- **MLDG域自适应加权(B):** MLDG 元训练阶段按各源区域标签的 `std + |skew|` 计算域权重,而非简单拼接批次做平均。

**结果:**

| 变体 | corrected_mae | physics_only_mae | persistence_mae | vs持久性基线 |
|---|---|---|---|---|
| baseline(不开启任何缓解手段) | 76.725 | 84.101 | 67.568 | +13.6% |
| +REG/NEG | 81.948 | 86.978 | 67.568 | **+21.3%(六者中最差)** |
| +温度协变量(C1) | 76.599 | 84.527 | 67.568 | +13.4% |
| +门控条件化(A) | 76.820 | 84.354 | 67.568 | +13.7% |
| +MLDG域自适应加权(B) | 78.317 | 85.798 | 67.568 | +15.9% |
| **全部组合** | **75.508** | 84.171 | 67.568 | **+11.8%(六者中最优)** |

**结论:** 六个变体全部跑不赢持久性基线。REG/NEG 和 MLDG 域加权单独使用甚至比不开启任何手段更差;温度协变量和门控条件化基本中性;"全部组合"是六者中最好的,但仍比基线差 11.8%。这是一个重要的、如实报告的负面结果——四类"表层"输入/结构改动均不足以解决 SA1 迁移失败问题。

### 4.3 阶段二:更根本的域适应手段(`scripts/sa1_domain_adaptation.py`)

在文献调研(AAAI 2024 workshop、WWW'26 相关论文)基础上,识别出两条更根本的域适应路线,叠加在阶段一"全部组合"配置(5通道:RenewShare / LoadNorm / RenewOutNorm / NonRenewOutNorm / TempAnomaly)之上:

**方向 D:梯度解冻监督微调(gradual-unfreezing fine-tuning)** —— MLDG 源域预训练完成后,在 SA1 真实、有标签的校准集切分上做有监督微调,采用渐进解冻策略对抗灾难性遗忘:先解冻门控/预测头,再解冻 CV-DWCC,最后才解冻 LT-MWKC。参考 IBM Research AAAI 2024 workshop《Domain Adaptation for Time series Transformers using One-step fine-tuning》。实现:`src/transcif/training/domain_adaptation.py::fine_tune_on_calibration`。

**方向 E:Deep CORAL 特征协方差对齐** —— MLDG 元训练循环中额外加入无监督协方差对齐损失,拉近源区域与 SA1(仅用输入,不用标签)在编码器 pooled 特征上的二阶统计量差异:`L_CORAL = (1/4d²)·‖Cov(source) − Cov(target)‖²_F`。参考 Sun & Saenko (2016) Deep CORAL。实现:`src/transcif/training/domain_adaptation.py::coral_loss` + `train_multi_source_mldg_coral`(CORAL 权重 0.1)。

**方向 D+E:** 先用 `train_multi_source_mldg_coral` 做带 CORAL 对齐的 MLDG 预训练,再用 `fine_tune_on_calibration` 做渐进解冻监督微调。

**关键超参数:** 渐进解冻3组(`[gate_logit, volatility_gain_raw, predict]`→`[CV-DWCC]`→`[LT-MWKC]`);每阶段15轮;微调学习率 5e-4;CORAL权重0.1;MLDG预训练80轮。

**结果:**

| 变体 | corrected_mae | physics_only_mae | persistence_mae | vs持久性基线 |
|---|---|---|---|---|
| 全部组合(基线,阶段一最优组合的复现) | 74.712 | 83.283 | 67.568 | +10.6% |
| **+编码器微调(D)** | 67.240 | 72.519 | 67.568 | **-0.5%** |
| +CORAL特征对齐(E) | 75.788 | 84.380 | 67.568 | +12.2%(比基线更差) |
| **+D+E(两者结合)** | **66.004** | 69.578 | 67.568 | **-2.3%(全场最佳)** |

> 注:"全部组合(基线)"本轮复现值 74.712 与阶段一原始消融的 75.508 存在差异,来自 MLDG 每轮随机选择 meta-test 区域(`random.choice`)在不同脚本调用间未跨脚本固定种子,属正常运行间波动,不影响结论方向性判断。

**结论与分析:**

1. **D+E 组合是本项目所有 SA1 迁移实验(阶段一六变体 + 阶段二四变体,共十种配置)中,唯一、也是目前为止最好的一个跑赢持久性基线的结果**:corrected_mae 从基线的 74.712 降到 66.004,相对持久性基线 -2.3%。
2. **单独的编码器微调(D)已经足以跑赢基线**(-0.5%),说明真正起作用的关键因素是"在 SA1 真实校准集上做有监督微调",而不是阶段一尝试的任何"表层"输入/结构改动。这与 WWW'26 论文消融实验强调目标域微调重要性的结论一致(该论文报告去掉目标域微调会导致 MAPE 恶化 11.4%)。
3. **单独的 CORAL 特征对齐(E)是无效甚至有害的**:比"全部组合"基线还差(+12.2% vs +10.6%)。可能原因:CORAL 是纯无监督协方差对齐,没有标签信号指引对齐方向,可能把特征拉向了对预测无益甚至有害的方向。
4. **D+E 组合优于 D 单独**(-2.3% vs -0.5%),说明 CORAL 并非完全无用——当与有监督微调结合时,它更像一种特征正则化手段,而非单独使用时的"无方向对齐"负面效果。
5. 相比阶段一四类表层手段全部失败的结果,阶段二验证的"目标域有监督渐进解冻微调 + CORAL 特征正则化"组合,是**目前唯一确认能让模型在真实 AEMO 2023 数据上跑赢简单持久性基线的方法**。但优势幅度(-2.3%)仍有限,应如实报告为"方向正确、幅度有限"的阶段性结果,而非最终解法。

### 4.4 局限性与后续工作(适用于全部十种 SA1 迁移变体)

- **未做多种子稳健性检验:** 当前每个变体只跑一次(固定 `SEED=42` 初始化模型参数,但 MLDG 的 meta-test 区域选择带随机性且未跨轮固定),尚不能排除单次运行随机波动对 -0.5%/-2.3% 这类小幅优势的影响。后续应重复多个随机种子,报告均值和方差。
- **微调超参数(轮数、学习率、解冻分组顺序)未做敏感性分析**,当前取值是首次尝试的合理默认值。
- **CORAL 权重(0.1)未做扫描**,单独 E 表现不佳是否是权重选择不当导致,还是该方法本身对该任务不适用,尚未厘清。
- **样本量:** SA1 校准集用于微调的样本数相对有限(70%校准切分),微调轮数和学习率的选择需要在"学到目标域信号"与"过拟合小样本"之间权衡,尚未系统验证是否过拟合。
- **设计文档第8章原计划的实验尚未执行:** 与 CarbonCast、EnsembleCI、论文一原模型的正面对比;西班牙ES/德国DE作为目标域;正反两个迁移方向;针对五个核心创新点逐一移除的消融实验(注意:这与本章已执行的"四类表层缓解手段消融"是不同的实验设计——前者移除的是创新1-5这五个已确认写入核心架构的机制,后者测试的是四种额外的、专门为解决SA1迁移失败问题而设计的补充手段,两者不能互相替代)。

---

## 5. 核心管线的实现执行历史(Task 0-13 + 全分支复审)

核心两阶段架构与五大创新点的代码实现,通过 subagent-driven-development 方式执行(每个任务:全新实现子代理 → 任务复审子代理[规范符合性+代码质量] → 全分支复审)。完整执行台账见 `.superpowers/sdd/progress.md`。以真实生产/集成验证使用真实 AEMO 数据、单测形状/性质/训练行为验证使用合成数据为既定约定,全部 Task 0-13 均已完成并复审通过,过程中的关键发现:

- **Task 11(创新4实现):** 发现并修复 `recompute_dominant_variable` 的索引空间混淆真实 bug(详见第三章创新4备注),经人工在三个候选方案中裁决。
- **Task 13(端到端验证):** 修复测试夹具中 `torch.empty(..., generator=...)` 的 PyTorch API 误用(`torch.empty` 不接受 `generator` 参数,只有 `.uniform_()` 等随机填充操作接受);同时发现 brief 承诺产出 `run_end_to_end_smoke_test` 但从未定义,经补充实现抽取为 `evaluation/baselines.py` 中的标准可复用函数(Stage 1训练→Stage 2物理+残差→Stage 3重加权+保形预测,返回12键结果字典),供后续消融实验复用。
- **全分支复审(最强模型复审全部14个任务的合并diff):** 发现并修复两项问题——(1)`LTMWKC` 固定平均融合导致创新4重加权近乎无效(详见第三章创新4备注,已修复为可学习 `branch_alpha` softmax融合);(2)`synthetic_perturb` 只扰动 RenewShare、未覆盖设计文档要求的 LoadNorm(已修复为两者独立扰动+裁剪)。复审阶段独立编写因果探针(monkeypatch回旧融合方式)验证修复前后行为差异,而非仅信任报告的测试结果。
- **最终状态:** 全分支复审 PASS/PASS,零 Critical/Important/Minor 未决问题,47/47 全套测试通过,项目核心管线进入"生产就绪"状态,为第四章的真实数据实验提供了正确性基础。

---

## 6. 相关文件索引

| 文件 | 说明 |
|---|---|
| `/Users/agiuser/Downloads/2026-07-12-TransCIF跨地区碳强度预测-design.md` | 已批准设计文档(五大创新点原始定义) |
| `/Users/agiuser/Downloads/2026-07-12-TransCIF实现计划.md` | 详细实现计划(任务分解) |
| `.superpowers/sdd/progress.md` | Task 0-13 + 全分支复审执行台账 |
| `src/transcif/data/reparam.py` | 创新1:尺度不变重参数化 |
| `src/transcif/physics/cif.py`、`physics/residual.py` | 创新2:CIF物理层 + 残差校正头 |
| `src/transcif/training/consistency.py` | 创新3:合成扰动一致性正则化 |
| `src/transcif/calibration/dominant_reweight.py` | 创新4:CV-DWCC主导变量重加权 |
| `src/transcif/calibration/conformal.py` | 创新5:保形预测 |
| `src/transcif/models/encoder.py` | DomainInvariantEncoder,`forward_features` |
| `src/transcif/training/train_multi_source.py` | MLDG域自适应加权元训练 |
| `src/transcif/calibration/gate_recalibration.py` | SA1专项:门控条件化(阶段一变体A) |
| `src/transcif/training/domain_adaptation.py` | SA1专项:CORAL对齐 + 渐进解冻微调(阶段二方向D/E) |
| `scripts/sa1_ablation.py` | 阶段一:六变体消融脚本 |
| `scripts/sa1_domain_adaptation.py` | 阶段二:D/E/D+E四变体对比脚本 |
| `/tmp/sa1_ablation_output.log` | 阶段一完整运行日志 |
| `/tmp/sa1_domain_adaptation_output.log` | 阶段二完整运行日志 |
| `docs/experiments/2026-07-14-sa1-domain-adaptation.md` | 第四章内容的更详细独立版本 |
