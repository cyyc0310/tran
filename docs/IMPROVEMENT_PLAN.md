# TransCIF 架构优化与改进路线图

> 基于 2026-07 对 docs/ 全部文档的深入阅读、29 区域 LORO benchmark 结果分析，以及 2022-2025 年顶会/顶刊文献调研撰写。

---

## 一、当前架构局限性

### 1.1 模型表达能力的根本约束

**现状**：当前 `AdaptivePersistDLinear` 仅 ~18K 参数，包含 AvgPool(25) 趋势分解 + 两层 Linear + 配置条件偏置 + 持久性门控。

**局限**：
- 单一 pooling 核大小无法同时捕捉日周期（24h）、周周期（168h）和更长期的负荷/天气模式；
- 线性趋势/季节分解无法建模可再生占比中的非线性 ramp 事件（如云穿、风切变）；
- 当目标区域的可再生结构（太阳能 vs 风电主导）与源区域差异较大时，线性模型没有足够的容量来学习该差异。

**参考**：
- `PatchTST: A Time Series is Worth 64 Words` (ICLR 2023) — patch 编码显著优于逐点 MLP
- `TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting` (ICLR 2024) — 多尺度分解可以同时捕获细粒度和粗粒度模式
- `iTransformer: Inverted Transformers Are Effective for Time Series Forecasting` (ICLR 2024) — 变量间交互比时间维注意力更重要

### 1.2 配置空间的粗粒度描述

**现状**：区域配置仅使用 `(mean_renew_share, ef_nonrenew)` 两个标量。

**局限**：
- 两个区域的 `mean_renew_share` 可能相同，但一个以太阳能为主（日周期明显），另一个以风电为主（随机性更强），导致"伪近邻"问题；
- 缺乏燃料结构信息（煤/气/油比例）、储能容量、跨区互联程度等；
- 配置维度太低，config-conditioned MLP 的有效输入空间只有二维，限制了条件化建模的细粒度。

### 1.3 跨域泛化的边界失效

**现状**：US_FPL、US_PJM、US_ISNE 等低可再生区域误差是监督上限的 2.5-3.1×；29 区域中仅 12/29 的纯 ZS 击败 persistence。

**局限**：
- 当前加权采样策略（`weight = 1/(config_distance + ε)`）是启发式的，没有域泛化理论保证；
- 对所有源域做平均训练，可能在训练集平均误差最小化的同时，在边界区域产生严重负迁移；
- Theorem 2 确认了 U 型难度曲线（中等可再生占比最易、两端最难），但模型没有针对性地为边界区域建模。

### 1.4 数据异构与缺失问题未系统处理

**现状**：三个司法管辖区（AU AEMO、UK National Grid ESO、US EIA-930）的数据在时间戳格式、字段定义、缺失模式、采样间隔和时区上不完全一致。

**局限**：
- 没有统一的 missing mask 处理机制，缺失值被简单插值或忽略；
- 不同来源的排放因子口径可能不完全可比（是否包含进口电力、输电损耗等）；
- 温度数据来自 Open-Meteo，与官方发电数据的来源层级不同。

### 1.5 不确定性的覆盖率尚可但区间过宽

**现状**：90% split-conformal 在 25/29 区域有效，但平均区间宽度约为 MAE 的 6.8×。

**局限**：
- 一次性 split-conformal 对分布漂移不敏感，无法自适应调整；
- 不分状态的全局 quantile 造成高波动时段的 interval 过宽、平稳时段的 interval 可能过窄；
- CRPS 竞争力不足，不适合需要概率校准的下游任务（如碳感知调度中的风险约束）。

### 1.6 工程可维护性问题

**现状**：模型核心定义内嵌在 `scripts/run_unified_eval.py` 中，消融实验和部署脚本各自复制了模型类。

**局限**：
- 任何模型变更需要修改多个文件；
- 缺乏统一的模型注册和版本管理机制；
- 模型核心定义内嵌在 `run_unified_eval.py` 中，消融脚本和部署脚本各自复制了模型类代码，修改一处需要同步多处。

---

## 二、性能优化方向

### 2.1 P0 — 统一实现和实验协议 [优先级最高]

**原因**：当前模型定义散落在 `run_unified_eval.py`、`ablation_study.py`、`deployment_warmup.py` 等文件中，代码重复且不一致。这是所有后续改进的前提条件。

**方案**：
- 从 `run_unified_eval.py` 提取 `AdaptivePersistDLinear` 及 `zs_plus_predict`、`train_zero_shot`、`evaluate_target_region` 等核心函数到一个统一模块；
- 所有脚本从该模块导入，消除重复定义；
- 建立统一的配置 schema（区域配置、训练超参、数据路径）。

**预期收益**：模型变更只需修改一个地方，消融实验可确保比较的是同一个模型。

### 2.2 P1 — 多尺度分解与模型容量温和提升

**原因**：当前 AvgPool(25) 单一尺度无法同时捕捉日周期 (24h)、半日周期 (12h)、周模式 (168h) 和短时 ramp (1-6h)。

**方案**：
- 替换单层 AvgPool 为多尺度分解：并行运行多个不同 kernel_size 的 AvgPool（如 7, 13, 25, 49, 73），每个尺度独立的 linear head，最后通过可学习权重或 attention 融合；
- 参考 TimeMixer 的 past-decomposable mixing 设计，但保持当前低参数预算（< 100K）；
- 保持 persistence gate 不变。

**预期收益**：改善峰谷预测精度；降低高波动区域的长 horizon MAE；对复杂可再生结构（风光混合）更鲁棒。

**风险**：模型容量增加可能导致源区域过拟合；需重新调 kernel sizes 和融合策略。

### 2.3 P1 — 域风险感知的训练策略

**原因**：当前均匀采样 + 距离加权的训练方案缺乏对边界区域的显式保护。

**方案**：
- 在训练中引入 domain-risk-aware 采样或加权：不仅按 config distance 加权，还按当前区域的训练误差分配采样概率（类似 Focal Loss / Hard Example Mining）；
- 将 leave-one-domain-out meta-objective 引入训练：训练时模拟缺少某个域，评估对该域的影响；
- 可选：在 loss 中加入风险方差惩罚项（参考 VREx），减少"训练集平均好但边界区域崩溃"的问题。

**预期收益**：提升 US_FPL、PJM、ISNE 等边界区域；降低区域间 MAE 标准差。

**风险**：过度优化最差区域可能损害中间配置区域；需要严格嵌套验证避免信息泄漏。

### 2.4 P1 — RevIN 可逆实例归一化

**原因**：可再生占比的绝对水平和方差在不同区域、不同季节间差异巨大，当前模型需要学习处理这种分布偏移。

**方案**：
- 在模型输入前做可逆实例归一化（RevIN）：减均值/除标准差，输出后再还原；
- 归一化统计量从输入窗口计算，不涉及目标标签；
- 参考 `Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift` (ICLR 2022)。

**预期收益**：减少 level bias，让模型专注于学习时序模式而非绝对水平；改善跨季节的 temporal-OOD 场景。

**风险**：归一化可能掩盖配置空间中的水平差异信息（需要保留 config bias）；与 config-conditioned bias 需要额外协调。

### 2.5 P2 — 自适应在线不确定性校准

**原因**：当前 split-conformal 是静态的，无法适应分布漂移和季节切换。

**方案**：
- 替换为自适应在线 conformal（参考 ICML 2022 Adaptive Conformal Predictions for Time Series）：根据近期 residual 覆盖情况动态调整区间宽度；
- 对不同状态（低/高可再生率、低/高波动、工作日/周末）使用状态条件化的 quantile；
- 加入遗忘因子处理非平稳性。

**预期收益**：在保持覆盖率的同时缩小区间宽度 20-30%；改善高波动时段的 calibration。

**风险**：在线更新存在响应延迟；突发事件下可能暂时失去覆盖。

---

## 三、模型替换或升级建议

### 3.1 P2 — 配置条件化 MoE（Mixture of Experts）

**原因**：不同配置区域的物理规律不同（太阳能 vs 风电 vs 化石主导），单一模型本质上是所有区域的折中。

**方案**：
- 设计 4-6 个轻量专家（每个类似当前 18K 结构）：
  - 低可再生/化石主导专家
  - 高太阳能专家
  - 高风电专家
  - 高波动 ramp 专家
  - 平稳 persistence 专家
- Gate 网络使用配置 + 最近 48h 的波动率和均值特征；
- 每个样本仅激活 2-3 个专家（稀疏 MoE），保持推理成本可控。

**预期收益**：减少边界区域折中；gate 具备物理解释性；总参数量 < 100K。

**风险**：专家可能塌缩到少数几个；gate 可能学习区域 ID 而非可迁移模式；路由策略需要额外调优。

### 3.2 P2 — 语义配置向量替代二维标量

**状态**：🔧 **部分完成（Stage A, 2026-08-12）**。US 8 区域的 7 燃料分项（煤/气/核/油/
水/光/风）已从 EIA-930 Adjusted 列提取（`scripts/data/extract_fuel_breakdown.py`），
UK 17 区域的 9 燃料 mix 从 Carbon Intensity API 保留
（`scripts/data/extract_uk_fuel_breakdown.py`）。config 向量从 2 维扩展到 9 维
（`[mean_rs, ef_nr/1000, coal, gas, nuclear, petroleum, hydro, solar, wind]`），
`AdaptivePersistDLinear` 和 `train_zero_shot` 已支持混合维度池（2 维区域自动零填充）。
新增 `cif_from_fuel_shares()` 多燃料物理分解（`physics/decompose.py`）。
**AU 4 区域待办**：需 NEMED venv 重跑拉 DUID 级燃料类型（见下方 TODO）。

**原因**：这是当前最根本的能力瓶颈。两个标量无法区分"20% 太阳能 + 20% 风电" 和 "40% 太阳能"。

**方案**：
- 将配置从二维扩展到 8-12 维：
  - `mean_renew_share`, `solar_share`, `wind_share`, `hydro_share`
  - `ef_nonrenew`, `ef_renew`
  - `storage_capacity_ratio`, `interconnection_ratio`
  - `load_factor`, `timezone_offset`
- 使用 FiLM (Feature-wise Linear Modulation) 或 hypernetwork 将配置注入模型；
- 缺失字段使用 learnable mask token。

**预期收益**：根本性改善跨域泛化；区分伪近邻；为 Theorem 2 提供更精细的迁移难度度量。

**风险**：新字段可能不完整或口径不一致；需要额外数据收集工作；配置维度增加可能在小源域集上过拟合。

> **AU 燃料数据 TODO**：`scripts/data/generate_nemed_regions.py` 当前只取
> `Plant_Emissions_Intensity` 做 renew/nonrenew 二分。NEMED 的 DUID 表本身包含
> `FuelType`（褐煤/黑煤/气/水/风/光/屋顶光伏），需调用 `download_genset_map`
> / `DUDETAILSUMMARY` 拉取并按燃料聚合。需独立 Python 3.11 venv（nemed pin
> pandas<2.0）。完成后 AU 4 区也将拥有多维 config。

### 3.3 P3 — 时序基础模型作为冻结特征提取器

**原因**：主流通用时序预训练模型（TimesFM、MOMENT、Chronos）已在百万级序列上预训练，可能具备跨域迁移能力。

**方案**：
- 仅做冻结 backbone 的 pilot 对照实验：
  - 将 `RenewShare` 输入 TimesFM / MOMENT / Chronos，取中间层或末层 hidden states；
  - 在其上接轻量 linear head + 物理重建层 + persistence gate；
  - 与当前 `AdaptivePersistDLinear` 做相同 LORO 评估。
- 如确有改善，再考虑 adapter/LoRA 微调。

**预期收益**：潜在提升跨市场泛化能力；提供与 SOTA 基础模型的对标。

**风险**：
- 通用时序预训练数据与电网 CIF 的分布差异可能较大；
- 推理延迟和显存需求增加（MOMENT 约 85M 参数，TimesFM 约 200M）；
- 模型许可和复现成本上升；
- 不建议当前阶段投入大量资源自行训练 Time-MoE 或 Timer-XL。

### 3.4 P3 — iTransformer 或现代 Transformer 对标

**原因**：为论文提供更强的内部基线。

**方案**：
- 实现一个 iTransformer 版本作为可选的 backbone（替换 DLinear）；
- 使用冻结的 iTransformer 配置作为监督上限；
- 不改变物理层和 persistence gate 结构。

**风险**：参数和显存需求显著增加；可能与低容量设计哲学冲突。

---

## 四、数据处理与特征工程改进

### 4.1 P1 — 缺失感知与质量分层

**原因**：不同数据源的缺失模式不同，当前简单插值可能扭曲时序模式。

**方案**：
- 输入增加 missing mask 通道（与 RenewShare 同维度）；
- 训练时对输入做 missing pattern augmentation（随机 mask 掉部分时间步）；
- 使用 Huber loss 或 trimmed loss 降低异常值的影响；
- 建立统一的数据质量评分：时间戳连续性、物理范围检查、负值检测、传感器延迟。

**预期收益**：提升实时流中的鲁棒性；降低插值误差对预测的放大。

### 4.2 P2 — ramp-aware 训练损失

**原因**：当前 MSE loss 对 ramp 事件的梯度较小，模型倾向于平滑预测。

**方案**：
- 在 loss 中加入 ramp 权重：对实际变化率较大的时间步给予更高权重；
- 使用 event-weighted evaluation：单独报告 ramp 时段（CIF 变化 > 阈值）的 MAE。

**预期收益**：改善高波动时段的预测；使模型对调度决策影响大的时段更准确。

### 4.3 P2 — 时间戳与时区标准化

**原因**：澳洲、英国、美国使用不同时区，夏令时规则各异。

**方案**：
- 统一将所有时间戳转换为 UTC；
- 记录每个区域的本地时间偏移，作为可选的配置特征；
- 所有周期性特征（如时间编码）基于 UTC 小时。

---

## 五、部署与可维护性建议

### 5.1 P0 — 统一模块化

**原因**：当前模型定义在多个脚本中重复，任何修改需要同步多处。

**方案**：
- 在 `scripts/` 下创建 `transcif_model.py` 统一存放 `AdaptivePersistDLinear` 及所有消融变体；
- 在 `scripts/` 下创建 `transcif_pipeline.py` 统一存放训练、推理、评估、ZS+ 校准函数；
- 所有脚本从此模块导入，消除多文件中的重复定义。

### 5.2 P1 — 实验可复现性

**原因**：论文复现是学术可信度的基础。

**方案**：
- 每次运行自动保存：
  - 完整配置（data path, hyperparams, seed）
  - git commit hash
  - Python 环境信息
  - 输入信息集标记（ZS / ZS-TTA / ZS+）
  - 所有指标
- 提供单条命令复现全量 benchmark：`python scripts/run_unified_eval.py --full`

### 5.3 P1 — 严格 zero-shot 信息集审计

**原因**：当前 `mean_rs` 可能从完整 CSV 计算，包含测试期信息。

**状态**：✅ **已完成（2026-08-12）**。`load_region_data` 现在只用训练段（前 80%）计算
`mean_rs` 和 `config` 向量；UK `ef_nr` 反估也限制到训练段。6 处脚本重复的 config
构造已同步修复。7 处内联的 `1/(dist+0.05)` 加权提取为共享 `config_weight()` 函数
（`physics/bounds.py`）。详见 `docs/PROTOCOL.md` 的信息集定义。

**方案**：
- 为每个实验明确记录三类信息集：
  - Training information（源区域训练数据）
  - Live input information（目标区域实时 RenewShare + 配置）
  - Calibration information（目标区域 CIF 观测流，仅 ZS+ 可用）
- 每个 fold 运行前自动检查是否读取了测试区间统计量。

### 5.4 P2 — 预部署难度估计

**原因**：新区域上线前需要知道预期性能。

**方案**：
- 基于配置向量、配置距离和源区域历史误差，训练一个轻量难度预测器；
- 预测器仅使用配置和源域统计，不读取目标标签；
- 输出：预期 MAE、预期 ratio vs persistence、置信度。

### 5.5 P2 — 版本管理和模型注册

**原因**：多个实验版本和消融变体需要有序管理。

**方案**：
- 每个训练好的模型保存为 named checkpoint（`epoch`, `model_name`, `git_hash`, `config_hash`）；
- results JSON 中记录模型版本标识；
- 提供 `verify_paper_numbers.py` 的增强版本，可对新结果做一致性检查。

---

## 六、优先级排序总结

| 优先级 | 方向 | 具体工作 | 预期时间 | 依赖关系 |
|:---:|---|---|---|---|
| P0 | 统一实现 | 提取公共模块，消除重复定义 | 1-2 天 | 无 |
| P0 | 信息集审计 | 修复 mean_rs 泄漏，明确 ZS/ZS-TTA/ZS+ 协议 | 1 天 | 无 |
| P1 | 多尺度建模 | TimeMixer 风格多尺度分解（< 100K 参数） | 3-5 天 | P0 |
| P1 | RevIN 归一化 | 可逆实例归一化 | 1-2 天 | P0 |
| P1 | 域风险权重 | Domain-risk-aware 训练策略 | 2-3 天 | P0 |
| P1 | 缺失感知 | Missing mask + 鲁棒 loss + 数据质量分层 | 2-3 天 | P0 |
| P2 | 语义配置向量 | 二维 → 8-12 维配置 | 3-5 天 | P0 |
| P2 | MoE 架构 | 配置条件化稀疏 MoE | 3-5 天 | P1 多尺度建模 |
| P2 | 自适应 conformal | 在线自适应不确定性校准 | 2-3 天 | P0 |
| P2 | ramp-aware loss | Ramp 加权的训练和评估 | 1-2 天 | P0 |
| P3 | 基础模型试点 | TimesFM/MOMENT 冻结对照 | 3-5 天 | P0 |
| P3 | iTransformer 基线 | 内部 Transformer 对标 | 3-5 天 | P0 |

---

## 七、预期性能目标

| 指标 | 当前值 | P0+P1 后目标 | 最终目标 |
|---:|---:|---:|---:|
| config-only ZS 中位数 ρ (vs PatchTST) | 1.24 | ≤ 1.12 | ≤ 1.05 |
| config-only persistence 胜率 | 12/29 | ≥ 22/29 | ≥ 26/29 |
| 最差区域 ρ | 3.11 | ≤ 2.0 | ≤ 1.6 |
| ZS+ persistence 胜率 | 29/29 | 保持 29/29 | 保持 29/29 |
| 90% interval 有效区域数 | 25/29 | ≥ 28/29 | 29/29 |
| 区间宽度/MAE | ~6.8× | ≤ 5.0× | ≤ 4.0× |
| 区域间 MAE 标准差 | 当前高 | -20% | -40% |
| 单 fold 复现速度 | 手动多步 | 一条命令 | 一条命令 |

---

## 八、相关文献索引

### 8.1 多尺度与高效时序模型

- `PatchTST: A Time Series is Worth 64 Words` — Yuqi Nie et al., ICLR 2023
- `TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting` — Wang et al., ICLR 2024
- `iTransformer: Inverted Transformers Are Effective for Time Series Forecasting` — Liu et al., ICLR 2024
- `DLinear: Are Transformers Effective for Time Series Forecasting?` — Zeng et al., AAAI 2023

### 8.2 分布偏移与域泛化

- `RevIN: Reversible Instance Normalization` — Kim et al., ICLR 2022
- `AdaRNN: Adaptive Learning and Forecasting of Time Series` — Du et al., NeurIPS 2022
- `Non-stationary Transformers` — Liu et al., NeurIPS 2022

### 8.3 时序基础模型

- `TimesFM: A decoder-only foundation model for time-series forecasting` — Das et al., ICML 2024
- `MOMENT: A Family of Open Time-series Foundation Models` — Goswami et al., ICML 2024
- `Chronos: Learning the Language of Time Series` — Ansari et al., TMLR 2024
- `Timer-XL: Long-context Transformers for Unified Time Series Forecasting` — ICLR 2025
- `Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts` — ICLR 2025

### 8.4 不确定性量化

- `Adaptive Conformal Predictions for Time Series` — Zaffran et al., ICML 2022
- `Conformal Prediction for Time Series` — Stankeviciute et al., NeurIPS 2021

### 8.5 缺失数据处理

- `PriSTI: A Conditional Diffusion Framework for Spatiotemporal Imputation` — ICDE 2023
- `NETS-ImpGAN: Networked Time-Series Prediction with Incomplete Data` — Zhu et al., ACM TKDD 2024
- `S4M: S4 for Multivariate Time Series Forecasting with Missing Values` — ICLR 2025
