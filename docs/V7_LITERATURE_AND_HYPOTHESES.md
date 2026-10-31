# V7：文献边界与可检验机制

独立审查；检索日期 2026-10-23。本文提出开发假设，不登记新的确认性结果；V6 六区 2026H1 真值和成绩均已查看，后续在这些数据上的实验属于开发复测。V6 的真实失败、offset 等价性和识别边界见 [V6 独立创新审查](V6_NOVELTY_REVIEW.md) 与 [两轮结果](COARSE_TO_FINE_RISK_RESULTS.md)。不修改 V6 冻结实现或协议。

## 1. 当前判断

更值得检验的是**源燃料联合轨迹经过目标粗构成的能源比值算子重新组合**，而非继续给同一 CIF 点模型加 offset 或 NNLS 方差。该机制能够改变小时相位、非对称性与尾部；但 analog trajectory、保留依赖、物理重组、聚合监督和在线混合均已有先例。现阶段只能主张一个明确的信息受限问题及待验证的机制组合，不能主张新优化算法、首次粗细迁移或任意 shift 下的覆盖保证。

需要先排除更简单的解释：原点模型只有四源、旧年份，而 V7 使用七源 2024 小时数据。若同七源、同年份、同 donor 权重的直接 CIF analog 已取得相同进步，则收益主要来自更新数据与支持度，不能归因于燃料重组。V6 的源选择结果和人工漂移结果均不能替代这个比较。

## 2. 输入预算与文献的真实可比性

目标部署允许：按当前 issuance 已发布的粗月燃料构成历史，以及已有 calendar。源训练允许：原七区 2024 合法小时燃料发电数据。源 2025 留一区域开发选择、2024H2 校准等细节以 V7 实验协议为准；每一折须排除被留出区的 proper 与 calibration 小时数据。目标当月尚未发布的实际构成、目标小时 CI/发电/需求、天气实况或根据目标真值优化的相位，均不是同预算输入。

检索先通过 Consensus 的公开主题关键词定位，再核查作者、会议或期刊原文。以下是有针对性的直接先例，不是穷尽系统综述；未复现实验不能写成已经击败这些方法。部分文献有额外小时输入，只能作高信息参考，不能删除其关键输入后仍用原方法名声称公平复现。

| 原始论文 | 实际机制或输入 | 对 V7 的限制与可比较部分 |
|---|---|---|
| [Zhang 等，WWW 2026，DGCFM](https://wangdan.people.ust.hk/Publication/WWW26-CIF.pdf)，§3.2、§5.1.2 | 跨区 CI foundation model，双图与历史 CI、电力流序列；所谓 zero-data 是未用目标数据集训练，不等于推理时没有当地小时上下文。 | 不能称首次跨区/zero-shot CI 预测。其历史小时输入与当前预算不同。 |
| [Yan 等，2025，EnsembleCI](https://arxiv.org/html/2505.01959v2)，§4.1 | 三模型动态集成；使用过去 24h 真值 CI、能源生产，以及未来 calendar 和天气预报。 | 动态集成本身非新；不满足无目标小时输入的预算。 |
| [Maji 等，2025 preprint，CarbonX](https://arxiv.org/html/2510.01521v2)，§4.2.1、§4.3 | 零微调 time-series 模型仍接收目标历史 CI 上下文；概率区间用滚动历史残差校准。 | 零微调与零目标遥测须区分，不能直接借用其本地区间保证。 |
| [Terrén-Serrano、Deshmukh 与 Martínez-Ramón，Nature Communications 2026，Probabilistic day-ahead forecasting of system-level renewable energy and electricity demand](https://www.nature.com/articles/s41467-026-69015-w)，Methods: Data/Data structure | CAISO 区域联合预测 demand、solar、wind；sparse weather-feature selection 与 multi-task GP，使用 HRRR 小时天气预报，并比较跨能源/跨节点联合模型及多元 proper scores。 | 是联合能源依赖和风险场景的直接近期先例，不能称 V7 首次验证 joint 优于 independent。其小时天气、当地能源训练数据和任务不同，不是粗月/calendar 预算下已复现的对照。 |
| [Bowen Zhang 等，AAAI 2026，Improving Day-Ahead Grid Carbon Intensity Forecasting by Joint Modeling of Local-Temporal and Cross-Variable Dependencies Across Different Frequencies](https://ojs.aaai.org/index.php/AAAI/article/view/41310)，[原文](https://arxiv.org/html/2601.06530v1)，Data Sets/Experimental Setup | 五变量为 CIF、load demand、renewable generation、non-renewable generation、temperature；过去 24h 输入预测未来 24h CIF。并行 multi-wavelet convolution 与动态跨变量 wavelet correlation；四个澳洲州，评价 RMSE/MAE/SMAPE。 | 是用户关心的小波 CI 方向的直接原文；依赖目标当地小时历史，不是当前月粗信息预算。多频/跨变量表示不是 V7 的创新；该文点预测结果不能当作概率区间或无小时输入迁移的证据。 |
| [Bowen Zhang 等，Sustainability 2024，Experimental Comparison of Two Main Paradigms for Day-Ahead Average Carbon Intensity Forecasting in Power Grids](https://www.mdpi.com/2071-1050/16/19/8580)，§3.3、§4 | 明确比较 source-disaggregated approach（逐 fuel 小时发电预测，再按固定 EF 比值合成 CI）与直接 CI 预测。输入含天气、历史价格/负荷及负荷预测；讨论 curtailment 与跨能源相互作用对独立 fuel 模型的影响。 | 是 fuel→ratio CI 最直接的领域先例。不能将 V7 重组公式、fuel 分解或跨能源依赖称首次；当地细历史与天气等输入不同，仍未在我们的预算下复现。 |
| [Kohut 等，Applied Energy 2025，Unified carbon emissions and market prices forecasts of the power grid](https://doi.org/10.1016/j.apenergy.2024.124527) | 摘要已给出能源构成估计、moving-horizon correction 与概率 CI/价格预测。 | 从能源构成计算 CI 并非新想法；本次只核实摘要，未核实全文输入，不能据此宣称可在当前预算直接运行。 |
| [Singh 等，UAI 2025，Learning from Label Proportions and Covariate-shifted Instances](https://proceedings.mlr.press/v286/singh25b.html)，[全文](https://arxiv.org/html/2411.12334v2) §1、§3 | **回归**中结合源 instance 真值、目标 bag 平均标签和 domain adaptation；设定同一条件标签规律、变化的协变量分布，理论使用随机 iid 成袋。 | 是“源细真值 + 目标粗标签迁移”最直接的近期先例。当前连续时间依赖、能源加权比值、延迟发布和预测未知未来构成，不由其定理直接覆盖。 |
| [Makhija 与 Saket，2024，Weak to Strong Learning from Aggregate Labels](https://arxiv.org/abs/2411.06200) | LLP/MIL 的弱到强学习；对特定 bag 构造给出 boosting 不可能性及受限正结果。 | 不是小时联合概率分布恢复算法，其不可能性也不能直接移植成当前任务的定理；它提醒 bag 拟合不能自动升级成细粒度正确。 |
| [Bergström 等，2024 preprint，Deep Temporal Deaggregation](https://arxiv.org/html/2406.12423v1)，§2、§5.4–5.5 | diffusion 从细粒度人类移动轨迹学习，条件输入是去掉时间的空间 occupancy 分布；用未参与训练区域的 heatmap 生成轨迹，并评价空间边际。 | “源细轨迹 + 目标聚合信息 → 未见域轨迹分布”已有直接先例。其 heatmap 是目标环境的聚合条件；当前还需预测未发布的未来构成。空间边际一致不证明细时间联合规律正确。 |
| [Zhang 等，NeurIPS 2020，Learning from Aggregate Observations](https://papers.nips.cc/paper/2020/file/5b0fa0e4c041548bb6289e15d865a696-Paper.pdf)，Assumption 2、§5.3 | 已有已知聚合算子的概率观测模型与似然；均值回归例子含 Gaussian 分布假设，条件独立的 instance 给出可计算的聚合分布。 | “把细分布经算子推到粗观测上训练”不是首创。月内强时间相关、不同燃料的联合依赖必须另建模，不能套独立小时方差公式。 |
| [Mosley 等，R Journal 2024，DisaggregateTS](https://journal.r-project.org/articles/RJ-2024-035/)，§2、§5 | Chow–Lin 等方法与 sparse temporal disaggregation，用高频指标解释低频总量；应用先分解 GHG 排放，再除季度 sales 得强度。 | 约束总量、再取比值已有实践；当前没有目标高频指标。不能把“比值的和”等同“总量的比值”。 |
| [Girolimetto 等，IJF 2024，Cross-temporal probabilistic forecast reconciliation](https://arxiv.org/html/2303.17277v3)，§3–4 | 概率分布的跨层级/时间线性协调、joint bootstrap、多步残差协方差。 | 保持跨变量/时间依赖并非新；固定线性求和约束和当前未知分母的 CI 比值不同，不能把 coherence 当小时正确性。 |
| [Correa 等，HESS 2026，Q-CODA](https://hess.copernicus.org/articles/30/4457/2026/)，§7.3–7.4 | daily-to-hourly 随机分解结合 KNN、copula 和总量/子日极值约束；区域化版本用经纬度、海拔插值参数，并做留站验证。 | fragments/analog 加依赖与物理约束已有近期强先例。其极值依赖假设和额外空间输入不是当前自动拥有的信息；需学习其假设失效实验。 |
| [Campbell 等，Scientific Data 2024，GODEEEP wind/solar](https://www.nature.com/articles/s41597-024-03894-w)，Methods、Technical Validation | 多年一致的风光小时模拟，使用天气模拟与 plant 技术配置；分别验证月发电与 BA 小时特征。 | 物理 fuel shape 不是新；月量拟合和小时形状验证是两件事。当前不引入其天气/机组元数据。 |
| [Xu 等，ICLR 2025，WR-CP](https://proceedings.iclr.cc/paper_files/paper/2025/file/283066055b0256ca8e3e0c8c96019357-Paper-Conference.pdf)，§4 | 联合 shift 下的 coverage gap 分析，importance weighting 和 Wasserstein regularization；多源目标是未知源分布混合。 | 不能用 mixture 名称或 Wasserstein 距离就继承覆盖保证；目标超出源联合轨迹支持时，粗标签也不能排除不可见 shape shift。 |
| [Schulz、Köhler 与 Lerch，Machine Learning 2026，Aggregating Distribution Forecasts from Deep Ensembles](https://link.springer.com/article/10.1007/s10994-025-06946-3)，§2、§6 | 比较 probability linear pool 与 quantile aggregation，系统分析组合后的分布、dispersion 与 calibration；讨论严重组件错设不能靠组合自动修复。 | 分布混合、分位数平均和参数平均须区分。选择 linear pool 不能自动保证 calibration，也不能将其他任务中的 quantile pooling 经验当当前联合路径的理论。 |

补充的经典最近机制：Schaake shuffle 已利用历史日期重排 ensemble 来恢复跨变量、空间、时间依赖；[Clark 等，2004 原文记录](https://pubs.usgs.gov/publication/70027101)。ECC 也将边际校准与 ensemble rank dependence 组合，[Schefzik 等，2013 原文](https://arxiv.org/abs/1302.7149)。因此“同一天联合抽燃料、独立抽样会破坏 copula”是合理的必要消融，不是可单独宣称的新发现。

## 3. 先把观测算子写对

令小时燃料能量为 \(g_{hf}\ge0\)，总发电 \(G_h=\sum_f g_{hf}>0\)，固定 proxy 排放因子为 \(e_f\)，则

\[
y_h=\frac{\sum_f e_f g_{hf}}{G_h},\qquad
m_f=\frac{\sum_{h\in M}g_{hf}}{\sum_{h\in M}G_h},\qquad
z_M=\sum_fe_fm_f=\frac{\sum_{h\in M}G_hy_h}{\sum_{h\in M}G_h}.
\]

等时长小时的算术均值满足恒等式

\[
z_M=\overline y_M+\frac{\operatorname{Cov}_M(G,y)}{\overline G_M},
\]

其中 covariance 用该月等权总体定义。该式是代数恒等式，不是新估计器；不等长观测须使用时长权重。由粗月能源份额直接得到的是 energy-weighted CI，而非平均小时 CI，更不是每天 24h 的均值或 shape。只学这一项 covariance correction 仍是结构化 offset，不能仅因此声称脱离 V6 offset 思路。

无目标小时分母时，不能从 \(m\) 单独求出 covariance，也不能恢复 phase。两个能源路径可有相同 \(m\) 和不同 \(y_{1:24}\)。条件源轨迹迁移是额外假设，而不是从守恒推出的识别结论。固定 EF proxy 也不包含进口、机组热效率、启停排放或生命周期差异。

诊断要同时量化三种误差：① \(z_M-\overline y_M\) 的尺度；②粗构成预测错误；③给定正确构成后，donor 的小时形状与联合依赖仍错误。若①只占 PACE/AZPS 失败的一小部分，则改善①不能解释大幅相位修复。当前已见目标仅用于这种开发归因，不能再视为盲测。

## 4. 候选 A：联合燃料轨迹的比值重组（优先）

对源 region-month 的有效小时定义 \(\mu_f\)，保留整日联合路径

\[
q_{dhf}=g_{dhf}/\mu_f,\qquad
T_m(q)_{dh}=\frac{\sum_f e_fm_fq_{dhf}}{\sum_fm_fq_{dhf}}.
\]

源 month 构成正确、\(\mu\) 与构成使用同一支持集时，代入源构成应重建源原始小时比值，这是单元可检验的 round-trip。目标则用当前合法的构成预测 \(\widehat m\)，按粗构成距离和季节选择源 donor；每条路径的所有燃料用同一天，保留能源之间的依赖和日均波动。该方法可改变 CIF shape，不等于给旧 CIF 模板加一个常数或统一乘倍率。

**另一项等价性边界：**令 donor 的源能源份额 \(s_f=\mu_f/\sum_j\mu_j\)。当相关燃料 \(s_f>0\) 时，上式精确等于

\[
T_m(q)_h=\frac{\sum_fe_f(m_f/s_f)g_{hf}}{\sum_f(m_f/s_f)g_{hf}}.
\]

因此它也是按 target/source 粗构成比例对源燃料路径做固定乘法 rescaling，再重算 CI。这是透明的情景重组，不是新学出的最优传输映射；“pushforward”在这里是概率分布经确定性函数映射的数学描述。fuel rescaling 的 \(m_f/s_f\) 也不是域适应中保证有效性的联合密度比。零支持燃料的平坦补全不属于上述可逆等价式。论文贡献仍须来自该结构假设在当前信息协议下的可检验增量。

**限制：**这里月均功率归一化不等于每条抽样日燃料能源份额恰为 \(m\)。即使源完整月份满足平均 \(q_f=1\)，kernel 倾斜和有限 ensemble 也可能破坏这个平均条件。只有相应支持集的加权 \(\sum wq_f=1\) 才能声称期望层面的构成守恒；逐月完整路径另行校正后才可能声称该月精确守恒。不得将当前 day-bank 重组笼统命名为 exact energy-conserving。

\(\mu_f=0\) 而目标 \(m_f>0\) 时，设 \(q_f=1\) 是平坦先验；须报告 unsupported fuel mass、其预测权重及敏感性，不能伪称从源观察到该 fuel shape。单燃料目标的固定 EF CI 应为常数，这是有用的极限检验。所有分母为零/极小、无有效整日、负净发电裁剪等处理都须显式记录，不能悄悄制造任意轨迹。

必要比较（同七源、同 2024H1 bank、同 calendar/粗构成、同 draw 预算）：

| 比较 | 识别哪种收益 |
|---|---|
| 同权重、同 donor 索引的 raw CIF direct analog | 重组算子是否增加价值，排除源年份/数量更新。 |
| 同一 fuel 边际但逐 fuel 独立 donor | 联合依赖是否增加价值；不能同时更改 donor 权重而声称纯 copula 消融。 |
| 无构成距离的 pooled bank | 收益来自构成条件选择还是仅 bank 支持扩大。 |
| 确定性 \(T_m(Eq)\) / mean-profile | 比较保留轨迹与 plug-in；明确 \(E[T_m(q)]\ne T_m(Eq)\)。 |
| 简单 source seasonal CIF 和同特征 loglinear/树回归 | 是否复杂机制超出一般条件拟合。 |
| lawful mix forecast 与标注为诊断的 oracle month mix | 分离未来构成误差和条件 shape 误差；oracle 不能参加主方法选择。 |

随机模式统一使用固定 seed 的相同 ensemble 大小，以 ensemble mean 作为点估计；mean-profile 是另一个 estimator，不把它标成原随机分布的精确均值。除逐小时 CRPS/coverage/IS，还需零均值 shape MAE、最低 CI 小时的圆周误差、日内相关、ramp/日均误差；依赖消融尤其需 joint path 的 energy/variogram score 或相应预定统计，单一边际 CRPS 不足以识别 copula 作用。

## 5. 候选 B：按发布延迟回测的燃料构成预测混合

为 latest、seasonal、seasonal+annual-drift 等简单粗构成专家，在历史月真正可发布后评分，分数必须对应该月过去 issuance 当时可见的 prefix；不能用今日完整历史重算过去专家输入。历史粗月可更新粗预测器或 expert 权重，不能当作目标小时校准标签。

分别将每个专家的构成预测经 \(T_m\) 推送 donor 分布，再做概率混合：

\[
P(y\mid\mathcal I_t)=\sum_k\pi_{k,t}\,(T_{\widehat m_{k,t}})_\#P_k(q\mid\mathcal I_t).
\]

这个式子与 \(T_{\sum_k\pi_k\widehat m_k}(q)\) 不同；应明确实现选择。若所有专家只有点构成，这个混合只表达专家分歧与 donor 变化，不涵盖完整构成预测误差。Dirichlet/log-ratio 或历史粗预测误差重采样也是既有工具，不能取一个分布名就称已校准。

可主张的机制问题是：**先做专家条件 pushforward 再混合**，在同样发布延迟和输入下，是否比先平均构成、单专家和普通回归改善分布？指数权重/凸混合/重要性权重不是新优化器；过往粗分数好不能保证未来小时风险。其有效性需要粗专家错误可迁移，以及给定构成、季节后 donor 联合路径近似覆盖目标。

强对照必须得到相同 lawful fuel-forecast 特征及更新时点；不得只给新方法 annual drift，然后把收益归于 transport。分开消融 coarse forecasting、donor selection、ratio recomposition、joint sampling、残差校准。先在源留一区域、2025 选择，六区 2026 只作开发；选择标准和倍率预算须在开跑前固定。

## 6. 候选 C：显式 clock-aware shape（当前不加入）

公开 EIA [About the EIA-930 data](https://www.eia.gov/electricity/930-content/about.html) 直接链接 [EIA930 reference workbook](https://www.eia.gov/electricity/930-content/EIA930_Reference_Tables.xlsx)。2026-10-23 下载的 BAs sheet 有 Time Zone 字段；仅读取注册表，未读取新小时标签。下载文件 SHA256：`7658291595aeebc93cad050267a48dc6b161038f39a8bf156a1405edf503b698`。

| BA | EIA 报送时区 |
|---|---|
| PACE、PSCO | Mountain |
| AZPS、SRP | Arizona |
| CISO、BPAT | Pacific |
| ERCO | Central |
| MISO、NYIS、ISNE、FPL、DUK、LGEE | Eastern |

EIA 页面另外明确 MISO、LGEE 全年报 EST，不随 DST；不能直接把 MISO 映成 Chicago 或把 LGEE 的报告 clock 映成 New_York DST。上述元数据描述**报告时间**，不是全 BA 的太阳时；区域跨经度、机组分布、光伏朝向和调度也不同。已统一为 UTC 的源记录不能再错误重复 localize。

父任务已决定当前不加 clock variant。将来如新增 longitude/latitude、太阳位置或明确 civil-zone 假设，须作为单独新增静态输入实验，冻结来源/版本与映射规则，所有对照共享。不得通过与目标小时真值对齐来定 phase；即使纠正 UTC 相位也不代表识别真实昼夜 shape。仅燃料构成可以选相似 donor，但不能推出太阳相位。

## 7. 贡献与证伪标准

| 可写的条件性主张 | 最低证据 | 当前不能写 |
|---|---|---|
| 在合法粗构成 + calendar 的限制下，显式 fuel ratio transport 比直接 CIF analog 更能迁移 shape。 | 同 bank/权重的 direct 对照；源 LODO 与新外部 holdout 均有增量；phase、shape 与 CRPS 同时报告。 | 首次碳强度迁移、首次 physics-informed CI、比值公式首创。 |
| 特定联合燃料依赖对该算子的分布质量重要。 | independent 仅破坏 copula，边际与权重一致；联合路径指标和失败案例。 | 联合采样/ECC 首创；仅凭一个平均 CRPS 推断依赖已正确。 |
| 发布时点下的粗专家混合能减少构成预测引起的风险。 | prequential archive、单专家/先混合构成/同输入回归对照、粗预测与细风险联动消融。 | 新在线优化、从粗月误差推出小时无分布假设覆盖。 |
| 可审核的信息协议与支持外失败边界。 | hashes/issue-time tests、nullspace twins、unsupported mass、冻结外部样本。 | 粗月拟合良好证明小时 calibration，或任意 concept shift 保证。 |

若 joint 与 direct 持平而均胜旧四源点模型，应把结果解释为支持与数据更新。若 joint 优于 direct 但 independent 持平，不能主张跨 fuel copula 是关键。若只在 oracle 构成中改善，应优先解决 coarse forecasting。若 source LODO 成立、六区开发仍失败，说明当前条件 donor 支持假设不够，不能通过目标后调倍率把失败改称确认成功。

最终论文级新颖性仍需真实、稳定、可归因的增量，而非给经典 analog、混合或比例重组起新名字。将来确认集必须是未查看的新 BA/年份，并在观察其小时真值和分数前封存候选、选择与分析。

## 8. 实现/诊断审查记录

2026-10-23：独立阅读 `fuel_transport_v7.py` 全模块和其 15 项测试，另阅读 `coarse_forecast_v7.py` 与 4 项测试；实际执行两组共 19 tests 通过。bank prototype 未发现必修问题：joint/direct 同权重同 draw；independent 保持 fuel 边际；H1 先截取后解析数值；μ 来自相同 complete-day observed-support；零支持 fuel 显式平坦先验；纳秒/微秒及拆批随机性一致。权重实现为 equal-region **prior** 后全局 kernel 归一，不强制 posterior region 等权，文档与实现一致。批准模块不等于批准最终实验的 LODO、校准或结果主张；这些须在 runner 联调时另审。

另独立阅读 `baselines_v7.py` 全模块和 8 项测试，实际执行 8 tests 通过（1 项 joblib 物理核探测环境 warning）。ExtraTrees/quantile HGB 及 feature normalization 只用 proper；H2 只构建 paired daily residuals 与 signed finite-rank CQR offset；HGB 权重缩为 mean one，避免固定 L2 单位随样本权重总和变化。CQR 是明确的 pooled-hour calibration，不能与 equal-region residual resampling 混称，也没有 target-shift 保证。该数组 API 不含日期，runner 仍须强制 split 与 heldout-region 排除。

最终 [诊断产物](../results/failure-diagnosis-v7/diagnosis_metadata.json) 已独立批准：136 份输入及全部输出 SHA256 一致；修正 inverse-CDF 分位定义与 native 年初 UTC 跨快照匹配后的 11 tests 实跑通过。另从六区 prepared 原始 CSV 独立复算等时均值、能源加权均值、covariance gap、逐月 gap、complete-day shape RMS 和低谷时刻，最大差异 `6.75e-14`。native QA 拒绝重复 `(region, valid_at)` 并要求与 prepared valid-hour timestamps 完全匹配。

在 extra delay 0 下，PACE/AZPS/PSCO 的真实 daily centered CIF RMS 分别为 140.42/114.26/72.07，旧 raw point 为 12.99/12.84/12.22，旧 ensemble mean 为 25.41/25.85/21.94。raw mean-profile 低谷 UTC 08/08/07，对应真实 19/20/19；ensemble mean 已移至 16/17/16。因此“所有 V6 预测仍错相 11–12h”是不准确的，主要问题同时包含幅度不足。以 2026H1 全部保留小时计算的能源加权减等时均值差为 −11.55/−23.19/−8.47；不能用此单项解释巨大的 shape 错误，更不能将半年 gap 当成每个预测日的合法 correction。

PACE 的 oracle 仅替换 shape 或 level 时，小时覆盖分别约 56.1%/47.4%，都未解决整体失败。oracle 替换使用目标真值、移除该组件的不确定性，而且不投影以保持组件恒等式；这些是机制诊断，既不是合法候选，也不是可达风险下界。bank 全体行的 p95/p01–p99 是未加权描述分位，不是 equal-region predictive quantile；max 支持范围与逐源统计更适合说明支持外问题。上述数字均为 V6 的事后开发诊断，不会将其伪装为新的确认性证据。

最终完整阅读 `run_fuel_transport_v7.py`、`FUEL_TRANSPORT_V7_PROTOCOL.md` 与 11 项 runner 行为测试，和其余 V7 core/coarse/baselines/diagnostic 合计独立实跑 **49 tests 通过，1 项环境 warning，1.74s**。已解决并复核：粗锚返回接口、候选网格与协议一致、从 committed source CSV 重算选择并验证 name/factor/status、完整 archive hash 预检先于读取标签、ESS/unsupported mass/fallback 落盘。LODO 同时从 H1 proper 与 H2 calibration 排除留出区；source 只用 2025H1 选择而不受 H2 压力成绩影响；exact direct/independent/pooled 强制沿用主的 forecast/temperature/bandwidth/factor；CQR 只有区间；full-seven-source 拟合与源预测一起封存；目标完整 grid seal 写完后才读取目标小时数值。

审查结论 **APPROVE 开始源开发实验**，无剩余必修实现问题。审查时 runner SHA256 `9489d2e9001bddbb0b75a88fd9bb7583bef67a8a0d247ff97b009faa6e5dc93b`，V7 protocol SHA256 `b353925c23438e2921758c904f1add86df08f351313a26c843a14e0c472bba6f`。这是实现/协议批准，不是方法有效性或论文新颖性通过；真实增量仍须按前述对照与支持边界评价。未修改任何 V6 冻结文件。

冻结前随后仅增加两项 manifest 守卫：复核 recorded delays/seed/samples，以及 completed 绑定 target score-seal SHA256；已独立确认不改变数值算法、候选网格或选择。源实验开始后未改 producer。新分析脚本另行审查 source-fixed 角色、等区/延迟聚合、六个月同步 paired bootstrap 与不同预算 V6 背景比较；这些描述分析不改变开发复测的证据资格。

新分析完整 13 项合成测试独立通过，含 V6 历史对照 full branch、不同目标数据修订及旧 daily bytes 篡改拒绝。新增路径诊断 9 项测试独立通过；Energy Score 使用 L2/sqrt(24) 与有限 ensemble 的完整自配对期望，Variogram Score 使用 p=0.5、276 个无序小时对等权；严格沿用源固定倍率后投影，并分开 stored raw point 与 scenario mean、flat/tied minima 与稳定相位。均未改冻结 producer。

## 9. 实际源开发结果的独立审计

[源分析](../results/fuel-transport-v7-source-analysis/analysis_metadata.json) 已完整生成。独立按 H1 宏覆盖、worst 与 violation 重新选择，未调用 producer 的 `select`；两轮所有角色、精确机制对照和联合跨轮选择均一致。另逐块读取 822,900 条 day scores，自行汇总到 5,460 个 region/delay/semester/configuration 单元及 macro，最大差分别 `9.09e-13`、`2.27e-13`，输出哈希均匹配。

| 源固定角色 | 2025H1 CRPS | 宏覆盖 | worst 地区/延迟覆盖 | 源状态 |
|---|---:|---:|---:|---|
| R1 latest joint，bandwidth 1，factor 1.5 | 30.8166 | 93.2069% | 74.7633% | failed，worst 未达 85% |
| R2 adaptive mixture，temperature .03、bandwidth .3、factor 1.5 | 26.5917 | 96.3363% | 86.3163% | failed，宏覆盖超过 95% |
| 两轮固定 best baseline / simple anchor，factor 1.5 | 35.7406 | 93.9737% | 72.9861% | failed，worst 未达 85% |
| 固定 CQR | 不定义 | 63.8893% | 0% | failed |

联合 primary 因预定最小违规选择 R2，但它不是 qualified primary。该配置 H2 的 CRPS 为 26.9504、宏覆盖 94.5288%、worst 83.0556%；H2 只作压力测试，未参与选择。这里有源开发的机制信号，也有明确失败；不能因 R2 的 CRPS 较低就删除 gate 上限或把 CQR 的跨域失败隐去。

## 10. 完整目标复测后的科学结论

目标 612 份预测档案哈希与 score bytes 已核；独立将 421,200 条方法日评分汇总为 2,340 个地区单元及宏平均，最大差分别 1.82e−12/4.55e−13。另用 difference-first 实现重算 17 个配对比较、1,160 个分层指标行的六个月 bootstrap，最大差 1.82e−12；证据在 [核验目录](../results/fuel-transport-v7-verification/)。

R2 固定主 CRPS 57.0041、覆盖 88.70%、宽度 342.1480、区间评分 418.4930；简单锚为 68.2532、77.82%、281.7025、589.7417。CRPS 和区间评分改善，但宽度增加 21.46%，覆盖仍不足；未通过预定继续标准。R1 覆盖 92.75% 的宽度为 501.1060，也不支持高效覆盖。LGEE 覆盖下降至 76.41%、最差额外延迟 90 日为 73.88%，不能说普遍修复。

R2 joint−independent 的 CRPS 描述区间为 [−0.4855, 0.0872]；路径 ES 为 64.0860 对 64.2305，VS 为 11.7283 对 11.5600，不支持一致 copula 优势。mean−mixture 的 CRPS 描述区间为 [−0.7570, 0.3338]；混合主要体现覆盖—宽度取舍。直接类比到重组有明显改善，但仍须加入同 donor 加性粗水平锚对照，不能把整个差值归因于独创 shape 算法。后续计划与已执行结果明确分开，详见 [V7完整报告](FUEL_TRANSPORT_V7_RESULTS.md)。

更有实质性的研究问题是明确粗观测的识别能力：若每种燃料的候选月路径都归一为均值 1，则它们产生同样的粗月组成，粗标签无法区分 shape 混合权重。需要证明在当前可用观测算子下哪些参数可识别，或者对不可识别的轨迹集合给出条件风险边界；不能仅增加 coarse-matching、NNLS 或 OT 损失就宣称恢复了 phase/copula。引入公开地理或预测天气则是另一个信息预算，须单独对照。这是待完成的理论/方法方向，不是当前已有定理或原创性认证。
