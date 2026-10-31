# V13 设计与原始文献审查

2026-10-24，Sol。运行前设计审查，不是 V13 结果。使用 scientific-critical-thinking 指引明确可证伪命题、监督时点和对照；未读取未查看地区标签、未实现预测方法。执行细节以同批冻结的 [V13 协议](WEATHER_COARSE_V13_PROTOCOL.md) 为准。

## 1. 从 V12 能推断什么

[V12](WEATHER_INCREMENT_V12_RESULTS.md) 的源选天气 donor 相对 native 仅改善约1.05%目标 CRPS，描述区间跨零；PACE 形状改善但小时覆盖仍不足，LGEE 形状反而恶化。QRF 原始 H2 误差库在 PACE 的逐小时支持诊断不佳，而 LGEE 的主要问题不能同样归为原始误差范围不足。上述是已查看目标开发证据，不能把两区名称或误差直接写成下一轮门控规则。

V13 检验两个不同命题：**G：**仅用源训练的合法特征能否识别天气增量何时有益，并优于固定强度/普通收缩；**P：**把合法粗组成预报不确定性通过 CIF 比值传播、再迁移源细尺度误差，能否在覆盖适当时优于完整端到端误差学习，而不靠统一放宽。G 的成功不证明 P，反之亦然。

## 2. 最小两轮及真正的比较对象

R1 固定 native、V12 tau=.5 天气端点、端点间燃料轨迹半收缩，以及 Ridge/ExtraTrees 三动作损失门控；不是继续用目标结果搜索天气强度。门控动作 `g in {0,.5,1}`，`q_g=(1-g)q_native+g*q_W(tau=.5)`；g 是整日查询级标量。不对每个 donor 单独门控。半收缩在 simplex 投影之后，通常不等于直接把差分天气强度设为.25，也不等于混合 CIF 分布或平均两条独立随机路径。

门控标签来自 proper H1 地理折外教师的确定性中心 MSE，逐日按三动作平均损失加25归一化。Ridge(alpha=10,proper标准化) 与固定128棵、leaf20、max_features=.7的 ExtraTrees 预测三动作损失后取最小值。归一化改变不同难度日的权重，不能声称直接最小化总体原始 MSE 或 CRPS。教师预测必须同时排除外层地区、教师自身地区；其 donor 天气预测也折外。教师是回顾性的地理折外训练例，不称当日实际可发布的按时间前推预报。

R2 只锁定 R1 **源选**天气策略。B/W 两中心比较 global、hetero、location_scale、QRF p=0/.5、标量粗冲击 independent/rank、物理比值 independent/rank 九家族，另保留 exact native/simple 和 fixed/selected donor。所有新风险参考共享相同合法粗预报情景摘要、监督日和随机数预算。门控和风险家族的普通机器学习版本均为强参考，不预设自定义分解会胜出。

源2025H1复用于两轮选型，两轮是同一自适应开发，不是两次独立确认。两轮网格、源回退、对照及停止门槛在第一轮源分数前同时冻结；两轮源选择完成、两轮目标预测全部封存后，才查看六个已查看目标分数。后续根据结果增加模型必须另开版本。

## 3. 粗误差与小时风险的分解合同

记 `R(m,q)_h=sum_f(EF_f*m_f*q_hf)/sum_f(m_f*q_hf)`。每个校准日 i 固定同一个 qbar_i；它由**相同128个合法路由 donor 燃料轨迹的平均**得到，再进入比值。它是全库加权均值的 Monte Carlo 近似，不是精确全库期望；也不等于先比值后取场景平均。

定义 `c_i=R(mforecast_i,qbar_i)`、`f_i=y_i-R(mtrue_month_i,qbar_i)`、`z_i=EF·(mtrue_month_i-mforecast_i)`。则：

`y_i-c_i = f_i + R(mtrue_month_i,qbar_i)-R(mforecast_i,qbar_i)`。

该恒等式只针对相同 qbar 和相同投影/中心合同成立，必须逐日数值核查。物理候选在查询日使用 `R(mquery_draw,qbar_query)+f_i`，不再加完整端到端 e_i。标量参考使用 `c_query+e_i+(EF·(mquery_draw-mquery_point)-z_i)`。后者只是标量替换近似；z_i 并非精确 CIF 日水平误差。

每个查询先固定相同128个粗情景和相同128个等地区细误差日索引。independent 与 rank 只改变其配对：rank 用源 z_i 与查询 EF 粗冲击的秩排列细误差整日索引，重复值用固定随机数打破；不新增或删除索引。这样两个组件的经验边际**完全不变**，才可把差异归于配对。不能用最近秩重新选一批 donor，随后把抽样质量变化误称依赖收益。

物理映射可在给定查询粗预报支持下改变小时支持范围，但不保证覆盖恢复。它假定 source fine error 及其与粗误差的关系可迁移；天气/组成、donor代表性、固定EF及分母非线性仍可导致失败。按同一月份重复到许多日的 z_i 不是许多独立粗误差样本，应报告地区月数量及逐月敏感性。

## 4. 合法时点及季节支持不是同一问题

校准月真实组成只能作**源监督响应**。它不必在那个校准日已发布，但必须在固定模型训练时点、且不晚于最早源评估2025-01-01已经发布；按冻结月表`period_end/available_at <= 2025-01-01`过滤；lag7是小时遥测延迟，非月表发布延迟，不能默认全部2024H2都可用。风险家族共用这一缩减后的日队列，缺失/未发布月都同样排除。exact旧native是单独较低监督预算的锚，不能悄悄重做成新队列后仍称逐元素相同。

查询粗情景及其摘要完全复用已封存 V9 合法谐波预报法及其前推误差合同，不增加读取当前真实月组成的“回测误差特征”。任何额外历史误差特征若未来加入，须满足其伪预报在当时仅用已发布历史、其真实月响应也在当前查询时点已发布。当前版本不另开此特征搜索。

H2误差用于H1评估仍有季节支持差异；排除未发布12月也不解决这一点。若未来扩大到H1风险库，不能用包含该H1后续月份的预测器制造历史伪预测后称时间合法；需更早训练前缀与合法天气，且所有对照同步改变训练支持，或者明确只作回顾性地理迁移试验。对H2重加权不能创造未观测的H1季节响应支持。

## 5. 粗监督不能独自识别小时尾部

两小时发电总量均为1、煤/清洁燃料EF分别为1000/0。每小时煤占比均为.5时，CIF为 `[500,500]`；第一小时全煤、第二小时全清洁时为 `[1000,0]`。两者聚合煤占比同为.5，聚合粗误差可完全相同，小时尾部却不同。因此粗统计只是约束/监督信号；细误差迁移假设不可被代数分解替代。

门控必须实际超过 native 和普通收缩；若全部回退到native，只说明保护起效，不说明天气增量可识别。物理候选应超过同预算标准分布参考和匹配的标量传播；rank 相对 independent 的增量另报，不能把物理支持改变与秩耦合的效果混写。保持覆盖后宽度、CRPS和IS都改善才是效率收益；只降低MSE、增加宏覆盖或扩大区间均不足。

## 6. 原始文献核查与新颖性边界

以下2026-10-24通过原始出版页/作者论文核查，均不提供本项目无目标小时标签条件下的自动保证。

| 原始研究 | 对V13的约束 |
| --- | --- |
| [Jacobs等，Adaptive Mixtures of Local Experts，Neural Computation1991](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf)；[Wang等，Characterizing and Avoiding Negative Transfer，CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Characterizing_and_Avoiding_Negative_Transfer_CVPR_2019_paper.html) | 以输入门控选择专家、抑制有害迁移已有直接先例。V13 normalized-center-MSE门控是标准成本预测/专家选择；这些研究不证明本任务只凭源数据能识别LGEE伤害。 |
| [Ben-David等，Impossibility Theorems for Domain Adaptation，AISTATS2010](https://proceedings.mlr.press/v9/david10a.html) | 原文在其分类学习框架分析迁移所需关系假设及无目标标签的局限。对CIF的设计启示是必须声明细误差关系可迁移，不能仅凭协变量相似声称保证；不将该分类定理直接当CIF回归定理。 |
| [Zhang等，Learning from Aggregate Observations，NeurIPS2020](https://papers.neurips.cc/paper_files/paper/2020/hash/5b0fa0e4c041548bb6289e15d865a696-Abstract.html) | 群体监督到个体预测已有概率框架，包含均值监督回归和等价关系下的一致性分析。“粗到细”不是充分新颖性；聚合约束也不自动识别小时联合尾部。 |
| [Clark等，Schaake Shuffle，JHydrometeorology2004](https://journals.ametsoc.org/view/journals/hydr/5/1/1525-7541_2004_005_0243_tssamf_2_0_co_2.xml)；[Schefzik等，Ensemble Copula Coupling，Statistical Science2013](https://arxiv.org/abs/1302.7149) | 通过排序重排保持边际并携带经验依赖已有成熟方法。V13的粗/细误差秩配对是相关思想的应用，不是新的copula或秩耦合算法；其依赖模板来自源误差，迁移有效性须实验检验。 |
| [Girolimetto等，Cross-temporal Probabilistic Forecast Reconciliation，IJF2024](https://danigiro.github.io/pubs/2024-ctprob.html) | 粗细概率情景与协方差/残差传播已有基础。该文使用线性协调；CIF的随机分母比值不能直接转成等权小时均值约束，也不因非线性映射就自动原创。 |
| [Ćevid等，Distributional Random Forests，JMLR2022](https://www.jmlr.org/papers/v23/21-0585.html)；[Romano等，CQR，NeurIPS2019](https://papers.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf) | 完整条件分布学习、异方差自适应和独立校准均是标准比较对象。普通均值分裂QRF不代表全部DRF方法；未实施conformal步骤就不能称CQR。 |
| [Tibshirani等，Conformal Prediction Under Covariate Shift，NeurIPS2019](https://proceedings.neurips.cc/paper/2019/file/8fb21ee7a2207526da55a679f0332de2-Paper.pdf) | 加权覆盖结论依赖相应分布变化与权重假设；不能用源/目标特征相似、粗统计匹配或经验秩重排代替条件响应迁移及交换性要求。 |

当前可诚实主张的是：在无目标小时历史、合法滞后粗统计和受控天气代理下，对**天气负迁移保护、物理粗误差替换、细误差依赖模板迁移**作严格分离、可复核检验。是否存在方法贡献取决于相对普通门控、标量传播和强条件分布学习的独占收益与后续独立确认；本轮在运行前不命名原创算法，不设自动原创或新地区开启规则。
