# V191草稿：迟发布月量不能自动识别地区小时偏置

2026-10-29，Sol。停止候选的识别/数据审计，非新预测法或突破。来源只用已有V188 proper样本，July cutoff，不使用cal/test小时或2025评分。此前动态隐状态想法不能直接升级为完整CIF实验。

## 候选及精确障碍

V152已有μ=a−κ，其中a=EF·M/sum(M)是能源锚，μ是时间平均CIF，κ=a−μ。候选令x=(a,κ)满足月状态式x_next=A x+b_r，月报只观测H x=a，H=(1,0)。若b_r已知，普通可观测矩阵[H;HA]在A12≠0时可满秩，但这不允许忽略未知目标地区b_r。

对任意常数d，取e=(0,1)，x'=x+e d、b'_r=b_r+(I−A)e d，则A x'+b'_r=x_next+e d，而H x'=H x。故任意长度、任意延迟的粗月量似然在该变换下相同（观测噪声也取相同分布），μ'=μ−d。增加一个乃至无限个月的同类月边际不解除该gauge。证明仅属于这个允许未知地区迁移项的状态空间模型；不是所有算法或任意物理过程的普遍不可能性定理，不是原创可观测性定理。

增广z=(a,κ,b_a,b_κ)，T=[[A,I],[0,I]]，不可观测方向v=(0,1,−A12,1−A22)。Tv=v、H_aug v=0。因此普通已知offset的满秩不能当成未知地区的latent风险已被月量识别。

## 数据和停止条件

V188固定原小时样本每地区最多Apr/May两月，实际六折连续转移19/17/17/17/18/20。每区仅一个转移时，允许地区intercept的回归设计D=[x,I_region]中I_region已满行秩，A预测列不增加rank；给A任意值均可由各区intercept吸收。公共intercept或b=g(月特征)可另外作迁移假设并被源外检验，但不能称源月量独立识别；V39–40已经检过相关粗误差/持续偏置适配。

不能虚称原始数据不存在：canonical Jan–May metadata中27源有5月、TEC4月、SPA2月。当前审计不启用这些额外早期fine标签；可另立协议重新分配相同小时数到更长时期以估计within-region动态。但这最多改善source A/b估计，不能单靠更多source标签消除未知目标b的gauge。

## 最小解除信息（仅针对这一维gauge）

已知A时，新增一个可靠本地线性统计c′z且c′v≠0可固定这一个平移自由度；不代表足够恢复整个小时CDF或原性能gate。

- 一个月同口径的真实时间平均CIF μ，联合已知a可给κ；这是本地响应标签，不再是严格zero-local-CIF合同。少数随机小时仅给有噪估计，未精确固定月均。
- 真实Cov(G,C)/mean(G)给κ；等价可给mean(C)与mean(GC)、mean(G)的联合矩。仅完整小时总G或再多燃料月总量不够，V15已有同G反例。
- 可信目标b的一个横截约束若与(I−A)e方向非正交也可破此gauge，但它是额外可检验迁移/物理假设，不可由source计数或月报直接认证。
- 月内燃料/CIF时序摘要可能区分具体物理例子，须检查其算子是否不消灭该方向；不是任意日总量或任意一条遥测都足够。V19/V23/V131–138已显示少量摘要有目标依赖，不能重新包装成首次发现。

## 不重复映射

|候选|既有出处|本轮边界|
|---|---|---|
|粗组成Kalman/季节状态|[V9](COARSE_DYNAMICS_V9_PROTOCOL.md)|普通状态空间不是新算法；V191特指不可观测κ及地区offset|
|月锚≠时间均值|[V15](IDENTIFIABILITY_V15_RESULTS.md)、[V152](V152_HOURLY_MOMENT_DIAGNOSTIC_RESULTS.md)、[V163](V163_LEVEL_SPREAD_DIAGNOSTIC_RESULTS.md)|恒等式/两小时物理witness复用，不称新辨识结论|
|粗反馈估计持久地区偏差|[V39](RESIDUAL_PERSISTENCE_V39_RESULTS.md)、[V40](COARSE_RESIDUAL_V40_RESULTS.md)|强行公共offset是新增假设，先前普通适配未解决|
|历史月量→小时风险|[V186](V186_MONTHLY_HISTORY_ZERO_RISK_RESULTS.md)、[V187](V187_EXTENDED_TWO_SCALE_ZERO_RESULTS.md)、[V188](V188_FORWARD_ZERO_RISK_RESULTS.md)|已有严格时间/地区切分与有限监督，不能改称hidden filtering就绕过|
|sharp set/容量/摘要|[V66](V66_MONTHLY_MARGIN_TAIL_IDENTIFIABILITY_RESULTS.md)、[V108](V108_SHARP_SET_METHOD_AND_PAPER_GATE.md)、[V132](V132_MULTISCALE_RISK_SIDE_RESULTS.md)|识别集/新增信息预算已做，不再重复当预测创新|
|源重权/校准|[V173](V173_MOMENT_PROTECTED_RISK_RESULTS.md)、[V175](V175_DIRECT_CRPS_MAP_RESULTS.md)、[V190](V190_SOURCE_EVENT_CIF_RESULTS.md)|不再调source权、阈值或重命名已否定机制|

尚未发现当前合同下有充分不同且可正面检验的新算法候选；这不是声称所有可能算法都已穷尽。V191只完成下面可复核的最小否证与真实支持库存，不做完整CIF重跑。长期目标active；额外本地统计未获取、扩展source时序未启用、未查看地区不开。
