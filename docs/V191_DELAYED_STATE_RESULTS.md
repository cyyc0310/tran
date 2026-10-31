# V191：延迟月观测隐状态候选停线

2026-10-29，Sol。已完成真实source监督量检查及独立Fraction审计，没有训练预测模型、没有cal/test小时评分或新增地区。**该候选不能借“动态过滤”绕过未知地区κ偏置，当前不推进完整CIF实验；不是新定理、创新成果或所有预测法的不可能性结论。**长期目标active。

状态定义、候选与V1–190重复映射见[冻结查验说明](V191_DELAYED_STATE_OBSERVABILITY.md)。V9已做普通组成Kalman，V39–40已做持续偏置/粗误差适配，V152/V163已有μ=a−κ，V186–188已有月历史与源外前向风险对照；不能换名重跑。V15两小时物理例子在这里明确复用，不当新witness。

## 完成的最小否证

令x=(a,κ)，a是月能源锚、μ=a−κ是小时CIF时间均值。候选模型x_next=A x+b_r，粗月观测H x=a，H=(1,0)。若未知目标地区迁移项b_r允许变化，取e=(0,1)、任意常数d：

x'=x+ed，b'_r=b_r+(I−A)ed。

则A x'+b'_r=x_next+ed，H x'=H x。对相同创新/观测噪声，并相应平移初始状态（或初始状态分布），**条件粗观测似然不变**，但μ'=μ−d。固定source prior不再平移会选定一个解，但那是新增迁移假设，不是月报数据识别了它。

独立精确例子A=[[1/2,1/4],[1/8,1/2]]中，已知b时可观测rank=2；未知常量b增广为4维后rank=3，gauge方向v=(0,1,−1/4,1/2)满足Tv=v、H_aug T^j v=0。12步全Fraction核验。再加同类月边际仍rank3；加一个可靠同口径μ或κ观测则该例rank4。

V15的煤/清洁EF=[1000,0]两小时例子分别给μ=250/500、共同a=1000/3，燃料月总量与完整小时G都相同；在相应b下两条常量状态轨迹给出相同12个月观测。只证明该限定子类物理非空；不能推广到所有EF相同的退化情形、任意调度物理约束或任意模型。

## 实际监督支持

V188锁定原sample/July cutoff下，每源最多Apr/May两月：

|fold|proper区|零状态区|一状态区|两状态区|连续转移|
|---|---:|---:|---:|---:|---:|
|0|19|0|0|19|19|
|1|19|1|1|17|17|
|2|19|1|1|17|17|
|3|19|1|1|17|17|
|4|20|1|1|18|18|
|5|20|0|0|20|20|

累计220个source月统计行、108条转移，跨折重用不独立。每region至多一转移且region intercept自由时，[X,I_region]中intercept已满行rank，A两列额外rank六折全部0。Fraction独立重算完整设计和两个显式斜率null方向；并从原sample重新核角色、profile映射、时效、源统计及转移绑定。κ_sample_proxy=a−sample均值带抽样噪声，**不是完整月协方差的精确估计**；这个噪声不会补足结构rank。

不能虚称数据不存在：更早canonical metadata为27源有Jan–May五个月、TEC四月、SPA两月。此次未启用这些额外fine标签；另立协议延长source时间支持可能帮助估计source动态，但不能独自解除未知目标地区offset的gauge。

## 最小新增信息与停止依据

仅针对这一维gauge：在A已知、其余可观测方向能估计的限定模型内，一个可靠本地统计c′z且c′v≠0可固定该平移。一个同口径月时间平均CIF μ或κ=−Cov(G,C)/mean(G)满足此条件；它们是新增本地响应/联合矩信息，不能继续冒称严格zero-local-CIF。几条小时随机点只能给有噪估计，不能认证整月μ；完整小时G也不够（上述例子保留G）。多一个或无限个相同燃料月边际仍不解除gauge。

已知目标b的独立横截约束也可能解除它，但需要新的可检验迁移/物理假设，不能从source计数认证；V39–40的普通粗反馈结果不能被忽略。月内燃料/CIF摘要是否足够须核其观测算子，V19/V23/V131–138已有目标依赖的价值试验，不重复包装。

因此停止的只是“现有月报会独立识别该隐状态，故直接上latent filter”的论证；不是证明所有公共offset模型或其他统计预测均失败。若后续另立可被源外反驳的offset迁移假设，应与V186/187/188及强直接、普通Kalman/锚/异方差同预算比较并保留完整CIF gate。本轮没有该后续拟合/性能结果，不开未查看地区。

## 文件与核验

主脚本 [v191_delayed_state_observability.py](../scripts/v191_delayed_state_observability.py)，输出`results/v191-delayed-state-observability-r1`；独立脚本 [v191_audit_delayed_state.py](../scripts/v191_audit_delayed_state.py)，有效输出`results/v191-delayed-state-observability-audit-r4`。两个进程均直接terminal exit0，无后台句柄。audit-r1也成功但未显式重算完整source设计的Fraction rank；r2补齐该检查，r3补齐Fraction逐燃料月量相等，r4另从原v153 timestamps按V188矩阵raw_id独立复核July cutoff。r1–r3均保留；r1执行源码按归档SHA精确重构并明确事后提取，不冒称当时快照，r2/r3源码在后续修订前精确保存。没有模型/预测/标签规则修改。

原始目标尚无突破；V191的交付是有出处的重叠映射、真实监督量及限定gauge否证，不是新算法阳性。

审计版本状态：r1退役为缺少完整source设计的显式Fraction rank；r2补齐该rank但缺少Fraction物理witness逐燃料月量断言，因此退役；r3补齐列和但尚未从原timestamp数组独立核源小时cutoff，因此退役。**r4为唯一有效最终审计**，包含上述完整rank及两个补充断言：逐燃料月合计严格相等、全部source raw_id对应原timestamp严格早于July cutoff。旧版均成功执行但范围较窄，并非数值反例或模型变动。最终50项归档hash与两个脚本AST通过，见`results/v191-final-validation-r1/summary.json`。冻结协议及producer未改。
