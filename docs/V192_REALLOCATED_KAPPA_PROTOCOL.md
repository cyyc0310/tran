# V192 固定源小时预算的跨月 κ 可预测性：有限开发协议

状态：待代码只读审阅；任何新增 held-out fine 性能尚未读取。不是原创算法、确认性新域试验或CIF突破。V191否定未知目标偏置下的识别论证；这里检验一个额外的、可被源外数据反驳的共享动态/偏置先验。

## 固定区域、时点与预算

唯一名单为V188 fold0的19个proper源。其余10个cal/test区不读取fine标签；不取2025或H2 query truth。在这19区内部leave-one-region-out，三origin固定2024-04-01、05-01、06-01，累计57个region-month。每fold目标region完全不参与fine/coarse拟合、标准化、动态参数或随机效应估计；只有该region已发布的公开月量是输入。

月量synthetic release=月首+1月+30天；每origin只取release≤origin的source监督。两种预算分配：spread取所有已发布2024月（分别Jan–Feb/Jan–Mar/Jan–Apr），recent取其中最近两月。每source区总标签数固定为其V188 fold0训练矩阵原小时数（874–942，19区合计17262，目标区删去后按其预算扣除）；两分配、全部方法均同source集合和总fine小时数。每区均分到合法月，余数按月序；各月使用独立固定seed=192和profile_id的无放回随机排列前n。训练月均是该样本的CIF时间均值，κ_sample=a_full_month−μ_sample，不冒称完整协方差。目标真值使用该held源query整月有效小时，仅在本fold所有方法/两分配预测封存后读取。

每个监督profile的历史特征在它自身月首构造，只用当时release合法的extended月量prefix。extended2022/23月量是已查看的synthetic retrospective输入，不是实际历史版本。coarse标签仅18个训练源且release≤origin；所有源raw timestamp<origin。完整source raw_id、profile、prefix、query prefix、源矩和模型参数均归档。当前目标月M及未来M不得进入预测；每origin首尾region做fresh未来月M扰动重拟合，预测应bit-exact。

## 八个固定方法 × 两分配

1. anchor：最新已发布月能源锚。
2. seasonal：去年同月能源锚，缺则anchor。
3. direct_ridge36：历史36维→μ_sample，源区/月等权，proper加权标准化，Ridge α=1。
4. direct_et36：同输入/监督/权重，ExtraTrees 64树、min_leaf3、max_features1、seed192、n_jobs1。
5. decomp_ridge36：共同coarse历史36维ExtraTrees预测a，再减同预算κ Ridge36。
6. decomp_et36：同a预测，再减κ ExtraTrees36。
7. dynamic_pooled：二维x=(a,κ)/1000的普通VAR(1)及共享截距，Ridge α=1；A奇异值上限.98。目标初始κ来自上述κ Ridge；目标已发布a为无噪观测，以源创新协方差进行Kalman条件更新，迭代至最新已发布月，再无观测外推到query月。
8. dynamic_random：同VAR但加入源region随机截距的L2收缩（斜率惩罚1、region偏差惩罚1）；目标region偏差取零，即共享均值，不用目标κ。其源内偏差不是目标可识别参数；这是标准partial pooling假设。

coarse a模型用同18源所有合法2022–24月监督，历史36维、同64/leaf3/seed192 ET；两分配共享，所有方法拥有相同coarse信息预算。训练μ/κ样本各区月等权（不按采样数扩大某月权重），coarse训练亦每区总权重相同。动态transition按区等权，完整state/transition与A/b/Q/P记录；没有合法转移显式失败，不伪造。Q、初始P仅source估计，加1e−8I保证正定；a观测采用精确线性条件化，更新P对称化。动态不假设已识别目标offset，也不输出分布自由保证。各μ预测clip[0,1000]；κ预测单独保留原值以诊断，μclip可能破坏a−κ恒等式，记录clip前值。

## 先预测后评分、预锁判据及停止

57×2×8=912月均预测记录，不做小时CDF评分。另每query保存true a/μ/κ、训练/输入边界，但这些真值只在本fold16个预测完成后生成。主指标月等权再region等权的μ MAE/RMSE；附κ MAE（预测κ按anchor/seasonal用0，direct按共同a_hat−μ_hat推导，分解/动态用自身κ）、a预测误差及a−κ误差分解，不把κ好坏等同于CIF分布改进。三origin各自和总体均报；两分配同query配对。19region paired bootstrap（seed192，2000重采样）仅开发描述；不能增加样本数或作确认性CI。

概率扩展触发条件预锁：某分解/动态方法在spread分配对两个强direct方法均达到总体μ MAE至少5%降低、至少12/19地区改善、paired差的95%描述区间上界<0，且May和Jun各自MAE均胜两direct；并对其recent版本总体MAE至少改善2%。未达到即停止本候选，不重调A/阈值/区名单。达到才另冻结概率扩展协议：同源小时和严格source/cal区域隔离，比较强直接/锚定/异方差及朴素保守法的CRPS、90%宏/最差覆盖、宽度/Winkler；原全部gate保留。月均成功不自动触发新地区或创新主张。

## 既有工作与文献边界

V9已有Kalman，V39–40已有粗反馈/region持续偏置，V152/V163已有a与κ分解，V186–188已有普通历史风险/前向切分，V191已有未知目标offset gauge。V192增量仅为把原相同fine标签数分散至更长的已发布源月份，以source内部跨区域、跨origin检验共享κ先验；不是新识别定理或新Kalman/层级方法。

[Dynamic hierarchical state space forecasting (Liu et al., 2024)](https://onlinelibrary.wiley.com/doi/10.1002/sim.10097)已研究跨单位借力与目标历史的层级SSM；本轮的普通partial pooling/动态控制不声称原创。[Spatial disaggregation of time series (Tobar et al., 2025)](https://arxiv.org/abs/2509.04065)依赖空间/辅助变量及anchoring条件，不可直接为此处zero-local-CIF提供识别保证。这里只据一手摘要核边界，未声称全面文献穷尽。

运行前只读review；预检不读held truth。生产输出与独立审计各新目录，atomic finalize且输入begin/end hash绑定。审计必须独立重建release/raw时效/角色/预算/抽样/prefix、独立source矩与Ridge/动态线性代数、叶均值及全部损失、同query分母和触发条件；共享树生长明确注明。
