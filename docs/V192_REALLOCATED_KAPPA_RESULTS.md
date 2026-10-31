# V192 跨月重分配源小时预算：未触发小时概率扩展

状态：正式912条月均预测及完整独立audit-r2均terminal exit0。四个预锁概率扩展触发全部false；未运行小时CIF分布评分，原研究目标保持active。

## 实验范围

仅用V188 fold0的19个proper源，在其内部逐区holdout，目标区完全不参与coarse/fine拟合。Apr/May/Jun三个origin共57个query；原cal/test其余10区、H2目标truth和2025标签不进入本轮评估。不是新地区确认或中国目标验证。

每个训练源保留原874–942个fine标签预算，排除目标区后每case总16320–16388小时。spread分布到所有已发布Jan–Feb/Jan–Mar/Jan–Apr月，recent只分最近两月；source集合、总小时数、coarse信息预算与8个固定方法相同，但这是**重分配抽样**，不声称原样本索引相同。Feb月量按旧synthetic规则于Mar31发布，Apr1合法，不是只有Jan可用。月监督是样本均值，κ_sample=a_full_month−μ_sample；目标误差对整月有效小时的时间均值评分。

**预算限制：上述相同预算只在单origin/单fit内成立。** 单origin每分配的19源pool为17262个标签，LOO fit为16320–16388；跨Apr–Jun去重的源标签并集recent=34532、spread=27317，二者不同。因此跨三期汇总不是“全研究唯一标签预算相同”的实验，不能把其差异唯一归因于月份覆盖。只解析114个source JSON的[预算库存脚本](../scripts/v192_budget_union_inventory.py)与`results/v192-budget-union-inventory-r1`独立复核此口径。冻结模型/协议未为此改写；June单origin本身仍是同唯一source预算，已预设的by-origin表中anchor MAE42.855981、分解ET46.242661、direct ET56.812543，仍未胜简单锚。

每个target/origin先封存两分配全部16个预测，再读取该query真μ/a评分。目标held fine不参与拟合/预测；容器确实解码整个既有NPZ bank，再按source raw_id访问，不声称物理上从未解码held数据。source各月历史特征只用当时发布prefix；目标只用预测时已发布月量。2022/23粗历史同旧synthetic retrospective合同，不是实际历史版本的前向确认。

## 月均 CIF 结果

单位gCO₂/kWh，先月等权、再19地区等权。RMSE是地区等权MSE开方；八法×两预算×57=912条，不是912个独立月份。

| 方法 | recent μ MAE | spread μ MAE | spread μ RMSE | spread κ MAE |
|---|---:|---:|---:|---:|
| 最新能源锚 |37.084711|37.084711|55.482362|4.092947|
| 去年同月能源锚 |38.603010|38.603010|56.466824|4.092947|
| direct Ridge36 |96.273613|70.017191|120.341970|76.825976|
| direct ET36 |52.876307|50.053145|66.135923|21.593138|
| 分解 Ridge36 |50.111524|52.964564|73.584268|13.625504|
| 分解 ET36 |44.176761|43.925404|63.866998|3.988982|
| pooled VAR/Kalman |120.061892|110.712495|134.248825|5.896863|
| random-intercept VAR/Kalman |124.293270|122.477930|149.508711|5.829004|

κ列的含义不同，不能隐去：锚/季节设κ=0；direct的κ是共同coarse a_hat−其μ预测的派生量，μ被clip时不保证等于a_hat−最终μ；分解与动态为其原始κ预测。因此该列不是把direct单独训练成κ模型的公平比较。

spread分解ET相对direct ET的μ MAE下降12.2425%，13/19地区改善，但地区paired描述区间为[−14.3921,3.4594]g，跨0；相对Ridge下降37.2648%、14/19改善，区间[−70.3764,9.7122]g亦跨0。相对最简单能源锚反而差6.84069g（18.4461%），仅8/19改善。不能只选择较弱direct控制宣称突破。

分解ET的spread相对recent仅改善0.56898%，低于预锁2%；其May/Jun虽均胜两个direct，但完整strong-direct/地区区间条件未通过。四个候选的预锁概率扩展触发全部false，按协议停止，不调动态惩罚、区名单或阈值，不追加小时CDF评分。

## κ 与动态假设的实际信息

在这个有限19源/Apr–Jun样本中，κ=0的MAE为4.092947g，κ ET为3.988982g，差0.103965g。这个小的点值差不等于稳定、普遍或CIF分布增量；source训练κ还带样本噪声。

共同coarse ET的a MAE为45.245667g，最新能源锚的a MAE为38.515528g。动态pooled/random的a MAE分别112.608291/124.065535g，而κ误差约5.9/5.8g：它们的月均失败主要表现为能量锚预测很差，不能唯一归因于κ无法迁移，更不是任意state-space模型不可能。spread改善一些方法的点值，但没有达到完整触发；不能说重新分配源预算从无价值，也不能将普通ET/partial pooling包装原创。

V191的未知目标κ-offset识别障碍仍在。这里施加共享源动态/目标随机效应均值为零的迁移先验，用源外数据检验预测，而非从目标月报识别offset。没有本地κ标签、没有额外可识别观测，Kalman名称不解除gauge。与V9、V39–40、V152/V163、V186–188的重叠及2024层级SSM/2025空间disaggregation一手文献边界已在[冻结协议](V192_REALLOCATED_KAPPA_PROTOCOL.md)列明。本轮是有边界的预算/假设开发试验，不是新算法或新定理。

## 明确事后的锚与κ组合负诊断

完整主结果已查看后，应主线程建议固定计算“最新已发布能源锚−四个κ预测”，八组合全部保留，不重训、不读新raw/M标签、不改变主触发条件。代码[v192_kappa_anchor_posthoc.py](../scripts/v192_kappa_anchor_posthoc.py)，输出`results/v192-kappa-anchor-posthoc-r1`，terminal exit0。主要诊断先clip至物理[0,1000]；raw只用于误差代数核验，456条组合有8条raw越界，不能把越界值冒称合法预测。

| κ来源 | recent clip μ MAE | spread clip μ MAE | spread raw代数 MAE | spread clip−anchor差的地区描述区间 |
|---|---:|---:|---:|---|
| 分解ET |38.655617|38.719949|38.721719|[0.634671,2.870255]|
| 分解Ridge |42.573532|48.400251|49.053671|[5.652615,20.580693]|
| pooled动态 |37.964497|37.948468|37.948468|[0.070160,1.698645]|
| random动态 |37.945920|37.896092|37.896092|[0.064201,1.595982]|

能源锚自身MAE37.084711；八组合均没有胜它。这是已查看同57query后的探索性负诊断，不能与预注册比较混成独立确认。raw误差恒等式为

`(a_latest−κ_hat)−μ_true = (a_latest−a_true)−(κ_hat−κ_true)`。

所以κ校正误差和未来a变化误差是有符号、可能相关的两个分量，绝对误差/风险不能直接相加；κ MAE略降并不保证最终μ MAE降低，更不能推出小时CDF/尾覆盖改善。clip后另算合法点误差，不能继续声称raw等式逐值保持。

## 预检、独立审计与归档

- 正式：`results/v192-reallocated-kappa-r1`，57个case/912条，session24807 terminal exit0。
- 无held评分预检：`results/v192-reallocated-kappa-preflight-r1`，session28545 terminal exit0；仅6个首尾query/12分配case、96预测，12 fresh未来M与12 fresh目标raw干预逐位一致，不冒称覆盖全部57query。
- 首个audit session45286在首query(Apr AVA)的score列顺序检查exit1，无summary；JSON的sort_keys字母序不同于producer DataFrame插入序。源码、失败log、SHA及原因保留`results/v192-audit-source-r1`。
- 只改audit：完整核JSON键集后按producer固定列顺序构表，所有字段/数值比较保留；helper/runner/协议和正式结果未改。有效审计输出`results/v192-reallocated-kappa-audit-r2`，session90247 terminal exit0；全57 query/114 allocation-refit、5130 source-profile records、1864296个源小时重复出现记录、912损失/32比较重建通过。source角色/抽样/原timestamp与release/prefix、coarse名单、源μ/κ与矩阵、Ridge/动态线代和ET加权叶均值均核验；maxscore7.28e−12、独立预测差2.52e−11。共享tree growing与refit特征构造，并另做prefix与36维特征算术核验，不冒称完全不同算法的端到端审计。

强树基线和partial pooling均为普通方法。地区bootstrap仅19个重采样cluster、2000次，57月共享训练源，反复开发已查看数据；描述区间不是新域分布自由保证。没有CRPS、90%覆盖、宽度或Winkler新结果，因此不宣称完整gate通过或CIF创新，不开未查看地区。

最终核验`results/v192-final-validation-r1/summary.json`：五个成功归档累计1093项输入/输出hash核验（包含重复依赖检查，非1093个唯一文件）、preflight summary传递绑定、退役audit源码SHA均通过；5个新增脚本AST通过。现有环境无ruff/mypy/black/pylint，未声称运行这些检查。所有本轮进程均terminal，未修改旧冻结输入/源码。
