# V193：CIF benchmark评估合同补全与验证

本轮补全的是类型、信息合同、评分cohort和失败保护，不是新预测/校准/选择算法。有效版本为**assessment-r3**；完整audit-r2已terminal exit0，最终核验通过。不训练模型、不打开新地区、不更改旧输入或旧结果，不声称CIF性能突破或新覆盖保证。

## 已有能力与实际补全

| 能力 | 原有位置与范围 | V193新增范围 |
|---|---|---|
| outer target/seed→电网/国家宏平均 | `src/transcif/benchmark.py`已有 | 没有重复实现训练sweep |
| 月量only、源/目标隔离、asof、先预测seal后评分 | `research/gate3.py`、`references.py`已有 | 把声明与artifact按key绑定，检查合同和同评分mask |
| weighted atom-CDF quantile、CRPS、点/区间指标 | `research/weighted_metrics.py`已有 | 直接复用，不另造加权概率算法；补Winkler |
| 条件部分识别界 | `physics/envelope.py`、V66/V72等已有 | identified-set与概率区间分型，拒绝把前者标nominal90%；只做toy guard，不重复锐界推导 |
| 全样本/accepted决策风险 | `evaluation/scheduling.py`已有 | 固定原地区权重条件化的选择风险、同时报告另一种非空地区条件宏平均及分母 |
| 输入、输出hash/审计 | 各pipeline已有 | 同key信息合同逐字段比较、显式校准白名单、缺标签/空接受与负例拒绝 |

新模块为[assessment_v193.py](../src/transcif/research/assessment_v193.py)，adapter为[v193_benchmark_assessment.py](../scripts/v193_benchmark_assessment.py)。它是可复用评估接口加旧gate3适配器，**没有把全部legacy研究脚本、generic benchmark或所有识别集统一迁移**。

## 固定数据、合同与有效评分范围

只读取旧`research-prefix/gate3/*_other400`已密封预测：BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS，2024全年366日、24h；prior/gaussian/diverse固定seed0，level_template/ridge/extra_trees用原reference。六法共15372个probability artifact，全部适配、写新seal后才读取旧目标hourly标签及旧score表；但这些是已查看开发数据，新seal不把它们变成新盲测。不是原三seed平均的旧macro原样复现，而是**选定seed逐window**复现。

每个地区每天独立检查24个精确UTC小时、全部valid及finite/正总发电。可评分2266个region-day、54384小时；缺标签296个region-day保留missing（不声称只要预测存在就有真值）。六法展开为13596 scored与1776 missing method-window；同forecast key、同label mask。各区完整日：

| BPAT | CISO | ERCO | FPL | ISNE | MISO | NYIS |
|---:|---:|---:|---:|---:|---:|---:|
|361|363|251|357|208|364|362|

总体和各阈值都保留缺失分母。地区/月完整表可见归档，不把这些天当独立电网样本；有效日不是随机缺失的保证。

源/目标区域不得交叠；同key的region/origin/available_at/period_end/sources/horizon/track/fit_end必须相同。唯一声明白名单差异为calibration_kind/regions/start/end。source residual transfer与none不同，不意味完全同预算或同处理：旧reference query base用2022–23拟合，残差来自另一2022模型在2023的误差，不能称最终base的严格held-out split-conformal，更不保证目标覆盖。

合同检查只认证**所声明字段和密封数组的一致性**，不能证明程序没看过标签、实际发布元数据真实或EF外生。月报仍为controlled synthetic30d，非真实issued-vintage；固定EF非负已报发电代理，不是物理全电网/消费侧/生命周期CO₂。不能将V193写成严格真实公开月报部署benchmark已完成。

## 旧指标复现与新增Winkler

单位gCO₂/kWh；小时加权后每区等权。闭端点central90%，名义90%只是定义，不是实际target校准承诺。分布mean点指标保持旧定义。完整13596窗口七项旧指标最大复现差1.85e−13。

| 方法 | MAE | CRPS | nominal90区间实际coverage | 平均宽度 | Winkler90 |
|---|---:|---:|---:|---:|---:|
| prior |61.2051|53.6336|48.4926%|82.5077|781.2413|
| gaussian |59.9022|52.2598|46.3729%|81.1285|725.8068|
| diverse |54.3715|46.6268|48.0685%|81.3220|582.8838|
| level_template |43.4615|32.3368|83.0912%|158.4274|280.4073|
| ridge |46.8680|34.2288|82.2655%|158.9416|273.4865|
| extra_trees |83.3538|67.8723|56.1586%|146.4286|830.4455|

Winkler为`U−L+20max(L−y,0)+20max(y−U,0)`。该表展示既有模型的另一评分视角，没有新模型增量。ridge的Winkler略低于level_template，同时CRPS较差，不能挑一种指标宣称完整胜出。

## 固定选择阈值：低宽度不是低风险保证

阈值{50,100,200,400,1000}均在评分前固定，仅由预测平均区间宽判断接受，30项完整结果在`selection.json`。未用truth选阈值或挑最优阈值。

主选择风险按**原地区等权，再对接受条件化**：地区r的每个有效小时先赋`1/(7×该区全部有效小时)`权重，然后在所有accepted小时上重新归一。另一口径是在有accepted样本的地区内各算条件风险后等权，地区数会变，两者都保存但不混用。forecast acceptance与有效标签上的acceptance也分别报告。

以预设100g阈值为例：

- diverse接受1725/2266个有标签day，涉及7区，原地区权重接受率72.8040%；accepted CRPS38.9475、coverage41.9973%，全样本则46.6268、48.0685%。cohort已变化，不能把较小accepted CRPS当整体改进；区间较窄亦未提供90%安全性。
- level_template只接受58个有标签day、1个地区，原地区权重接受率2.2952%。accepted coverage100%只描述这很小子集，不是七区覆盖保证。
- ridge与extra_trees在100g下没有accepted标签，条件风险为**None**，不是0误差。另50g下level_template亦无接受；全部空结果明示状态/分母。

没有新增选择保证、跨域交换性或分布自由风险声明。identified set只做类型安全toy检查；本轮没有真实非平凡部分识别集与概率区间的同estimand联合实证，不用宽[0,1000]盒子“包含率100%”装成果。

## 版本、完整审计与修复记录

- r1曾成功运行，但same-cohort只查key集合，未查同key下的信息合同；退役保留`results/v193-benchmark-assessment-r1`，当时四文件源码/hash在`results/v193-assessment-source-r1`。
- r2加入逐字段合同与仅校准白名单、20个拒绝负例；完整运行及audit-r1成功。随后发现月表错误地平均daily RMSE；其原audit没有独立重建月表，不能据其通过称月RMSE已核验。r2/audit-r1退役保留，五文件源码与pre/post hash在`results/v193-assessment-source-r2`。
- r3只把月表RMSE改为`sqrt(mean(daily RMSE²))`（每有效日均24h，即小时加权MSE开方）。462个region×month×method均修复，最大改变量11.48879g，不能把这个数据修复说成无关紧要。window_scores/region_metrics/macro/selection四主输出须与r2逐字节一致。
- 有效完整审计为`results/v193-benchmark-assessment-audit-r2`：独立重建全部旧指标/评分mask/Winkler、30项选择风险及两类地区权重、地区/macro以及462行月表。真实每区首尾完整日×6法×24h=2016个单点独立CDF平方积分，与core逐小时CRPS及saved day均值对照；这不是全13596窗口的独立CDF算法重算。

负例涵盖未来release/period、target混入fit/cal、虚假region/temporal split、identified set冒充nominal90%、interval冒充CDF、非法weights/nonfinite、不同forecast/cohort/scoring mask、同key信息字段差异及白名单越权。解析Winkler例子、40个独立toy CDF积分与空selection None同时核验。无新训练/模型搜索。

## 与文献的重叠及发表边界

[EnsembleCI（E-Energy 2025）](https://arxiv.org/html/2505.01959v2)已经跨11电网评估、提供代码/数据，并与CarbonCast比较；其描述明确采用历史能源/碳/天气序列。V193的month-only、zero-local-hour合同与其遥测丰富设定不同，但多电网预测/开源比较本身不新。

[Uncertainty-Aware Decarbonization（HotCarbon 2024）](https://arxiv.org/html/2407.02390v2)已做CIF conformal区间与调度风险，使用2021训练、2022源内时间验证/测试；[概率CIF分解（Energy Reports 2024）](https://doi.org/10.1016/j.egyr.2024.05.002)已有分位数概率预测。因此calibrated interval/selective decision不能作为首次机制主张。既有V66/V72的部分识别与选择前沿也不能抹去。

可能的窄发表价值是**有严格信息边界、可重现失败保护和同cohort结果的CIF资源/评估协议**，而不是把现成技术拼在标题就成为原创框架。定向文献核查不证明这种组合“世界首次”，本轮未继续无边界搜索或宣称可发表已获认证。要形成完整benchmark发布，仍需数据/许可证及accounting card、公开版本与release-vintage track、预算和source/cal角色ledger、非平凡identified-set同estimand适配及跨pipeline测试；真实中国小时准确性目前不能验证。当前只交付prototype assessor和一条已有gate3 adapter，未实现这些剩余项，也未取得新域性能或新概率保证。

所有本轮V193进程的最终状态/hash见`results/v193-final-validation-r1/summary.json`；原研究目标保持active。

最终核验：有效producer-r3的50项、audit-r2的57项输入/输出hash全部通过（共107次，含重复依赖），r1四文件/r2五文件退役执行源码SHA均匹配；4个新增Python脚本AST通过。有效audit最大旧指标差1.85e−13、独立Winkler差4.55e−13、2016小时CDF逐小时差1.99e−13、其day平均差5.68e−14，462行修复月表与30项选择汇总全部通过。r3 session79099与audit-r2 session2437均terminal exit0；20个负例、40个toy CDF与empty-selection None检查通过。当前环境无ruff/mypy/black/pylint，未声称运行这些工具。
