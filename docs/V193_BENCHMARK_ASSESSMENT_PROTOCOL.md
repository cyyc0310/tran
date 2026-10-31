# V193 统一CIF评估合同补全（冻结前审阅稿）

本轮只新增V193文件，不改旧framework/输入/结果。没有新模型、调参、地区或性能优势主张。贡献是可复用、类型明确、同cohort的评估接口与失败保护；不是selective/conformal/部分识别的新方法或保证。

## 已覆盖与本轮缺口

`benchmark.py`已有outer target/seed、target/国家macro；`research/gate3.py`已有month-only LORO、源拟合/残差、预测seal、源角色/asof审计、同月block比较；`references.py`有原点冻结的monthly asof；`weighted_metrics.py`有weighted atom-CDF quantile/CRPS；`physics/envelope.py`有条件容量识别界；`evaluation/scheduling.py`有finite-mechanism决策和全样本/accepted风险；V66/V72已有月量sharp-tail集/逻辑弃权。不能将这些重命名为V193创新。

缺口是上述pipeline没有统一typed forecast artifact + score cohort合同。V193补：预测分布/名义概率区间/条件identified set分型；预测来源/校准角色和时效声明的可检查一致性；相同forecast keys/label-mask；明确missing/empty-selection；weighted CRPS复用旧实现，另补interval score/Winkler和固定选择策略下风险/分母。

## 固定旧预测、全部日级评估

只读`results/research-prefix/gate3/{BPAT,CISO,ERCO,FPL,ISNE,MISO,NYIS}_other400`已有密封预测。固定2024全部366个00UTC origin、24h；6法prior/gaussian/diverse seed0及level_template/ridge/extra_trees seed−1。共7×366×6=15372个概率artifact。identified-set仅作toy类型/拒绝校验，不加入真实预测排行；固定[0,1000]盒子没有新增识别信息，不当成有效风险成果。没有新decoder或模型拟合，旧source预算沿原seal，不将不同方法decoder质量成本说成相同训练算法。

预测阶段先验证旧seal及source provenance，构造全部artifact index和mean/lower/upper等truth-independent数组，再对全体写新seal；之后才读既有目标hourly CSV和旧score表验证。它们均为已查看开发标签，新seal不是新的盲测。全部2562 forecast windows保留。按旧规则逐地区、逐预测日检查：该日必须恰有24个精确UTC小时且全部valid、truth finite/正G，才对该day评分；不完整label的window保留missing状态、0 scoring hours，所有方法使用完全同一个mask，不能暗中按方法择日。所有已评分probability指标须逐window复现旧window_scores（包括原分布mean的MAE/RMSE/CRPS/90%coverage/width/shape/level）；本轮不更改原point definition。

## artifact类型和声明边界

- `predictive_distribution`：有限weighted particles，使用旧weighted_metrics；central90% inverse empirical-CDF quantiles，闭端点。`nominal_alpha=.1`是区间定义，不是target覆盖保证。
- `probability_interval`：独立lower/upper + nominal alpha，允许没有CDF，因此CRPS必须None，不凭区间臆造uniform分布。
- `identified_set`：条件lower/upper、assumptions必填，nominal alpha必须None，不得报coverage90/Winkler/CRPS；可报truth containment这个条件诊断。本轮仅以toy区间验证类型与拒绝逻辑，不报告真实[0,1000]盒子的containment作研究收益；V66/V72非平凡月内tail集合属于不同estimand/cohort，本轮不将其同pointwise概率coverage横比。

统一合同只检查显式元信息/数组/哈希的一致性，不认证算法从未访问标签、实际发布vintage真实或参数外生。旧month-only是controlled synthetic30d，不升级成真实issued-public exact-asof。目标区必须与fit/residual calibration地区均隔离。source-temporal residual transfer允许source区重叠，明确不是新目标保证；若声称region-split conformal则fit/cal地区必须不交，若声称temporal-split则fit_end≤cal_start。旧Gate3 query direct base用2022–23拟合，residual来自另一个2022模型的2023误差，标source_residual_transfer，不能当最终base的严格held-out split-conformal。

## 固定选择风险与权重

接受规则只由密封预测90%区间平均宽≤固定{50,100,200,400,1000}g决定；不按truth选择阈值、不选最优阈值或声称选择后coverage。identified set不参加点风险比较。

主指标依旧每区按有效小时权重、再地区等权；RMSE由地区MSE开方后地区等权，与旧gate3汇总一致。另报按月/区域表。不以日数或seed伪增独立region。每个阈值报告预测总体/有效标签/accepted的window与hour分母、原始全样本风险；选择风险主口径为固定原地区权重条件化：sum_r (accepted_loss_sum_r / all_hours_r) / sum_r (accepted_hours_r / all_hours_r)。同时另报有accepted样本地区的条件risk均值及其地区数，明确两个口径不同。空接受或某区无接受时None/explicit status，不能伪造0误差。报告accepted MAE/CRPS/interval coverage/width/Winkler，不把coverage变宽或拒绝更多当完整方法胜出。

Winkler/interval score：U−L+(2/alpha)(L−y)1(y<L)+(2/alpha)(y−U)1(y>U)。central90% alpha=.1。所有full/selection指标依同一truth mask；不是任意nominal calibration保证。

## 验证和失败条件

冻结protocol/helper/runner后执行。Negative checks必须拒绝未来available_at/period_end、target参与source或cal、声称region/temporal split却重叠、identified set冒充nominal90%、interval冒充CDF、非有限/负权、错或重复forecast cohort；空selection必须None。小维weighted CRPS用独立CDF平方积分校验、Winkler用解析例子；真实每区首尾完整可评分day×6法×24h共2016 CRPS单点以独立CDF积分验证（若缺完整day明确失败，不事后换地区）。全部旧metric逐window复现，全部Winkler与聚合/接受分母由单独审计重建；共享树/模型不存在新拟合。只有限数值核验不声称全面独立软件证明。

只新增`assessment_v193.py`、V193 runner/auditor/tests/protocol/results和新results目录。输出fresh、输入begin/end hashes，旧artifact hash不符则停止。未通过不放宽合同；任何修复保留失败source/log与版本。没有CIF概率优势/新域保证，原研究目标仍active。

## r2 合同修订与r1保留

r1曾成功处理15372 artifact，13596 scored method-windows（2266 unique scored region-days）、1776 missing method-windows；但validate_cohorts只校key而未校同key的信息合同，故退役保留原输出/四文件源码与provenance。r2默认同key合同逐字段严格相等；唯一允许按预先method声明白名单不同的字段为calibration_kind/regions/start/end。region/origin/available_at/period_end/sources/horizon/track/fit_end不得豁免。白名单不为算法信息预算相同背书：reference有源残差处理而inference没有，必须明确记录，不能称完全同预算算法比较。r2重新冻结后完整重跑，六法旧forecast未变。

## r3 月表RMSE修复

r2完整评估及其原audit均成功，但月表对daily RMSE直接平均，错误；原audit未独立重建月表，不能称已核这项。因此r2及其audit仅保留退役证据，r3为唯一有效版本。r3只把月RMSE改为sqrt(mean(daily RMSE²))；所有有效day恒为24h，等价小时加权MSE开方。新增audit独立核每region-month-method全表，其他地区/macro/selection结果须与r2精确一致。不改旧预测、方法、阈值或模型。修订前五文件源码及pre/post SHA保存在v193-assessment-source-r2。
