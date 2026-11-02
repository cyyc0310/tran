# 历史资源参照与日幅度开发证据

对应[结果报告](../../RESOURCE_AMPLITUDE_RESULTS.md)。这是三个已见国家、624个唯一目标小时的自适应开发，不是新国家确认或中国精度验证。

`manifest.json`绑定本包文件及归档成员的SHA。`results/predictions.csv.gz`无损还原149760行原预测；`members.tar.gz`保存360个原始成员文件，包含原始全局队列`row`索引。`cohort_with_labels.csv.gz`沿用前轮2400行队列，可按`region + valid_at + label_row`绑定预测与评估标签。细燃料固定因子可独立重建代理CIF，标签仅用于评分后打包，不进入部署输入。

可从包内复算2160组区域、720组配置、80行宏指标及原始粒子CRPS。源CQR仅扩张区间端点，不能将原始CRPS描述为扩张后分布的评分。`results/independent_review.json`记载Sol独立复核；200份源过去月功率凭证随9个模型训练审计封存。`source_heads.tar.gz`为本项目自产模型，只供受信任本地复现。

`historical_climate.tar.gz`含50个原始ERA5年度点响应、对应请求/检索/单位/格点receipt、600个月度资源、120行气候均值及完整性manifest。可独立复核DST、闰年、辐射区间与五年等权汇总。`collection_before_review.tar.gz`在该归档内，保留最初采集代码和manifest；审查修复后已仅通过缓存恢复重新验证。天气原始公众可用性仍为False；原生签发GFS缓存、外国完整小时能源原始数据及旧轨迹银行不在本包内，因此本包不等于全原始采集/重训包。

`snapshots/`保留新采集、方法、runner、测试、预测前协议和本轮报告。原始输入seal仍使用原工作区绝对路径，换机器应按manifest映射，不悄改原封存。两条源标签轨道分别比较；完整源年银行的监督更多，不能把它与小银行的差值归给天气。

`source_resource_diagnostic*`和`posthoc_*`为评分后的源区诊断，使用源小时标签、未拟合新模型；固定功率曲线没有产生新目标CIF成绩。它们不属于冻结预测，也不提供目标精度或覆盖保证。

结论：太阳能比例有局部效率收益，英国仍失败；风光原始CRPS改善，但高覆盖伴随加宽。比例、普通Ridge、功率曲线均为参考，未建立原创方法或满足未知国家晋级条件。
