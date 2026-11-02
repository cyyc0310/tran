# Energy-Charts 中国输入预算口径更正

更新：2026-11-01。代码核查发现，`src/transcif/research/multicountry.py`的`GROUPS`包含`other_renewable`，共六个粗组；部分Energy-Charts协议和结果却把它写成中国可用的火、水、核、风、光五组预算。**这类旧结果实际允许了第六类目标月信息，不能作为严格中国五类预算的主结果。**保留它们作为“额外可见其他可再生”探索/敏感性运行，不删除旧预测或数值。

## 影响范围

Energy-Charts脚本凡从`multicountry.GROUPS`、`LATEST`、`HISTORY`构造目标特征，均应按六组信息预算重新解读，包括五国LOCO、CZ/RS扩展、九国六组LOCO、对应容量与源筛选、伪目标选择/门控/样本重加权/因子收缩，以及Energy-Charts到SMARD德国留出。其数值仍可回答六组可见信息下的代理预测表现，不能回答中国只有五类公开月报时的性能。

以下轨道使用的不是该六组特征：官方US↔GB消费强度任务和EIA-923输入臂基于`cross_country.COARSE`的五类预算；2026-11-01此前已审计的`run_episodic_coldstart_transfer.py`与五类容量/因子时间核验使用独立的`GROUPS5`五类特征。以各自协议为准。

## 严格五类重跑

新增[严格五组模型模块](../src/transcif/research/multicountry_cn5.py)，公开输入、月份额、小时形状参考与模型特征均只有火、水、核、风、光五类；`other_renewable`仍在代理小时标签总发电量中，不提供给模型。严格九国外层结果已重跑：

- [严格五类九国基线](ENERGY_CHARTS_NINE_COUNTRY_CN5_RESULTS_20261101.md)：比例重组MAE 62.16、经验覆盖85.2%；因子历史头MAE 63.19、覆盖78.9%。
- [严格五类年度装机增量](ENERGY_CHARTS_NINE_COUNTRY_CN5_CAPACITY_RESULTS_20261101.md)：容量因子头相对无容量头MAE改善0.35，但没有超过比例重组或解决覆盖。
- [严格五类伪目标源筛除](ENERGY_CHARTS_PSEUDOTARGET_SOURCE_DROP_CN5_RESULTS_20261101.md)：平均MAE改善0.76，区间分数恶化、覆盖下降；不能冻结。

最新研究计划、摘要、论文图表或方法结论应使用上述严格五类结果。历史六组输出仍可用于量化额外字段的信息价值，但必须标清预算。今后Energy-Charts五类实验使用`multicountry_cn5`；任何引入其他可再生月信息的设计需另立更高预算臂。
