# 美英双向pilot：机器证据

此目录保留`cross-country-pilot-r1`原始预测、结果与哈希清单，并新增一项事后全可评分月份拆分。

- [新逐地区×月份×方法评分](monthly_posthoc_all_methods.csv)：426行，六方法，所有已评分GB/US目标小时。
- [新评分摘要和哈希](monthly_posthoc_all_methods_summary.json)：含目标已打开状态、原始索引回连检查、预测时间戳校验、代码哈希及CSV哈希。
- 原始总结果：[region_metrics.csv](region_metrics.csv)、[country_metrics.csv](country_metrics.csv)、[provenance.json](provenance.json)、[预测封存](prediction_seal.json)。

新分析使用[budget-training-r2证据包中封存的精确`cross_country.py`](../cross-country-budget-training-r2/code/cross_country.py)，其SHA与pilot原`provenance.json`中记录的`src/transcif/research/cross_country.py`一致。逐月小时加权MAE对每个地区和方法都复原原封存地区MAE。

该新增评分是事后风险拆分，未用于选模或改参；所有2024目标标签已打开。详情见[分析报告](../../CROSS_COUNTRY_PILOT_MONTHLY_BREAKDOWN_20261102.md)。
