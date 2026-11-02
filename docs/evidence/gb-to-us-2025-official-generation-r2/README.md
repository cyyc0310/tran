# GB→US 2025官方发电强度封存与评分证据

这是冻结协议下的开发时间留出，非盲地域测试、非as-issued回放。预测文件在读取EIA目标标签前封存。

- `sealed_predictions.csv.gz`: 七BA全年预测，61,320行；最后一个有效小时因目标生成标签缺失而不参与评分。
- `prediction_seal.json`: 预测文件哈希、生成时间及标签文件哈希。
- `calibration.json`: 2024源校准期各模型的固定偏差与90% split-conformal绝对残差半径。
- `ba_metrics.csv`, `macro_metrics.csv`: 冻结主分析；区间在目标上的宏覆盖只有22.19%–71.13%，均低于标称90%。
- `ba_month_metrics.csv`, `ba_month_macro_metrics.csv`, `ba_month_pairwise_vs_anchor700.json`: 事后区域×月份诊断，不用于选模。
- `model_audit.json`, `status.json`: 源/目标数据量、模型配置、标签/特征版本与来源哈希。

主结论和方法限制见[结果报告](../../CROSS_COUNTRY_2025_OFFICIAL_GENERATION_TRANSFER_RESULTS_20261102.md)及[冻结协议](../../CROSS_COUNTRY_2025_OFFICIAL_GENERATION_TRANSFER_PROTOCOL_20261102.md)。结果中另列的原始未偏差修正MAE，是目标评分后的敏感性诊断，不属于冻结主指标。
