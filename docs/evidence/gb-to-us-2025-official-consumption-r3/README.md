# GB→US 2025官方消费强度封存证据

本证据包对应英国单源到美国七BA的消费侧2025开发时间留出，非盲地区确认、非as-issued回放。

- `sealed_predictions.csv.gz`: 每个BA 2025年1—10月小时预测，共51,072行。
- `prediction_seal.json`: 预测在读取2025目标`CO2.CER`之前封存的哈希与样本数。
- `calibration.json`: NESO GB消费强度2024源校准偏差和90% split-conformal半径。
- `ba_metrics.csv`, `macro_metrics.csv`: 按BA和BA等权宏汇总的冻结主指标。
- `ba_month_metrics.csv`, `ba_month_macro_metrics.csv`: BA×月结果，供失效诊断。
- `source_audit.json`, `target_feature_audit.json`, `target_scoring_audit.json`: 源训练/校准年、目标输入合同、标签访问顺序与目标评分样本。
- `model_audit.json`, `startup_hashes.json`, `status.json`: 燃料响应、模型来源、数据/代码SHA-256与状态。

目标月输入由EIA-923 2024/2025当前年度终版回算；源NESO历史燃料量也为当前修订序列。结果解释见[报告](../../CROSS_COUNTRY_2025_GB_TO_US_OFFICIAL_CONSUMPTION_RESULTS_20261102.md)和[冻结协议](../../CROSS_COUNTRY_2025_GB_TO_US_OFFICIAL_CONSUMPTION_PROTOCOL_20261102.md)。
