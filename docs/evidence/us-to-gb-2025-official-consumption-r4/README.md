# US→GB 2025官方消费强度迁移封存证据

按[冻结协议](../../CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_CONSUMPTION_PROTOCOL_20261102.md)执行，属于开发时间留出，非盲地域确认、非as-issued历史回放。

- `sealed_predictions.csv.gz`: GB目标2025年1—10月逐小时预测，7,296行。
- `prediction_seal.json`: 预测标签读取前封存的SHA-256与行数。
- `calibration.json`: 美国2024源域七BA等权残差估计的固定偏差与高斯尺度。
- `monthly_metrics.csv`, `metrics.csv`: 逐月及十个月等权汇总的点/区间指标。
- `target_feature_audit.json`, `target_scoring_audit.json`: 目标特征预算、标签时间窗、评分样本和访问顺序。
- `calibration_support.json`: 2024美国七BA源校准样本数。
- `startup_hashes.json`: 脚本、协议、源标签、目标公开燃料文件及来源文件哈希。
- `status.json`: 方向、时间拆分、目标预算、标签定义和开发状态。

固定因子重组是发电侧代理，虽在消费标签上表现较好，不能解释为已建模消费潮流。详细解释见[结果报告](../../CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_CONSUMPTION_RESULTS_20261102.md)。
