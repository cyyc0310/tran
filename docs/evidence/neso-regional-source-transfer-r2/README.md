# NESO 14区源域扩展 r2 证据

最终冻结协议：[`NESO_REGIONAL_SOURCE_TRANSFER_PROTOCOL_20261101.md`](../../NESO_REGIONAL_SOURCE_TRANSFER_PROTOCOL_20261101.md)。结果分析见[`NESO_REGIONAL_SOURCE_TRANSFER_RESULTS_20261101.md`](../../NESO_REGIONAL_SOURCE_TRANSFER_RESULTS_20261101.md)。r2在修正候选月份与实际目标BA可用交集的协议措辞后重跑，替代探索性r1。

## 内容

- `regional_carbon_intensity_history.csv`与`regional_carbon_intensity_history.manifest.json`：运行时冻结的NESO官方区域估算/预测源数据及获取信息。
- `sealed_predictions.csv.gz`：两种GB源策略在相同2024 US BA目标小时上的预测、源校准参数和时间戳。
- `monthly_metrics.csv`、`region_metrics.csv`、`country_metrics.csv`：逐月、逐BA与宏平均指标。
- `prediction_seal.json`：预测文件SHA-256、行数和标签使用时序。
- `regional_source_audit.json`、`post_run_verification.json`：来源口径、覆盖范围及最终校验。
- `startup_hashes.json`、`target_scoring_hashes.json`：预测前代码/协议与输入SHA、预测后目标标签SHA。

## 最终校验

14区方案和全国单区方案各自使用同一目标样本。封存预测SHA匹配；运行结束代码/协议和输入哈希差异均为空。共计560个方法×BA×月评分行，40个BA×月、28,983小时，49小时缺目标响应。预测之前没有读取目标响应；目标标签在本项目此前实验中已经打开，故此项仍是开发结果。

NESO区域响应是官方区域估算/预测序列，不同于历史区域燃料构成或直接排放监测。14区输入共用GB全国粗月结构，不代表各区自己的历史燃料月报。
