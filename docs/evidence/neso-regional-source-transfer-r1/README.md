# NESO 14区源域扩展 r1 证据

该初跑已被按最终协议重新封存的[r2证据包](../neso-regional-source-transfer-r2/README.md)取代。r1保留原始输出供审计；正式结果引用r2。

冻结协议：[`NESO_REGIONAL_SOURCE_TRANSFER_PROTOCOL_20261101.md`](../../NESO_REGIONAL_SOURCE_TRANSFER_PROTOCOL_20261101.md)。结果解释和BA级分析见[`NESO_REGIONAL_SOURCE_TRANSFER_RESULTS_20261101.md`](../../NESO_REGIONAL_SOURCE_TRANSFER_RESULTS_20261101.md)。

## 内容

- `regional_carbon_intensity_history.csv`与`regional_carbon_intensity_history.manifest.json`：运行时冻结的NESO官方区域估算/预测源数据及获取信息。
- `sealed_predictions.csv.gz`：两种GB源策略在相同2024 US BA目标小时上的预测、源校准参数和时间戳。
- `monthly_metrics.csv`、`region_metrics.csv`、`country_metrics.csv`：逐月、逐BA与宏平均指标。
- `prediction_seal.json`、`regional_source_audit.json`、`post_run_verification.json`：预测哈希、来源口径和最终校验。
- `startup_hashes.json`：代码/协议与源输入文件SHA-256。

## 最终校验

14区方案和全国单区方案各自使用同一目标样本。封存预测文件哈希匹配；启动时记录的所有代码/协议和数据SHA在运行结束仍匹配。共计560个方法×BA×月评分行，40个BA×月，28,983小时，49小时缺目标响应。

这个包包含事后目标预测与评分结果，且目标标签在此前项目开发中已打开；它不可标为独立确认。NESO区域响应为其官方估算/预测序列；不同于区域发电燃料构成或直接排放监测。14区输入共用GB全国粗月结构，不代表各区自己的历史燃料月报。
