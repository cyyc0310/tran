# 双标签轨双向迁移配对证据包（权威复跑）

US↔GB 2022训练、2023校准、2024开发目标；发电代理与官方消费侧响应分开拟合评分；并列小时回算与EIA-923月量输入、五类直接模型、年度容量模型及固定EF小时重组。

- `endpoint_metrics.csv`：按月等权、目标区等权的方向级汇总。
- `region_metrics.csv`、`monthly_metrics.csv`：逐区、逐月结果。
- `*_predictions.csv.gz`、`prediction_seal.json`：预测及哈希封存。
- `*_calibration.json`：源2023按方法校准偏差与尺度。
- `startup_hashes.json`、`target_scoring_hashes.json`：代码/输入及评分响应哈希。
- `input_audit.json`、`common_support.json`、`eia923_monthly_ba_mix.csv`、`status.json`：信息口径和运行边界。

复现：

```sh
PYTHONPATH=src:scripts python scripts/run_paired_endpoint_track_transfer.py \
  --output docs/evidence/paired-endpoint-track-transfer-r2
```

目标响应此前已查看，因此这是开发回放，不是盲测。结论见[结果报告](../../PAIRED_ENDPOINT_TRACK_TRANSFER_RESULTS_20261101.md)与[冻结协议](../../PAIRED_ENDPOINT_TRACK_TRANSFER_PROTOCOL_20261101.md)。
