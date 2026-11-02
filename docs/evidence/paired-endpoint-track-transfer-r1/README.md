# 双标签轨双向迁移配对证据包

US↔GB 2022训练、2023校准、2024开发目标；分别用固定EF发电代理与官方消费侧强度响应；并列小时回算和EIA-923月量输入，五类直接模型、年度容量模型及固定EF小时重组。

- `endpoint_metrics.csv`：按月等权、目标区等权的方向级汇总。
- `region_metrics.csv`、`monthly_metrics.csv`：逐区、逐月结果。
- `*_predictions.csv.gz`：预测封存后的小时结果，文件校验和在`prediction_seal.json`。
- `*_calibration.json`：按方法分开的源2023校准偏差与尺度。
- `startup_hashes.json`、`target_scoring_hashes.json`：输入/代码和评分响应哈希。
- `input_audit.json`、`common_support.json`、`eia923_monthly_ba_mix.csv`、`status.json`：输入口径与运行边界。

复现：

```sh
PYTHONPATH=src:scripts python scripts/run_paired_endpoint_track_transfer.py \
  --output docs/evidence/paired-endpoint-track-transfer-r1
```

目标标签过去已在研究过程中查看；这是开发回放。报告见[结果](../../PAIRED_ENDPOINT_TRACK_TRANSFER_RESULTS_20261101.md)及[协议](../../PAIRED_ENDPOINT_TRACK_TRANSFER_PROTOCOL_20261101.md)。
