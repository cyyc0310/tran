# 双标签伪目标同队列重评分证据包

复评分US→GB 2024伪目标残差封存预测，分别使用固定EF发电代理和NESO消费侧官方强度；重用既有2023H2留BA校准参数，并与同一预测时间队列的七个基线逐值核对。

- `monthly_metrics.csv`：标签轨、输入臂、方法及月份级分数。
- `summary_metrics.csv`：月等权汇总。
- `pseudo_hgb_vs_recomposition.csv`：伪目标HGB对固定EF强重组的配对差。
- `integrity_comparison.json`：与双轨配对回放的时间/预测一致性，以及与原逐月分数的复算差异。
- `provenance.json`、`startup_hashes.json`：数据源、预测、校准、标签、脚本和协议哈希。
- `input_audit.json`、`common_support.json`、`status.json`：输入、支持和解释边界。

复现：

```sh
PYTHONPATH=src:scripts python scripts/audit_paired_pseudotarget_endpoint_tracks.py \
  --output docs/evidence/paired-pseudotarget-endpoint-tracks-new
```

报告：[同队列结果](../../PAIRED_PSEUDOTARGET_ENDPOINT_TRACK_RESULTS_20261101.md)；[冻结协议](../../PAIRED_PSEUDOTARGET_ENDPOINT_TRACK_PROTOCOL_20261101.md)。目标曾用于开发，不属于盲测。
