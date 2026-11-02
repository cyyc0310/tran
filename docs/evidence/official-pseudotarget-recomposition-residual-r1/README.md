# 官方消费强度伪目标重组残差证据包

本包检验US BA伪目标训练的小时残差修正，能否迁移到GB官方消费强度标签。完整解释见[结果报告](../../OFFICIAL_PSEUDOTARGET_RECOMPOSITION_RESIDUAL_RESULTS_20261101.md)，预测前冻结的执行规则见[协议](../../OFFICIAL_PSEUDOTARGET_RECOMPOSITION_RESIDUAL_PROTOCOL_20261101.md)。

## 产物

- `prediction_seal.json`和两个`*_prediction_seal.json`：两种输入预算下GB目标预测哈希。
- `*_US_to_GB_predictions.csv.gz`：小时回算与EIA-923输入臂的2024预测。
- `*_selection.json`：源BA留出选择的残差模型和收缩率。
- `*_source_calibration.json`：2023H2源BA留出校准参数。
- `source_h1_validation.csv`：2023H1源端模型与收缩候选的逐BA验证结果。
- `monthly_metrics.csv`、`summary_metrics.csv`：目标逐月与宏观结果。
- `startup_hashes.json`、`target_scoring_hash.json`：运行输入、代码和评分标签的SHA-256。
- `input_audit.json`、`common_support.json`、`status.json`：数据覆盖、共同支持与开发/盲测状态。

## 复现

```sh
PYTHONPATH=src:scripts python scripts/run_official_pseudotarget_recomposition_residual.py \
  --output docs/evidence/official-pseudotarget-recomposition-residual-r1
```

输出目录只创建一次。目标2024官方标签此前已用于开发；尽管本轮脚本先封存预测再读标签，该评估仍不是盲测确认。
