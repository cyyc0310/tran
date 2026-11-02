# US→GB两月总发电量趋势复核证据

2026-11-02。该包对应[冻结协议](../../CROSS_COUNTRY_US_TO_GB_VOLUME_TREND_PROTOCOL_20261102.md)、[结果报告](../../CROSS_COUNTRY_US_TO_GB_VOLUME_TREND_RESULTS_20261102.md)和[runner](../../../scripts/run_us_to_gb_volume_trend_experiment.py)。主运行在`results/us-to-gb-volume-trend-r7/`；这里保存可随仓库追踪的紧凑证据。

## 运行合同

- EIA-930七BA训练2022–2023、来源校准2024；NESO 2025国家`intensity.actual`为目标标签。
- 目标仅使用`t−2`五类月份额、日历项及一个两月对数总量变化`t−2−t−3`。
- 燃料月特征先从独立完整发电输入文件生成，再与小时官方强度标签配对。候选与基础HGB共享七BA全部有效训练/校准支持。
- 预测与来源校准值先封存，随后才读NESO标签。标签SHA-256与此前正式US→GB 2025评分文件相同：`c6bb4e11b293457232b757b5e149db0a45f731bd1d6db27a7747baea10e42f14`。

## 主要文件

- `sealed_predictions.csv.gz`：逐小时输入、预测与时间戳。
- `source_calibration.json`：逐方法来源偏差、90%半径和每BA训练/校准支持行数。
- `overall_metrics.csv`、`monthly_metrics.csv`：封存预测的总体及逐月评分。
- `prediction_seal.json`、`scoring_manifest.json`、`audit.json`：预测/校准哈希、特征合同、标签哈希和评分状态。

复跑命令（从仓库根目录）：

```bash
python scripts/run_us_to_gb_volume_trend_experiment.py \
  --output results/us-to-gb-volume-trend-replay
```

`--target-labels`默认为仓库内与冻结SHA匹配的NESO 2025标签。预测数据、源训练/校准输入、runner及协议的哈希写在机器审计文件中。
