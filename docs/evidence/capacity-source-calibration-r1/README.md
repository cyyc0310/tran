# 年度装机源端校准诊断证据包

冻结协议和结果解释见[协议](../../CAPACITY_SOURCE_CALIBRATION_AUDIT_PROTOCOL_20261101.md)及[结果报告](../../CAPACITY_SOURCE_CALIBRATION_AUDIT_RESULTS_20261101.md)。

## 文件

- `region_year_fuel_mix.csv`：7个美国BA的2022/2023煤/气容量份额及小时燃料回算火电因子。
- `loba_predictions.csv`：每个BA留出、用其他6区2022拟合后对留出区2023的三种预测。
- `macro_metrics.csv`：逐BA等权MAE/RMSE。
- `input_hashes.json`：8个小时源文件与容量预算文件哈希。
- `startup_code_hashes.json`：诊断脚本、标签定义和协议哈希。
- `post_run_verification.json`：留区样本数及哈希核验。

输出保存在[`results/capacity-source-calibration-r1/`](../../../results/capacity-source-calibration-r1/)。复现命令：

```bash
PYTHONPATH=src:scripts python scripts/audit_capacity_source_calibration.py \
  --capacity-json data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json \
  --output results/capacity-source-calibration-r1
```

这是美国BA留区开发诊断，不是跨国确认。
