# EIA-923月量与官方消费强度迁移证据包

US↔GB双向、小时回算月量与EIA-923真实BA月量配对开发回测。目标是官方EIA/NESO估算的消费侧强度。协议和结论见[结果报告](../../EIA923_OFFICIAL_CONSUMPTION_INPUT_BUDGET_RESULTS_20261101.md)。

## 文件

- `results/prediction_seal.json`：16份目标预测文件的SHA-256、小时数和方向。
- `results/*_predictions.csv.gz`：两种输入臂、US→GB及GB→US的小时预测。
- `results/monthly_metrics.csv`、`region_metrics.csv`、`country_metrics.csv`：逐月、逐区及宏观评分。
- `results/eia923_monthly_ba_mix.csv`：解析后的BA×月官方燃料发电量与五类粗份额。
- `results/startup_hashes.json`：运行代码、协议、官方源响应、小时输入、EIA-923档案及年度容量输入哈希。
- `results/target_scoring_hashes.json`：预测封存后读取的2024响应文件哈希。
- `results/common_support.json`、`input_audit.json`、`status.json`：小时支持与时间口径审计。

## 复现

在仓库根目录运行：

```sh
PYTHONPATH=src:scripts python scripts/run_eia923_official_consumption_input_budget.py \
  --output docs/evidence/eia923-official-consumption-input-r1/results
```

输出目录只创建一次。2024标签曾在此前开发中打开，所以本证据是开发回测而非盲测；本次脚本仍先写出并哈希全部预测，再读取标签评分。EIA-923采用当前年度最终档案，延迟只按`t−2`模拟，不代表as-issued历史版本。
