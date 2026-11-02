# EIA-923月统计伪目标迁移证据包

该证据包配对比较 EIA-930 小时回算月量与 EIA-923 官方 BA 月统计在 US→GB 留 BA 伪目标训练中的差异。研究协议及解释见[结果说明](../../EIA923_PSEUDOTARGET_MONTHLY_TRANSFER_RESULTS_20261101.md)。

2026-11-01更正：`results-r2/`是加入旧表`WAT`水电映射后的权威运行；`results/`保留修复前输出，仅供追溯。两次运行预测数值及评分相同，但修正后的输入表和封存哈希以`results-r2/`为准。

## 内容

- `input/f923_2020.zip`、`input/f923_2021.zip`：EIA 官方旧年度档案，含获取 URL、HTTP 元数据和哈希的 `input/acquisition.json`。
- `../eia923-ba-mapping-r1/f923_2022.zip` 至 `f923_2024.zip`：原 BA 映射审计所用年度官方档案。
- `results-r2/eia923_monthly_ba_mix.csv`：修正映射后的 BA×月燃料量和五类粗份额。
- `results-r2/*_predictions.csv.gz`：两个预算臂的目标小时预测。
- `results-r2/paired_prediction_seal.json`：两臂的预测哈希与预测数量。
- `results-r2/country_metrics.csv`、`results-r2/monthly_metrics.csv`：评分数据。
- `results-r2/startup_hashes.json`：代码、协议和输入哈希。
- `results-r2/target_scoring_hashes.json`：预测封存后读取的 GB 标签文件哈希。
- `results-r2/support_audit.json`：两臂共同 BA 小时支持。

## 运行

从仓库根目录执行：

```sh
PYTHONPATH=src python scripts/run_eia923_pseudotarget_transfer.py \
  --output docs/evidence/eia923-monthly-pseudotarget-r1/results-r2
```

脚本会生成只写一次的结果目录。EIA-923 档案为当前年度最终值；尽管特征采用精确 `t−2` 月，仍不代表历史 as-issued 发布版本。
