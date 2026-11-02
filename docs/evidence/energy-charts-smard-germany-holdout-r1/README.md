# Energy-Charts源训练到SMARD德国留区证据

本实验使用五个Energy-Charts国家作为源，预测SMARD德国2023H2小时生成强度代理。研究问题、映射规则和解释边界见[冻结协议](../../ENERGY_CHARTS_TO_SMARD_GERMANY_HOLDOUT_PROTOCOL_20261101.md)与[结果报告](../../ENERGY_CHARTS_TO_SMARD_GERMANY_HOLDOUT_RESULTS_20261101.md)。

## 产物

- `DE_2023H2_predictions.csv.gz`、`prediction_seal.json`：4,417个目标小时和SHA-256封存。
- `monthly_metrics.csv`、`summary_metrics.csv`：逐月和月等权宏观评分。
- `source_calibration.json`：五个源国2023H1的加权绝对残差90%半径。
- `input_audit.json`：Energy-Charts逐国接口映射及SMARD年度质量审计。
- `startup_hashes.json`、`target_scoring_hashes.json`：源数据、SMARD原始分类档案、代码和目标评分输入哈希。
- `preseal_target_audit.json`、`status.json`：目标响应只在预测封存后评分，以及此前目标已查看的开发状态。

在仓库根目录复现：

```sh
PYTHONPATH=src:scripts python scripts/run_energy_charts_smard_germany_holdout.py \
  --output docs/evidence/energy-charts-smard-germany-holdout-r1
```

输出目录只创建一次。SMARD/Energy-Charts目标2023数据使用当前归档值，输入时滞按`t−2`模拟，不构成历史as-issued回放；德国目标已用于早前开发评估。
