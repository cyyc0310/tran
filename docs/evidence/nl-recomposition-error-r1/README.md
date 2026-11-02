# 荷兰比例重组误差分解证据包

该包是对NL已查看2024外层预测的事后误差诊断，不用于选方法或参数。

- `monthly_decomposition.csv`：t−2热电份额、源火电因子、目标oracle替换下的逐月MAE/偏差。
- `hourly_decomposition.csv.gz`：逐小时目标代理、粗热电份额和预测/诊断量，包含目标标签，仅用于结果解释。
- `summary.json`：8,784小时汇总指标。
- `input_hashes.json`、`startup_code_hashes.json`、`input_audit.json`：来源、代码及口径审计。
- `post_run_verification.json`：输入、代码与产物完整性核验。

结果报告见[NL分解报告](../../NL_RATIO_RECOMPOSITION_ERROR_DECOMPOSITION_RESULTS_20261101.md)，冻结设计见[协议](../../NL_RATIO_RECOMPOSITION_ERROR_DECOMPOSITION_PROTOCOL_20261101.md)。完整机器输出位于[`results/nl-recomposition-error-r1/`](../../../results/nl-recomposition-error-r1/)。

复现：

```bash
PYTHONPATH=src:scripts python scripts/diagnose_nl_recomposition_error.py \
  --output results/nl-recomposition-error-r1
```
