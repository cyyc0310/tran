# 伪目标火电因子响应收缩证据

五国LOCO开发回测，使用源端伪目标H1标签选择比例重组与因子历史头之间的收缩系数，H2作区间校准，目标2024预测先封存后评分。全部国家已查看。方法与限制见[协议](../../PSEUDOTARGET_FACTOR_RESPONSE_SHRINKAGE_PROTOCOL_20261101.md)和[结果](../../PSEUDOTARGET_FACTOR_RESPONSE_SHRINKAGE_RESULTS_20261101.md)。

## 内容

- `macro_metrics.csv`、`country_metrics.csv`、`monthly_metrics.csv`：逐月、逐国和宏平均指标。
- `prediction_seal.json`：目标预测SHA-256和覆盖小时数。
- `alpha_selection.json`：每个外层折伪目标H1所选收缩系数及嵌套留伪目标诊断MAE。
- `*_alpha_scores.csv`、`*_nested_alpha_cv.json`：伪目标网格分数和留伪目标评分。
- `startup_hashes.json`、`input_audit.json`、`post_run_verification.json`：代码、数据、映射与输出完整性记录。

完整预测保存在[`results/pseudotarget-factor-response-shrinkage-r1/`](../../../results/pseudotarget-factor-response-shrinkage-r1/)。

复现命令：

```bash
PYTHONPATH=src:scripts python scripts/run_pseudotarget_factor_response_shrinkage.py \
  --output results/pseudotarget-factor-response-shrinkage-r1
```
