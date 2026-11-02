# 生成侧伪目标残差配对复验证据

与官方消费强度伪目标残差运行使用同一模型代码和源端切分，仅响应换成统一固定EF发电侧代理。分析见[结果报告](../../PSEUDOTARGET_RECOMPOSITION_RESIDUAL_GENERATION_PROXY_RESULTS_20261101.md)，规则见[冻结协议](../../PSEUDOTARGET_RECOMPOSITION_RESIDUAL_GENERATION_PROXY_PROTOCOL_20261101.md)。

`results-r2/`为修正目标年份筛选写法后的权威运行。`results-r1/`两臂预测和源端选择已经封存，但未进入目标评分，原因是pandas年份索引比较发生异常；该目录保留作为失败尝试记录。

权威产物：

- `results-r2/prediction_seal.json`与`*_prediction_seal.json`：两输入臂预测文件哈希。
- `results-r2/*_US_to_GB_predictions.csv.gz`：GB 2024小时预测。
- `results-r2/*_selection.json`、`*_source_calibration.json`：源端收缩选择与H2留BA校准。
- `results-r2/source_h1_validation.csv`、`monthly_metrics.csv`、`summary_metrics.csv`：逐BA源端选择、逐月目标评分及汇总。
- `results-r2/startup_hashes.json`、`target_scoring_hash.json`：代码/输入及目标原始燃料数据哈希。
- `results-r2/input_audit.json`、`common_support.json`、`status.json`：小时输入覆盖与开发状态。

从仓库根目录复现：

```sh
PYTHONPATH=src:scripts python scripts/run_pseudotarget_recomposition_residual_generation_proxy.py \
  --output docs/evidence/pseudotarget-recomposition-residual-generation-proxy-r2
```

此处2024目标发电侧代理由NESO小时燃料序列按冻结固定EF计算，目标国家曾用于开发，不是盲测确认，也不是独立实测碳强度。
