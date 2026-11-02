# 伪目标粗组成条件堆叠证据包

五国逐国留出开发实验。方法使用源端国家轮流伪目标，将只含t−2及更早粗月信息的预测器输出与其查询小时标签配对，学习目标无小时标签时可用的粗组成条件凸组合。五个外层国家都已查看；这不是最终未查看国家确认。详情见[冻结协议](../../PSEUDOTARGET_COARSE_CONDITIONAL_STACK_PROTOCOL_20261101.md)和[结果报告](../../PSEUDOTARGET_COARSE_CONDITIONAL_STACK_RESULTS_20261101.md)。

## 内容

- `country_metrics.csv`、`macro_metrics.csv`、`monthly_metrics.csv`：外层逐国、宏平均和逐月评分。
- `prediction_seal.json`：五份2024外层预测SHA-256和小时数。
- `pseudo_target_selection.json`：每个外层折的伪目标选型、门控惩罚及优化摘要。
- `*_inner_penalty_cv.csv`：伪目标内层门控惩罚选择。
- `startup_hashes.json`：原始国家数据、协议和代码哈希。
- `post_run_verification.json`：预测、输入和代码哈希核验结果。
- `input_audit.json`：Energy-Charts来源数据映射、时间分辨率及缺失审计。

逐小时预测保留在[`results/pseudotarget-coarse-conditional-stack-r1/`](../../../results/pseudotarget-coarse-conditional-stack-r1/)。复现命令：

```bash
PYTHONPATH=src:scripts python scripts/run_pseudotarget_coarse_conditional_stack.py \
  --output results/pseudotarget-coarse-conditional-stack-r1
```
