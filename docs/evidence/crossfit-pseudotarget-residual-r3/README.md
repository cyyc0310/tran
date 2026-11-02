# 跨拟合伪目标残差传输及源端回退 r3

协议：[残差传输](../../CROSS_FITTED_PSEUDOTARGET_RESIDUAL_PROTOCOL_20261101.md)、[源端收缩](../../CROSS_FITTED_PSEUDOTARGET_BACKOFF_PROTOCOL_20261101.md)。结果：[实验报告](../../CROSS_FITTED_PSEUDOTARGET_RESIDUAL_RESULTS_20261101.md)。

## 复现

```bash
PYTHONPATH=src:scripts .venv-research-py312-20261101/bin/python scripts/run_crossfit_pseudotarget_residual.py \
  --output results/crossfit-pseudotarget-residual-r3
```

运行以`startup_hashes.json`固定代码和输入。五个外层2024预测先分别写盘并汇总到`prediction_seal.json`，随后才评分固定因子小时响应。残差权重由源2023年7–9月选择，区间仅由源10–12月交叉拟合绝对残差校准。

## 证据文件

- `results/`：五个目标预测、预测哈希、逐月/逐国/宏指标、源端收缩CV和已选系数、源端校准、状态和哈希。
- `code/`：r3运行器、强基点和伪目标专家依赖、协议与结果报告快照。
- `input/`：Energy-Charts获取清单与质量审计。

大型国家小时序列仍留在本地忽略的数据目录，其SHA-256在启动清单中。数据为修订序列，模型把t−2及更早聚合量当作可见粗月信息，不是历史as-issued回放。
