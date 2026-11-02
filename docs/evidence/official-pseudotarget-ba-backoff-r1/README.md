# US→GB官方消费侧伪目标专家回退 r2

协议：[冻结协议](../../OFFICIAL_PSEUDOTARGET_BA_BACKOFF_PROTOCOL_20261101.md)。结果：[实验报告](../../OFFICIAL_PSEUDOTARGET_BA_BACKOFF_RESULTS_20261101.md)。

## 运行

```bash
PYTHONPATH=src:scripts .venv-research-py312-20261101/bin/python scripts/run_official_pseudotarget_ba_backoff.py \
  --output results/official-pseudotarget-ba-backoff-r1
```

运行前，`preseal_target_label_audit.json`记录目标标签目录为空。运行器写出并哈希封存8,784个GB预测后才调用目标标签提取脚本。`target_scoring_hashes.json`记录事后提取的GB响应文件哈希。

## 证据文件

- `results/`：源伪目标逐BA验证、选定α/τ、独立源校准、GB预测文件、预测封存、逐月/汇总分数、输入和启动哈希，以及目标标签哈希。
- `code/`：本轮运行器、官方标签面板/特征共享代码、既有容量数据读取和协议快照。
- `input/`：EIA/NESO采集及覆盖清单、构造后的官方强度面板清单。

大型原始EIA/NESO文件留在本地忽略的数据目录；文件身份与派生面板摘要由清单和哈希记录。GB 2024此前已在另一个实验打开，因此本轮是方法开发，不构成未看目标确认。
