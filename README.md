# TransCIF

面向**无本地遥测电网**的概率调度迁移与碳强度预测研究基线。源电网提供监督标签，目标电网只提供预测时刻已经公开的配置和预报。当前实现包含训练、无标签推理、概率评估及负荷时段选择。

当前核算边界为六类燃料的**发电侧直接 CO₂ 强度**，单位 gCO₂/kWh。公开输入包括装机容量、历史已发布能源占比、外部固定排放因子、成本先验，以及已发布的天气和负荷集合预报。无本地遥测意味着不读取目标域历史 CIF、逐时发电量或实测负荷；仍需要公开预报，缺失时会报错。

旧实验分支、错误协议、重复脚本、结果快照和过时文字已从工作树删除。历史实现由 Git 保留；原始 `data_2023/` 未修改，也未自动转换为合规实验数据。

## 使用

Python 3.10+；建议独立虚拟环境。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_experiment.py --demo --output results/demo
```

`--demo` 生成明确标注的人工电网，只用于验证完整流程。它使用硬排序模拟调度，模型使用凸松弛调度；两者仍共享简化的物理假设。模拟分数不能作为真实电网性能证据。

真实数据需按 [数据协议](docs/DATA.md) 提供 manifest：

```bash
python scripts/run_experiment.py --manifest data/public/manifest.json \
  --target TARGET --leave-jurisdiction-out --output results/public-run
```

目标没有评估标签时将其 `labels` 设为 `null`；也可传 `--no-target-scoring`，训练与预测均不打开目标标签。运行输出 `model.pt`、`forecasts.npz` 和包含划分、来源、哈希及评估的 `summary.json`。已有运行不会被覆盖。

已有模型可以直接在新电网推理：

```bash
python scripts/predict.py --checkpoint results/demo/model.pt \
  --manifest results/demo/synthetic_data/manifest.json --target target_mixed \
  --origin '2023-03-20T00:00:00Z' --output results/prediction.json
```

## 实现

| 模块 | 作用 |
| --- | --- |
| `data/asof.py`, `dataset.py` | 按发布时间选配置和完整预报批次；输入、标签分别读取 |
| `models/dispatch.py` | 配置条件 TCN、相关潜在状态、联合发电/CIF 情景 |
| `physics/dispatch.py` | 容量约束凸调度及隐式梯度；显式返回未满足负荷 |
| `training/` | 源域 CRPS、路径 energy score、燃料占比监督；源验证集选模型 |
| `models/baselines.py` | 同输入岭回归、公开能源占比常数；源验证残差路径构造分布 |
| `evaluation/` | 点预测、概率与路径指标；均值/CVaR 连续时段选择 |
| `experiment.py` | 排除目标及覆盖重叠区域，时间前推，预测完成后才评分 |

模型不会拟合目标排放因子或添加 CIF 残差修正。源验证残差仅用于两个对照基线。所有超参数和源域选择都应在查看外层目标分数前固定。

## 研究状态

这是可运行、可审计的新起点。尚未提供真实公开预报档案的接入器、真实电网基准结果或跨域覆盖保证。调度层为单节点逐时松弛，不含输电、进出口、储能、爬坡和启停；潜在状态的相关性不能替代这些约束。能源占比不能与容量因子混用；不同核算口径不得直接比较。

均值/CVaR 调度评估衡量固定能量负荷的平均 CIF 代理成本，不代表因果减排。论文问题、近期文献和下一步判据见 [研究设计](docs/RESEARCH.md)。
