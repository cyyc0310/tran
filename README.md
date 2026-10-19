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

命令行现在默认 `--objective point`：MAE 主导训练，按源域宏平均 MAE 选 checkpoint；概率评分作为辅助。`--objective crps` 可复现概率主导模式。点预测单独输出 `point_g_per_kwh`，评估同时记录显式点预测、情景均值/中位数 MAE，避免把更换点估计方式当作模型收益。

`--objective crps --selection-metric mae --point-estimator median` 保留分布训练并按源域中位数 MAE 选模，是 MAE 任务必须比较的低成本基线。`--point-estimator mean/median/physical` 显式确定点输出并写入 checkpoint；MAE 主导模式默认使用已监督的物理点输出。

更强的条件编码器和非线性树对照：

```bash
python -m pip install -e '.[dev,trees]'
python scripts/run_experiment.py --demo --demo-seed 19 --objective point \
  --backbone context --tree-baseline --output results/point-development
```

`context` 使用膨胀残差卷积、全时域均值上下文、配置 FiLM、公开净负荷特征及可学习调度平滑度。`--steps`、`--learning-rate`、`--batch-size`、`--schedule cosine` 和 `--sampler region_balanced` 可用于源域开发；外层结果不能用于选择这些设置。模拟器仍只有四个固定电网配置，更换随机种子不代表新的配置分布。

源域留组机制集合和外层电网/种子对照可选开启：

```bash
python scripts/run_experiment.py --demo --mechanism-ensemble --output results/transfer
python scripts/run_benchmark.py --manifest results/transfer/synthetic_data/manifest.json \
  --mechanism-ensemble --seeds 0 1 2 --output results/outer-benchmark
```

集合模式中 `--scenarios` 是每个机制的路径数，其他概率基线同步使用相同总路径数。`--inner-unit jurisdiction` 将同国源域一起留出；覆盖范围有直接或间接交集的源域始终成组。至少需要两个独立源组；集合包含全源模型及每个留组模型，训练成本随源组数增长。

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

集合运行另保存 `ensemble.pt` 和 `mechanisms.npz`；将推理命令的 checkpoint 换成 `ensemble.pt` 即可得到配置支持诊断、模型间分歧和四种时段方案。配置超出源支持或任一机制的平均缺电比例超过 1% 时，推荐时段返回 `null`，候选方案仍保留用于审计。这是启发式弃权规则，没有可靠性保证。

## 实现

| 模块 | 作用 |
| --- | --- |
| `data/asof.py`, `dataset.py` | 按发布时间选配置和完整预报批次；输入、标签分别读取 |
| `models/dispatch.py` | 配置条件 TCN、相关潜在状态、联合发电/CIF 情景 |
| `models/context.py` | 全时域公开协变量上下文、配置调制和门控残差编码 |
| `physics/dispatch.py` | 容量约束凸调度及隐式梯度；显式返回未满足负荷 |
| `training/` | 可选 MAE 主导点预测或 CRPS 主导分布训练，源域选模与训练诊断 |
| `training/transfer.py`, `models/ensemble.py` | 独立源组交叉拟合、源留域尺度选择、有限机制集合及配置支持诊断 |
| `models/baselines.py` | 同输入岭回归、MAE 提升树、公开占比常数；源验证残差路径构造分布 |
| `evaluation/` | 点预测、概率与路径指标；均值/CVaR 连续时段选择 |
| `experiment.py` | 排除目标及覆盖重叠区域，时间前推，预测完成后才评分 |
| `benchmark.py` | 固定协议逐目标/种子运行，先平均种子再汇总电网与国家 |

模型不会拟合目标排放因子或添加 CIF 残差修正。源验证残差仅用于两个对照基线。所有超参数和源域选择都应在查看外层目标分数前固定。

## 研究状态

这是可运行、可审计的新起点。尚未提供真实公开预报档案的接入器、真实电网基准结果或跨域覆盖保证。调度层为单节点逐时松弛，不含输电、进出口、储能、爬坡和启停；潜在状态的相关性不能替代这些约束。能源占比不能与容量因子混用；不同核算口径不得直接比较。

均值/CVaR 调度评估衡量固定能量负荷的平均 CIF 代理成本，不代表因果减排。论文问题、近期文献和下一步判据见 [研究设计](docs/RESEARCH.md)。
