> 历史入口快照：2026-10-31 整理前的 README，状态以当前根目录 README 和四日成果报告为准。

# TransCIF

面向**无目标电网逐时遥测**的概率调度迁移研究基线。源电网提供监督标签；目标电网只提供预测起点前已公开的装机、能源占比、固定排放因子、成本先验及天气/负荷集合预报。目标历史 CIF、逐时发电和实测负荷不进入训练或推理。

核算边界为六燃料（煤、气、核、水、风、光）的**发电侧直接 CO₂ 强度**，单位 gCO₂/kWh。模型是单节点容量约束松弛，不含输电、进出口、储能、爬坡或启停。时段选择评分是平均 CIF 代理成本。

**当前证据：**核心调度模型已跑通人工数据训练、无标签推理和概率评估；真实数据采集与 staging 已实现，尚无满足 issued-public 协议的真实电网训练基准。新增有限信息研究模块已完成三关试验及七区固定EF代理验证，候选采样器未通过方法优势门槛，见[三关结果](docs/THREE_GATES_RESULTS.md)。历史 v0.1 的 29 区域成绩不能归入 v0.2。研究定位见 [RESEARCH](docs/RESEARCH.md)，开发结果见 [RESULTS](docs/RESULTS.md)。

## 安装与运行

Python 3.10+；在可用的独立虚拟环境内执行：

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
python scripts/run_experiment.py --demo --output results/demo
```

`--demo` 是四个固定配置的人工电网，仅用于开发和流程验证。默认 `--objective point`，按源验证 MAE 选 checkpoint，并监督确定性物理点输出；`crps`、`joint` 分别提供分布评分与联合发电评分。`--point-estimator` 显式选择 `physical/mean/median`。各燃料的边际中位数仅为监督统计，不代表一条可行调度。

真实输入须先满足 [数据协议](docs/DATA.md)。已有输出不会被覆盖：

```bash
python scripts/run_experiment.py --manifest data/public/manifest.json \
  --target TARGET --leave-jurisdiction-out --output results/public-run
python scripts/run_benchmark.py --manifest MANIFEST --seeds 0 1 2 --output results/benchmark
python scripts/predict.py --checkpoint results/demo/model.pt \
  --manifest results/demo/synthetic_data/manifest.json --target target_mixed \
  --origin '2023-03-20T00:00:00Z' --output results/prediction.json
```

目标无评分标签时将 manifest 中的 `labels` 设为 `null`，或训练时传 `--no-target-scoring`。所有目标预测先保存，再读取目标标签评分。

可选对照包括 `tcn/context/shared/covariate` 编码器、资源校准、仅源域学习率/架构选择和留源组机制集合；选项及预算见两个运行入口的 `--help`。它们保留为消融，不作为已验证的贡献。树基线需安装 `.[trees]`；GRIB 解码与完整天气测试需安装 `.[weather]` 及可用的 ecCodes 动态库。

## 代码与产物

| 入口 | 职责 |
| --- | --- |
| `data/asof.py`, `dataset.py` | 按发布时间选完整预报和配置；公共输入与标签分离 |
| `data/{eia930,neso,aemo,gfs,gfs_decode,gfs_window}.py` | 原生数据整理、归档天气采集/解码；不自动认证为合规模型输入 |
| `models/`, `physics/dispatch.py` | 潜在调度状态、相关情景及容量约束物理解码 |
| `physics/envelope.py` | 指定容量/EF/总发电量条件下的确定性 CIF 可行范围 |
| `inference.py` | 单模型批量推理；与训练过程分离 |
| `training/fit.py`, `transfer.py` | 源监督、时间选模、留源组选择与支持诊断 |
| `evaluation/`, `benchmark.py` | 点/概率/路径评分、时段选择、先种子后电网/国家汇总 |
| `experiment.py`, `scripts/` | 实验编排、命令行、数据审计 |

单次运行保存 `model.pt`、`forecasts.npz`、`summary.json`，包含输入及源码哈希、划分、点估计定义、来源与评分。机制集合另存 `ensemble.pt`、`mechanisms.npz`，保持机制身份；配置外推或预测缺电触发启发式弃权，不提供覆盖保证。

清理范围、内部 Python 接口迁移与验证记录见 [重构记录](docs/REFACTOR.md)。本地原始数据和历史结果保留在忽略目录；Git 中的旧实现及其数值不代表当前版本的真实电网性能。

## 有限信息研究试验

`src/transcif/research/` 是独立的三关可行性实验：受控机制歧义、同预算推断比较、七个 EIA 地区的留区预测。它只需要日历与滞后月度发电统计，使用源域拟合的低维正值发电成分模型；不依赖目标负荷预报，也不等同于上面的容量约束调度模型。协议见 [RESEARCH_GATES_PROTOCOL](docs/RESEARCH_GATES_PROTOCOL.md)，前两关的逐关复核见 [GATES12_STAGE_REVIEW](docs/GATES12_STAGE_REVIEW.md)。

```bash
python -m pip install -e '.[dev,research]'
python scripts/run_research_gates.py --stage prepare --source /path/to/eia-2022-2024 --output results/my-research
python scripts/run_research_gates.py --stage gates12 --output results/my-research
python scripts/run_research_gates.py --stage gate3 --output results/my-research
```

第三关拒绝覆盖已有实验；前两关应使用新的输出目录以保留旧结果。原始 EIA staging 只读。实验中的月度统计为回溯构造、假设延迟30天发布，标签为固定排放因子的非负已报发电代理；这不构成真实发布档案或中国小时准确率验证。研究模块不会读取旧版本按目标 CIF 拟合的排放因子。

本轮正式结果在 `results/research-prefix/`。`results/research/gate3/` 是修复原始记录时间边界前的已替代诊断运行；请以[完整报告](docs/THREE_GATES_RESULTS.md)及其正式证据链接为准。

后续V2已实际尝试CDF风险分配、强采样基线、多尺度动态校正和发布延迟干预：5,460组受控采样、七区2023探索验证共68,580条方法日评分。新候选尚未形成相对强基线的优势，见[V2结果与原创性判断](docs/V2_EXPERIMENT_RESULTS.md)、[候选推导](docs/CDF_CANDIDATE_DERIVATION.md)。这是独立探索，不替换旧三关状态；没有用2024小时标签选择V2模型。

V3已扩大为三轮推断搜索、12种水平/形状方法及新的2025全年、2026上半年时间检验。2025季节粗水平相对近期模板降低CRPS13.1%–19.3%；2026固定形状组合比季节模型再降低1.7%–2.7%，但短延迟仍有更强的简单对照，且90%区间覆盖不足。当前最有证据的优化空间是粗统计时效性与水平/形状分开迁移，尚未形成充分原创的统一最优算法。全部296项测试通过；完整数字、失败区域、冻结协议和独立Sol复核见[多轮优化搜索报告](docs/OPTIMIZATION_SEARCH_RESULTS.md)。

沿此方向的V4/V5又完成三轮方法开发与两组新地区测试：支持窗口GP和水平/形状残差传输在V4六个新地区仍严重欠覆盖；V5利用已发布粗统计的变化与年龄归一化水平风险，在另四个新地区将覆盖从同信息固定MoE参考的77.30%提高到94.70%，但区间宽度增加82.76%，CRPS均值恶化5.99%。相对固定GP参考CRPS基本持平，尚无全面概率优势。两组地区不同，不将跨版本覆盖差异归因算法；全部366项测试通过，执行及独立复核均用Sol。见[V4正负结果](docs/TRANSFER_V4_RESULTS.md)、[V5完整结果与创新边界](docs/RISK_TRANSFER_V5_RESULTS.md)和[核验证据](docs/evidence/transfer-v4-v5-verification.json)。

V6已完成“粗回测误差→小时风险”的两轮开发、同一批另外六个新地区2026H1的完整确认和90次人工机制拟合，全部使用Sol。两轮预测与联合源选择均先封存再评分：固定主覆盖77.84%，加粗锚后77.85%，没有达到90%；完整锚定方法相对同点同倍率的简单锚覆盖增加7.11个百分点，但宽度增加24.97%、CRPS恶化，未实现维持覆盖同时缩窄。精确无锚消融也未显示完整模型的锚定增益。简单global使用锚有局部改善，仍不足以支持算法优势或可靠覆盖的论文主张。全量431项测试通过；见[V6完整结果与收益归因](docs/COARSE_TO_FINE_RISK_RESULTS.md)、[独立新颖性审查](docs/V6_NOVELTY_REVIEW.md)及[V6核验记录](docs/evidence/coarse-risk-v6-verification.json)。

V7已诊断PACE等地区的训练支持、形状收缩、水平偏差及真实信息年龄，并完成燃料轨迹比值重组、粗预测分布混合两轮开发。六地区已在V6看过，本轮为开发复测。源固定R2相对同七源更新的简单锚定，CRPS降低16.48%、形状MAE降低19.71%，但覆盖仅88.70%、区间宽度增加21.46%，尚未达到可靠且更窄的目标；joint与independent消融也未显示一致优势。全部源候选未通过完整覆盖门槛。全量502项测试通过，数值经独立Sol复算；下一步优先补同轨迹加性锚定并区分组成预测与形状映射瓶颈。见[V7结果与下一步](docs/FUEL_TRANSPORT_V7_RESULTS.md)、[失败诊断](docs/FAILURE_DIAGNOSIS_V7.md)、[文献边界](docs/V7_LITERATURE_AND_HYPOTHESES.md)和[核验记录](docs/evidence/fuel-transport-v7-verification.json)。

V8已完成上述两项归因试验：同donor、同抽样下，R2重组相对加性粗锚的CRPS降低6.89%，但宽度增加20.35%，覆盖仍88.70%；简单水平校正解释了直接迁移到重组的大部分改善。固定合法路由、仅换成事后实际月组成的非法oracle将重组CRPS由57.004降至43.943、宽度缩窄10.08%，改善主要来自水平，LGEE仍严重欠覆盖。下一步优先合法粗组成动态预测及其小时风险传播，保留条件迁移失效诊断；这尚不是已验证的原创算法。预定继续门槛未过，未消耗新的独立确认地区。全量523项测试通过，执行和独立复核均用Sol。见[V8结果与路线决定](docs/FUEL_ATTRIBUTION_V8_RESULTS.md)、[预先固定协议](docs/FUEL_ATTRIBUTION_V8_PROTOCOL.md)及[核验证据](docs/evidence/fuel-attribution-v8-verification.json)。
