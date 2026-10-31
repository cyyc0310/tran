# V280 结果：PACE/LGEE 官方 EIA 小时 CO₂ 标签外部压力检验

运行日期：2026-10-30。冻结方案见 [V280 protocol](V280_EIA_OFFICIAL_LABEL_PACE_LGEE_PROTOCOL.md)。本轮共 2 个 BA、24 个 BA-month、每月 100 次随机抽取 12 个本地完整日。输出指标按 BA 等权汇总，区间宽度和 Winkler 分数单位为 g CO₂/kWh。

## 结果

| 方法 | 小时覆盖 | 日同时覆盖 | 平均区间宽度 | Winkler-90 | 事后稳健集合包含率 |
|---|---:|---:|---:|---:|---:|
| support_minimum | 99.67% | 97.95% | 501.50 | 503.71 | 92.38% |
| fixed_four | 99.67% | 97.95% | 501.50 | 503.71 | 92.38% |
| full8 | 99.67% | 97.95% | 501.50 | 503.71 | 92.38% |
| source_loo | 75.52% | 51.37% | 239.08 | 620.99 | 不适用 |

三种小时通道方法的逐 BA-month、逐重复结果完全相同（各指标最大绝对差为 0）。因此本轮没有证据表明支持感知通道选择改善了小时区间；把可用小时字段从 3–4 个增加到 full8 也没有降低风险半径。`source_loo` 更窄，但总体覆盖不足，且错误主要来自 PACE。

| BA | 方法 | 小时覆盖 | 日同时覆盖 | 平均宽度 | Winkler-90 | 事后集合包含率 |
|---|---|---:|---:|---:|---:|---:|
| PACE | support_minimum / fixed_four / full8 | 99.34% | 95.95% | 682.60 | 687.02 | 91.99% |
| PACE | source_loo | 51.35% | 4.05% | 239.08 | 1002.62 | 不适用 |
| LGEE | support_minimum / fixed_four / full8 | 100.00% | 99.95% | 320.41 | 320.41 | 92.77% |
| LGEE | source_loo | 99.69% | 98.70% | 239.08 | 239.35 | 不适用 |

`support_minimum` 使用 3 或 4 个小时数值字段（含总量）：PACE 12 个月均为 4 个；LGEE 9 个月为 3 个、3 个月为 4 个。结果说明字段预算减少本身不是这次检验的增量来源。PACE 的局部风险校准把区间推到极宽仍未达到 90% 的日同时覆盖；LGEE 则明显过覆盖。单一固定的 90% 区间机制不能同时兼顾这两种风险状态。

## 解释边界

- 中心是目标 BA-month 的 EIA 小时 CO₂ 强度按正发电量加权的月均值，属于 oracle；本实验只考察给定月中心后的小时/日形状风险，不证明月前 CIF 强度可预测。
- EIA 标签是基于 BA 报告发电和年度 BA 排放因子估算的生成侧 CO₂ 强度，不是直接 CEMS 真值。EIA 说明其小时估算使用年度 BA 煤、气、石油因子，其他燃料使用全国统一因子；源数据与本实验的小时燃料类别共享，不能视为完全独立观测。[EIA-930 方法说明](https://www.eia.gov/electricity/930-content/about.html)
- LP 采用美国 CEMS 的季度经验 EF 支持作为跨系统压力合同。它不是 PACE/LGEE 目标因子的置信集。目标 CEMS EF 与此支持不相容、或 EIA 发布标签的映射规则不同，都可能造成不必要的宽区间。
- `ex_post_robust_day_set_containment` 使用测试日完整小时燃料结构计算，只作事后诊断，不是预测时可用输入，也不提供预测保证。
- PACE/LGEE 先前已出现在项目代理数据实验中；这只是新标签源的区域压力检验，不是未查看地区确认。只有两个 BA，不作 BA 层面的总体统计推断。

## 对创新方向的决策

本轮不支持将“按 EF 支持挑最少小时字段”单独作为论文核心创新：它与固定四字段及 full8 无可测差异。下一轮优先检验 **标签/EF 支持错配诊断与可识别风险半径分解**：区分（a）由抽样日的小时燃料信息可识别的形状误差，（b）月度边际下仍无法识别的组内调度误差，（c）目标排放因子/标签制度的跨系统误差。重点查明为什么 PACE 的 source-only 风险远高于 LGEE，以及为什么目标本地校准达到高小时覆盖却仍未稳定达到 90% 日同时覆盖。只有在不利用测试日结构的条件下，能针对失效地区降低区间宽度并保持日覆盖，才值得冻结方法并做真正未查看区域的外测。

## 文件与复现

- 分析脚本：[v280_eia_official_label_pace_lgee.py](../scripts/v280_eia_official_label_pace_lgee.py)
- 冻结方案：[V280 protocol](V280_EIA_OFFICIAL_LABEL_PACE_LGEE_PROTOCOL.md)
- 指标：`results/v280-eia-official-label-pace-lgee/run-r1/{macro_metrics,ba_metrics,ba_month_metrics,replicate_metrics,field_budget}.csv`
- 数据与代码哈希：[metadata.json](../results/v280-eia-official-label-pace-lgee/run-r1/metadata.json)

所有指标 CSV 已完整写出。首次运行在写 metadata 时因 tuple 月份键不能 JSON 序列化而退出；之后按已生成的 CSV 与输入文件哈希补齐 metadata，未改变指标计算。脚本已修正该序列化问题。避免覆盖已有冻结结果；需要复跑时将输出目录配置为新的 run 版本。
