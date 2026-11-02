# 滞后净受电信息未带来可信的消费侧小时 CIF 增量

更新：2026-11-01。协议见[冻结实验定义](NET_INTERCHANGE_INCREMENT_PROTOCOL_20261101.md)。这是已看过 2024 标签的开发集结果，不是盲测。

## 结果

M 与 M+X 使用相同训练、校准和目标评分支持：8 个目标区域、44,794 个目标特征小时，44,766 个有标签评分小时。下面对区域做等权宏平均。`Δ` 为 M+X 相对 M 的变化；误差指标正值为变差，覆盖负值为变差。

| 方向/方法 | MAE M → M+X | 形状 MAE M → M+X | CRPS M → M+X | 90%覆盖 M → M+X | 宽度 M → M+X | Interval score M → M+X |
|---|---:|---:|---:|---:|---:|---:|
| GB→US HGB | 128.85 → 129.94 (+0.85%) | 22.31 → 22.01 (−1.33%) | 102.04 → 103.15 (+1.09%) | 47.36% → 46.10% (−1.26pp) | 206.81 → 204.12 | 1186.61 → 1209.09 (+1.89%) |
| GB→US Ridge | 372.96 → 381.85 (+2.38%) | 37.44 → 38.34 (+2.41%) | 336.60 → 344.84 (+2.45%) | 16.16% → 18.44% (+2.28pp) | 230.39 → 234.65 | 5548.78 → 5700.05 (+2.73%) |
| US→GB HGB | 81.81 → 83.97 (+2.64%) | 34.06 → 33.12 (−2.76%) | 58.71 → 60.44 (+2.95%) | 71.56% → 68.12% (−3.44pp) | 222.47 → 217.53 | 455.39 → 474.90 (+4.28%) |
| US→GB Ridge | 209.99 → 211.32 (+0.64%) | 22.84 → 22.85 (+0.01%) | 172.49 → 174.95 (+1.43%) | 9.71% → 9.07% (−0.64pp) | 224.39 → 217.31 | 2231.40 → 2316.92 (+3.83%) |

形状 MAE 在两方向的 HGB 上各有 1.3–2.8% 改善，但点 MAE、CRPS、interval score 都恶化；宽度缩窄伴随覆盖继续下降。Ridge 也没有呈现稳定收益。90%覆盖只有 9–72%，远低于名义覆盖；这套源区残差尺度在跨域目标上不可信，不能靠此次加流量特征修复。

## 解释

这是一个重要的负结果：即使在海外有完整的月度净流量，并严格按中国式滞后公开预算模拟，单一月度净受电占比也不足以稳定约束小时消费侧强度。月统计可能帮助粗略区分小时轮廓，但无法识别小时流向、跨区来源强度、交换的边际碳因子和目标域水平偏移。局部形状指标的小幅变好不能抵消整体误差和覆盖损失。

英国 `IMPORTS` 字段被排除，因为它是总进口而非净交换；净流量由 NESO 官方需求表中十条互联线的正负带符号 MW 流量求和得到。NESO 定义正值为进入 GB、负值为流出 GB。EIA-930 将负 `Total interchange` 定义为净流入，因此 `-Total interchange` 对齐为正向净受电。GB 分母使用 TSD，US 使用 Demand；这两个官方需求口径仍有范围差异，故本实验只能评估方向性增量，不能把跨国效应解释成严格同口径结构系数。

## 研究决策

暂不把 `lagged net import share` 作为跨域方法创新点，也不冻结该特征进入中国预测器。可保留的论文贡献是明确量化：月度构成和月度净流量提供的信息仍不足以实现可信小时消费 CIF 迁移；消费侧必须把来源地/跨省流量及对应碳强度作为独立数据预算与识别条件。下一步应做来源强度混合的部分识别/敏感性界限，或先将主论文贡献限定在发电侧统一迁移基准，并把消费侧作为数据可识别性边界。

## 复现与数据

- 运行：[脚本](../scripts/run_net_interchange_increment.py)
- 机器指标：[逐目标区域](../results/net-interchange-increment-r3/region_metrics.csv)、[方向汇总](../results/net-interchange-increment-r3/endpoint_metrics.csv)
- 训练/校准和预测散列：[校准参数](../results/net-interchange-increment-r3/calibration.json)、[预测封存](../results/net-interchange-increment-r3/prediction_seal.json)、[代码与输入散列](../results/net-interchange-increment-r3/hashes.json)、[共同支持/时效状态](../results/net-interchange-increment-r3/status.json)
- 原始英国流量：[来源清单](../data/neso-interconnector-flows-r1/source_manifest.json)

来源口径：[EIA-930 interchange 定义](https://www.eia.gov/electricity/930-content/about.html)、[NESO Historic Demand Data 字段说明](https://www.neso.energy/data-portal/historic-demand-data/historic_demand_data_2024)、[NESO Historic Generation Mix 字段说明](https://www.neso.energy/data-portal/historic-generation-mix/historic_gb_generation_mix)。
