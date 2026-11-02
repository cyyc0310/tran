# 同小时净流量Oracle仍未形成双向可靠增量

日期：2026-11-01。按[冻结协议](HOURLY_FLOW_ORACLE_INCREMENT_PROTOCOL_20261101.md)运行。目标小时的实际净互联流量是不可部署Oracle输入；此处只用它诊断消费侧缺失信息，不能把表格当预测成绩。

## 配对结果

模型只在相同特征完整支持上训练、校准和评分。汇总先对目标区域等权。单位gCO₂/kWh。

| 方向/模型 | M MAE → Oracle | 形状MAE | CRPS | 90%覆盖 | 宽度 | Interval score |
|---|---:|---:|---:|---:|---:|---:|
| GB→US HGB | 131.44 → **128.85** (−1.97%) | 22.70 → 26.28 (恶化15.78%) | 103.78 → **100.06** (−3.59%) | 45.60% → 48.74% | 206.81 → 213.22 | 1195.43 → **1098.15** (−8.14%) |
| GB→US Ridge | 381.40 → 385.03 (恶化0.95%) | 37.24 → 37.26 | 343.87 → 346.74 | 11.84% → 12.32% | 230.39 → 234.82 | 5658.63 → 5692.18 |
| US→GB HGB | 81.81 → 88.04 (恶化7.61%) | 34.06 → **31.53** | 58.71 → 63.54 (恶化8.24%) | 71.56% → 61.85% | 222.47 → 210.00 | 455.39 → 500.42 |
| US→GB Ridge | 210.01 → 212.70 (恶化1.28%) | 22.84 → 24.58 | 172.48 → 177.56 | 9.75% → 10.45% | 224.61 → 211.32 | 2230.13 → 2412.48 |

净流量Oracle在GB→US提升树上改善点MAE、CRPS和interval score，代价是形状误差上升和区间变宽；其覆盖仍仅48.7%。US→GB提升树的点误差、CRPS、覆盖和interval score均恶化。Ridge两向均未改善整体概率分数。方向之间没有稳定增量，区间也远未可靠。

## 研究含义

结合月度净受电负结果，现有证据不支持“消费侧失败只因净流量时间聚合太粗”。当小时流量本身已知时，增量仍依迁移方向和模型而变，且没有来源碳强度信息；因此下一步应把**流量方向/来源碳强度及其标签边界**作为独立信息预算，或明确给出缺失来源信息时的敏感性界限，而不是继续堆叠月度净受电特征。

这是解释假设，不是因果归因：US/GB的净流量、需求分母、统计边界及官方消费标签的跨区排放分摊均不同；目标标签在既有开发中已查看。此结果不改变中国发电侧主线，也不能支持中国消费侧小时精度主张。

## 复现

- [运行脚本](../scripts/run_hourly_flow_oracle_increment.py)
- [方向汇总表](../results/hourly-flow-oracle-increment-r2/endpoint_metrics.csv)
- [逐区表](../results/hourly-flow-oracle-increment-r2/region_metrics.csv)
- [目标支持审计](../results/hourly-flow-oracle-increment-r2/feature_support.json)
- [预测封存](../results/hourly-flow-oracle-increment-r2/prediction_seal.json)
- [代码/输入哈希](../results/hourly-flow-oracle-increment-r2/hashes.json)
- [机器状态](../results/hourly-flow-oracle-increment-r2/status.json)

代码和输入哈希均匹配；M与Oracle各方向/模型的目标评分小时相同。脚本实际运行并完成，无测试套件运行。
