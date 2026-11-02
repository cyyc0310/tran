# NESO 2025风电预报形状迁移时间留出证据

状态：按冻结协议完成预测封存后评分；属于GB同国家时间留出，不是新国家盲确认。目标为固定因子发电侧代理，不是官方消费侧CIF。

## 数据和封存

- NESO来源：Embedded Solar and Wind Forecast Archive 2025；原档案SHA-256：`8a9e75dace7f99590f00ffd025a817a397ba63e55ab88d03194f31d7c0e468b9`。
- 预测起点：每个UTC交付日前一日09:15 UTC之前的最新签发风电预报；M为GB t−2月份额。
- 预测签发信息覆盖359日/17,232半小时；完整小时共同支持8,616小时。
- 预测文件SHA-256：`0f1a2089100a539f46d35ba611082e05658a0994a073621d68cf7cc50e4a1513`；协议SHA-256：`b9c85652cd009ff0d2af39fd13bab064a94ba602a082d0d185bdac9366106425`。预测封存记录`prediction_seal.json`确认封存前未读取目标2025发电结果。
- 标签按跨国开发协议的固定因子映射从NESO半小时发电量生成，合并为小时代理后评分。所有分支使用同一个US 2023源偏差与90% Gaussian半径；没有使用GB 2025误差重校准。

## 结果

详细机器指标在`scored_summary.json`，逐月指标在`monthly_metrics.csv`。冻结的主比较为原始五类粗重组M与风电形状替换M+F：

- M：MAE 67.09，日形状MAE 26.24，固定半径经验覆盖89.06%，Winkler 293.87 g/kWh。
- M+F：MAE 65.60（改善2.23%），日形状MAE 21.72（改善17.20%），覆盖90.03%，Winkler 293.75。区间宽度两者同为260.91 g/kWh。
- 配对日bootstrap的M+F−M MAE差为−1.49 g/kWh，95%区间[−2.25, −0.76]；359日中165日M+F误差较低，逐月胜出9/12月。
- 预先登记的五个百分点MAE门槛未达到。故按协议停止这一单变量风电形状分支，不训练专门风险头，也不称为算法创新。结果支持：签发风电形状在该GB代理上有小幅、可检测的时间外信息价值；但该增量不足以证明新方法成立或能直接迁移到中国。

## 复现文件

- 冻结方法与通过门槛：[2025时间留出协议](../../NESO_WIND_FORECAST_SHAPE_2025_TEMPORAL_PROTOCOL_20261102.md)
- 原始2024开发实验：[初始证据包](../neso-as-issued-vre-shape-transfer-r1/README.md)
- 此目录文件：`wind_forecast_features_2025.csv`（预测输入）、`preflight_manifest.json`（原始档案/覆盖哈希）、`predictions_2025.csv`（封存预测）、`prediction_seal.json`（预测校验）、`scored_summary.json`和`monthly_metrics.csv`（评分）。
