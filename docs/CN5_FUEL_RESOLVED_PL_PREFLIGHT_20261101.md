# 波兰外部确认的预测前字段审计

更新时间：2026-11-01。此审计仅检查Energy-Charts类别、时戳及完整性，不读取CIF误差或模型分数。协议与确认结果分开保存。

## 结果

- 数据文件：`data/energy-charts-v2-panel-r1/pl_2022_2024.json`，API报称PT1H、Europe/Warsaw；2023–2024输入序列时戳连续。
- 燃料映射按[确认协议](CN5_FUEL_RESOLVED_PL_CONFIRMATION_PROTOCOL_20261101.md)：硬煤与褐煤归coal，煤气化副产气进入other；核电按Eurostat 2024无核电独立统计设零；风只映射陆上风。接口未提供单列waste，故不补零或并入其他燃料，作为明确未解析范围。
- `other`与`other_renewable`映射序列的空值各有12,719格，未插值。将已存在组分全部完整、有限、非负作为标签/月份额完整行门槛后，2023–2024只保留4,825小时。
- 2024年6月完整行覆盖56.7%；7–12月完整标签覆盖100%。但严格六个月CN5输入历史只支持9–12月，固定目标窗因此为2024年9–12月，共2,929小时：9月720、10月745、11月720、12月744。
- 1–8月不参与评价。其原因是字段/历史可用性，不是看过模型误差后挑选月份；波兰预测目标标签误差在预检阶段未计算。

## 限制

这不是完整波兰发电清单。未单列waste可能被Energy-Charts `others`吸收，也可能缺失于代理分母；接口类别未能区分这两种情况。其他可再生类别的间断缺测进一步限制了样本时间。故后续结果只检验固定Energy-Charts映射代理上的外部表现，强度数值及方法优劣不能被解释为波兰官方CIF精度。

可复现预检命令：`python scripts/preflight_cn5_fuel_resolved_pl.py`。机器证据及文件哈希见[`preflight.json`](../results/cn5-fuel-resolved-pl-preflight-r2/preflight.json)。
