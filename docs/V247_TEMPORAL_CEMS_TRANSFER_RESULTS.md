# V247：2024供体到2025新时间窗，热输入增益继续存在

日期：2026-10-30。按[V247协议](V247_TEMPORAL_CEMS_TRANSFER_PROTOCOL.md)完成。该结果确认月度燃料效率迁移，不代表小时全网CIF预测。

## 设计与数据

2024 EPA CEMS直接归档为供体，2025 EPA CEMS为目标；两年档案都按Zenodo/PUDL公开版本使用。目标BA从上一年供体中同样剔除。CO₂和热输入均只用测量标记，非混合主燃料标签。利用EIA-860对应年份的厂址BA crosswalk。源档案：2024 CEMS MD5 `2dfd1a51f0cc1e4689fa5f3e5bdf0b53`，2025 CEMS MD5 `f6a73c504c86cd90b72a727cc3c15bb0`。2024供体包含50个BA/10,135,957条有效机组小时；2025目标49个BA/10,018,395条。配对有49个BA、586个BA-month、1,251个fuel-cell。供体支持覆盖目标筛选样本99.765%的毛发电量。

2025目标端采用按燃料聚合的月度发电量与额外月度热输入。目标发电量乘2024其他BA同月份同燃料的CO₂/MWh供体EF，中位绝对百分比误差5.42%、平均7.29%；目标热输入乘同供体CO₂/mmBtu，中位0.316%、平均1.214%。目标BA月均化石机组强度绝对误差中位28.22→1.60 g/kWh、均值39.19→7.15 g/kWh；全队列排放WAPE 4.78%→1.18%。

| 2025目标燃料 | 发电量 | generation-only WAPE | heat-input WAPE | 判断 |
|---|---:|---:|---:|---|
| Coal | 691 TWh | 4.41% | 2.31% | 改善但残差仍明显 |
| Pipeline Natural Gas | 1,548 TWh | 6.43% | 0.16% | 显著改善 |
| Natural Gas（单列） | 88 TWh | 12.00% | 0.89% | 显著改善 |
| Diesel Oil | 1.6 TWh | 9.87% | 20.56% | 变差，低样本/异质性风险 |

热输入的信息增益从2025同年同月空间留区（V246）复制到2024→2025跨年供体，因此不只是同年供体对目标设备/时期的近邻记忆。对主流煤气组合，基于热输入的迁移在多数目标BA月改善燃料转换效率差异。

## 明确失效案例

DUK与WALC的煤电热输入转移仍明显偏差。DUK煤电目标实测约632.7 g/kWh，2024供体generation因子约934.2 g/kWh；用热输入后预测等效约840 g/kWh，绝对WAPE仍32.1%。WALC煤电目标约720 g/kWh，热输入路线仍有34.0% WAPE。主要天然气类别在这些BA更接近供体。说明“CO₂/mmBtu更可迁移”不是普适定理；煤质、共燃、主燃料分类、特定监测/热输入口径或区域特殊机组仍需要目标本地碳含量/热值、排放、机组效率或更细群组信息。不能为了整体均值改善而隐藏这两个反例。

## 论文判断与下一步

V246候选通过了时间外推的一关：两年完全分离供体仍支持**月度热输入是大价值本地粗信息**。这比单纯的“月度全燃料类型可得”更精确，且可转成字段预算/信息价值贡献。不过该研究仍是化石机组月均强度的核算迁移，恒等式本身标准；CEMS没有提供平均电网净发电分母，也没有小时覆盖/区间结果。因此目前判定为**强实证研究候选，未达到完整算法创新突破**。

下一步只在主流煤气类别上进入直接小时目标：用2024供体 EF 热输入系数与2025目标月本地粗热输入，重建实测CEMS小时CO₂风险集合；比较简单加性/乘法锚、hourly generation-only、heat-input normalized、目标极少数小时实测校准，并报告90%整日覆盖与区间宽度。DUK/WALC煤电保留为压力区。若不控制覆盖、不优于强基线，则不冻结成方法稿；若通过，后续需再将CEMS化石毛负荷与EIA930全BA发电总量配平。

可复现产物：[`v247_cems_temporal_heat_input_transfer.py`](../scripts/v247_cems_temporal_heat_input_transfer.py)、[`summary.json`](../results/v247-cems-temporal-heat-transfer/summary.json)、[`BA-month-fuel预测表`](../results/v247-cems-temporal-heat-transfer/target_2025_ba_month_fuel_predictions.csv)、[`BA-month误差表`](../results/v247-cems-temporal-heat-transfer/target_2025_ba_month_metrics.csv)。
