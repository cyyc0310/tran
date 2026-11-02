# 英国官方月报输入的跨国小时迁移信息价值实验

冻结日期：2026-11-01；目标2024已用于先前开发，因此本实验是已查看英国目标上的输入来源消融，不是盲测。它承接ET 5.4与NESO组成对齐审计，回答“真实官方月度统计口径能否替代从小时回算的CN5月输入”。

## 固定实验

- 源：美国六个EIA平衡区，2022训练、2023源校准；目标：英国NESO GB 2024年小时固定EF发电侧代理。源和目标定义、EF、训练权重、模型参数、校准方法均沿用[美英首轮冻结协议](CROSS_COUNTRY_PILOT_PROTOCOL.md)。
- 两个输入臂只改变英国目标2024的五类` t−2 `月份额：
  - `NESO_reconstructed`: 从当前修订NESO小时发电系列回算完整GB本地生成份额；
  - `DESNZ_ET54_MPP`: 从2026-10-29当前ET 5.4工作簿的MPP月度净供电燃料值构造份额。
- ET 5.4映射沿用[输入对齐审计协议](GB_OFFICIAL_MONTHLY_INPUT_ALIGNMENT_PROTOCOL_20261101.md)。月份额都按来源内部五类和归一化。ET 5.4缺失月/燃料不填零；目标预测只保留两臂都有合法月份额的小时。
- 以完全相同的US训练集和2023校准，配对运行六个已冻结模型：700火电锚、源火电EF锚、日历Ridge、CN5 Ridge、CN5提升树、CN5小时轮廓重组。不能按英国目标结果重训或选型。
- 输出逐方法两臂点预测哈希；在预测封存后计算MAE、去日均形状MAE、Gaussian CRPS、源校准90%覆盖、宽度、区间分数及配对变化。

## 解释边界

ET 5.4月表仅含MPP、目标小时序列是GB、工作簿是2026修订版本。故该实验检验官方统计来源更换的预测敏感性，不等于真实历史发行版回测；不能外推到全体英国发电商、美国输入或中国NBS。ET 5.4只是输入预算替代臂；消费侧官方强度不参与本实验。

## 冻结来源

- ET 5.4当前版页面/数据定义：[DESNZ Energy Trends: UK electricity](https://www.gov.uk/government/statistics/electricity-section-5-energy-trends)。
- 工作簿：ET_5.4_OCT_26.xlsx；公开附件下载地址在报告证据清单保存，SHA-256由脚本写入结果。
- 目标小时：`data/neso-gb/df_fuel_ckan.csv`，读取发电燃料列，不读取`CARBON_INTENSITY`。
