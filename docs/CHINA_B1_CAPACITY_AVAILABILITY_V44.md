# 中国B1装机信息可得性核查

2026-10-27，Sol。用户目前确认的是完整粗月度发电B0；中国库存B1仍为 **not verified**。本次仅查公开说明，不代表新数据集已下载、转换或与月报配对，没有新拟合或性能结果。

## 已确认与未确认

- **公开入口存在**：GEM Global Integrated Power Tracker（GIPT）提供中国省份×技术×状态汇总入口，以及机组容量、燃料、状态和位置等字段。容量是nameplate或最佳可得报告容量；这是研究者整合的设施清单，不是中国官方登记库存。
- **油电也有来源**：Global Oil and Gas Plant Tracker已覆盖油、气发电，不能把可得信息概括为只有煤、气。油气tracker的省份字段及覆盖说明支持继续核查，但不认证中国省级全量库存。
- **完整分母未核**：GIPT保留各tracker门槛：煤≥30MW，油气在中国≥50MW，生物质≥20MW，地热≥1MW。部分小机组被记录不证明门槛以下完整。V34的other包含工业副产气、余热等；其完整覆盖、互斥燃料归类及跨tracker去重尚未核。
- 因而不能把coal+gas部分容量归一化后称为V34四POS的q，也未认证完整C_POS。不同技术、机组/设施定义、多燃料和转换记录还须另核。
- **当前版本不等于历史版本**：GIPT当前September 2026版混合多个tracker不同更新月份，发布月不是统一研究截止日。旧版本的原始字节、当时公开时间及省级同年库存尚未核；不能用当前修订档或start/retirement年份冒充历史首发信息。
- **省厂址不等于月报成员**：地理省份与坐标描述厂址，不认证该设施是否属于用户粗月报的发电统计范围；全口径/规上、自备电厂等范围需匹配。不能称其与美国EIA860登记BA或EIA930发电报告口径一致。
- **NEA本入口不足**：此次核到的全国月度电力统计发布只列水/火/核/风/光装机，没有省级coal/gas/oil/other四分表。这只说明所核入口不足，不推断所有官方资料均无此信息。
- GEM方法说明多数数据为CC BY 4.0，特定TZ来源记录另有许可；实际使用时需保留来源及对应许可。公开可获取与完整、可比、历史可得是不同问题。

## 三个原始来源与实际阅读范围

1. [GIPT官方入口](https://globalenergymonitor.org/projects/global-integrated-power-tracker/)：阅读Overview、What's inside、Methodology和FAQ；核到中国省级汇总入口、当前September 2026版本、字段和容量定义，未打开容量汇总数值表。
2. [GEM GIPT官方方法说明](https://www.gem.wiki/Global_Integrated_Power_Tracker_Methodology)：阅读Inclusion criteria and source releases、Capacity and dates、Geography、Updates、Citation and data use及相关release notes。核对覆盖门槛、混合源版本、位置含义、历史更新记录和许可；未下载workbook。另只读GOGPT官方说明中的oil覆盖、State/Province字段及更新说明，未消费机组数据。
3. [NEA全国电力统计发布](https://www.nea.gov.cn/20251026/70d3b88f62f1401bbe47a8212e06ad8e/c.html)：阅读2025年1—9月发布表的指标类别与统计月份注释，不提取容量值、不据此建立省级库存。

当前可表述为“部分省级库存代理有公开来源”，不能表述为“完整中国B1已取得”。完整四POS分母、历史可得性及与用户B0月报范围一致性继续标为not verified。
