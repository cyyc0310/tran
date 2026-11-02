# 2025美英跨域验证：美国官方小时碳强度标签数据审计

日期：2026-11-02。此处接续四项研究任务：跨国迁移、源域伪目标学习、小时结构迁移、低信息预算校准。中国仍按两个独立终点处理：发电侧CIF与消费侧CIF。

## 本轮发现

此前计划用EIA-930逐燃料小时数据在2025年反向验证GB→US，但目标七BA中存在缺列且官方月度总量证明部分缺列燃料实际非零，故不能把它们填成零来构造小时CIF。该不完整性审计保留在[evidence/eia930-2025-label-support-r1](evidence/eia930-2025-label-support-r1/results/summary.json)。

进一步检查EIA官方数据接口后，发现可直接取得EIA-930估算的逐小时碳强度，不必从不完整的逐燃料小时列重建标签。已抓取BPAT、CISO、ERCO、FPL、ISNE、MISO、NYIS七个BA的两个独立标签：发电碳强度`CO2.GER`和用电/消费碳强度`CO2.CER`。原始JSON、规范化CSV、请求参数和哈希见[机器证据README](evidence/eia930-official-co2-2025-r1/README.md)及[采集清单](evidence/eia930-official-co2-2025-r1/acquisition_manifest.json)。

| 目标轨道 | 官方变量 | API返回数（含端点） | 2025对齐小时/缺失 | 当前用途 |
|---|---|---:|---:|---|
| 发电侧 | `CO2.GER`，lb CO₂/kWh | 7×8,761 | 7×8,760 / 每BA缺1 | 按小时结束时刻减一小时并限制到2025后，每BA评分8,759小时；年内最后一个标签缺失，不补值 |
| 消费侧 | `CO2.CER`，lb CO₂/kWh | 7×8,761 | 7×8,760 / 每BA缺1,094 | 年内共同有效窗口截至2025-11-05 08:00 UTC；主分析固定为1–10月 |

EIA文档说明其小时标签是基于BA上报净发电量的官方估算：煤、气、油用BA机群年度排放因子，其他燃料用全国因子；消费强度通过MRIO模型给跨BA进口/出口分配排放。EIA同时指出，潮流报告误差会影响消费强度。故这些变量是口径清楚、可用于任务基准的官方估算标签，但不是机组烟囱实测，也不与项目固定EF标签等同。[EIA方法及标签定义](https://www.eia.gov/electricity/930-content/about.html)

NESO官方API提供2022–2024英国国家`intensity.actual`历史序列，当前取得52,486个半小时记录，其中56条实际强度为空，另有122个预期半小时起始时点无记录（2023-10-20 22:00至10-22 19:00 UTC共91期，2024-06-11 23:00至06-12 14:00 UTC共31期）；原始区间请求与逐请求摘要保存在[采集清单](evidence/neso-national-intensity-2022-2024-r1/manifest.json)。这提供了可与EIA消费轨迹对接的英国官方源域响应。NESO将其描述为基于计量发电的估算，不应表述为排放实测。[NESO Carbon Intensity API](https://api.carbonintensity.org.uk/)，[NESO国家数据说明](https://www.neso.energy/data-portal/national-carbon-intensity-forecast/national_carbon_intensity_forecast)

## 时间与口径边界

1. EIA接口值是**2026-11-01抓取时的当前历史值**，不是2025年预测当时保存的发行版。EIA月度`2025 Final`于2026-09-14发布；该年度文件只用于本轮燃料缺列支持审计，不能进入任何预测特征。[EIA-923发布页](https://www.eia.gov/electricity/data/eia923/)
2. 英国API历史值也是当前返回的估算序列，不是当时发行快照。故下一步最多先做一项明确标注的**当前修订值时间留出**，不能称作真实as-issued前瞻回测。
3. 发电标签和消费标签必须分别拟合、校准、评分；EIA指出消费标签受MRIO潮流平衡误差影响。两个终点不得合并成一个指标。
4. 下一步须先冻结时间戳对齐、EIA/NESO单位换算、2025共同目标小时、源域训练/校准年与预测特征合法性，再运行GB→US。若目标特征只能用目标年之后发布的最终月报回算，则结果属于修订输入敏感性实验，不能支撑“当时可直接部署”。

## 研究意义

此审计确认官方强度标签可支持独立时间留出，但三个任务不构成统一的双向同口径排行榜。GB→US发电侧结果中，源校准燃料重组MAE 87.06、覆盖66.86%，差于`anchor700`的MAE 81.88、覆盖71.13%。US→GB消费侧中，发电重组代理MAE 83.28、覆盖86.07%，好于源均值但仍欠覆盖且未建模流量。反方向GB→US消费侧，`anchor700` MAE 92.09、覆盖61.16%，好于源燃料重组MAE 103.87、覆盖57.13%；MISO重组覆盖仅0.9%。目标标签均是当前机构估算，目标月量是终版回算的`t−2`模拟，且GB消费强度源校准只有一个国家。详细见[GB→US发电](CROSS_COUNTRY_2025_OFFICIAL_GENERATION_TRANSFER_RESULTS_20261102.md)、[US→GB消费](CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_CONSUMPTION_RESULTS_20261102.md)、[GB→US消费](CROSS_COUNTRY_2025_GB_TO_US_OFFICIAL_CONSUMPTION_RESULTS_20261102.md)。这些结果不能合并为双向可信迁移证明。
