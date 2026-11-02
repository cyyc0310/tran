# 真实预报档案的最小可用性核查

2026-10-31，仅元数据审计。天气是此前已授权的独立附加信息预算，当前月报/装机成绩不变。本页没有天气模型成绩，也没有新增地区小时标签。

后续已完成[两个原生24h窗口提取及缓存验证](GFS_NATIVE_PILOT_RESULTS.md)，本文保留最初元数据阶段的范围；仍没有CIF天气模型成绩或区域天气认证。

[Open-Meteo Historical Forecast](https://open-meteo.com/en/docs/historical-forecast-api)说明其连续序列拼接各运行的前几个小时，不能直接作为日前24小时预测时已可见的天气。[Single Runs](https://open-meteo.com/en/docs/single-runs-api)支持初始化运行，但多数模型从2026-04起，IFS从2024-03起；文档还将早期IFS档案标为49R1 hindcasts，不能只凭初始化字段认定历史实时可用。初始化也不等于公开时间，文档给全球模型约4–6小时计算/分发延迟。[Previous Runs](https://open-meteo.com/en/docs/previous-runs-api)提供固定提前量，多数模型从2024开始，GFS温度较早，不能由温度可用推断旧风/辐照同样可用。这些差异使其不能无说明回填当前2022–2023三国试验。

为保留现有时期，本轮另核查[NOAA GFS公开运行档案](https://registry.opendata.aws/noaa-gfs-bdp-pds/)，只请求HEAD与小型.idx文本，未下载约500MB的全球GRIB主体。

| 运行UTC | 提前量 | 对象Last-Modified UTC | 索引可见变量 |
|---|---:|---|---|
| 2022-01-01 00 | 24h | 2022-01-01 03:43:28 | 2m温度、100m U/V风、地表向下短波 |
| 2023-07-01 00 | 24h | 2023-07-01 03:38:48 | 同上 |

索引中短波是18–24h平均预报，不是瞬时辐照，必须保留时间区间语义。两个运行都有当年对象修改时间，说明存在可追溯的旧运行候选；这不认证全部日期、全部提前量、变量值质量或空间代表性，更不等于已完成合法天气输入。

[审计证据](evidence/legal-weather-archive-audit-r1/manifest.json)记录URL、HEAD、索引原文、SHA和抓取时间。下一步应先少量提取：核对GRIB运行/valid-time/step及索引字节范围、在预测起点前留足公开延迟、按同一起点生成完整24h输入、检查全国/BA空间采样；城市单点不能默认代表全国风光。小规模通过后才评估全量检索成本与模型信息价值；不给接口占位返回数值，不使用事后天气补洞。
