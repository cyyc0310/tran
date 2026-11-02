# R6b GFS对象预起点可用性审计协议

日期：2026-11-01。只核验已封存R6b天气对象的来源、内容封存和S3 `Last-Modified` 与既定forecast origin的先后；不读取小时标签，不重算预测或指标，不据结果选模型。

## 固定审计对象与判定

- 使用`results/weather-transfer-expanded-r1/collection_complete.json`指定的24个窗口及各自`window-*/manifest.json`。逐项核对清单SHA。
- 每窗固定检查lead 24–72的49个元数据记录；核对初始化为预测起点前一日00 UTC，GFS源URL属于`noaa-gfs-bdp-pds.s3.amazonaws.com`，并有保存的GRIB对象`Last-Modified`、ETag和响应元数据。
- 记录每个对象最后修改时间相对forecast origin的小时余量。若任一缺记录、对象时间晚于起点、来源桶不匹配或封存哈希不符，则该窗口判为未通过。
- “预起点公开档案证据”只表示存档对象在AWS公共数据桶的S3服务端修改时间早于预测起点。它不表示已获得逐用户访问日志、源站发布时间/可下载时间的独立回执，也不表示已核验当时的ACL或网络延迟。因此不得写成精确运行可用时间或实时业务SLA证明。

## 范围与隔离

该审计在R6b目标已经查看之后执行，为对象级来源事实核验，不作模型选择或新确认。R6b原始`eligible_for_issued_public=false`状态和输出封存保持不变；若审计通过，只在补充结果中提高对“预测前已公开归档”的证据等级，并保留上述限制。

来源说明：[NOAA NCEI GFS页面](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)列出GFS四个每日运行周期和NODD数据访问；[NOAA Open Data Dissemination数据清单](https://www.noaa.gov/nodd/datasets)将GFS链接到AWS公共桶；[AWS Open Data Registry的GFS说明](https://registry.opendata.aws/noaa-gfs-bdp-pds/)介绍数据产品、周期与开放使用条件。
