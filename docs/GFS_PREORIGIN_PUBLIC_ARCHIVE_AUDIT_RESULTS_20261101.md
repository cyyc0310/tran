# R6b天气输入：24窗均有预起点公开S3档案时间戳

完成时间：2026-11-01。此项是对已封存R6b天气文件的来源审计，不重训、不重算预测或指标，也未读取小时标签。R6b目标国家和日期此前已用于开发，审计不构成新确认。

## 核验结果

脚本检查24个固定窗口、每窗lead 24–72的49个GRIB对象，共1,176个对象。每一份窗口manifest都与总完成清单中的SHA一致；每份lead元数据也与对应窗口manifest内的文件SHA匹配。所有对象均来自`noaa-gfs-bdp-pds.s3.amazonaws.com`，且保存有S3 `Last-Modified`和ETag。

| 项目 | 结果 |
|---|---:|
| 窗口数 | 24/24 |
| 对象数 | 1,176/1,176 |
| 修改时间晚于预测起点的对象 | 0 |
| 最紧预起点时间余量 | 19.867小时 |
| 最大预起点时间余量 | 20.398小时 |
| 元数据/窗口/总清单SHA不匹配 | 0 |

每窗49个lead的最新S3 `Last-Modified`均早于该目标日00 UTC forecast origin约19.9–20.4小时。样例对象在预报初始化日约03:50 UTC写入存档，预测起点为次日00 UTC。S3对象修改时间是服务端对象元数据，和今天的重新下载时间分离；这比仅凭GFS初始化时间推断预报可用性更有力。

NOAA NCEI说明GFS为NCEP预报系统，并列出NODD 0.25°/0.5°预报及00/06/12/18 UTC运行周期；NOAA数据集目录将GFS链接到AWS，AWS Open Data Registry说明该GFS数据的公开使用条件。[NCEI GFS数据与周期](https://www.ncei.noaa.gov/products/weather-climate-models/global-forecast)、[NOAA NODD数据集目录](https://www.noaa.gov/nodd/datasets)、[AWS GFS公共数据说明](https://registry.opendata.aws/noaa-gfs-bdp-pds/)。

## 如何解释

结果支持将R6b天气输入描述为：**使用对象级时间戳核验的、在预测起点前已存在于NOAA公共S3档案的历史GFS预报**。此前R6b将其记作`eligible_for_issued_public=false`，因为没有逐次原站回执；本审计提高的是档案可见时间证据，不修改原预测seal或原评分输出。

剩余限制：本审计不能还原具体客户端何时完成下载，也没有2023年的公众ACL历史或服务延迟SLA；GFS的单点变量仍是天气情景特征，不是完整电厂装机/负荷加权预报。所以上述信息足以支持“公共档案在预测原点前已有该对象”，但不支持精确实时可用承诺、区域天气误差保证或中国预测精度。

## 对四项研究重点的影响

- **跨国小时预测/合法信息合同：** 天气分支现在有了强得多的预起点档案证据。R6b已看开发集上，M+C天气HGB相对无天气HGB的MAE、形状误差与WIS均改善，宏覆盖提高且平均区间变窄；最差配置仍约65%覆盖，故不能称作可信区间方法。
- **训练时模拟中国信息：** 该天气收益主要出现在年度装机预算，M月报预算仍严重欠覆盖。年度装机对中国字段的增量已在另一五国M+C消融中未复现，因此天气+容量的组合不能直接转成中国主结论；五类M天气表现应继续作为重点对照。
- **可迁移燃料关系：** 本审计不提供新燃料模型证据，现有强重组和普通变化量模型保持原判断。
- **月报适配/风险：** 有效平均风险分数改善，但地区尾部仍失效。下一步应该按预先固定门槛检查天气增量的跨国家/地区异质性及风险排序，不再扫描半径来追求90%覆盖。

## 证据

- 协议：[对象级可用性审计协议](GFS_PREORIGIN_PUBLIC_ARCHIVE_AUDIT_PROTOCOL_20261101.md)
- 代码：[audit_expanded_gfs_preorigin_public_archive.py](../scripts/audit_expanded_gfs_preorigin_public_archive.py)
- r2结果：[完整对象时间表与摘要](../results/gfs-preorigin-public-archive-audit-r2/)
- 原始R6b天气预测：[结果报告](EXPANDED_ANCHOR_ISSUED_RESULTS.md)、[封存证据包](evidence/expanded-anchor-issued-r6b/README.md)

r2逐对象表SHA-256：`9f41dac54adcfd3053b4ae9019272698d247f03df29595f4a45d64c4e4868cee`。审计只读取已封存的窗口元数据与公共来源文档，不读取预测结果或评分标签。
