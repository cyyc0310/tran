# 英国ET 5.4历史月报附件追溯：页面有快照，表格无公开可取版本

日期：2026-11-02。该审计推进美英双向小时迁移的真实预测时信息合同。结论只针对检索到的2024年9月一版ET 5.4工作簿，不推断所有历史月份都不可恢复。

## 核查路径

DESNZ官方电力统计页说明ET 5.4月表每月发布且滞后两个月，早期版本可向统计部门索取；Energy Trends collection保留各期发布日期。[官方ET 5.4说明](https://www.gov.uk/government/statistics/electricity-section-5-energy-trends)、[Energy Trends历史发行集合](https://www.gov.uk/government/collections/energy-trends)。

在2024-09-26发行后一天的网页快照（Internet Archive时间戳2024-09-27 18:17:23 UTC）中，页面仍链接当期原附件`ET_5.4_SEP_24.xlsx`，原始资产路径为`https://assets.publishing.service.gov.uk/media/66f422dfa31f45a9c765ec02/ET_5.4_SEP_24.xlsx`。快照HTML已保存在[证据目录](evidence/uk-et54-vintage-archive-audit-r1/)。

原附件URL的Internet Archive CDX精确查询返回空列表。直接请求官方资产URL后，服务器将其重定向到当前`ET_5.4_OCT_26.xlsx`，响应`Last-Modified`为2026-10-27；因此当前可下载内容不是2024年发行版本。网页快照证明了2024年页面曾发布该附件链接，但不能恢复表格数值或哈希。

## 对四项研究目标的影响

- **优先级1（美英双向）：**历史美英回测可继续用严格`t−2`滞后月量模拟；GB官方ET 5.4输入实验已证明当前修订官方输入与NESO小时回算不同，但不能标成AS-issued。已封存的2026双向前瞻预测仍按各自成熟日期评分。
- **优先级2（源端伪目标训练）：**输入合同不能简单用“历史公开页面上的现值”代替。GB源/目标伪目标若要模拟实际发布，应保留版本日期；取不到旧版时，先用严格滞后模拟并单独报告修订风险。
- **优先级3（燃料关系）：**现有M+F实验使用目标小时细燃料回算的`t−2`份额，是中国五类M之外的信息预算，仍只能作上限/机制诊断。
- **优先级4（月报适配与可靠性）：**真实历史适配回测目前缺英国原始月报版本；不应把当前修订月报和旧小时标签配对后称作as-issued预测。可继续报告非as-issued输入敏感性与封存前瞻预测。

若要补齐2024英国版本，现有公开网页和Internet Archive快照不足；需向DESNZ统计团队申请旧版ET 5.4工作簿，或对未来每次发行从现在起归档。美国EIA发行包可逐月还原，需继续将其作为单独、地理一致的预算验证，不把州月报硬配到不一致的BA小时标签。

## 可复核证据

- Internet Archive网页快照：`https://web.archive.org/web/20240927181723/https://www.gov.uk/government/statistics/electricity-section-5-energy-trends`
- 快照原始HTML及提取结果：[证据README](evidence/uk-et54-vintage-archive-audit-r1/README.md)、[机器清单](evidence/uk-et54-vintage-archive-audit-r1/archive_lookup.json)
- 页面附件原始URL目前指向的最新资产由官方ET 5.4页面列出；官方说明历史表可申请，不能仅据链接保留推断历史值仍公开。
