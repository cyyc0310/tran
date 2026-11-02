# 年度装机的年份、版本与地域审计

2026-10-31。用户已允许年度煤/气装机作为独立预算。本轮补查来源，并不把缺少版本证据的数据自动视为历史预测时可见。

美国EIA-860原7区和新增4个供体均有2020统计年份；官方档案当前Last-Modified为2021-09-09，按次日已可用进入既有软库存试验。原生zip SHA、主燃料/OP/登记BA筛选、零不等于物理缺席等限制保留；未认证首次发布的全部原始字节。4供体由同一curator生成，原7区重算与旧表atol1e-8一致。

英国[DUKES 2021页面](https://www.gov.uk/government/statistics/digest-of-uk-energy-statistics-dukes-2021)发布于2021-07-29，但目前官方xlsx的HTTP Last-Modified为2022-03-10，工作簿内部修改时间同为2022-03-10。5.7表有明确2020年装机，可以提取All generating companies的coal/gas，却仍是UK而非GB口径。不能只按网页原始发布时间把当前工作簿用于2022年1月伪预测；可单独设计修改日期之后的队列或版本敏感性，不能无说明替换旧实验。

德国[2021年报发布说明](https://www.bundesnetzagentur.de/SharedDocs/Pressemitteilungen/DE/2021/20211201_Monitoringbericht.html)日期2021-12-01。当前[官方PDF](https://data.bundesnetzagentur.de/Bundesnetzagentur/SharedDocs/Mediathek/Monitoringberichte/monitoringbericht_energie2021.pdf)正文第1页却标“截至2022-03-15”，HTTP Last-Modified为2023-11-02；因此报告标题年份、正文版本日和当前文件版本必须分开。第61页包含2020年褐煤20.6GW、硬煤23.8GW、气31.7GW，前文说明常规机组按所在地≥10MW范围且MaStR基于2021年11月登记，既有范围又有修订。不认证它与2021-12首版字节或数字完全相同。

| 软容量煤/(煤+气) | WRI 2021静态快照 | 当前官方年报的2020数值 |
|---|---:|---:|
| 德国 | 66.16% | 58.34% |
| UK（用于GB的代理） | 29.13% | 13.35% |

两者不是同年份、同覆盖和同容量定义的严格配对，不能把差异直接称作WRI错误率，更不能据此推断换表后预测会改善。它们说明静态辅助库存与同口径年度预算不能混用；英国差异尤其足以影响简单锚的水平。WRI两国的容量年份缺失，继续单列。

[审计JSON](evidence/public-capacity-annual-audit-r1/audit.json)记录原生文件SHA、当前版本、提取表位和处置。年报数字未进入本轮新模型，不触碰原预测。中国省级煤/气装机尚无本项目已核验且绑定发布时间的年度表，不能拿中国全国比例替代上海；月报无装机版本继续保留。

后续年度主实验至少记录：统计年、发布日期/可用日、正文版本日、当前文件SHA、地域与机组范围、主燃料及双燃料处理、缺失与陈旧标记。可声明的历史可见性要与证据强度相符，现有三国月报本身也仍是历史修订档案加模拟t−2时效，不因此升级为真实as-published回放。
