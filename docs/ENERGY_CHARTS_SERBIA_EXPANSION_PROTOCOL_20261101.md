# 塞尔维亚源域扩展：六国整国留出协议

冻结日期：2026-11-01，生成任何目标预测前。问题是：添加一个高火电、无核电的公开小时源域，是否改善中国式滞后粗月预算下的跨国小时预测，尤其改善目标域最差表现？这是已查看候选国家上的开发实验，不是盲测确认或真实CIF实验。

## 外层设计

- 国家：NL、ES、BE、AT、SE，加RS（塞尔维亚）。每次整国留出，其他五国训练/校准；目标标签只在预测文件全部封存后评分。
- 源训练：2022小时固定因子发电代理；源校准：2023H1；目标：2024。目标输入仅使用当地日历和从当前修订小时数据聚合、模拟`t−2`及更早的粗月发电量。
- 主对照：同一冻结模型集合、同一样本权重、同一月/国家评分和同一源90%经验残差区间规则。方法为`anchor700`、`macro_recomposition`、`ratio_recomposition`、`ridge_history`、`direct_latest/history`、`factor_latest/history`。不按RS或中国的预测输出来选择方法。
- 对NL/ES/BE/AT/SE，新增六国运行与五国基线r3的同方法配对，以估计仅增加RS一个源域的增量。RS留出折是新开发目标，单独报告，不与既有五国结果混成最终确认。
- 每个国家先月等权，再国家等权；报告MAE、去日均形状MAE、90%经验覆盖、宽度、区间分数、最差目标。国家数是统计单元，不将小时当独立域样本。

## RS数据合同与预先限定

- 输入：Energy-Charts API v2 `public_power`，2022-01-01至2025-01-01边界，小时MW、`Europe/Belgrade`；归因Energy-Charts.info，CC BY 4.0。原始响应和API元数据在本协议冻结前下载并哈希。
- 细类映射：`fossil_brown_coal_lignite→coal`，`fossil_gas→gas`，`biomass+others→other`，常规河流/水库水电→hydro，`wind_onshore→wind`；抽蓄、跨境、负荷、派生比例不进入本地发电分母。
- API未列出的nuclear、solar、oil、hard coal、offshore wind、other renewables在本回测映射中设为零占位并逐项记录。这表示接口未提供这些序列，不是独立证实真实发电为零。RS的标签因此只是已映射类别上的固定因子代理。
- 返回序列中的空值不按零填补；只保留映射类别均完整的小时。粗月量需达到本地日历小时数99%。`others`与`biomass`在24小时缺失，`wind_onshore`在2022年有1,948个缺失小时；预计2022可用训练样本少于完整年份，实际数量在运行审计中报告。
- API修订小时数据回算月统计，不是历史as-issued月报。抽蓄/跨境统计、源国家别的净发电口径和固定排放因子均限制结论，不称实测CIF。

## 判读门槛

将RS保留为有价值源域，需看到至少一种可部署参考在原五目标的配对国家宏观MAE或区间分数改善，同时不使最差目标的覆盖或点误差显著恶化；只增加单变量坐标范围、只改善RS本国折、或以加宽区间换覆盖都不算普遍增量。若RS源样本稀少、标签类别缺口令指标不稳，结果只记为数据支持敏感性，不推进方法创新主张。

## 可复现输入

- 原五国原始数据及校验清单：`data/energy-charts-v2-panel-r1/`。
- RS原始API响应与获取记录：`data/energy-charts-v2-panel-r2/rs_2022_2024.json`、`acquisition_manifest.json`。
- 官方API文档：<https://api.energy-charts.info/>；发布说明为CC BY 4.0并要求归因Energy-Charts.info。
