# V33：公开装机信息的获取与字段审计

2026-10-26，Sol。当前阶段只获取官方年度档案并检查工作簿字段，不转换容量数值、不训练模型、不读取旧目标或真正未知地区的标签。

## 研究动机与预算

V32的季节相位假设未通过，MISO持续水平偏差仍在。下一步检验结构信息是否比继续扩充时间特征更有价值。B0继续指原有公开月报；B1明确增加事前公开的机组装机结构。B1若胜B0，首先是附加信息价值，不是方法创新。方法增量仍须与B1同预算下的简单装机锚、普通回归等比较。

只收集EIA-860的2020—2023年度ZIP，这些年份已有2025Jan1前公开的官方记录。当前下载的年度档案是检索时版本，不能冒充原始历史vintage。后续若做按月起点的源训练，须使用当时已发布的年份；一种保守、待字段审计后锁定的政策是使用origin年份减2的库存，不能用2025前可得替代每个历史起点可得。

EIA主页面给出的四个下载入口为：

- `https://www.eia.gov/electricity/data/eia860/archive/xls/eia8602020.zip`
- `https://www.eia.gov/electricity/data/eia860/archive/xls/eia8602021.zip`
- `https://www.eia.gov/electricity/data/eia860/archive/xls/eia8602022.zip`
- `https://www.eia.gov/electricity/data/eia860/archive/xls/eia8602023.zip`

获取时逐档exclusive写入，记录最终URL、HTTP头、UTC检索时间、字节数及SHA256。原始ZIP不解压写入任意成员路径，只枚举文件、工作表与表头。失败记录保留；不能把未完成下载当作可用数据。

## 本阶段允许的字段检查

检查Plant和Generator工作簿是否存在、sheet名称、Plant Code／Generator ID、Balancing Authority Code、主能源来源、nameplate／夏冬容量、状态及多燃料字段的可获得性。只输出字段名与档案结构，不显示或转换机组容量、发电量或排放数值。不同年份的实际schema必须逐一核对，当前instructions不能替代旧表头。

后续数值整理另立合同，只允许七个已查看源的记录；须先按plant的BA筛选，再转换容量值。不得按州界、公司所有权或近邻坐标替代BA，也不得把未匹配记录默认为零。机组按Plant Code＋Generator ID唯一计数，多燃料容量不能在多个主燃料类别重复累加。

容量对应的是库存及额定口径。EIA的Operable包括运行、停运和备用；多燃料表还记录能够使用但未在上一年使用的燃料。当前EIA说明书允许nameplate小于net summer capacity并要求说明原因，因此nameplate本身也不是已认证的未来出力上限。本阶段不据容量硬截CIF，也不从无某类记录推断该燃料不可能出现。

若后续仅有煤／气容量而POS月量还包括油／other，`POS量/[小时×煤气容量]`只能称部分容量归一化的利用强度代理，不能称完整容量因子或把大于1直接判错。是否整理其余POS容量、怎样归类能源代码和状态，需要字段审计后预先固定，并保留缺失及未覆盖部分。

## 公开来源证据与限制

EIA-860主页面说明年度档案包含plant、generator、状态及多燃料文件：[官方数据说明](https://www.eia.gov/electricity/data/eia860/)。独立只读核查找到以下公开时间证据：2020数据于2021-09-09公开，见[EPA eGRID2020技术指南§2.4](https://www.epa.gov/system/files/documents/2022-01/egrid2020_technical_guide.pdf)；2021 final于2022-09-22公开，见[EIA当时通告](https://www.eia.gov/about/new/?r=1480)；2022 final于2023-09-19公开，见[EIA当时通告](https://www.eia.gov/about/new/?r=1120)；2023 final于2024-09-23公开，见[EIA当时通告](https://www.eia.gov/about/new/?r=750)。EIA的分页URL可能移动，须同时保留通告日期与正文年份，不能只凭当前页标题；2021通告标题存在年份笔误。

[当前EIA说明书](https://www.eia.gov/survey/form/eia_860/instructions.pdf)用于核查BA、状态、容量及多燃料语义，尚未逐年认证其与旧schema完全一致。[GEM当前GIPT页面](https://globalenergymonitor.org/projects/global-integrated-power-tracker/)有中国省级汇总入口，但最新版本为2026Sep，且煤电／油气等存在不同容量门槛；它可以作为中国落地的公开来源候选，不能直接用作2025可得或完整支持证明。本阶段不打开该汇总数值表。

## 下一阶段的条件

只有档案完整、所需旧字段真实存在、区域映射和缺失处理可审查后，才锁定七源数值整理与B0／B1同队列实验。数据增量、普通基线增量、专门风险机制增量分开判断；仍不打开未知地区、不报告当前尚不存在的容量实验或小时覆盖结果。长期目标active。
