# V247 协议：2024供体到2025未见BA的热输入迁移

## 目的

验证V246的最小信息增益能否跨年份保留。供体系数仅从2024 CEMS实测CO₂+实测热输入估计；目标标签与本地燃料发电量/热输入来自2025。每个目标平衡区从2024供体中排除相同BA，减少区域身份泄漏。

## 估计

按EIA-860当年Plant Code→Balancing Authority映射CEMS。仅保留CO₂与heat input measurement indicator都标记`Measured`/`MEASURE`、正运行时长、毛负荷、CO₂质量、热输入、且机组Primary Fuel Type为单一燃料标签的记录。将小时率乘以运行时长后按BA×月份×主燃料聚合。

对每个2025目标BA-月-燃料格，只用2024同月份、同燃料的其他BA估计：

- generation-only donor EF = 供体CO₂吨 / 供体毛发电MWh；
- heat-input donor EF = 供体CO₂吨 / 供体热输入mmBtu。

目标预测分别为2025本地月发电量×generation-only donor EF、2025本地月热输入×heat-input donor EF。只有两个系数都存在的fuel cell进入配对比较；结果同时报告覆盖率。BA-month误差包括等权绝对百分比/强度误差和跨样本总绝对排放误差占实际总排放的WAPE。

## 结果解释约束

目标标签为CEMS覆盖化石机组毛负荷强度，不代表BA全网CIF。目标月的热输入是当月本地粗统计，故这检验月内/月底排放核算与冷启动区域迁移，不是目标月开始前的小时日前预测。CEMS primary fuel是机组属性，不等同于每小时实际燃料。实验没有用小时因子结构，也没有评区间覆盖或预测区间效率。

