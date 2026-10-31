# V280：EIA官方BA小时CO₂标签上的PACE/LGEE冻结外部标签检验

冻结日期：2026-10-30，下载或解析PACE/LGEE CO₂结果之前。本实验检验V279支持区间感知最小观测规则能否从“项目固定EF生成代理”转到EIA公开BA小时生成侧CO₂估算标签。PACE/LGEE此前是项目的困难区域，但这两区的EIA官方BA CO₂工作簿此前未进入V273–V279实验；因此本轮是**新标签源的区域压力外测**，不是新地区确认。

## 数据和标签

- 目标BA：PACE与LGEE；目标期：完整本地月2025-01至2025-12。
- 官方单BA工作簿：`https://www.eia.gov/electricity/gridmonitor/knownissues/xls/PACE.xlsx`、`.../LGEE.xlsx`。
- 标签：`CO2 Emissions Intensity for Generated Electricity`，报告正发电小时；按本地完整24小时组日。
- 每BA-month中心用目标工作簿月度正发电加权EIA小时强度计算，属于oracle中心；只评价小时/日风险形状与区间，不评价月前中心预测。
- 月燃料边际及小时燃料组合按V275同口径聚合八类类别。煤、气、油使用V269 r2对应季度的美国CEMS经验支持；`other`固定400，非化石固定0。该EF支持为跨系统压力合同，不是PACE/LGEE目标因子的置信集。

## 冻结方法和信息预算

每个BA-month随机抽12个完整日。使用抽中日的本地小时聚合发电字段，联合整月燃料边际，通过V279同一逐小时LP，在CEMS支持区间端点求CIF可行范围。每个抽中日的分数是24小时中相对于oracle月中心的最大端点距离；90%有限样本分位数给剩余日期构造中心对称区间。相同随机划分比较：

1. `support_minimum`：V279规则，即测总量、活跃煤/气/油分项；若`other`活跃另测thermal；
2. `fixed_four`：总量、thermal、coal、gas；
3. `full8`：八类小时量，上界参考；
4. `source_loo`：七个已下载供体BA（BANC、BPAT、CISO、ERCO、FPL、MISO、NYIS）同月2025名义小时日最大偏差池，不含PACE/LGEE目标。

目标小时字段只用抽中12日校准风险；测试日覆盖、整日同时覆盖、宽度和Winkler评分只用EIA官方目标CIF。另报告可行集稳健包含的全小时事后诊断，明确其使用测试日完整燃料结构，不作为预测时可用输入或覆盖保证。每BA单独报告，因只有2个BA，不对BA bootstrap做推断。

## 预设解释

- 支持感知方案在两个BA均维持至少90%日同时覆盖，且比固定四字段更窄/Winkler更低，才算V279机制在EIA估算标签上的初步复制。
- 对source baseline只比较目标月中心条件下的覆盖与评分；由于其不读取目标小时字段，不报告LP集合包含。
- 任何差异都只能说明在EIA估算标签和该跨系统EF压力合同下的形状迁移；不得称作直接CEMS真值、全网实测或中国验证。该标签由燃料发电与年度BA因子估算，来源与小时燃料活动量共享。

## 复现

运行：`python scripts/v280_eia_official_label_pace_lgee.py`。原始工作簿与哈希保存在`results/v280-eia-official-label-pace-lgee/`。
