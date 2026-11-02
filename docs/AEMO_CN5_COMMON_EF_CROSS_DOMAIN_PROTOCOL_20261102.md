# AEMO跨域迁移的统一固定因子标签诊断协议

冻结日期：2026-11-02。该诊断检验现有US/GB→AEMO结果中的失配是否主要来自目标与源采用不同排放因子口径。只重新构造目标评分标签；复用已封存预测，不重训、不按新标签选择模型。

## 预测封存与目标支持

- 预测来自`results/aemo-cn5-cross-domain-transfer-r2/predictions.csv`，必须先核验SHA-256为`bcc11f64680df9c122655f5d8c1ffe337baf6f9a22cd10fa85f9efae2bd9bbb1`，并核对原预测封存中的AEMO SCADA/GENUNITS/注册快照哈希。
- 比较US-only、GB-only、US+GB三个既有源训练臂，以及已封存的anchor700、source thermal EF、calendar Ridge、coarse Ridge、coarse boosting和coarse recomposition六种输出。
- 目标期为2023-03至2023-12，主地区固定为NSW1、QLD1、TAS1。延续原实验预测标签读取前冻结的CN5月支持≥95%门槛；另要求本协议重建的燃料映射小时支持≥99%，防止遗漏燃料改变分母。
- 每个地区先逐小时算分，再地区等权汇总；同时保留全部逐区、逐方法结果。SA1/VIC1仍只作支持失败诊断，不进入主汇总。

## 统一源目标标签

源标签沿用`cross_country.py`的直接CO₂固定因子映射，单位g/kWh：煤1000、气400、油800、其他化石/生物质400，水/核/风/光为0。目标侧从AEMO SCADA逐DUID五分钟样本按小时/地区汇总，使用同一固定因子和同一归一分母计算小时标签。

注册燃料到源模型类别的冻结映射：black coal、brown coal→coal；natural gas (pipeline)、coal seam methane、coal mine waste gas→gas；diesel oil、kerosene - non aviation→oil；其余已审阅的thermal源（landfill biogas methane、bagasse、biomass and industrial materials、other biofuels、primary solid biomass fuels、other solid fossil fuels）→other；hydro/nuclear/wind/solar各自映射到同名类别。未知或多类别冲突的DUID不静默映射，按未映射正向MW计算小时支持率。

目标标签为**统一固定因子发电代理**，不是AEMO官方小时CIF、烟囱实测或消费侧强度。它消除源/目标EF表定义差异，但不能消除AEMO与EIA/NESO在调度、统计边界、燃料归类、单位/时钟及SCADA抽样方式上的差异。原机组注册CO₂-e代理保留作为不同标签定义的压力测试，不与本标签并为一组。

## 评分与解释

复用原封存预测中的来源偏差和尺度，不用AEMO标签重新校准。报告MAE、日内形状MAE、Gaussian CRPS、90%经验覆盖、90%宽度和区间分数，逐区并等权宏平均。另与原AEMO CO₂-e代理报告作同一预测、同一支持小时配对差异，识别标签定义敏感性；不得据单一AEMO开发样本宣称普遍迁移。

若统一因子标签下锚700因定义与标签高度重合而机械占优，应明确指出这是标签构造结果，不作为算法证据。只有粗模型/重组在非退化燃料组合和至少两个目标区同时改善，且区间评分与覆盖不因改变标签定义而失真，才能支持“迁移机制而非EF口径”解释；否则保留为基准敏感性分析。

## 可复核材料

新脚本在预测哈希与输入哈希核验后才生成目标标签，并将标签哈希、代码/协议哈希、支持统计和逐项分数写入新的结果目录。它复用此前已经封存的预测，故是回顾性目标标签敏感性分析，不是盲区确认。
