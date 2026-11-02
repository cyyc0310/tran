# CN5燃料组成响应头：波兰外部确认协议

冻结时间：2026-11-01，评分前。法国已提供一次外部正向信号后，选择此前只作为字段缺口候选、未评分CIF误差的波兰作为第二个外部目标。候选算法和六个基线完全沿用法国冻结版本；仅按目标公开类别预先登记数据映射及可评估月份，不重训选型或修改模型超参数。

## 盲测状态与目标窗

- 目标：波兰（PL），Energy-Charts `public_power` 2024小时序列，Europe/Warsaw时区。
- 先前CN5严格LOCO九国中不含PL。前置证据清点仅发现PL的原始公开能源数据曾在取数清单中，未发现PL作为该燃料组成响应方法的CIF误差、预测或校准结果；本协议把PL 2024该方法视为未评分外部确认目标。
- 预测窗固定为2024年9–12月，共4个连续月份。纯字段完整性预检显示：这四月有完整六个月滞后CN5特征，且映射后标签达到当地月小时数99%；1–5月缺完整标签，6月标签覆盖不足，7–8月缺完整六个月输入。上述裁窗只根据输入/标签存在性和完整性确定，不读取误差或CIF均值。
- 年度标签映射存在明确范围缺口：接口不提供独立`waste`序列，代理不把它自行插补到某燃料；若相关发电没有被API的`others`类别吸收，则不在本代理分母内。结果只针对此可映射的固定因子代理，不代表完整国家发电排放。
- 波兰没有2024核电生产，核能按0入映射，这是来自Eurostat的独立年度类别信息；Energy-Charts响应没有核能序列与之相符。[Eurostat 2024核能生产统计](https://ec.europa.eu/eurostat/fr/web/products-eurostat-news/w/ddn-20260129-2)。该计划数据表报告波兰2024年海上风电装机为0，Energy-Charts仅列风电陆上序列。[欧盟成员国风电部署计划数据](https://webgate.ec.europa.eu/circabc-ewpp/d/d/workspace/SpacesStore/fad3f719-dfed-44c2-909f-568e5a18575a/download)

## 固定类别映射

| 代理燃料组 | Energy-Charts生产类型 |
|---|---|
| coal | `fossil_hard_coal`, `fossil_brown_coal_lignite` |
| gas | `fossil_gas` |
| oil | `fossil_oil` |
| other combustion | `fossil_coal_derived_gas`, `biomass`, `others` |
| nuclear | 0（Eurostat记录2024无核电生产） |
| hydro | `hydro_run_of_river`, `hydro_water_reservoir` |
| wind | `wind_onshore`（官方统计显示2024无波兰海上风电运行） |
| solar | `solar` |
| other renewable | `other_renewables` |

抽蓄、储能、负荷、跨境电量及API派生比例均排除。存在序列的空值不填0；一小时只有全部映射的非零组分均有有限、非负值时才进入标签，其他小时保留缺测。缺少接口序列的技术须在映射表中明示，不能被解释成普遍“物理零”。因`other_renewables`及`others`序列存在间断缺测，协议固定采用完整行，不插值、不按误差剔除时间。

## 训练、候选及校准

- 训练源：NL/ES/BE/AT/SE/CZ/SK/HU/SI，等权国家，2022小时代理标签。PL不进入训练、选型、残差模型或校准。
- 目标预算：日历与仅以`t−2…t−7`历史五类月发电份额计算的CN5特征。不使用波兰小时燃料结构、年度容量、消费流量或目标标签。
- 候选`fuelshare_history`及全部实现、种子、EF和训练流程沿用[法国冻结协议](CN5_FUEL_RESOLVED_FR_CONFIRMATION_PROTOCOL_20261101.md)：以源小时样本训练煤/气/油/其他四个热电份额头，投影到非负单纯形后与源区小时形状合成CIF。
- 对照固定为`anchor700`、`direct_history`、`factor_history`、`macro_recomposition`、`ratio_recomposition`和`fuelshare_history`。
- 风险半径沿用九国源内逐国H2交叉拟合残差；按国家等权估90%绝对误差分位。不使用PL标签做偏差修正或校准。
- 所有PL目标小时预测、风险半径及输入/代码哈希先写盘封存，只有之后显式评分才读取响应标签。

## 解释门槛

按4个月等月权和有效小时池化双重汇总；报告每月及全窗MAE、去日均形状MAE、90%经验覆盖、区间宽度、区间分数和有效小时。比较门槛与法国一致：相对比例重组MAE至少改善5%，形状误差不恶化超过2%，覆盖不低于90%且相对比例重组下降不超过3个百分点，区间分数改善且不靠加宽区间。

即使通过也只代表单一国家、四个月、Energy-Charts固定EF代理上的第二次外部复制；小时相关，不能作独立样本显著性或整国覆盖保证。由于标签类别缺口，本次不能支持完整发电侧CIF准确度，更不能支持消费侧或中国小时精度。失败时不得按PL改映射/超参数重跑并继续称作确认。

## 复现产物

字段完整性预检：`scripts/preflight_cn5_fuel_resolved_pl.py`；确认运行：`scripts/run_cn5_fuel_resolved_pl_confirmation.py`；预检证据：`results/cn5-fuel-resolved-pl-preflight-r2/`；最终封存目标结果：`results/cn5-fuel-resolved-pl-confirmation-r3/`。r1/r2仅保留运行审计文件，不作为最终评分结果。
