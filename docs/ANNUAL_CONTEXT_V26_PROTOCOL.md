# V26：未执行条件草稿／PAUSED：年度细分不在已确认字段内

2026-10-26，Sol。**本草稿没有冻结或正式执行。** 用户最新确认只有月度火电与可再生能源发电量，未确认年度煤／气发电细分；因此三正式入口在任何IO前无条件拒绝，无CLI绕过。以下保留的是先前条件设计，不能当成实际数据合同或已执行结果。仅七个已查看源区2025H1，原V25全部1145共同有效源日，未查看地区保持关闭。前提是存在过去年度燃料**发电量**细分；用户只确认月／年公开燃料统计，未确认此字段，因此本轮是受控信息价值实验，不是中国渠道验证。源年度是完整12条同边界月proxy的能量加权年度代理，不认证官方年报、发布档案或完整小时普查。

## 固定资料与时点

父V25最终证据清单 `740e8b1f18c93ec4bb63136da59cbb026ff24233587c17ca1569b2e6d90f5ad3`；V24清单 `b2f98a68f8bebf64852c75c8a9eba6d373132bbb0a52648a4ab262b9538ecf98`；年度输入 `results/coarse-fields-v24/annual_bounds.csv` SHA `5ef5561ec4d57888c84e45a317df8281f9222be7eeef1ad096b45112fc94fd70`。完整闭包与新producer/tests/protocol、三预审、环境绑定在新freeze。仅新增V26文件，不更改父产物。prepare/predict/score分别拒绝覆盖，预测全七区seal后才能评分原七源2025H1小时响应。

直接复用V25 MG law draws、原conditional IDs、POS fallback IDs及bank q、rawCIF、donor_mix，不重新拟合月law，不读新proper细数据，不读本地小时G/天气/日反馈/目标。M8结果仅背景，不作为同预算参考。年度所有行先以字符串读取；按schema=eight、region、年度期间结束与available_at均≤各自context origin筛选，再转燃料数值。仅完整年度有效；不补缺月、不复制年度成月份、不读取未来年度数值。2023代理年在2024-01-31可用；2024只有五源完整且于2025-01-31后可用，ERCO/ISNE不能伪造2024年。

## 五个ordinary臂

1. `MG-recompose_conditional`：逐raw、fine_draw、IDs精确等于V25原MG，不加年度信息的背景。
2. `AY-fixed_anchor`：最新合法年度POS内部比例固定，使用V25的加性锚公式。
3. `AY-fixed_recompose`：相同固定比例、相同q，使用原物理比值重组。
4. `AY-residual_anchor`：年度锚＋普通源比例残差，使用加性锚。
5. `AY-residual_recompose`：相同残差比例、相同q，使用比值重组。

设a_t为查询区最新合法年度中coal/gas/oil/other四项归一比例。每粒子MG五组draw的POS总份额p保持不变，其它四个非排放项保持不变。固定分支四项为p*a_t。残差分支取有效POS供体j的月内部比例r_j、该供体自己的日期之前最新合法年度比例a_j，得到 `r=Proj_simplex(a_t+r_j−a_j)`。这只是标准加性delta/残差迁移，不命名为原创机制，也不称充分age-matching。

有效POS供体若有V25 fallback则为j，否则为原i；比例、源年度context和日期均用同一个有效POS供体，整组POS q及其support mask也随j。非POS q/support仍取i。缺查询年度或最新年度POS=0，两AY分支退回原MG比例；查询年度可用而源年度缺失或POS=0时，残差delta=0，退固定年度比例。不得混称为原MG回退。每种缺失、零POS、projection幅度、context年份／可用时点／年龄均留档，查询POS=0不会误失败。

源缺某fuel时其q=1沿用原库的显式默认；年度锚可能给该fuel正质量。这是新的形状先验假设，不是该源提供了这种fuel的真实形状。逐粒子报告无源fuel形状支持的POS能量比例，POS support随fallback更新。没有宣称新年度约束识别了小时调度，也不强制预测月真总量守恒。

加性锚 `rawCIF_i + EF·(fine_draw−donor_mix_i)` 与V25完全一致；重组为既有 `EF·(fine_draw*q)/sum(fine_draw*q)`。每对anchor/recompose共享fine_draw与q。V25基线重组必须复算后raw严格identity，IDs/fallback/fine/groupdraw exact，否则预测中止且不seal。本版没有容差降级；如发现布局问题，须保留失败记录、另行审查新版本，不得运行中切换规则。固定与残差的差还可能包含投影、source-context缺失与非线性中心，不称纯风险机制。

## 评分、归因与失败关闭

181起点×128粒子×24小时；原五倍率0.75/1/1.25/1.5/2，既有裁边与缩放、逐日评分口径。每源有效日等权再七源等权。90%区间门槛宏90–95%、worst≥85%，失败fallback继续false；不扩倍率。CRPS、覆盖、宽、IS、上下漏、日level和中心化shape误差齐报。先封存全部七区再读已查看2025源fine，不做模型训练或新增风险拟合。

对每个倍率报告四个匹配对比：fixed recompose−fixed anchor、residual recompose−residual anchor、residual recompose−fixed recompose、fixed recompose−MG背景。全年主结果保留原1145日共同队列；年龄／年度发布前后结果只作预定描述，不按分数选窗口或删缺源。context审计报告query/source年龄支持、missing/zeroPOS与unsupported比例；不能把同年度181日起点视为181独立年度样本。源proper H1的年度新鲜度与2025query通常不同，特别2025Jan及ERCO/ISNE更陈旧，因此普通残差的不足不等于所有时效方法失效。

新增年度信息造成的收益只能称信息价值；普通残差超过固定锚也不自动原创。异方差／条件分位数等完整同预算强参考、年度only和真实中国字段/vintage仍未完成。本轮不决定paper已达标，不授权打开未知区。只有真正独立机制在完整同预算强参考上满足原覆盖效率与增量门槛，另立冻结确认合同。

三份预审先完成再prepare。实际审计应不导入V26 producer，独立选年度、重建有效供体/回退/投影/两比例、路径和评分，并核首尾hash；原receipt不回填。所有失败日志保留，不因结果不好改变预算、fallback或门槛。
