# V20 native-feature强控制的源区数据先决阶段

2026-10-26，Sol。只生成已有native架构的训练特征档，不拟合新的point/risk模型，不评分，不提出方法创新。所有V7–V19冻结字节保持不变。本轮代码、测试、协议需先通过根代理通用审查、独立Python审查和科学审查，之后才可正式prepare/predict。无target或未查看地区入口。

## 两类teacher与原查询copy

仅七源BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS。每个outer held源r，对每个s≠r构造teacher；bank严格为S\{r,s}，两地区fine标签均不进入该bank。42有序outer/teacher任务共享21个无序双排除bank。所有bank固定2024Jan25–Jun30完整valid UTC日、原region-month observed-support均值归一化、原unsupported fuel q=1 prior，fit cutoff固定2024Jul1；不把daily归一化改名复用。

每任务分两类，entry/split/时间语义严格分开：

- H1_retrospective：2024Jan25–Jun30全158calendar days。**回顾性地理OOF teacher，不是chronological hindcast**。bank可包含其他proper源在查询日之后或同日的H1 donor，明确chronological_hindcast=false、within_proper_future_donors=true；保存每origin可用bank中非过去donor行数及实际抽中数。原始时间不平移、不伪造prefix bank、不调用predict后绕其cutoff guard。用途仅为部署2025前已经可知的proper训练表征；不能将其评分包装成逐日滚动预测表现。
- H2_hindcast：2024Jul1–Dec31全184calendar days。相同H1 bank此时已在所声明数据集时间边界之前，chronological_hindcast=true、within_proper_future_donors=false，全部非过去donor计数必须为0。真实发布/训练就绪vintage未认证，不能据此声称已验证现实实时可用性。

另七个2025Jan1–Jun30 query native从已完成V18注册源归档逐元素copy原origins/raw/donor IDs，center为该raw的128粒子均值，原六源bank行顺序由已授权H1源日重建并保存。copy身份只用np.array_equal，不用allclose替代，也不读取2025 fine。

因此严格91个主预测档=42 H1 retrospective+42 H2 hindcast+7 querycopy；不是49，也不能把附属bank文件计为额外预测。21个bank provenance NPZ单独注册并封存，含q、raw_cif、donor_mix、region-month mean/support、row IDs、地区、日期、月份、双排除与cutoff。所有供体行和归一化月份均在H1。

## 同一native接口与合法粗前缀

只复用原V14 legal_coarse_law(monthly,region,days)、V11 _bank/_indices及recompose_paths；原seed17023、128粒子、Hellinger bandwidth.3、month kernel和source均衡prior均不变。donor用每origin的point mix选择，再将同128 coarse draws与抽到的donor q逐粒子重组。不是仅调用bank.predict(point_mix)从而丢掉coarse draws。

coarse law使用源monthly已完成且available_at≤origin的前缀，其历史误差在各自pseudo origin也只见当时已发布前缀。monthly读取先按源registry、period_end/available_at≤本轮最后H2 origin过滤，再转mix数值；每条teacher保存coarse mix/draw/metadata、最大已用period_end和available_at及全部raw/center/donor IDs。H1 coarse前缀合法不使其H1 donor bank变成时间合法；二者必须区别披露。

旧coarse提取器曾读取fine燃料聚合的事实沿用，不宣称新增粗统计来自已认证的独立真实渠道。本新生成器只解析源H1 fine和继承的已发布源粗统计；从单体hourly CSV扫描time文本后，丢弃H2/2025行，才对完整valid H1日转换generation数值。H2 fine响应、2025 fine和target monthly均不解析；不调用会打开target monthly的旧_load_training，不构造weather/ShapeCache/Context。

## 不变性、输出与后续用途

代码闭包继承V18的58文件加本轮4文件=62。输入exact21：V18 manifest/freeze/prediction_seal/completed四份、V14R1 prefit、source monthly、Python binary、七源hourly、七V18 native档。V18 manifest SHA固定efa95d7d79284d3157426fae9cd46785d75355fa0daee4fa479e1edc9571a3aa；source路径/七fine hash继承V18，monthly路径/hash继承被V18绑定的V14 prefit，query档hash对应V18 manifest。运行环境版本、路径及摘要冻结。仅计算旧文件byte hash不等于读取其未来标签作特征。

排他创建，任何partial预测/bank/收据均拒绝覆盖。全部91主档+21bank文件先完成严格路径/grid/hash、数据形状、完整日期、donor范围、双排除、H1归一化与采样/比例公式核验；H1/H2语义flag必须匹配对应split。bank通过已注册H1源数据独立重建，预测用相同原接口完整重建逐元素比对；query与原V18逐元素比对。输入/代码/环境/起始freeze在核验结束再次检查，随后才生成prediction_seal和completion。输出没有scores或训练新模型。

这个数据阶段并未完成native-feature residual refit强控制。后续若使用这些H1 geographic-OOF表征，native-only、packet-only和native+packet必须全部在相同H1 proper窗口、同预算和相同外层排除下重训；H2继续分riskfit Jul–Sep/riskpool Oct–Dec，缺少有效fine响应不捏造。本轮对calendar day全预测，不通过H2真值mask挑日。H2标签只能在本阶段完成封存、后续新阶段获授权后用于源训练；禁止用已评分2025H1误差训练候选。没有target授权、独立确认、实际预测增量或机制突破结论。
