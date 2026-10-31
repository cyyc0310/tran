# V16A：合法滞后日组成反馈与创新条件形状的源区筛查

2026-10-25，Sol。目标是寻找超出简单信息更新的机制增量；不预称新算法。只使用七源BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS的2025H1开发评价，不读取任何目标/未查看地区数值。V7–V15冻结文件不变。文献和科学风险见`results/breakthrough-search-v16/design-review.md`。

## 信息与时间合同

明确新增信息预算：同口径八燃料**日总能量**、总能量和日期，模拟日末+7日/30日可得。不是当前/未来日真实组成，不是真实公开vintage认证，不假定中国实际存在这种发布渠道。日总能量可反算过去发电加权日CIF，所有同预算基线都可使用；它不等于过去等权小时CIF，也不含逐日24小时燃料/CIF轨迹。源hour监督只用于2024Jan25–Jun30 donor。不把“没有名为CIF的字段”当无本地碳信息。

隔离聚合器读取注册七源2022–2025H1已有hourly表，仅将完整有效UTC日压为八能量总量。它确实读取源评价期燃料值，必须披露，不能称整个实验从未读过评价hourly数值。预测模块只读压缩日表、源2024H1 proper小时、已冻结V14预测/coarse particles；全部预测封存后，评分模块才读2025小时CIF。此为先前已查看源区的开发，不是盲确认。

每次forecast均要求`day+1d+lag<=origin`；预序列误差的预测在该历史日00UTC仅用当时已发布日表，误差响应只有在其自身发布后才能进入当前抽样。2024H2开始保留至多168个可见日误差，按相对年龄exp(−age/56)抽128个整组成向量。源proper donor创新用其2024H1真日组成减自己的同规则预测；这些训练响应在最早2025源评价前均已发布。proper源地区与评价地区严格LODO排除；评价地区的过去粗日统计属于明确允许的本地上下文，而不是全局源训练标签。

## 固定简单预测与形状对照

日组成四个普通预测：latest；最近56日同星期加权均值（无同星期回全部，衰减尺度28日）；最近28日EWMA（尺度14日）；两阶年谐波ridge季节基线+标准local-level Kalman残差（q=.0001、r=.0025、初始P=.01，最多730个日历日历史；少于30响应回EWMA）。均允许全部八燃料simplex，不把历史零值当未来结构零。

粒子组成=当前点+同模型已发布预序列误差，再投影simplex。用于形状匹配的是**实际投影后冲击**；保存原冲击、实际冲击及投影量。local标准化同样先把历史误差应用到当前点、投影后计算，避免把被截断的冲击说成真实组成变化。

每个forecaster比较九个小时构造，保持128粒子和共同随机数：

1. additive：原V14 native粒子的零日均CIF形状+新日组成EF锚。明确锚是发电加权日量，未宣称是等权日CIF。
2. monthly_q：新日组成粒子×原月归一化donor q；用来揭示继续混入供体日幅度的影响，不把它当守恒日组成。
3. daily_native_ids：新粒子×每燃料**日均归一化**q，保留原native donor身份。
4. daily_analog：标准日组成状态/日历条件analog，kernel与原bank同Hellinger bandwidth .3、month .5、地区等权prior，但比较真实donor日组成。不是按查询未来真值选择。
5. forecast_analog：query预测点匹配donor的teacher预测点，使用相同kernel，明确两边都是forecast state。
6. particle_analog：每个query组成粒子匹配donor真实日组成，不使用innovation变量。这是“是否仅因用了粒子实现组成而获益”的必要强控制。
7. innovation_carbon：在forecast_analog权重上，按每个组成粒子的投影后冲击，与proper源donor的预序列冲击作三碳轴匹配。两边条件均为forecast state+forecast innovation；不能拿query预测点与donor真值的普通daily_analog当唯一强对照。
8. innovation_full：同机制用全部八组成轴，作为碳投影信息丢失的强控制。
9. innovation_permuted：把proper源创新在相同源地区/月内用固定seed打乱，再做三碳轴匹配。它破坏创新与同日state/shape原配对，不能单独隔离innovation−shape条件依赖、用作置换显著性检验或作因果归因。

三碳轴=(.4δ正EF份额,.6δ煤,.4δ油)，其和保留粗EF冲击，但不对小时比值充分；风/光或gas/other互换仍会改变形状。proper冲击按当前合法状态analog权重均值/标准差标准化；query冲击按当前已发布本地误差均值/标准差标准化，各轴sd下限.005。标准化距离取轴均方，likelihood=exp(−distance/2)，固定25%base+75%conditional混合。不是新copula定理；改变donor边际和ESS，不能解释成固定边际的纯依赖消融。保留full8、打乱控制和ESS/unique donor记录，避免只凭新命名声称创新。

日归一化q每fuel日均为1；任何日均为0的fuel固定回平坦1（包括donor某日没有该fuel而query粒子有份额的情况），不声称保持夜间零/机组可行性。每日相同fuel份额点/粒子与该q重组精确保持对应日燃料能量比例。这个变化与innovation条件化分开比较。proper H1与误差H2起存在季节支持差，记录历史数、响应时刻和选择ESS；同bank不等于同分布。

## 原控制、评分与阶段决策

从V14正式seal绑定原source native、coarse draw和donor indices，重建2024Jan25–Jun30六源bank；native raw必须逐元素完全相同，否则停止。原native无新日信息，是额外信息价值参考；四个simpleforecaster×六个普通hour arms均共享完整新增预算，是机制强控制。proper小时先按时间字符串筛选，再数值解析；不会因构造bank而解析2024H2小时值。

完整7源×2lag×4forecaster的56个归档（每个含九arm），加七native控制，共63个归档先封存。再读取2025hourly truth，评分原五倍率(.75,1,1.25,1.5,2)，均围绕各ensemble均值缩放并clip[0,1000]。主指标CRPS、90%小时覆盖/宽度/区间评分，另报双尾、日水平/形状MAE。与V14完全相同的1,145个有效源地区日，地区内日等权、地区间等权。30日全表作压力诊断；主选择只用7日，不能后验换lag。

按宏覆盖[90%,95%]且最差源区≥85%筛后最小CRPS；无人通过显式按违反程度回退。候选只在carbon/full两种innovation机制中选；打乱arm不参选。baseline在六个普通hour arms×四个forecaster中选；native单独选倍率。源pilot筛查只有候选通过上述覆盖且相对native和最佳同预算baseline均CRPS至少下降5%、区间评分下降、宽度不增加才为true。全部曲线/失败选择保留。

此处只比较local粗误差bootstrap+source shape，尚未完成相同日预算的直接Ridge/ET、异方差和QRF端到端强参考，未做新H2小时风险校准。因此即使源筛为true，也不能宣称最终概率优越、原创性或自动打开目标/新区；先增加强控制并源冻结方法，必要时再复测已查看目标。任何新区仍需明确增量后的独立确认合同。A/B是开发筛查，不重定义持久goal的突破标准。
