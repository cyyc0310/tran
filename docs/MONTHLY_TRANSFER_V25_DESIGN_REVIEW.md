# V25：月度M8／MG九臂普通预测的科学设计审查

2026-10-26，Sol。本文件提出源only普通预测合同，不是已执行实验或最终生产审批。只读旧协议／代码定义和已完成结果；未打开新fine、目标或未知区数值，未运行任何真实预测，未改旧封存字节。后续实际protocol、代码和证据闭包须另行三审。

**建议执行月度M8／MG的三类ordinary预测及匹配消融，共九臂，先检验联合源先验能否在隐藏燃料分配后仍保持覆盖效率。** MG固定为一个正EF合并组P={coal,gas,oil,other}加nuclear/hydro/wind/solar四单项，共五坐标；other也被隐藏，不得沿用V24“other仍已知”的CGO+gas结果。不是中国thermal认证映射，亦不预设新算法或成功。

## 1. 先关闭最容易混入的额外信息

- 仅七个已有源、2025H1完整共同评价日、七outer留区；所有源预测和完整grid封存后才评分。这是重复使用已查看源的开发结果，不是独立确认。无目标授权。
- 本地查询信息只包含合法发布月prefix、日历及声明的同预算元数据。无daily／T8／小时netG／Gmin、无本地小时CIF或由它们反算的信息。最低预算无天气；不能把既有历史GFS proxy当真实publication认证来补差。
- M8可以看八燃料月份额；允许独立coarse adapter把合法八列输入聚合为五组，但MG模型接口只能返回／持有这五组；必要的group求和不是使用隐藏细分预测的权限。MG的harmonic、prequential残差、support、距离、fallback、选型均只读五组。不能先做八维本地预报／校准再合并，也不能让隐藏coal/gas比例通过支持掩码、缓存键或原native供体权重返回。
- 本轮不执行年度支路。年度不能复制成12个月；Y-only需另行定义年度状态和源端年内分配层。

“无新fine parse”必须按阶段说清：训练和预测可以仅复用R1proper归档而不开小时源文件；R1不含2025评价真值。实际小时评分必须使用已绑定的2025真值cache，或另行写明全预测seal后才允许读取这七个已查看源的2025响应。旧CRPS／MAE表不能代替新路径的真实评分。没有合法评分真值通道时，不得声称完成预测实验。

## 2. proper库与发布时点

使用2024Jan25–Jun30源proper q、rawCIF、monthmean、donor_mix、support、地区／日期及父R1哈希。对outer r，fine库必须只含其它六源，整个查询、路由和源先验不得包含r。

R1保存的21个bank是**双排除五源bank**，并非七个现成六源bank。若从它们拼出六源库，须按(region,day)唯一去重，同一行所有q/raw/mix/support均要求精确相同；记录每行来自哪些父档、全局顺序和新row-ID映射。不能将多个pair-bank拼接后按重复出现次数加权，也不能重parse源fine而继续称纯归档复用。源行q按proper的region-month observed support均值归一化；它不是整个自然月公开表的重新认证。

所有组成prefix满足period_end、available_at<=origin。历史月误差只在响应也已发布后进入库；每条pseudo forecast仅见其pseudo origin的prefix。外层r的本地月历史作为合法查询信息可用，但其fine不进入源库／任何全局拟合。这种权限与“完全删除r所有信息”不同。

本九臂不拟合H2误差律或新point模型，因此不要求为它制造三排除内层risk教师；以后若训练pseudo-target风险再按既有严格嵌套合同处理。不要以预防未来问题为由把本最小试验扩成复杂风险机制。

## 3. 同频率的两条粗预测必须各自合法

固定现有harmonic思想、相同60月窗口／短历史回退、同一pseudo-origin规则、最多24个已释放历史误差、相同128粗粒子和日期seed。M8用八维单纯形及七维Helmert基；MG用五维单纯形及四维基，直接拟合五组。维数、simplex投影、coldstart和支持规则均需参数化，不能调用写死八列的helper后靠删列掩盖泄漏。

**共粗化一致的静态prior要提前锁定。** M8均匀八项聚合到MG是(.5,.125,.125,.125,.125)，不是MG均匀(.2,.2,.2,.2,.2)。冷启动若各用均匀，会把先验变化混入字段损失。推荐用静态粗化一致prior；若为了完整复用旧M8必须保留其它策略，清楚披露并给合成边界测试，不能归因于信息频率。MG不能用r当地隐藏八燃料历史估prior。

各轨的误差月份、日期uniforms尽可能配对，但MG误差向量由五维自己的历史预测和五组真值计算；“相同抽样索引”不意味着可以偷用M8误差再合并。过去月从未出现某组的support截断是所选预测模型的假设，不能称未来物理不可能；保留其诊断和冷启动规则。

粗点g0只决定该方法路由，联合粗draw g^(s)只进当前次传播。若让draw同时改变路由，也是可定义的方法，但会另改变耦合；此最小合同建议保持路由由point决定，减少差异。

## 4. MG的隐藏细分必须和fine形状联合

令源供体i的八项mix为m_i、q_i(h,f)为同一个proper记录。正EF组质量p_i=Σ_{f∈P}m_if。p_i>0时定义组内比例r_if=m_if/p_i。MG粒子g给出正EF组份额g_P和四个零EF单项；重建

`m_f^(s)=g_P^(s) r_if (f∈P)`，其余四项直接等于对应g项。

**同一i决定r_i和q_i**。不能独立抽组内煤气比例和另一个人的完整q，否则破坏待检验的源联合先验；也不能仅取全源r均值后称保留分组不确定性。每个粒子记录coarse-error index、donor ID、r、m、q来源和最终组份额一致性。M8则直接使用自己的八维draw，不再估隐藏分配。

这定义的是`Q(g | 合法月历史) × π(i | 可见粗点,日历)`及其r_i/q_i映射。它不是从本地月表识别出的真实P(r,q|g)，也不保证源r与当前g的真实依赖。先验跨地区可迁移就是本轮可反驳假设。

### 零组支持必须有显式预案

- 若g_P=0，组内r任意不影响路径；可用固定占位并单列计数，不宣称取得信息。
- 若g_P>0而所抽i有p_i=0，r_i无定义。不能设r=0使g丢质量；不能借j的r却继续用i在整个正EF组无支持的平坦q并宣称联合先验完整。
- 建议预定组级fallback：在同outer proper、组有支持的供体j中按当前方法供体权重限制到supported行并重新归一化抽取，**一起替换r_P与q_P**，其它组仍来自i。单列j、触发频率及它打破跨组整日依赖的局限；同budget各匹配臂用相同fallback随机流。其它单燃料在donor无支持时保留明确的既有q=1模型先验及支持计数，不当物理证据。
- 若整个proper库无该组支持，最保守的本小实验可失败关闭并报告unsupported，不能悄悄抹掉查询正组。若要继续生成必须在首评分前另锁纯prior分配和形状风险（例如组内纯燃料混合，而非确定中点），同时声明fine形状没有源依据；这不自动保覆盖，也不属于已完成合法回退。

若新增fallback使M8与旧native不再身份相同，保留原native独立背景，并明确新方法的差别；不要以“完整harmonic/native”为名覆盖旧路径。

## 5. 三臂必须是可解释的匹配对照

令a_i为地区均衡source prior；两种重组共用同一日历核、proper库和uniforms。global可以是calendar-matched global，但不得偷偷加本地fine组成条件。conditional仅乘可见组成的固定Hellinger核（旧bandwidth.3可沿用），M8距离在八维，MG在五维；不能从八维权重直接复用MG。地区均衡指prior而非条件后posterior必然等地区，ESS与地区集中度需记录。

| 臂 | 粒子构造 | 匹配目的 |
| --- | --- | --- |
| conditional_additive（M8、MG） | 使用conditional_joint完全相同的i、fallback j、r及fine mix m；`rawCIF_i + [EF·(m−m_i)]`，后项对整日广播 | 给标准线性EF校准全部相同隐藏比例先验；与conditional_joint匹配传播映射，保留各自clip行为 |
| global_recompose（M8、MG） | calendar／地区prior下抽i，按本轨coarse draw构造m，与同donor q重组 | 相对conditional_joint仅移除可见Hellinger权重；MG因此会同时改变隐藏分配和q的联合先验 |
| conditional_joint（M8、MG） | 可见组成固定核下抽i，按同donor的r/q联合重组 | 标准conditional joint prior候选，不称本地细分或fine形状已被识别 |
| conditional_shuffled（M8、MG） | 保持conditional_joint的128个fine-mix向量和128个完整q张量各自的多重集，预定seed仅置换q | 隔离完整fine-composition/q配对；M8是必要负对照，不能命名为仅MG隐藏比例或输出copula的作用 |
| conditional_plugin_split（仅MG） | 使用相同conditional权重在supported行上的平均组内r，粗draw及q保持joint版本 | 强普通点拆分参考；同时改变比例散布及其与q依赖，不是纯方差实验 |

MG plugin的平均比例来自source proper支持行，不能由本地隐藏月细分得来。组内r方差坍缩可能改变R的中心；shuffled虽然保留两个输入多重集，经过非线性R后并不保留CIF边际／均值／宽度。必须报告实际中心与风险评分，不能将其差值单独称尾部copula贡献。若M8粗draw流与donor流独立且路由仅依赖point，M8母体生成律原本就是Q(m)×π(q)，shuffle的母体分布不变；有限128样本的评分差应首先视为配对／Monte Carlo敏感性，不能虚构新的独立性贡献。MG含同i的r才引入设计中的m/q关联；两轨shuffle差也不是自动成立的因果差分。置换粒子必须包含完整24h×8fuel q，不独立洗牌小时／燃料；置换索引与重复值处理在评分前固定。它是source-adaptive依赖假说消融，不是因果检验或新copula算法。

粗粒子与供体随机数须为独立流、但在方法之间用common random numbers，避免复用同一个uniform无意制造关联。128情景抽完整日向量，不逐小时独立抽；不再叠加一套完整源残差或完整donor方差。保留现有[0,1000]评分clip、统一factor网格和原point定义，记录clip前后width／center／新增饱和和分母为零计数。R的零分母CIF未定义，只能用提前固定的模型回退（原m·EF可沿用）并披露。

### 精确恒等式与不能声称的守恒

必须有MG信息隔离的合成变形测试：固定proper bank、dates、seed及所有本地月份的五组值，任意重分配正EF组内coal/gas/oil/other，使各组和不变，MG全部point/draw/support/权重及预测路径应逐元素不变；M8允许变化。这检查误复用八维缓存／残差／support旁路。隐藏细分的评分通道不得返回预测器。

合成和归档层应验证：组份额Σ组内m=g；原source mix时R(m_i,q_i)=rawCIF_i；同mix时additive校正0；给定g和r的一致拆分；所有概率权重归一。M8 generic harmonic law应先与原八维helper在相同输入的point／draw核对；若bank顺序、CDF、seed和kernel均相同，可额外要求conditional native路径身份，否则只称可比较背景。

q是按源region-month observed均值归一，不是按每个预测日强制每fuel日均为1。独立抽日、query路由和缺失支持会使新月的整月总量不一定等于所抽月量。**本轮不宣称目标月燃料总量或CIF等小时均值精确守恒**。任何额外的整月raking都改变路径依赖与风险，应另作方法，不能插入本九臂。

## 6. 评分、背景和可反驳增量

建议本轮先固定唯一主发布延迟，沿用月表自身available_at，不能拿小时lag7充月发布。M8四臂、MG五臂，统一五factor形成45条主曲线；actual主计数以最终协议为准。M8与MG共享日期、128样本、proper支持与评分规则，分开记录各轨source选择。

粗评分：两轨都在同一五组函数上评组成CRPS／joint energy score、组误差及发布年龄；隐藏组内分配与完整八维EF粗强度只在评分侧使用已封合法标签。若月细分缺失，保留未评分月，不删共同小时评价日。每月重复日预报按既有规则汇总并报告独立地区月数，不能当成数百个独立粗标签。月粗A是发电加权函数，绝不拿它直接当真实等小时日均。

小时评分：CRPS、90%覆盖／宽／IS、上下尾、逐源worst，以及等小时level和中心化shape指标。覆盖资格仍宏90–95%、worst≥85%，全grid、固定factor配对和source选后结果并报；无合格项则fallback记failed。不让“比M8更保守”或仅变宽成为MG成功依据。

至少保留完整M8 harmonic/native及旧V9作为强背景，披露窗口、proper支持、倍率和随机法不同。**M8是MG的信息较丰富参考，不是MG同预算baseline。** MG应首先与其自己的additive/global/plugin普通对照比较；拿合法MG打败用错中点的弱锚也不能构成完整方法优势。后续主张优于强概率模型，仍须同月信息的direct Ridge/ET、location-scale、QRF/EMOS等完整预算控制，本九臂不冒称完成。

可证伪解释提前固定：

1. MG conditional不胜MG global：停止把普通组成条件化包装成机制；不据此断言月粗信息无用。
2. MG相对M8差距大且隐藏组内粗A也失配：该源先验未补足字段损失；不能说增加复杂小时传播已解决。
3. 粗A改善但hourly CRPS／覆盖效率未改善：尚未证明粗预测帮助小时风险；源形状／G–C依赖迁移仍是假设。
4. additive已解释全部收益：保留标准方法，不将比值重组／联合拆分命名为新算法。
5. 覆盖不合格、width靠clip下降、IS恶化或source支持fallback集中：不晋级，不转向target调参。
6. 若MG同预算条件机制在覆盖合格时超越其普通参考达到至少5%CRPS下降、width不增、IS下降，才提出更强同预算概率控制及独立确认合同；source gate不是创新证明或未知区授权。六个月CI只作开发描述。

## 7. 本轮可能新增的知识

[V9](COARSE_DYNAMICS_V9_RESULTS.md)已经跑过月动态＋条件残差；[V14](WEATHER_COARSE_V14_RESULTS.md)已经有粗细误差／比值和rank对照，均未建立强效率增量。V25的明确差别是**在本地月燃料分组被隐藏时，仍共同抽样源组内比例与fine形状，并用完全相同信息的标准线性／全局／条件映射及plugin/shuffle反驳假设**。是否缓解现实字段不足可被此次预测否定；组件本身仍是普通conditional analog／mixture／比值传播，不称原创。

聚合监督及非线性概率协调已经存在：[NeurIPS2020聚合观测学习](https://proceedings.neurips.cc/paper/2020/file/5b0fa0e4c041548bb6289e15d865a696-Paper.pdf)、[UAI2025细源监督与粗目标标签](https://proceedings.mlr.press/v286/singh25b.html)、[2026非线性概率协调](https://arxiv.org/html/2604.26668v3)。不能仅靠“粗目标＋细源”“H/G非线性”或“保留相关”主张新颖。真正潜在贡献仍需对明确字段合同、未识别部分和跨源可迁移假设给可验证适用边界，再有超越同预算强参考的独占增量。当前只建议冻结最小预测合同，不预告突破。


本文已纳入根代理随后确定的九臂修订：conditional anchor而非global anchor、M8与MG对称shuffle、仅MG plugin。它仍是设计建议；尚未读未稳定生产实现或对其签发执行审批。若最终protocol变更这些匹配或发布门禁，应重新审查对应字节。
