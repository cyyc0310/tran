# V14 冻结协议：天气门控与粗误差到小时风险

2026-10-24，Sol。V13原网格在其首轮源评分前已冻结；本V14仅作空校准响应队列的运行时修复，在V14首轮源评分前冻结。V13源首轮分数已经查看，不能称全新盲选；未改变网格、参数、选择或晋级门槛。本文件不是结果。只使用七个既有源区和六个已查看目标，新地区保持封存。科学动机、原始文献和可识别性边界见 [设计审查](V13_DESIGN_AND_LITERATURE.md)。


修复来源和元数据计数见 [V14校准队列修复说明](V14_CALIBRATION_REPAIR.md)。唯一新增运行合同：从冻结源月表确定具有至少一条已发布H2响应月的`calibration_sources`。实际为BPAT/CISO/ERCO/FPL/MISO/NYIS六区；ISNE仍在七源proper、donor、gate训练和源评估中，只不作为没有合法月监督的H2风险响应区。外层留出六区之一时风险响应区为五个，留出ISNE或预测目标时为六个；所有新风险家族使用完全相同的有效日队列。不能因响应缺失放宽日期、补标签或删去ISNE评估。

V14首轮重新运行原六分支，所有预测数组、逐日/地区/宏评分和源选择应与保留的V13首轮完全相同；只允许版本、路径、封存时间与运行元数据不同。先验证此身份，才将其源选policy传入V14第二轮。V13第二轮在构造空ISNE响应前失败，未完成源预测/评分，且当时两轮均未生成目标预测或目标评分；V14不把运行失败当性能证据。

## 1. 地区、时点和共同信息

- 源：BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS；已查看目标：AZPS/DUK/LGEE/PACE/PSCO/SRP。
- proper/donor：2024-01-25至2024-06-30。风险监督：源2024H2的完整地区日，另要求该月真实粗组成在2025-01-01前满足冻结月表的`period_end <= cutoff`且`available_at <= cutoff`。所有新风险家族共用完全相同的过滤队列，不为某方法补回未发布月份。
- 选型：源2025H1地理外推；目标复测：2026H1。已有目标与源选期均是反复使用的开发数据。监督发布日期模拟与月表原始vintage认证不足继续披露。
- B使用既有合法月组成/总量、日历、区位及既有前推粗预报法。lag7是小时遥测延迟，不是月表发布延迟；月表遵循自身冻结的`period_end/available_at`（当前源表常见月末后30天）。W额外使用与V12相同的单点、固定提前量GFS预报代理、缺失掩码及时间段定义，不更换天气地点/变量/来源。
- 每配置128条完整24小时情景；seed17023；spread倍率`.75/1/1.25/1.5/2`。倍率只由源选定。先按既有合同裁剪到`[0,1000]`，以初次情景均值为中心放缩，再裁剪；主指标始终评价最终发布情景。
- 所有新风险模型共享相同合法V9谐波前推粗情景的摘要和支持元数据，不能调用当前真实月组成作查询特征。共25列：`mdraw-mpoint`的8燃料均值/8标准差，EF粗冲击的.05/.5/.95分位数及均值/标准差，released_months、prequential_months、horizon_months、release_age_days。列次序、计算函数和缺失规则由冻结代码/归档审计绑定，两个预算和全部新风险家族一致使用。不另搜索新增历史误差特征或读取新数据。

## 2. 主B精确身份和天气门控

主B `native_recompose`及`simple_anchor`逐元素复用并重新生成核对V12共同支持归档。旧V7–V13冻结代码/协议及V13失败输出保持原字节。新风险参考不宣称等于native。

固定天气端点为V12差分天气tau=.5：`q_W=a*simplex24(s+.5*(delta_query-delta_donor_OOF))`，其中`delta=pred_W-pred_B`。对查询日动作`g in {0,.5,1}`定义`q_g=(1-g)*q_native+g*q_W`。g为查询整日标量，所有情景/燃料使用相同g，不对donor另设gate。g=0/1直接返回相应端点，确保完整结构情景精确等于native/fixed天气归档。g=.5是投影后的燃料轨迹收缩，通常不等于直接tau=.25，也不是CIF路径平均。

燃料日幅度、unsupported flat prior、零燃料日与联合donor身份保持。CIF等权日水平、目标月组成及原逐小时燃料零值不作为守恒声明。记录燃料幅度误差、投影/零值新增质量、gate动作分布和裁剪变化。

门控教师来自proper H1严格地理折外。外层评估区与教师自身区不进入教师模型/donor/缺失填补；donor天气预测也地区折外。教师为回顾性跨地区训练例，非按日时间滚动预报。门控训练不使用H2小时误差或2025H1选择标签。

教师的`qbar_g`为相同128条合法抽样donor轨迹的平均，再计算`c_g=R(mforecast_point,qbar_g)`。三动作标签为`MSE(y,c_g)/(mean_g MSE(y,c_g)+25)`，单位归一化前MSE为`(g/kWh)^2`。这训练的是归一化确定性中心MSE，不等于总体原始MSE或概率评分，也不等于结构情景均值的MSE。

`gate_ridge`：proper特征标准化、Ridge alpha10，预测三动作损失；`gate_forest`：ExtraTrees128棵、leaf20、max_features=.7、seed17023。均使用等地区训练权重，取预测损失最小动作；并列按动作0、.5、1顺序。缺失填补/标准化仅在相应proper训练子集拟合。

## 3. 同时冻结的两轮网格

| 轮次 | 预算 | 模型名 | 主候选资格 |
| --- | --- | --- | --- |
| R1 | B | native_recompose、simple_anchor | 否；强锚和B参考 |
| R1 | W | fixed_donor、constant_half、gate_ridge、gate_forest | 是；四种天气策略 |
| R2 | B | native_recompose、simple_anchor | 否；精确强锚 |
| R2 | W | fixed_donor | 否；固定天气诊断 |
| R2 | W | selected_donor | 是；R1源选策略 |
| R2 | B/W各一 | global_total、hetero_total、location_scale、qrf_0、qrf_50、level_independent、level_rank、physical_independent、physical_rank | 仅W家族有资格 |

R1共6分支，R2共22分支；每分支全部5个spread倍率。R1记录`locked_policy=primary.model`；R2的`weather_policy`逐字继承该源选策略，不根据目标效果选择不同policy、动作或参数。R2所有W风险中心都使用此锁定策略。

源选择使用等地区宏覆盖`[.90,.95]`及最差地区覆盖`>=.85`。通过者按CRPS、budget、model、factor字典顺序择优。若无人通过，最小化`max(.90-macro,0)+max(macro-.95,0)+max(.85-worst,0)`，再按上述CRPS及字典顺序；状态保持failed。每分支、主候选、B最佳、native/simple及纯CRPS诊断均保留，诊断不替换主候选。source-best W标准参考只在global/hetero/location_scale/qrf_0/qrf_50源选记录之间用相同规则选，不读取目标分数。

## 4. 新风险家族与物理分解

所有新风险确定性中心使用同一128条合法加权抽样轨迹的`qbar`再进入CIF比值；这是Monte Carlo近似，不声称精确全donor库期望，也不声称等于结构情景点均值。B使用native q，W使用锁定策略q_g。相同预算的全部新风险家族使用同一qbar、组成点和结构基础中心`structural_center`。location_scale的实际方法中心另加完整24小时条件位置向量，保存`location_prediction`；其`center_level/shape`指标评价这个修正后的中心，其余家族中心仍为结构基础中心。

对校准日i，定义：

`R(m,q)_h = sum_f(EF_f*m_f*q_hf)/sum_f(m_f*q_hf)`；
`c_i=R(mforecast_i,qbar_i)`；`e_i=y_i-c_i`；
`fine_i=y_i-R(mtrue_month_i,qbar_i)`；`z_i=EF·(mtrue_month_i-mforecast_i)`。

逐日验证`e_i=fine_i+R(mtrue_i,qbar_i)-R(mforecast_i,qbar_i)`。真实月组成仅作已发布源监督响应；外层评估源区不得进入风险库、gate、跨地区尺度/位置/QRF拟合。局部合法滞后粗历史作为查询信息的权限不因此删除。

- global_total：确定性中心加等地区抽样完整e_i；不再额外叠加粗情景或donor方差。
- hetero_total：标准日水平L、中心化路径D双通道尺度学习；标准化/恢复均保留同日L/D配对，不假设独立。
- location_scale：ExtraTrees128预测完整24小时条件位置向量，先减去地区折外位置，再对所得误差学习配对L/D尺度。用于预测持出地区C的尺度，其训练地区D误差的位置预测同时排除C与D；完整最终尺度模型使用所有地区折外位置残差。不得通过其他地区的位置拟合间接引入C的H2标签，避免虚假窄尾。
- qrf_0/qrf_50：V12式多输出均值分裂、全训练叶地区先验加权整日误差CDF；p=0/.5的全局分布混合，非路径算术平均。128棵、leaf20、max_features=.7、bootstrap及固定seed；记录实际sklearn版本/权重语义，不继承CQR覆盖保证。
- level_*：使用`c_query+e_i+(EF·(mdraw-mquery_point)-z_i)`。该标量替换是参考近似，不把z称为精确日均CIF误差。
- physical_*：使用`R(mdraw,qbar_query)+fine_i`。粗情景只传播一次，不再加完整e_i或donor方差。

对level和physical每个查询都固定同128个粗情景、同128个等地区源误差日索引。independent为固定随机配对；rank只根据查询EF粗冲击与对应源z_i的秩置换这128个整日索引，固定seed打破并列。不得改变任一边际的粒子/索引多重集；不得逐小时各自选择误差。level/physical的匹配配对也共享索引和粗粒子，隔离非线性映射贡献。

## 5. 封存、分析和晋级

两轮全部源预测先封存再评分；两轮源选择完成后才能生成目标。两轮全部目标预测封存以后才进行任何目标小时评分。R2冻结文件绑定R1源选择、源seal、完成记录和policy字节。输入、代码、协议、环境、预测及评分闭包以SHA-256固定，未完成/不匹配/非有限/不等评分支持均失败关闭。输出目录必须全新，不覆盖旧实验。

主指标沿用V12同18项：CRPS、90%小时覆盖/宽度/IS、上下尾、整日路径覆盖、日水平与中心化形状MAE/CRPS/覆盖/宽度、确定性中心水平/形状误差、中心化路径energy score。所有方法逐区/宏平均并报，保留PACE/LGEE；完整倍率网格只诊断，不能目标调参。

配对差沿用V11/V12冻结的`take/paired/METRICS`合同，源和目标都用6个H1同步月份块、5000次、seed17023；区间仅为反复开发样本描述，不包含源选/跨轮开发不确定性。报告三动作gate频率与相对固定天气/半收缩、B/W同家族、physical-vs-level相同配对、rank-vs-independent同factor及各自源选factor、标准参考、跨轮source-fixed主候选的完整比较。

实用gate：R2源选W主候选通过源门槛，目标宏覆盖至少.90且最差至少.85；相对native、simple、source-best B及source-best W标准参考，CRPS至少降低5%、CRPS配对区间上界<0、宽度不增加且IS降低。主候选若本身为同一个W标准参考，不作自身5%比较，该项标为相同基准并要求其余非自身对照完整；不得换成目标次优参考。

机制gate：只从W physical_independent/rank中按源门槛及同回退规则选择物理候选；要求相同覆盖门槛，以及相对上述锚/强标准参考和与其配对方式相同的W level参考满足相同CRPS/区间/宽度/IS要求。另报固定相同factor的physical rank减independent，未有匹配增量不得声称秩耦合贡献。机制候选不能由目标结果挑选。

上述gate分开记录；任何缺项都失败关闭。当前组件为标准方法，不自动给出原创机制结论；`custom_innovation_gate=false`、`new_holdout_triggered=false`。无论开发分数如何，均不据本轮自动开放新地区；需要清楚的性能证据、随后另行冻结的天气代理/地点/提前量/缺失与独立确认合同。

## 6. 预定解释边界

门控退回native不等于天气有可识别增量；物理传播变宽而覆盖提高不等于效率改善；日水平覆盖不等于小时或整日覆盖。源月粗误差会跨日重复，报告有效地区月数；季节H2→H1及误差支持问题不因新分解自动解决。PACE/LGEE只能用于预定诊断，不训练专用规则。所有事后分析须明确标记，不回填为预注册成功机制。
