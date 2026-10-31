# V47：完整开发日期上的普通水平与小时风险对照

2026-10-27，Sol。父V46清单e0597281ae5ce839df6588b0f7a7bb2af78cd8bb49d99ccb84ad2b7ccd78257d。V46的32个开发日期发现普通水平控制已能改善覆盖，但不支持新门控创新。本轮扩展至原七个反复开发地区2025年1至6月每日UTC零点，不开PACE/LGEE、真正未知地区或新增供体2025细标签。

## 输入与日期

固定181个日期，7区共1267 query，原V29的每日MG、Scale-anchor raw/fine_draws/query_mean、Scale-recompose及V25 M8参考逐日期/粒子配对，禁止将月初粒子跨日套用。不重新拟合MG、小时shape bank、旧条件比例分布或任何风险尺度。

评分队列只按V29 day_metrics的region/origin/arm/factor metadata固定：BPAT176/CISO178/ERCO180/FPL179/ISNE89/MISO166/NYIS177，共1145个UTC日，旧每对10arms×5factor完整，无重复。1267query全预测，122缺日保留支持标记；不按新分数改队列。完整预测seal后才按七源原source_days在2025-01-01至2025-07-01窗口转换有效generation；函数先筛时间和有效24小时，随后转数值。实际支持若不等metadata则失败保留，不删日重试。既有V46的32日已查看，新增1113开发日不算独立确认。

## 每日特征与固定水平头

沿V31合法三月context：period_start<query、period_end<=query、available_at<=query。只将被选中历史月的四个零因子份额及total_mwh转数值，POS由补集；不读query当月真实细份额或未发布历史。每天的month_sin/cos仍以所属月份为离散相位，month_age仍整数月差；release_age按实际每日时差更新，最近三月允许随发布改变。保持原31列Size及36列SizeCap/DirectEF定义。新增daily入口只放宽日期为181个UTC日，不修改旧V31月初入口。原六个月初特征须与V34/V42逐值相同。

库存沿原query2023 OP代理：所有query在2025年，year−2选择不变。读取已封OP表相同year/region的八类容量，求四POS类别容量及内部q，核其与V34六个月初query_feature_SizeCap的q、pressure和EF锚一致。这是重放已有合法库存输入，不能作为物理出力硬上限。中国同口径完整B1尚未验证。

固定12头：Legacy/Expanded × Size,SizeCap,DirectEF × Ridge,ET。Legacy Size两头取V30审计，Legacy另四头取V34，Expanded六头取V42审计。保留原训练X/y/keys/regions/weights、scaler、projection和超参数；source fine只2022-01至2024-06、Jan2025快照，outer始终排除。不新选训练月或供体。Ridge按原coef/intercept重放，不fit；ET按同X/y/weights/参数确定性重建，每outer6次共42次，核importance及六个月初的原始预测、EF、投影量和CIF。ET重建不是新监督数据或新算法，但如实计fit42。每日仍调用月度监督头，月内release_age存在超出训练月初取值的可能，不称为已接受日监督或风险校准的模型。

加Capacity-OP、CoalOilUpper、固定Mix-Router-DirectEFET-w0.25三个水平，共15。q在半年内固定，Coil路由沿旧V43 old_weights[8]，先核二值及六个月初复制；Mix=.75Coil+.25Expanded-DirectEFET，先EF混合再乘同日MG。全部15水平与V42/43/V34六个月初CIF须在1e−9内一致；参考raw及MG literal exact。ET月初replay核projection而不是只核feature_importance；无重新选择参数或新gate。

## 完整匹配控制

A=原Scale-anchor raw，U=fine_draws dot EF8，P=MG_POS*(query_mean4 dot EF_POS)，每个S均为同日同粒子月度水平。全部15 S都输出Replace=A−U+S、Keep=A−P+S、CenterOnly=A+mean_particle(S−P)、FlatKeep=Keep每粒子日均重复24小时。再加原Scale-anchor/Scale-recompose/M8-reference，共63臂：B0 18、B1 44、M8 1。不纳入V44增益gate，当前问题是普通水平/风险与日期稳定性，不把删gate后普通基线的成绩称为方法创新。后续候选必须面对本轮完整强对照。

Keep−Replace=U−P含非零均值，不能称纯方差消融；CenterOnly与Keep只有raw ensemble mean曲线相同，clip后可不同。FlatKeep的日均分布只在raw/clip前保留。非Flat逐粒子居中形状保持，不等于分位数/clip后形状保持。原raw不clip，评分再沿固定helper；S不识别完整r4，无物理重组、月能量或新center风险校准保证。

## 评分、诊断与执行

沿原先clip→粒子均值point→factor→clip，固定(.75,1,1.25,1.5,2)，primary1；90%区间排序第7/122。原八指标、path simultaneous coverage和spread clipping全部保留。七地区内先平均，再地区等权，逐地区与去地区敏感性均报；日/小时不当独立域或独立样本。nominal/efficiency仅描述，相对原Scale-anchor factor1的macro宽度/CRPS/IS；不冻结或开未知域。

63×5×1145=360675日指标、2205地区、315宏、2205去地区、1267支持；60×1267=76020 raw诊断。两个原Scale参考1145×5×2=11450旧八指标+clipping按round_trip读取，重算<=1e−9后复制原numeric；同时覆盖另算。全部15月初levels、原参考raw/MG/日期、Center/Flat与形状恒等式逐一检查，tol1e−9/rtol0。

输出每源一个compressed NPZ（raw63×181×128×24、levels15×181×128与合法context provenance）及JSON审计，raw诊断76020。评分逐源载入以限制内存，但读取任何truth前已验证全部七archive的grid/hash。exclusive results/daily-bridge-v47，prepare/predict/score各一次，失败保留，不因观察超时重启。五producer/test/protocol、两实际落盘预审、当前直接依赖与父清单预锁；只解析父1047 metadata并校验本轮直接输入，不重遍历旧全部数据。正式数值读取在freeze/prediction_started后，truth在完整seal/score_started后。

独立临时公式验证本轮15水平与63路径；重建42 ET、Ridge系数重放、全部月初replay、path/diagnostics及评分/聚合和原11450numeric exact。不能只核极少日期便声称完整。若扩展后优势消失或仍被CenterOnly解释，记录否定结果；下一步转向可识别风险与剩余调度风险的分离，而非再挑水平模型/尺度。长期目标active。
