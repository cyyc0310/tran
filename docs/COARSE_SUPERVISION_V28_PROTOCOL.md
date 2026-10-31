# V28：固定算法与小时库，仅改变源粗统计监督时间窗

2026-10-26，Sol。运行前合同，七已查看源2025H1开发，未查看地区不开。用户确认有完整电源类别月发电量；本轮仍采用火电内部细分未确认的MG保守代理（POS＋核水风光），不自行加入本地年度细分。当前任务是检验较长期、较近期源粗监督能否改善隐藏燃料组成预测，**不是新算法或已认证中国类别／发布档案**。

上一轮普通ET／折外残差部分修复MISO，但全部35配置不合格，约99%的宏CRPS改善来自MISO。现有比例head只用2024H1 bank的36个源月；学习源火电内部比例不一定需要同步小时q。V28改变源比例监督窗，而小时库、残差池、原月预报、供体IDs、q和随机数不变，先把信息利用与原创机制分开。

## 父证据、部署时点与合法输入

父[当前V27证据](evidence/group-regression-v27-r1-verification.json) SHA `25f0df9f032d1d0d8ecd1169be6ed2ebc06700bf5e5ec26f3d3ce19de8f4b18e`，其完整闭包与V25原月预报／R1银行、V27实际审计均绑定。V27 R1只是报告措辞澄清，没有producer或数值重跑；不把正式freeze的275inputs回填成R1清单的276。

新增粗监督只用父链已有七源月表，**固定部署截止2025-01-01T00:00Z**，适用全部2025H1起点。每个outer在任何燃料数值转换前排除该区的所有月行；每个LOSO fold再从所有历史中排除teacher区。按period_end和available_at均不晚于截止、固定窗口起止、合法月期间筛字符串后才转八燃料数值。未来／未发布／outer隐藏数值不得进入fit，时点不能以query新月份滚动扩展source标签。source八类作为源监督允许，查询仍只有MG粒子。

三个预定同粗表定义、嵌套窗口：

| 窗口 | 期间 | 对比解释 |
| --- | --- | --- |
| S24 | 2024-01至2024-06，period_end≤2024-07-01 | 与V27 bank标签口径相邻但不保证相同，不能称精确旧基线 |
| L24 | 2022-01至2024-06，period_end≤2024-07-01 | 相对S24加入较早源粗监督 |
| LR24 | 2022-01至2024-12，period_end≤2025-01-01且已发布 | 相对L24加入较近期源粗监督，旧代理通常到2024-11 |

同一窗口内每region-month唯一，重复拒绝，日期／发布时点和simplex有效性检查。按实际完整行使用，不补月，不造年度。POS=0没有r标签，排除监督并记录；所有训练域与缺失月保存。三个窗口必须按行键嵌套。共同截止保证六月标签可以在部署时已知，但不声称是2024-07-01可得或2024逐日hindcast。月表是修订EIA及模拟发布代理，不认证真实历史vintage。

## 固定学习流程：粗监督与小时残差池分开

X只五维组成，y为POS内部煤气油other四比例。每window／outer独立训练两模型，参数与V27完全相同：Ridge(alpha=1,fit_intercept=True)；ExtraTreesRegressor(n_estimators=128,min_samples_leaf=2,max_features=1.0,random_state=17023,n_jobs=1)。不加新特征、天气、年表或新的调参网格。

每个final和LOSO训练权重依自己的正POS标签队列为`N/(R*n_region)`，地区等质量、区内月等质量、均值1。固定Ridge alpha下相对平均loss正则强度随N变化，固定ET leaf2的有效平滑也随样本数变化；因此这是固定ordinary训练流程的数据窗口消融，不是纯信息量定理或完全固定复杂度比较。

final训练outer之外六源的粗历史。每个teacher的LOSO模型排除outer和teacher全部粗月，不可只排teacher2024H1。用于小时传播的残差池**仍只有旧proper bank的36个source-month**：

`delta_bank_j = r_bank_j − raw_fit5(X_bank_j)`。

这里r_bank及X_bank来自已有proper银行的eligible-day月组成，不替换成完整月粗表标签。LOSO残差是在部署前资料上做的回顾性地区交叉拟合；LR24的其它源较晚粗标签可以参与fit，但全部不晚于部署截止，不能称teacher在2024H1当日可取得的预报误差。更多粗月只用于head训练，不能增加q／残差抽样池或让月份重复权重流入抽样。source coarse标签与bank标签定义稍有差异；S24与旧V27对比包含口径变化，只有同粗表窗口间对比能隔离加入旧／近期标签的效果。

查询头取`raw_fit6(MGdraw)`；插件r=`ProjSimplex4(raw_fit6)`；残差r=`ProjSimplex4(raw_fit6+delta_effectiveBankMonth)`。raw模型输出不先投影，和V27同定义。源真组成条件拟合、查询粗预测粒子积分依赖Q(g|history)及P(r|g)迁移，不能说粗预报误差或目标条件分布已自动校准。

## 21臂与精确背景

前三臂分别为原MG、原V27 Ridge残差重组、原V27 ET残差重组；raw/fine/IDs等与各自已封存归档exact一致，仅作为背景，不重新挑选模型或因新结果改旧评分。

其余每窗口×每模型三臂：plugin_recompose、residual_anchor、residual_recompose，共18。命名例如`S24-Ridge-plugin_recompose`。每个残差anchor／ratio对fine_draw及q完全相同，原加性公式和比值映射不改；不新增插件anchor以扩大搜索。

POS有fallback时，delta所属月、q及support均随有效j；非POS q/support仍原i。query POS=0不误失败。缺源fuel时q=1仍是显式平坦先验，逐粒子计unsupported POS能量，不能称有源形状监督。MG输入组draw和非POS份额原样保持；分配后的组和仅容正常浮点误差，旧MG重建raw/fine及两个V27背景仍须exact，失败不seal、没有tol降级。

保存所有final／LOSO训练key／权重／日期／预测／残差、bank月映射、投影前后、query支持距离和组内EF越界。bank EF范围与长粗head训练EF范围分开；标量EF越界不当完整四维凸包或完整CIF路径支持测试，最近训练行距离不当凸包距离。新增coarse监督月份不增加独立小时残差标签数。

## 评分、门槛与归因

181起点×128粒子×24小时，21臂×原五倍率0.75/1/1.25/1.5/2=105曲线。先全部七区预测封存，再读七已查看源2025H1小时响应评分；无新proper小时解析／新目标／未见区。实际共同1145日，预计120,225日／735地区／105宏行。初clip→point→spread→clip、CRPS和90%经验区间端点及其它全部指标与V27相同。

按每区日均再七区均，宏覆盖90–95%、worst≥85%；失败fallback保持false，不扩大倍率。固定factor匹配与各臂源规则选择分开。所有窗口／模型结果完整报告，不按结果删负面区或只挑宏覆盖。

预定匹配对比共28组，每组七区×五factor=35行，合计980：

- 每window×model：残差重组−残差锚、残差重组−插件重组，共12组。
- 每model×三variant：L24−S24、LR24−L24，共12组，分别代表较早与较近期监督的加入。
- S24 Ridge残差重组−原V27 Ridge残差重组，S24 ET残差重组−原V27 ET残差重组，原V27两重组各−MG，共4组。前两个不称纯时间长度效应。

新增源标签或数据更新带来的收益首先归信息利用，不自动成为算法创新；旧源已反复开发，不能用source赢家替代独立确认。若维持覆盖的效率仍无增量，不放松门槛，不把source支持扩展当普适成功。若ordinary长粗监督已解释全部增量，应将其保留为强基线；专门机制必须再超越完整同预算条件概率控制，达到既有≥5%CRPS下降、宽不增、IS下降且覆盖合格，才另立冻结确认。没有新增本地细信息，也不宣称完整月绝对量已充分利用。

三份预审后才能prepare，三阶段独占写、不覆盖失败。独立actual审计须不导入V28 producer、重建每outer各window粗队列／排除／权重，重新拟合全部final和LOSO（全六源正POS时294模型），独立检查bank残差及query映射／路径／全部105评分曲线。原三背景可引用已审V25/V27及exact数组比较，不必重复旧98模型拟合；新增294不可仅接受stored delta。首尾字节哈希向量、失败历史与原收据保留，最终科学主文审定后才封清单。
