# V190：源事件层影响确实改变CIF law，但未提供强基线之外的效率增量

2026-10-29，Sol。有效r2正式和全量审计均terminal exit0，7900条CIF评分完成；336个逐参考开发比较中40项通过，但**H2全部强参考联合gate为0**。候选可达到覆盖双门槛，却明显更宽、未胜强直接/简单锚/异方差/普通source-spread控制，不能称原创突破或晋级新地区；目标active。

## 实验与原始目标的关系

本轮没有重做零事件分类，而是用同一合法source activity与森林kernel构造完整CIF概率law。对16个源层（每fuel的源当月M>0且小时zero/nonzero），保存所选层总概率，仅把层内源贡献向等源质量收缩；CIF-W1影响选择器与只看集中度选择器、固定oil层、全局源均衡、固定kernel删源平均、ESS/source-spread锚混合和朴素宽分布比较。强参考为同合同direct20/84、锚残差law及proper地区OOF HGB异方差，未用原全年2024方法冒充July前向基线。

这个层是proper source atom的元数据，不是目标未知当前月事件标签或真实未来事件概率。候选不保证其他重叠层质量/能量矩也不变。W1混合恒等式是基本概率性质，类别重权、group DRO和Wasserstein条件稳健已有工作；本次最多检验特定CIF风险影响组合是否有额外价值，不能因换一个名字宣称首创。[文献核查](V190_SOURCE_EVENT_CIF_LITERATURE.md)列原作及V119/V140/V159–175的既有实现边界；相关原作包括[类别条件Wasserstein DRO](https://arxiv.org/abs/2109.03676)、[ECTR](https://arxiv.org/abs/2601.22944)和[Robust Validation](https://arxiv.org/abs/2008.04267)。没有独立复现这些论文的全部代码，也没有证明文献不存在相同组合。

## 固定信息与校准合同

模型cutoff为2024-07-01。只原V188 proper小时子集，实际Apr–May，六折17262/16417/17630/17649/18542/19368小时，不补样；2ET+7HGB=9风险模型/折，正式54个。monthly_context内部既有harmonic Ridge辅助计算的输出未用于本实验，不混入风险模型数。

校准仅原cal地区Apr/May，共55月；六折10/10/10/10/8/7月、5/5/5/5/4/4可用地区，当前月synthetic release及整月有效小时UTC逐一≤July cutoff核验。proper/cal/test地区分离。July模型回算早期cal时可用晚于cal origin的其他区proper标签，因此是**July固定时点的跨区回顾性校准，不是cal逐月实时前向**，cal误差与H2不具已证明交换性；数万小时也不是数万独立校准源。[分层数据推断文献](https://arxiv.org/abs/2306.06342)同样要求相应结构性假设，本轮未声称满足其新域保证。

H2固定原158月/27地区，Q3 78月/26区、Q4 80月/27区；不是all29，也不是新未触碰集合。先对该折全部cal/test query构造law并seal x/w哈希，再读cal真值求半径，test重建核同seal后评分。无目标当前月M/小时真值进入构造，12全law未来M/raw=None检查通过。55cal+158test=213查询；16law×48clock=163584哈希。同一query在半径策略中重复，不当新增独立样本。

每法自身pooled/max-region半径；9修改法还报告共同physical84_bg固定半径，共50记录/test、7900条。uniform_range以1025中点Uniform近似计CRPS、[0,1000]全支持区间计覆盖，**不是该law的90%等尾区间**；两个cal标签重复只是表结构。

## H2完整主表

以下27地区等权，地区内按有效小时加权，自身recalibrated/pooled。单位为固定EF代理CIF对应的g/kWh，完整16法均列出。

|方法|CRPS|宏覆盖%|最差覆盖%|宽度|Winkler|
|---|---:|---:|---:|---:|---:|
|anchor_calendar|39.646|88.36|51.85|245.282|372.088|
|direct20|40.871|84.98|37.43|197.414|360.233|
|direct84|39.107|88.61|50.14|212.659|331.548|
|hetero84|42.981|86.08|49.22|232.801|373.085|
|physical84_leaf|45.533|95.46|72.85|385.169|417.294|
|physical84_bg|45.554|96.47|80.65|398.973|421.087|
|uniform_range|163.169|100.00|100.00|1000.000|1000.000|
|source_uniform|60.614|98.72|85.35|497.098|504.538|
|source_delete_mix|45.725|96.53|81.41|399.940|421.753|
|event_mass_full|45.396|98.19|86.75|425.030|435.266|
|event_risk_half|45.012|97.80|87.00|417.121|429.678|
|event_risk_full|44.936|98.11|87.16|435.453|446.415|
|oil_nonzero_full|44.966|96.78|82.28|403.996|422.888|
|uniform_mix10|47.731|99.31|90.49|505.466|509.156|
|ess_mix_anchor|42.402|96.71|78.62|388.014|409.568|
|source_spread_mix_anchor|38.860|95.43|66.09|350.703|385.942|

full候选宏98.1109%/最差87.1603%，half为97.8012%/87.0018%，均过H2覆盖双门槛。但full对physical84_bg CRPS仅改善1.357%，宽增加9.143%、Winkler增加6.015%；没有覆盖保持的效率增量。它对direct84 CRPS反差14.905%，宽+104.765%。强direct自身覆盖不足，并不能使候选的更宽区间自动成为突破。

同一CIF law的fixed pooled结果：full宏98.1168%/最差87.1603%、宽437.913；自身半径宽435.453。故本轮变化确实涉及law分位，不只是另校一个半径；但这个事实不等于有效机制或唯一归因。full max-region自身宽459.841，未修复效率不足。

Q3 full自身pooled最差89.4475%、宽444.040；Q4最差80.2536%、宽425.659。H2合并最差87.1603%不能冒充各季度稳定。季度地区/月构成不同，不能只凭点值变化断言机制漂移。

## 预定强比较及有限正例

full自身pooled的地区配对CRPS差（新−参考）：

|参考|差值|地区描述范围|判断|
|---|---:|---|---|
|physical84_bg|−0.6184|[−2.3982,1.0663]|范围跨0，且变宽|
|event_mass_full|−0.4595|[−1.6715,0.7406]|CIF影响选层未给出稳定额外收益|
|direct84|+5.8289|[−0.6229,12.4611]|点值更差、范围跨0|
|ess_mix_anchor|+2.5336|[0.3529,5.0066]|普通ESS控制更好|
|source_spread_mix_anchor|+6.0761|[2.4975,9.9536]|普通spread控制更好|
|uniform_mix10|−2.7945|[−5.0856,−0.8293]|仅对朴素保守法有局部效率收益|

必须保留有限正例：对uniform_mix10，full CRPS改善5.855%、宽缩13.851%、Winkler降12.323%，该单参考gate通过；对很弱的source_uniform和uniform_range也通过。但H2的24个单项通过仅来自这三参照，其他period再有16项，共40/336。每候选/策略/校准需同时胜14个预锁适用参照，H2八个组合全部联合gate失败；不能把逐项通过、相关重复或事后最优方法冒充完整门槛。

全部bootstrap为10000地区配对开发描述，未作多重修正，也未整合训练源抽样不确定性。27地区与已查看2024数据不能升级为独立确认。

## 支持不足仍未识别

六折“源当前月核电M>0且小时核电=0”的层始终absent；这不等于模型完全没有核电零atom（源月本身无核电的世界仍可能提供零atom），也不等于真实未来核电零风险为0。候选没有给缺失层伪造概率。单源层不作均衡、缺类保持None/status规则保留；本轮数据中其他已观测层至少有两个支持源，不把源数当独立有效样本保证。

event/source重权使用同physical84_bg atom支持；full在全部158个test月的support-below/above与基线逐项相同。它只能改变已有CIF概率质量，不能创造未观测类别/新CIF支持，也不能由源事件频次识别目标未来事件质量或条件shift半径。W1分数衡量候选改变CDF的幅度，不是预测收益方向或概率置信上界。

因此负结果仅反驳本次“层内等源稳定化+最大CIF影响选层”的效率假说，不证明所有源支持信息无价值。也不能唯一归因油电或CISO：更新可能改变其他重叠层质量/均值，law没有额外矩保护。

## 完整审计与保留的失败事实

r2审计重拟合54风险模型，重建全部213查询/163584 law哈希与同数量事件层诊断、55cal及全部7900分数。每个真实cell独立验证层内权重/源均衡/固定kernel删源平均，选层按独立目标最大值1e−9容差核验；不是独立bit级重演argmax。源索引/UTC/current-M release、角色、source short/long prefix、cal完整月时效、activity/残差标签与proper OOF均核验。共享冻结特征及tree-growing，不能称第二套完全独立学习算法。

真实CRPS全部以CDF平方积分反导式核对，最大评分差5.57e−11；独立概率权重差1.33e−14。另12首尾query×两clock×8物理法×11合成y=2112次直接CDF积分，最大差2.45e−10。三个root表与producer逐fold CSV序列化绑定，336强比较与all-strong gate完成；没有用toy替代真实分数核查。

r1 smoke和六折无评分preflight成功，但HGB原始结构化nodes字节指纹跨进程不一致。首次诊断的padding猜测未成立；随后三次fresh fold0精确定位45处num_threshold的+0/−0、padding差0，其他字段相同。7 HGB在完整17262源行与34query×48状态上预测strict bytes一致，45节点数组对/28预测数组对已保存并再次核验。原r1 smoke/preflight模型本体未保存，不能冒称直接比对了那两个原对象。

r1 formal因复合shell未中止而短暂启动，但由入口指纹保护拒绝exit1，仅manifest、无完成fold或评分。r2仅在指纹中把num_threshold数值为0的符号位规范化，模型不改；其他阈值/字段不放宽。r2每fold两freshfit，9模型完整source/全部query预测严格bytes相同，预检108=54selected+54repeat风险拟合，无held评分。正式仍54模型；旧源码/log/归档保留，详见[阈值修复](V190_THRESHOLD_FINGERPRINT_ADDENDUM.md)。

首个r2 audit在fold0末评分表比较因tolerant_coverage列顺序不同exit1，无成功summary；保留源码/log后，仅把独立评分dict该列移到与producer相同位置，全部字段/公式/键/容差不变，toy全字段顺序与数值核验后完整重跑。成功audit输出仍audit-r2、实际重跑log为audit-r3；不掩盖这次失败。

成功句柄：r2预检8024、正式33117、完整审计61043，均terminal exit0；失败formal74327、首次诊断29812、首audit67519均exit1记录保留。有效归档全部hash和四新增脚本AST通过；环境无ruff/mypy/pylint/black，不宣称运行。`results/v190-final-validation-r1`核r2预检201输入/2输出、正式204/70、audit276/3，另以原source快照映射核验退役smoke/preflight/diagnostic和失败manifest；不称当前runner可重现成功r1正式。

## 决策

已经把V189支持集中度问题接到真实CIF law和覆盖/宽度/CRPS，而非孤立零分类诊断。结果是：支持层内重权能改变law并提高覆盖，但效率与强参照不足，新增CIF几何选择未显示稳定价值；不继续在本支线扫描alpha、类别、阈值或挑地区，不能将普通class reweighting改名为创新。

当前没有原创机制成立或原目标完成的证据。任何后续源支持不确定集都必须说明目标shift预算/新事件概率从何合法识别，不能仅从source计数推出；若需新信息，先核真实可得性及发布版本。没有该后续实现/评分，不打开真正未查看地区或新增2025评价，原完整gate保持，目标active。

有效归档：`results/v190-source-event-cif-preflight-r2`、`results/v190-source-event-cif-r2`、`results/v190-source-event-cif-audit-r2`。协议：[V190合同](V190_SOURCE_EVENT_CIF_PROTOCOL.md)。共享已查看2024、合成发布、非负生产/固定EF proxy及未核验中国实际月报历史版本限制全部保留。
