# V189：油电条件风险收益的源块删留稳定性

2026-10-29，Sol，在任何V189拟合/评分前冻结。动机来自已查看V188油电例外，明确是针对性开发压力测试，不是新颖算法或未触碰确认；不推及其他燃料/CIF。

## 固定数据与删除名单

完全保持V188的2024-07-01截点、六折proper/cal/test角色、合法原抽样小时子集、extended query prefix及158 H2查询（Q3 78/Q4 80、27/26/27地区）。实际源小时Apr–May；早年只有粗表。训练当前源M release≤cutoff，原索引/时效由V188全审计核验，本轮再核原样本绑定。query协变量合法滚动，监督不更新；已查看2024/回顾性质量和合成发布限制不变。

删除名单固定原source roster全部29地区，不能按油电事件数或收益筛选。对每个全局删除源d：每折若d在proper则删除该源全部原训练小时，不补样并重算剩余地区平衡权重；若d不在proper，沿用baseline概率。SPA等proper零小时块是预定零删除量placebo，必须精确保持概率。粗容器/查询协变量不删除，只删除供体小时监督，不能叫删除真实系统/物理容量或因果贡献。

六折proper分别19/19/19/19/20/20，baseline六次加116次proper删除=122拟合块；每块仅油电direct43/direct84/positive-month conditional43/84四森林，488森林，32树/leaf64/原seed。clock用同剩余正月小时48状态地区平衡均值，空状态按同剩余全部正月小时回退。source positive由源已发布当期M定义，不引入目标当期状态；每块/类别记录小时、源地区集合、加权事件质量。单类训练保留并标记不可识别新事件，不伪阳性；完全没有正月监督则明确标不可估计并终止，不事后删case。

## 对照与分母

全部30场景（baseline+29删除）均输出5个无条件部署法：direct43/84、persistence×clock/conditional43/conditional84。持平只最近合法月状态。3个正月条件头仅在实际current-positive时评估，不当部署筛选；input_fuel_positive仍为合法部署分层。没有学习新月头，不能称本轮改进全部月状态方法。

每场景同一剩余源小时预算，删除后不会保持baseline总训练量；这种变化是源块压力合同，不是增加数据或声称控制了样本量的纯因果实验。全部剩余源参与，不重新挑供体。122块中baseline必须精确复现V188全部5部署/3条件概率；非proper/零量删除精确复现baseline。每折先构造该折全部场景/查询预测再读该折真值。

30×158×5=23700小时记录、30×61×3=5490油电正月条件记录，总29190；61月12区及input-positive56月11区为冻结V188条件任务分母，不以实际正月选部署样本。无新小时或新地区。所有H2/Q3/Q4与all_available/input-positive按原小时权重再地区等权报告。

## 预定统计与证伪解释

部署8对：direct84/direct43；持平conditional43/84各对持平clock；持平84对持平43；持平43/84各对direct43/84。条件3对：43/clock、84/clock、84/43。30场景×11对×3period×2cohort=1980描述比较。另对29删除的两个conditional-clock优势相对baseline做地区配对变化：29×2×3×2=348项。bootstrap10000 seed189。所有结果完整保留，不以最优删除场景替代部署法。

报告每个删除源的事件/非事件小时和地区支持、baseline地区平衡有效质量、删前后预算及优势变化；事件集中度为描述，不称物理/因果归因。若一两个预定删除使整体收益消失，说明本合同对这些源敏感，不唯一证明机制由它们产生（训练量/权重/支持同时改变）。若所有删除仍有收益，只说明这项普通风险控制的有限稳定性，不是时间不变/跨域保证或CIF收益。原V188油电正月条件收益与部署direct差异并列报告。

## 预检与审计

无held评分预检重拟合全部122块488森林，六折首尾query对baseline与全部proper删除做完整未发布M/raw=None干预；baseline概率与V188精确、零量placebo精确。保存指纹及训练provenance。正式重建须指纹相同，完整hash begin/end、atomic fold提交/resume，保存场景概率和来源映射。

完整审计再次488森林，核每块剩余source索引/目标/权重/时效/类别、全部4740场景查询及29190评分，独立经验叶均值和Brier展开，root/macro/比较绑定；共享树生长/特征helper范围明确。目标保持active，未通过CIF覆盖/效率gate不得开放未查看地区。后续假说必须另协议，普通删源分析本身不是原创。

## 拟合前补充：affected集合（主结果不变）

全样本删源结果包含该源不在proper或无小时的折，其baseline复用会稀释敏感度。因此预锁affected分析：删除源d仅保留d属于该折proper且原fine小时数>0的折；按该固定训练可用性选取全折查询，绝不按表现/目标真值筛选。baseline在完全相同的query/地区/月集合重算原小时加权再地区等权。保存每源affected fold、query、地区、period覆盖及input-positive条件任务分母；SPA等无eligible折源明确不可估计，不以零差冒充稳定。

对每删源、H2/Q3/Q4、all_available/input-positive计算cond43/84各自对clock的优势，再计算删除减baseline的地区配对差。29×3×2×2=348个预定身份，若空集合则保存status/no-eligible、regions=0及None数字，不静默丢失；有效数由训练/定义任务覆盖决定。全样本主结果1980/348仍保留，affected并非另独立验证或因果分析。审计从原V188 source region/索引独立验证affected而非信任producer标记。
