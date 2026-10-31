# V46：六个月初的小时水平—风险桥接对照

2026-10-27，Sol，新小时预测/评分前固定。父V44清单`5d91faa3d8a25a94f8b6100bd1bd7fc462edb2c52254216b336e70f596f2f8fd`。本轮只用旧七个反复开发地区、2025-01至06六个月初的原128粒子，零新拟合。此前V44/V45的月度结果不是小时覆盖证据；当前实验是稀疏月初日的开发性桥接，不是独立确认或完整半年评估。

## 固定输入、配对与支持

小时底座为已封V29 Scale-anchor未clip raw(181,128,24)、对应fine_draws(181,128,8)、query_mean(181,128,4)和MG_draws；按日期精确取六个月初。V42/V43/V44月度预测的origins/keys和MG必须与该切片逐值一致，不把月初第s粒子配给其它日的新seed。不重载/扩充hourly shape bank，不重新拟合任何旧模型。

预测仍输出42个地区日期。评分只用旧V29 day_metrics的region/origin/arm/factor可用性metadata预锁的32对：Jan1七区全缺；ISNE只Feb1/May1；其余六区Feb至Jun各5日。metadata每对原有10arms×5factor，无重复。不能读取分数选日期。评分seal后逐个精确UTC日[start-date,next-date)调用原source_days；只转换该32日有效24小时的generation_*。若与旧支持不符则失败核查，不静默改cohort。原其它日期、PACE/LGEE、真正未查看地区和新增供体2025fine均不转换数值。

## 九十九条固定路径

先定义45个新标量月度level S(6,128)：V42原Legacy/Expanded六头共12、Capacity-OP、旧CoalOilUpper、V43固定Mix-Router-DirectEFET-w0.25，以及V44全部30gate。全部使用原CIF_draws；Base Coil用原二进制规则重建。无新r4，scalar EF不识别完整四燃料比例，也不声明月总量约束。

设A=原Scale-anchor raw，U=对应fine_draws dot EF8，P=MG_POS*(原query_mean4 dot EF_POS)。固定两个模式，对所有45个S：Replace=A−U[...,None]+S[...,None]；Keep=A−P[...,None]+S[...,None]。Keep−Replace=U−P，保留原水平残差和同donor/粒子形状的配对，未另抽样。U−P粒子均值未必0，所以两模式不仅差方差，必须记录每query mean/std(U−P)及mean_hour(A)−U。

再对三个预锁强S（Coil、固定Mix、V44 SizeCapET/CIF/HistoryForecast-ET）各加两个普通控制：CenterOnly=A+mean_particle(S−P)，与Keep在clip前有同一ensemble mean整条曲线但保留原散布；FlatKeep=该S的Keep路径沿24小时取每粒子均值并重复，在raw、clip前保留日平均粒子分布但移除小时shape。它们用来解释中心/散布/形状来源，不算创新。

加原Scale-anchor和Scale-recompose两B0路径、V25 M8-recompose_conditional高信息参考，共90桥接+6普通控制+3原参考=99arms。M8只作高信息预算参照，不是B0或oracle。桥接四个Size头（Legacy/Expanded Size-Ridge/ET）属B0，其余B1，V44 HistoryForecast gate整个预测仍B1；source pool扩大不等于方法创新。中国完整B1库存仍未验证。

raw加常数保持每条粒子的小时差分/居中形状，不保证跨粒子分位数形状，也不保证日均等于月度S。clip后单轨迹形状也可改变。additive桥接不等于燃料物理重组，任何非零POS/零POS粒子不一致仅报告，不临时修补。原水平残差配对不等于新中心风险已校准。

## 固定评分与判断

沿原scaled_paths：先clip[0,1000]，粒子均值得point，再以point为中心应用factor并clip；保留(.75,1,1.25,1.5,2)全曲线，primary factor1。不改变MC顺序、分位数定义或根据truth拟合scale。90%端点是排序第7/122值，不改成线性分位数。

原八指标全部保留：CRPS、逐小时coverage90/width90/interval-score90、上下尾漏率、point level/centered-shape MAE；另报原score_arrays的24小时同时覆盖作诊断，不能把逐小时90%当联合90%。32日的768小时不是独立样本。先地区内平均再七地区等权，保留逐地区及去单地区敏感性。

99×5×32=15840 day，3465 region，495 macro，3465去地区。两个原Scale参考在32日/五factor的320条原八指标与clip值直接复制，重算须在1e−9内相符；其余15520条计算。新path同时覆盖只按原helper计算，不冒称旧CSV已有该列。原三个参考的预测raw逐值复制。

报告42×96=4032个新路径/query诊断：raw clipping、零POS粒子却非零/正小时路径计数、月度S与实际raw小时均值、继承level残差mean/std、单轨迹shape变化。自重建A−U+U和A−P+P的最大差≤1e−9。Keep与对应CenterOnly的raw ensemble mean差≤1e−9，FlatKeep保留每粒子24小时平均；这些是代数核对，不是能量/风险保证。

primary factor1分别报告：相对原Scale-anchor factor1的macro coverage不降且width不增，以及达到macro逐小时coverage≥.90、最差地区≥.85而macro width/CRPS/interval-score均不超过原Scale-anchor factor1的描述标记。显示完整五factor与原基线可达点，不靠事后挑factor说有新机制；不插值曲线，不以加宽单独提高覆盖称进步。B0/B1/M8预算分列，任何标记均不构成冻结或未知区资格。

## 执行与验证

固定五个新producer/测试/合同、两落盘预审、父清单和所需旧预测/可用性metadata、七源hourly文件及实际使用helper哈希。只解析父manifest元数据并校验本轮依赖，不重遍历旧1005数据。读取cached hourly预测数值只在freeze和prediction_started后；评分hourly真值只在完整prediction_seal、score_started之后。exclusive目录results/hourly-bridge-v46，prepare/predict/score各一次，失败保留，不因观察超时重启。

保存七NPZ/JSON（raw99×6×128×24与level45×6×128）及4032诊断，先全grid/shape/配对检查和seal再评分。所有新路径在32truth前生成；这次允许旧开发源指定32日的hourly评分，不是新地区或新fine训练。未来若扩展每日181origin，需另锁合法历史、meta输入和日期粒子耦合，不延用月初粒子索引。

独立临时公式重建全部桥接/普通控制/诊断、99×5评分及聚合、原320numeric exact，atol1e−9/rtol0；键/日期/MG和参考raw literal exact。无需新fit或大型审计框架。若point level改善却覆盖/效率不改善，保留否定结果，下一步分别定位未解释level、shape和函数量不足；不得由本轮标量桥接推断完整燃料或小时风险被识别。长期目标active。
