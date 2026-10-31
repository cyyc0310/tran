# V50：分离校准地区后的标准风险强基线

2026-10-27，Sol。父证据为V49清单8269659f1f9e5d4d6c06fca3a9b3b74c2299d9c6c3fd2242db5bdd0475335530。本轮先回答标准风险方法能否改善区间效率，不预设新的方法贡献。本文为实际读取外部细响应和拟合前的实验合同，结果另报。

## 固定数据流与预算

原七地区BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS逐一为outer=o。点预测器仅用其余原六源：H1小时库、2022Jan–2024Jun细组成监督、Jan2025模型快照及原六源中心训练记录。六源Scale的七个ET、Size-ET和SizeCap-Ridge每outer各拟合一次，同一个函数预测本outer的2025H1和所有外部校准地区，不能按校准地区重新拟合。

外部校准固定SWPP/AVA/BANC/JEA/SC/NWMT/SCL/WACM/SEC/SCEG十个旧开发地区。它们不进入本轮点函数或风险头训练，SCL不因POS监督稀少而删除。这些地区早已参加其他开发，其中九域细月监督曾进入V42；它们不是新独立确认集。这里只建立本轮算法内的训练/校准分离，不抹去开发历史。

风险头的proper资料只用V49 outer=o索引中其余六held的H1日（2024Jan25–Jun30），每个held的原点教师排o和held、只见五源。原七源CSV按H1时间窗口先筛再转换generation，键须精确匹配V49预锁H1支持；不读取该CSV的H2或2025细响应作风险拟合，不另造NPZ二进制前缀解析器。V49保存的预测与合法context逐键配对，不能通过反向任务把outer响应纳入。风险头的五源训练误差向六源部署仍有算子变化，本轮如实保留；cal与target使用同一个六源部署函数。

原七地区2025Jan1–Jun30共181个查询日，最终评分只用父V48固定的1145个有效地区日。它们已反复用于开发，不作新确认。PACE/LGEE、真正未查看地区和外十源2025细分仍关闭。

B0三个base为Scale-anchor、B0-SizeET-ReplaceR、B0-SizeET-AlignedKeep；B1另有SizeCapRidge-ReplaceR/AlignedKeep。B0只需要合法粗月度发电量和总发电量，query/predict可不提供容量而独立返回三个B0输出，必须与完整五base的前三项一致。B1另用按year−2规则已知的登记库存代理：2024外部查询用2022年，2025原源用2023年；不把nameplate当出力硬界，不用epsilon填缺失容量。中国完整同口径B1可得性仍未认证。

## 发布时间情景与逐小时覆盖对象

一次固定校准截止为2025Jan1 00:00 UTC，之后半年不加入新的细标签或滚动校准。小时响应的可得性只作明确假设：available=UTC日末+lag天，lag依次固定90（主情景）、30、7天。校准观测日在2024Jul1至Dec24的177日网格内，并须available<=Jan1；prepare只根据time/region/valid固定各情景、各源的完整24h有效键，每情景十源必须全部非空。没有成绩驱动的删源、删日或替换lag。

日末定义为次日00:00，因此90天情景最多到Oct2、30天到Dec1、7天到Dec24；实际有效支持由元数据决定。EIA常规次日发布日程不认证我们2026下载值的历史首发版本，以上不是历史vintage复现。月度粗输入仍按每个查询起点逐条核period_end与完整available_at，不因回溯而放开。

所有校准分数为逐小时分数，目标是逐小时边际覆盖的经验比较。另报24小时全部覆盖比例只是路径诊断，不使用日内max分数，也不授予同时覆盖保证。地区/时间交换性、源选择和历史发布版本均未认证，不能因使用conformal公式就自动声称新域覆盖保证。

## 点函数重建与固定特征

每outer以V29冻结的两个pair bank及query的四项bank provenance构建原有序六源bank，保持q内存布局与行映射；query原档只读bank_regions/bank_days/bank_row_ids/bank_cutoff，不加载其余target字段。Scale拟合复用旧V29七ET，查询使用新显式region/date适配器；不修改旧SOURCES、日期常量，不调用旧background受限入口或伪装地区。

中心头采用V30/V34原六源记录、权重、参数和train-only scaler，不再排第二源或改变质量。JSON数字token保留文本，先核身份与日期，再转换对应训练字段。七outer共63次点模型拟合（56ET+7Ridge）。

打开外十源响应前，必须对原七源全部181日做旧缓存重放：MG与日期/键、训练cohort/权重/scaler、donor/fallback/effective顺序、Scale均值/尺度/OOF与fine、两个中心以及五base raw/point。Scale-anchor取V47按名，另四个取V48仅75新臂中的相应名字，禁止套138总臂索引。MG浮点口径tol1e−12，其余公式/raw tol1e−9、rtol0；身份、键、索引与原字面缓存项目须exact。超过容差即保留失败，不用目标fine调整浮点或静默换流。

Point一律先clip各raw粒子至[0,1000]，再对128粒子平均。风险特征按预算固定：B0为Size原31列的粒子均值，加9个预测/小时特征，共40列；B1为SizeCap原36列均值加相同9项，共45列。附加项依序为point_hour、std_hour、point_day_mean/min/max、mc_day_mean_std、mc_centered_shape_rms、UTC hour sin/cos。std均采用ddof0；日均std为各粒子的24h均值在粒子间的std；形状RMS为先逐粒子去日均，再对粒子方差取小时平均并开方。proper/cal/target均从同一clip后ensemble计算，无truth-derived、source-ID、库存混入B0或B1预测偷渡。

## 六个固定标准风险方法

所有方法使用同一base的proper信息，不能读取外十源响应训练。proper各源总质量相同，源内日与小时等分，HGB样本权重为N_day/(6*n_source_days)逐小时重复，总权重24*N_day；风险mean OOF每fold重新平衡。

1. GlobalABS：初始上下界均为point，校准scale=1。
2. MCStd：初始上下界均为point，scale=max(postclip hourly std,1 gCO2/kWh)。
3. Component：proper误差分成同日level与零均值shape，分别除max(MC日均std,1)与max(MC形状RMS,1)。保留两分量的同日配对，用query尺度重组完整误差路径，按源/日等质量取无插值5%/95%分位数，叠加point为初始区间，校准scale=1。禁止把两个分量各自分位数相加，也不作Monte Carlo抽样近似。
4. HETScale：HGB拟合log1p(abs(truth−point))，query scale=max(expm1(clip(pred,0,log2001)),1)，中心保留point。
5. HETLocationScale：风险mean head按六个proper源逐一排除，预测该源的signed error并clip到[−1000,1000]；以这些risk-head OOF偏差定义log1p(abs(error−bias))尺度标签。另拟合全proper mean与scale两个头，query中心为point+clipped bias，scale同上。共6fold+2=8 HGB。这里只排风险mean head自己的源响应，没有重建第三源排除的base教师，不能称严格source-held验证或据此报告验证收益。
6. CQRplus：两个HGB分别拟合signed error的5%/95%条件分位数，先对交叉的两值逐元素排序并保留crossing诊断，加point构初始界，scale=1。非负校准分数只允许外扩，明确是CQR+变体，不冒称原始有符号CQR收缩规则。

HGB全部固定max_iter128/max_leaf_nodes15/min_samples_leaf40/l2_regularization1/learning_rate0.1/max_features1.0/early_stopping=False/random_state17023。普通scale/mean/logscale用squared_error，CQR用quantile=.05/.95；模型完整get_params与训练键、权重、目标和自生成模型文件必须保留。每base的HGB数为1+8+2=11，35个outer/base共385 HGB；加63点模型共448 sklearn fits。合成测试可用明确标记的private快速参数，正式入口不得覆盖默认或把测试拟合当正式128轮。

## 两个校准规则与第二次封存

统一逐小时score=max((lo0−y)/s,(y−hi0)/s,0)。每source等质量、源内日/24小时等分，分位数使用无插值经验CDF并明确端点。

- equal_region：有限等源经验分布的90%分位数，不添加iid独立小时数的虚假保证。
- HCP：保留1/(K+1)的正无穷质量，有限分布权重为K/(K+1)。K=10时取有限等源分布99%分位数；若K不足使所需分位数>1，则q=∞并保留该标记。不能因最终裁切[0,1000]就隐藏无限阈值。

最终界为clip(lo0−q*s,0,1000)、clip(hi0+q*s,0,1000)，q=∞显式输出[0,1000]，避免∞乘法产生NaN。所有lag/rule/base/method的q与原七源全部181日最终界必须再次完整seal，之后才允许原七源2025 truth评分。校准后不能回补外十源训练、重新拟合、选择规则或依据目标成绩改参数。

原五个固定spread=.75/1/1.25/1.5/2另列未校准参照：沿旧native_center规则先clip raw，按其point均值扩散，再clip；取排序第7/122粒子作界。所有五factor全部保留，不挑最优点当预先校准结果。新方法只定义区间，不给它们补造CRPS或联合概率路径。

## 阶段、核验与成功条件

prepare→predict→calibrate→score每阶段一次，exclusive results/disjoint-calibration-v50，失败保留marker/log，不因观察超时重启。九个producer/test/protocol文件、实际落盘code/math整体预审、父manifest及本轮直接输入先固定。只解析父1370 metadata并核当前直接依赖，不重遍历旧全数据。

predict先完成63点fit和385风险fit、70外部与7目标完整预测、原181日缓存重放及训练审计；prediction_seal之后才进入calibrate外十源细响应。calibration_seal保护全部q/最终界；score只转换原七源2025的预锁1145日，失败不得按成绩删日。新增逐日区间指标206100行，地区1260、宏180、去一地区1260；原始参照另28625日、地区175、宏25，支持1267行。逐日覆盖/宽度/IS/上下尾/路径诊断全部保留，汇总先地区内平均再七源等权，不能按地区日数池化。

代码与合成测试、全量数据流/身份/预算/标签窗、真实缓存parity、校准阈值与最终界和逐日评分均需复核。核验可以从已保存模型和训练审计重放，不冒称每个448模型都再次独立重拟合；注明复用库、已有缓存和各核验范围。若某阶段失败，先定位实现或定义问题，不以失败数据调整算法选参。

本轮没有预设“专门机制已胜出”。若标准方法已解释收益，论文应把已有改进归给相应基线，继续评估统一信息预算基准和信息边界；若仍有稳定缺口，再提出具体可检验的新机制。平均覆盖提高但只靠加宽、少数地区极端漏覆盖、B1收益冒充B0，均不足以冻结。新方法、真正未查看地区确认与投稿贡献仍需后续证据，长期目标保持active。
