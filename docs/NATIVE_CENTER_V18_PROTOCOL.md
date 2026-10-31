# V18：保留native结构的最小中心增量反驳

2026-10-25，Sol。七源BPAT/CISO/ERCO/FPL/ISNE/MISO/NYIS、2025H1原1,145个有效源地区日；没有target或未查看区域入口。旧V7–V17保持冻结。本轮只检验：同一合法本地packet下，V17全局direct预测是否丢失native已有的fine结构。不提出新机制、新算法或独立确认。先写完代码、合成测试、协议并通过根代理通用审查、独立Python审查和科学审查，之后才可正式prepare/predict/score。

## 固定信息与中心

沿用V17主lag7的D2=(日H,G)、D8=(八fuel日能量)、T8=(四个固定UTC6h块H,G)、G5=(日H,四块G)及其完整可推导信息。H=Σ(EF/1000)×fuelenergy，G为正值proxy发电总能量，不是signed netG。全部packet仍须day_end+7日才可用；最近可见日之前28个calendar days、raw能量EWMA尺度14日，缺失规则与V17一致。V17隔离聚合器已经读过源2025小时燃料进行包压缩；本轮不重新聚合、不宣称源信息从未被查看。V17 lag30完整封存结果仅为既有背景，本轮不新增lag30预测、不换lag调参。

预测输入限V17已封存native、packet、metadata、features和fitted centers。ET点**必须**从models索引对应的centers[ET_global]取得，逐元素核验ET_hetero/ET_QRF的点完全相同；不以包含2024Q4残差的raw情景均值替代该proper-only拟合点，不重新训练或读取fine训练标签。没有给direct ET重新加入native features，故本轮未完成该强控制。

令N[d,s,h]为V17原native的128条路径（原V14身份），n_h=mean_s N_s,h，c_h为V17 ET拟合点，δ=c−n。P_day将24小时δ取等小时均值并广播；P_block对四个固定UTC6h块分别取等小时均值广播；P_full为identity。三个普通ET中心投影均采用raw_new=N+αPδ，固定α∈{0,.5,1}，四包完全相同网格。α=0直接返回独立copy，raw与native及所有donor IDs逐元素相同。任意α仅整组平移128粒子；裁剪前粒子相对差异和noise保留，最大浮点误差须≤1e-9，不重新抽残差、不添加coarse error bootstrap、不修改风险模型。

额外普通raw_day：用V17共同raw能量EWMA日H/G锚a，δ_day=a−mean_h n_h，广播后同样三个α。全部四包所有query锚在1e-9内必须与D2相同。只有T8开放raw_block：四个预测EWMA块H/G比值减native四块等小时均值后广播。特征切片有显式V17 schema/shape，且全部feature、anchor通过PacketHistory逐query按合法时点重算并交叉核验；不静默使用硬编码列。加权块H/G广播是普通heuristic，不是等小时CIF真值或物理H/G守恒；其他包不伪造块H。无历史沿用V17共同500+missing indicator，保留预测日，不删日。

## 完整网格与诊断

总35个NPZ：7个native，以及7源×4包。D2/D8/G5各ET_day、ET_block、ET_full、raw_day四method×三α=12 arms；T8多raw_block，共15 arms。合计51个packet-method-alpha配置，加native共52；每个原五倍率(.75,1,1.25,1.5,2)，共260条源宏曲线。α=0重复控制保留，不删减赢家预算。各归档保留181个calendar forecast days，评分交集固定原1,145个有效源地区日。

每个arm保存全部raw路径、原native donor IDs、请求point、实际未clip均值、clip后的realized center、shift、ET point、day/block anchors、原native边界饱和率、新超出[0,1000]的路径比例、裁剪前noise身份误差。请求中心=理论n+αPδ，unclipped center=实际raw均值，realized center=clip(raw)均值。评分先clip raw至[0,1000]，其均值为point；再按原倍率围绕point扩散并clip。另保存每日未clip90%宽度、base clip后宽度、最终扩散裁边率。边界裁剪会改变实际中心和区间宽度，不能把裁边变窄称作新的uncertainty机制。

评分使用冻结V10八指标：CRPS、小时90%覆盖/宽度/IS、上下尾miss、point level/shape MAE；经验区间为ceil order-statistic（128样本下第7与122顺序统计量），不使用线性插值。地区内有效日等权、地区间等权；全倍率、全projection、全α和分地区结果都保留。

## 不变性与封存顺序

新代码闭包严格继承V17的54文件再加本轮4文件=58；prepare绑定全部code/input/environment/Python binary hash。V17 freeze、data_seal、prediction_seal、source_completed、reference region_metrics、所有registered prediction/audit和packet SHA逐项验证；source_completed必须绑定相同freeze/prediction及reference output。V16 core manifest SHA固定03b38cd84d39390fa02947ed31fcff6110942e8f9ab6afb78a8282c16cefedde，验证其source_completed、prediction_seal和region_metrics的manifest/output父子绑定。仅读取已注册源区输入，没有目标配置。

预测完成全部35归档，核完整注册路径/grid、每个method/α、origins、native/IDs身份，独立重建平移公式与所有中心/clip诊断，再写prediction_seal。只有之后score才可读取源2025小时fine truth与继承reference表的数值。score入口先完整验证seal/grid/全部hash，再读truth；计算结束再验证闭包、输入和起始freeze/prediction字节。排他创建，部分或完成尝试不覆盖。真实runtime故障保留attempt并按新wrapper/version处理，不改旧冻结文件、样本或门槛。

## 参考与源选择

主要同包比较是本轮所有普通projection与raw anchor对照，各自使用相同native uncertainty和相同packet。强背景必须完整列出：native五倍率；V16四种日组成forecaster的lag7 additive全曲线；同包V17 ET_global/ET_hetero/ET_QRF三种完整五倍率曲线。继承reference region_metrics须通过原manifest/completion哈希验证，不后验筛行或重新拟合。V16 additive使用其原日fuel预测和预序列粗残差法，与本轮确定性中心移动并非同一信息摘要/风险生成流程；V17 direct模型没有独享native输入，所以这些是明确披露的背景，不声称完成native-feature refit控制。native八指标评分须与封存V17在1e-9内一致。

每包在宏覆盖[90%,95%]且最差地区≥85%的配置中最小CRPS；全部失败按违反程度最小，再CRPS、packet/method/α/factor固定排序破平局。native单独选factor。T8 source_information_gate要求其覆盖通过，而且相对D2、D8、G5各自最优完整普通配置以及native各CRPS至少下降5%、width非增、IS下降；不能只战胜无新增信息的native就归因于T8。另完整报告同method、同α、同factor的T8−G5各地区八指标差值；raw_block没有合法G5 counterpart，因此不编造匹配。默认不作H因果结论。

custom_method、mechanism_breakthrough、target_test_authorized始终false。源选择是反复开发，没有独立确认；native-feature refit、同预算coarse-CIF TSFM及其预训练截止/2025污染审计、统一proper-source尺度量化/噪声、经度太阳相位、现实发布vintage与真正采集成本均未完成。本轮信息设计pilot不授权目标或未查看区，不把这些晋级前未做项目称为passed。
