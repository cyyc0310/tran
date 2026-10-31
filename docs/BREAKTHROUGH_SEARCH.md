# 持久研究目标：继续直到找到可验证的突破点

2026-10-25，Sol。用户要求继续，不把候选方向、oracle改善、测试通过或常规组件改名当作目标完成。本文件是可更新的研究状态，不替代各实验首次运行前冻结的协议。

2026-10-28 V139更正：旧V120的589区月含76个容量不相容月；未检查全月容量条件的端点捷径导致44条物理区间不包含代理风险端点。V120原性能表不再作为整体证据，V121/V122及V126/V127的源支持校准也需重审。代码已修正，全分母比较见[V139](V139_PHYSICAL_RESERVE_CALIBRATION_RESULTS.md)，影响范围见[更正说明](CAPACITY_CONTRACT_CORRECTION.md)。此前下文相关候选评价应结合此更正阅读。

## 完成判据

保留原任务：有限、合法已知信息下，目标地区没有已有小时CIF时的可信可迁移预测/统一基准；同时重视真实算法贡献。允许探索清楚标明的新增信息预算，禁止把能够反算小时CIF的燃料遥测伪装成零本地小时信息。原创性需正面比较相关原始论文；普通analog、IPF、残差bootstrap、门控或rank机制本身不够。

正式晋级沿用已约定的性能要求：源选冻结后，在已查看开发目标上相对简单锚、最强结构和同预算强条件模型形成可归因增量；90%宏小时覆盖≥90%、最差地区≥85%，CRPS至少下降5%、区间评分下降、宽度不增加，配对描述区间支持而不是只靠点估计。不能把这些开发区间称独立确认。正结果后再冻结独立确认合同，才打开真正未查看地区。最终论文创新与确认仍须审查，不能只因某次开发gate通过就结束目标。

## 已确立的证据

- V14：天气门控主要回退保护LGEE，未胜固定收缩；物理粗误差传播和rank耦合输给匹配水平基线。无晋级。
- V15：精确粗总量识别加权聚合量，而不识别小时风险；15预算×2EF后验诊断与624实际LP端点核查通过。总排放燃料+煤的近零范围依赖真实日油量及固定EF简化，删除未来日总量后目标小时界宽105.798。它是信息边界进展，不是预测突破。V15完整证据已封存。

## 当前V16源区筛查

| 支路 | 信息预算与假说 | 当前状态 |
| --- | --- | --- |
| A：daily context | 新增日八燃料能量，日末+7/30日模拟延迟；用local prequential daily composition law取代月law，区分日q归一化、普通forecast/particle analog、innovation碳轴/full8/打乱控制 | prepare/predict/score均exit0；63归档及实际独立审计PASS；417925评分行，1145有效地区日；source_pilot_gate=false |
| B：generation context | 保持旧monthly+独立netG预算；7/30日滞后、28/56日历史形状，native精确身份、row-only负对照、conditionalanalog、三种IPF和64/64混合 | predict/score均exit0；28归档及实际独立审计PASS，160300评分行；未提供覆盖与效率联合增量 |

A只有七源入口，63完整归档封存后评分；B只有七源入口，28完整归档封存后评分。proper donor均为2024Jan25–Jun30，held源区排除；源评价2025H1。所有V7–V15生产字节保持。两个pilot都尚缺完整同预算H2风险强参考，源筛通过也不代表正式突破或自动开放目标。

V16正式进程均已由工具确认exit0，无运行待轮询。A完成收据`results/daily-context-v16/source_completed.json`，prediction seal `6ce41e1d4cc880225baafbd5cfb6878ab189b97f49145412f845031ebd0448a9`；B收据`results/generation-context-v16/completed.json`，prediction seal `aa09a9f494cc6704c43d3026343fa95ec0f8e751c47b846e8a4a53f97e169af7`。源小时评分均在对应完整预测seal后进行；隔离日聚合器读取小时源数据的事实单列披露。

B完成负证据：固定lag7/window28/factor1.25，native CRPS26.694362、覆盖92.9545%、最差85.2273%、宽176.272555；普通analog_g CRPS26.358455（约降1.26%）、覆盖92.5495%、最差82.9072%、宽171.234167。同倍率rake_mean_prior CRPS26.822470、覆盖93.2375%、最差84.7775%、宽179.012506；其他rake/混合亦未形成优势。lag7全曲线同时满足宏覆盖[90%,95%]及最差≥85%的仅native与row_only。故历史总发电形状的这组标准实现未提供晋级依据；不能外推为所有总发电信息无价值。均为源开发结果，不能用于宣称已解决PACE/LGEE。

上述B数值现已由独立实际审计及科学/事实复核确认。A源选ordinary为lag7/EWMA/daily_native_ids/f1.25：CRPS25.447761、覆盖93.7091%、最差89.3727%、宽184.997357、IS206.237578；CRPS比native降4.67%，但宽+4.95%、IS+1.97%。候选innovation_full CRPS25.540229，未胜同预算最强ordinary或particle_analog25.524431。A gatefalse，不能把新增信息和日归一构造收益算成innovation机制突破。[完整结果](CONTEXT_V16_RESULTS.md)已获独立科学复核，doc SHA256 `25fee4afa8c51bfeca3d240e27437d439e5979e76052f33c1045dbd00a0ae8e0`。

保留有限正结果：[普通additive补充](CONTEXT_V16_PARETO_NOTE.md)固定EWMA/7日/1.25的CRPS25.685008、覆盖92.4288%、最差87.5%、宽170.875452、IS197.820632，即覆盖门槛合格时CRPS降3.78%、宽降3.06%、IS降2.19%。它未达5%且形状MAE仅降约0.042%，不是机制创新。不能为了强调未晋级而笼统声称所有简单法无效率改善。补充已独立科学核对。

V16核心证据[context-v16-verification.json](evidence/context-v16-verification.json)封存57代码、27输入、153产物。独立[manifest-review](../results/breakthrough-search-v16/manifest-review.md)重hash全部237项，核126条receipt/预测/输出链接及实际集合；manifest SHA256 `03b38cd84d39390fa02947ed31fcff6110942e8f9ab6afb78a8282c16cefedde`。A审计脚本为已执行inline原文事后提取，日志保留原tool输入输出及截断，不冒充文件方式执行。新补充、字段与数学notes在核心manifest之外，各有独立审查。

新增[风险对象数学澄清](COARSE_RISK_IDENTIFICATION.md)：区间均值µ与支持[0,B]约束的是能量加权超越占比p_E，`max(0,(µ−τ)/(B−τ))<=p_E<=min(1,µ/τ)`；它不是危险小时数或未来覆盖，有限固定权重时也非sharp。经典矩界本身非创新，已有重组守恒时再加同一约束冗余。独立数学复核PASS。

## 下一轮V17：固定维数发布包，不预设原创机制

根代理已授权实现而未授权跳过审查：D2（日H/G）、D8（八燃料日量）、T8（4个UTC6h块各H/G）、G5（日H+4块G），H为固定EF等效能量。G5分离总发电时序与排放分子时序。相同7日主延迟/30日压力，Ridge/ET/异方差/QRF先检验信息；point proper2024H1，riskfit2024JulSep，riskpool2024OctDec，外层held源全排除。所有预测先完整封存再score，无target入口。

V17由v12_code_review实现，root独立通用审查、独立Python/科学审查均通过，24focused tests通过。prepare/predict/score已均由工具确认exit0（旧sessions62364/70589/13765均终止），无待轮询正式进程。54代码/17输入及四包各8460记录验证通过；预测seal `962ab2a882612cf1943a45fa43efee90884b3cf66997ff7cf3401033ad89d020`，完整63预测先封存再score。生产/test/protocol四文件保持冻结。

四个冻结pins：module `35beda22c1c691d4799ba0c155186019d7f2f7f2b0a5dc599e9d4bd25c9e65fc`；runner `ff2093c51973fababb25299cadd16bfdee6d03366efa1dacf66372393f22f181`；tests `89d913ed0c0e21dc2daa2a5c4171f7125ffd83357af7196fc8d7fb4f08cc8e72`；protocol `631154554ccd02af2f57949afd63dc76d423beb158a274f8fa1b93336214efd0`。运行对全部query及proper/riskfit/riskpool实际D2日锚核对1e-9身份；信息gate同时比较D2/D8/native，H时序对G5另描述bool。三审见`results/local-sketch-v17/*review.md`。

T8给历史块粗CIF信息，8标量不等采集成本或Shannon信息；量化/噪声/相位、同预算coarse-CIF TSFM与预训练污染检查、真实渠道在晋级前仍未完成。普通模型获益只能证明信息价值，新的机制贡献仍需单独同预算检验。

V17主lag7四个新包均0/20配置满足覆盖gate，不能后验换lag30压力表（D2一项/T8一项/G5两项/D8零项）。源T8 fallback为ET_hetero/f1.5，CRPS29.998103、覆盖95.8212%、最差83.2870%、宽236.080566、IS257.404504；同model/factor G5的CRPS/宽/IS更好，两个flag均false。固定1.25 ET-global T8为CRPS29.553619、level26.927809、shape26.549782；native26.694362/30.207553/19.954894。不能由当前全局direct+risk流水线失败推断T8无信息，也不能把初次clip后ensemblemean误差唯一归因point learner。[结果文档](LOCAL_SKETCH_V17_RESULTS.md)已获独立科学最终PASS，hash `bcedde62b80720d151e2b9fbca595b09a890f2b5b1d099105ebab3cd5cd71048`，对应review `60cd89c29b29adb4cb70a5f8e774c698b6839b87b1f9682bfbe7c3c0d2037c96`。

独立实际审计session78133首次exit0/PASS：63/54/17闭包、四包8460全压缩、188925日/1155地区/165宏、共同1145days、全部124551168raw逐值重构、allCDF/shareduniforms/cohorts/anchors，2310固定评分行最大差2.27e-13、112固定QRF查询权重差2.78e-17。没有全forest/尺度重拟合。`numerical-audit.md` SHA `ecdfa18dc54597c80e2000c63c2c8af866125bc02f4fc9ddc0a3a49832086ef1`；audit script `a5101be02c31b740eade721c053976dc2e784457d1f54f8aeab44b67d6399e1a`；首audit.log `7a93054e3b15f534849996b305c06224317641cd326da79006425095a23edfbd`。固定1.25全部12个ET新配置均CRPS/宽/IS劣于native，CRPS全范围29.4178–36.1703，不笼统写29–31。

V17最终[evidence清单](evidence/local-sketch-v17-verification.json)已封存54代码/17输入/143产物，hash `fa56414ac3e5cd4e066668cfc2d62cbc10cb16628806c1db85e3060bf1c13861`。独立全214项hash/closure、127条收据链接及完整目录核对PASS，新增独立`manifest-review.md` hash `3d466c564b59208cbec23a9d5de35bdb91c9c277158d4293f3378cb97d4ba3ec`（该复核报告在既有manifest之外，未改旧清单）。结果与审查等已列产物不再改动；V18另版。

## V18：先反驳信息利用方式，不同时换risk

root授权v12_code_review独占实现新`native_center_v18.py`、`run_native_center_v18.py`、对应tests/protocol。仅主lag7，固定native128整日风险与IDs；从V17封存`centers`取ET point（不能rawmean混Q4残差），对ET−native mean作day/block/full投影，alpha0/.5/1；另普通raw_day与T8 raw_block锚广播。alpha0必须精确native；raw块锚是weighted量，广播只叫heuristic不称真块均值或物理守恒。存requested/realized center、裁边和饱和率，裁边变窄不算新risk机制。暂不加本地coarse errorlaw/IPF/新校准。

V18完整35归档（7native+7×4包），共同12arm/alpha、T8十五arm/alpha，总52×5factor=260源曲线。D2/D8/G5与T8同投影同alpha强控制，另保留V16additive及V17完整三ET风险参考曲线并绑定其父收据。root通用、独立Python/科学均批准，19focused tests通过；prefreeze修复合法raw H/G为1000+ulp时的支持guard（仅1e-9浮点余量，不改值/公式/gate）。

V18已正式freeze，prepare session77381确认exit0，check验证58代码/152输入。原正式predict session16627现已exit0，日志`results/native-center-v18/predict.log`。四生产/test/protocol冻结：module `425d2e9459e4394b5b47a8dc9b314cb7dd48bd86b60bb1c1eea23f11ac57549e`；runner `25f7bd0b118cb4456b922b2d76dbd43f6a8375dc0d8d7f13ca00c44091c35af6`；tests `3a3ad1fd9629ef237b4942f06f773b67822075956f794b0b92820911bd178513`；protocol `e2bd84712d43db270d36ce82c2e4dba8b4b6efeb638158404c6a5065c21fb10b`。Python review `779da4ff752804bd31b4c1a3311cd2f61c87b4beb894dc692520eec0340979b5`，科学review `1e226455c8cfdd34f9a8737b22c049fbd3a15e2b17ebf9ccaef87a191aacd637`。完成评分与审计结果见下文。

无论普通中心是否有效，method_breakthrough固定false；明确中心增量后再另版同中心比较风险传播。普通中心shift在preclip保留native不确定性，任何变窄须区分更好中心支持较小factor、初次/扩散clip效应，不能归给新risklaw。

现实字段初查新增[中国本地信息记录](CHINA_LOCAL_INFORMATION_FIELDS.md)：核对国家能源局2024原始DOC、江苏2026披露通知和湖南2026出力细则解读；找到分类发电/机组支持与合法日前预测的字段线索，但未取得实际历史vintage或认证T8现实可得。只读字段/制度说明，没有新增区域运行数值读取。

V18原predict16627及score65738现均由工具确认exit0；35完整预测hash/父合同通过后评分，共297700日/1820地区/260宏，source_information_gate=false。源选T8 ET_day/.5/1.25为CRPS24.931876、覆盖93.9956%、最差85.4877%、宽176.272949、IS196.203187；相对native CRPS降6.60%，宽微增0.000224%。G5同结构CRPS25.057603、覆盖94.2251%、最差87.0028%、宽176.220407。T8对G5仅约0.50%、对V16additive约2.93%，不证明分时排放独有价值或原创机制；D2/D8选alpha0。T8 raw_block/.5/1.25 CRPS23.984642、形状MAE18.450372，但BPAT覆盖83.4754%，仍失败。见[已审结果](NATIVE_CENTER_V18_RESULTS.md)，独立实际/科学核对与封存见下文。

V18独立实际审计首attempt84393 exit0/PASS：全部198503424新raw独立重建，119alpha0精确，所有clip/汇总，3640固定Jan/Apr评分行显式pairCRPS复核max6.40e-14；未全重训V17中心。最终结果doc `cebd3ad0679335b5b68c1cf2dbcada69090631515963f76e3589ecca029efc24`，科学review `f03a11326353a83e6c2314f6fb05c885aa3fa23398ee6f2d150d043e9091e4cd`。最终清单[evidence](evidence/native-center-v18-verification.json) `efa95d7d79284d3157426fae9cd46785d75355fa0daee4fa479e1edc9571a3aa`封58code/152input/83artifact；独立全293hash/closure及35+28库存核PASS，新增manifest-review在原清单外。新next-information-design.md亦在核心清单外，不改旧封存。

## V19：按风险函数审视最小信息

纯合成新witness/module `346c9778d1b0fc2b1c3671785891bb00def89f057159cf10d029997af1677285`、test `4bc6cf8ba70aed75c95d93d471ae776ff20bfa180b03a922348f51e9cf4f1598`，独立Python数学批准（Fraction秩/逐项风险，Decimal根guard回归），正式writer首运行exit0。JSON `55a4452051b32e9de32f48269bb1536c6b3b18b04db4f6744ccb7f95dc4aee1c`：同D8/T8/G24可有strict>600小时0vs12；额外4固定小时标签仍0vs8；再加一时间矩仍0vs2。秩/零维数4/20、8/16、9/15。仅证所列观测在此可行子类不足，不证普适最小budget或未来预测。

新mean诊断5文件三审批准、20focused tests通过，严格只七源2024H2同期oracle，prepare首运行exit0、run原session3411 exit0。runner `d4b5f4825c69e5be21770a6a7d121f20eb7cedf5100b24f1a9d90b58b072eac0`、test `186057db7945cb7dc9739d68479025c957a522b15ee9c2a6486518365fb29690`、protocol `37872a5f9f8c2520f7454490e2129277deafbd384cff2ba4fa32a499125d4fd4`；freeze `8a1f9677f74b6c7d140fc972286f92a988d7639338a8edc7c23a6041d551b2b6`封5code9input。prefreeze修精确端点knapsack、parent-retarget、rounded endpoint认证与共享算术区间，正式运行无失败/无生产字节变更。一个合成旧assert未适应新增算术区间的失败工具记录保留，修为approx后20passed，不伪装从无开发测试失败。

实际1000地区日/2000行；BPAT180、CISO181、ERCO69、FPL180、ISNE26、MISO182、NYIS182。D2 finite/simpleCV/quadCV/fullG宏均值范围宽958.333/131.959/109.631/80.622；T8 833.333/61.649/50.489/35.261。真实abs水平口径差地区等权2.042/.628（T8为四块加权均值等权平均与日等小时均值差）。这量化G-only二阶量对历史均值外界的价值，不是小时tail或预测覆盖；不能把88.56/93.94%界宽缩减当native预测区间缩窄。三guard计数0。

独立实际审计首session27388 exit0：10000LP全通过，2000行全部字段max3.4106e-13，LP原始budget残差1.6653e-13、对偶gap2.22e-13；80位真加权均值在共同算术区间内，5正式JSON见证精确有理数核验通过。audit code `3ac5e43ad36a24270ad0b1f236e05ec00c25d6e834f02baf429e76bd655eb7e7`、log `225f2127fb4049ae0f2339324242995f2681c2c2cafa331c9475c8f8e095c7d0`、report `0d09ee133ea9c53ebceab60758fa4f4d654156a215e95cfb1cd154b68ed197c4`。正式阶段无失败或生产变更。

[风险与最小信息最终结果](RISK_INFORMATION_V19_RESULTS.md) `568b531cbc526c5ca9d3d9e1f5f0841c13b58f278c52f1d9f7d3e001b81ee8fa`，独立科学review `03dc9f1ac557ab0a11946b3ece9c53aae14bfd20f3e906de521e2c85b0ebb253`。新增严格限定类下的最小值：从仅4块mean500、G全1、coal/hydro可行类开始，只允许精确小时CIF点查询，对任意路径零误差确定strict>600小时数，最坏恰需20不同查询（每块5充分；flat500回答轨迹留两个未查点，750/250对手证明必要）。不是普适预算或实际概率预测标签需求，不称新理论。

V19核心[evidence](evidence/risk-information-v19-verification.json) `e2d30122f3daa245740efe902164a6e1367ec1fe906ebfdc589ff1944cc820fc`封5code9input17artifact，独立metadata/hash闭包复核PASS，全31文件hash/库存/3完成输出/父绑定一致；新增manifest-review（原清单之外）hash `4747923e6171ae37c3392d39eab13f34906221538775fbbd5d96358e0eeed714`。下一步面向可约束的具体tail事件设计最小信息/认证支持诊断，区别未来粗量前推与块内剩余风险；有mean界不等于对tail毫无约束，需核哪些超越数/事件确实能排除，strict阈值及整数性需明示。只有合法信息与强参考的联合增量出现，才设计专门机制或讨论源冻结后的目标开发复测；未知地区仍关闭。目标继续active，不降格为报告/源pilot完成。

## V20：超标小时计数与原生特征补齐（2026-10-26）

Tail V20A的4个生产/测试/协议文件经通用、独立Python数学及科学审查后冻结；prepare及正式run均由工具确认exit0，原session16490已关闭。沿用V19七源1000完整有效地区日，事先固定strict阈值100/300/600/900，比较四块H/G（T8）、再加四块G二阶量（T8_G2，必要外集）、再给全部小时G（T8_G24，仅HG/support可行类sharp）。原始十进制字符串用Fraction精确处理，保存完整可行计数集合和端点，非仅松弛连续界。输出12000日行、48000块行、84地区与12宏行。尚待独立实际数据审计与最终科学核对，不能提前称审计PASS。

初步宏结果：τ100的日计数span为20 / 16.691 / 16.017，τ300为20 / 16.281 / 15.477；G二阶量消除部分歧义，但完整小时G仍留下约15–16小时的计数跨度。τ600、900在本1000日样本实际超标数全部为0，必须披露，不能用这两阈值声称正尾事件预测有效。这些都是同期历史oracle识别范围，不是小时预测区间、CRPS或未来覆盖改进。四代码冻结：module `a00b0534e99ad8bc96f0a6947b321fdd9c9cc04358880b1fa1a6bb126968d0df`，runner `bcd2981294b8c76f40d56bcb7d53e9dd3b043d9b140da25b21cd4ad9985581a3`，tests `6880bebcac00d611050f9ee20345420865d5d8e5a9bd58f3d94c907414675d10`，protocol `0fef49af47161395d5ce349a29081ddef52b3d988bd4997a24064898cd54ef68`。

Native V20补齐是独立的数据先决条件，不是新增预测机制：42有向外层/教师对，各有H1离线地理折外回顾表征与H2真正时间回测，共84教师档；另7个原V18 query精确副本及21个双排除bank。H1教师允许其他proper地区同日/未来H1 donor，必须标chronological=false，不能称当时可发布预测。H2 bank均早于query；真实数据发布时间仍未认证。没有读取新目标或2025细响应。

该数据阶段首次正式prepare exit0；首次predict原session22648现已由工具确认**exit1**，日志保留在`results/native-hindcast-v20/predict.log`。所有教师写出后，在最终逐值重放检查`run_native_hindcast_v20.py:258`抛出“Prediction values, sampler IDs or exact parent-native identity changed”。目前无prediction_seal/completed，不能下游使用、不能宣称91+21完成。冻结4文件不改，先另写只读诊断定位字段/数值/RNG/布局差异；任何修复必须另版并保留失败证据。

V21普通native-offset残差强对照仅有未冻结实现草稿，尚未prepare/predict。计划统一H1训练窗口、统一`y−native_mean`响应，比较native_only、四packet_only、四joint输入；packet_only仍用native作为offset，不能称完全无native架构。固定native整日噪声，先补齐普通原生特征参考，防止把旧V17缺native输入造成的差异误归为新机制。父V20必须真正完成且独立实际审计PASS、证据清单封存后，才允许V21正式冻结运行。当前不打开未知地区。

### 同日后续：Tail已审计，Native仅校验器恢复待运行

Tail首实际审计session22248由工具确认exit0：全部12000/48000/84/12行和1000mask精确一致；1024000子集证书、66547条实际有理数可行构造、14336LP（6972不可行证书、7199正margin最优、165零事件可行，数值未决0）。最大margin差3.41e−13、CIF尺度等式残差8.88e−13。独立report `e1935e30cec79af9000a6ae6db38d5d3ae0a4fb3723e58a9783526d215588f65`，首log `55870041fa5feb022bb8edc18cd970f710df30271ecb3e041fa8c9832bd41b74`。实际τ100/300有正事件地区日820/487，真实超标小时总数19608/7805；600/900均0。主结果见[TAIL_COUNT_V20_RESULTS.md](TAIL_COUNT_V20_RESULTS.md)，待最终科学review封存。

新增两项独立数学核对：固定G的exact K集合在当前单H/同支持模型中其实必为整数区间。H≤τΣG时候选k由最小k个G和S_k检验`τS_k<H`；H>τΣG时由最大k个G和L_k检验`H≤τΣG+(B−τ)L_k`。所以块内K=0的sharp认证只需`H≤τmin(G)`，四个块最小G能复现全G对全日K0的认证。此排序/knapsack是经典结果，不能称新理论、不能称普适最小采集成本，也不复现所有K风险。新V22拟比较同维4个Gmin与4个G二阶量，原1000日/4阈值/4预算；只source历史自适应探索，先冻结审查，尚未正式执行。

Native V20失败已只读定位：NPZ保存q数值不变，却将原mixed strides变C连续，改变浮点求和次序；首档raw最大差2.274e−13，中心1.137e−13，IDs/粗量/所有其它字段精确。仅恢复原q布局即首档所有字段bitwise相等。原112NPZ、四冻结producer、失败log/freeze全部保留，无原completed。新V20R1仅校验：21bank对fresh重构逐字段含dtype精确后，用fresh bank原布局复算84teacher和7query，仍全部array_equal，失败则拒绝；新receipt诚实记录原exit1及独立audit待定。新3文件已通用与独立Python预审通过，科学预审中，无正式prepare/verify。65code141input，独立原生actualaudit另待R1完整exit0；R1通过也不等于新的模型增量或突破。V21父门改为R1独立审计PASS清单，不制造原V20成功。

### 最新完成状态：Tail、R1与V22均封存；V21模型比较完成待独立审计

Tail主文与测量note已获科学/通用最终PASS；[最终清单](evidence/tail-count-v20-verification.json) `7c2311d8dc8224778b45ea41a5386df5c7e5e5bea8c72d80db842d43b6cbf7d2`封4code/11input/19artifact。独立34项hash/库存/V19父31条目及收据绑定PASS，清单外sidecar `e762f109d8b2f2096b8f8b3a018e1689354e1a76a6aba51c1db06525cb1eb96e`。旧Tail输出与两篇最终文档不再改动。

R1已三审后正式prepare49569、verify50672均exit0，全21bank/84teacher/7query逐字段dtype及array_equal通过；原V20仍exit1、无原完成。独立actualaudit首33659 exit0，1838592个donor IDs精确、44126208个teacherraw独立重建最大差4.55e−13，H1未来样本421946/H2为0；没有解析H2/2025细响应。[R1主文](NATIVE_HINDCAST_V20R1_RESULTS.md) `ea447371db8cb6baa53c1a660ddbaaee852b85a57334e5bbbca49fc290b60cbb`与最终两审PASS。[最终清单](evidence/native-hindcast-v20r1-verification.json) `7a39eb59d98ce3323e443c49e3ead987813be855f49d3c21865155d71beaec8f`封65code/141input/19artifact，最终独立audit字段PASS而旧R1收据pending不回填。独立核225映射条目（222不同路径）及V18父293项PASS，清单外sidecar `2ce0abe31fcd321216275d5314b31dff84e4530a0965b2dccab460e17a0890f2`。只读早期diagnostic未保存112项digest向量，不能称独立保留pre-R1字节基线；R1freeze以后逐项hash及完整值重放有效。

V22 exact Gmin诊断三审/27tests后prepare与run20803 exit0，源1000日/4阈值/4预算，16000日行/64000块行/112地区/16宏。首actualaudit32049 exit0，112000候选证书、105831条有理数最小位置见证、2688LP（1220不可行/1303正margin/165零事件，0实际数值未决），旧三预算12000/48000曲线精确继承。Gmin vsG2日集合更紧658/更松519/相同2822/互不包含1；不是一种摘要支配另一种。

V22宏span在τ100为G2 16.691 / Gmin16.834（差0.86%），τ300为16.281 /16.033（改善1.52%）；600/900小幅改善但真实无正事件。K0认证严格复现fullG，所有证书仅BPAT：τ100 0→2日，τ30056→89，τ600179→180，τ900180相同。不要把这一风险函数的代数充分性/BPAT历史收益称多地区小时尾部突破；τ300五源span略改善、ERCO/MISO更差，和K0增量归因不同。[V22主文](RISK_STATS_V22_RESULTS.md) `b23cec00b32ace29eaf19e1f34ad39251998550f3482844dca536c7729ccf3b9`已两审PASS；[清单](evidence/risk-stats-v22-verification.json) `f6e022f8a286f5acbb0aa1b2c8f6946fee5814a8be6b463b8ed63b3cd7e28c6d`封8code18input20artifact，独立46条目/43不同路径及父闭包PASS，清单外sidecar `0e9210366f2f077153e860b6b496aa55cd8094aff60d6824eae1573b97a7a069`。exact/sourceadaptive，无量化/真实渠道/未来收益/新算法结论。

V21普通native-offset强对照四文件三审通过、root14tests exit0，69code158input。正式prepare52644、predict85113、score28067现已全部工具确认exit0，7compact NPZ+7fit/asof JSON完整seal后才评分，source_completed已在。全55arm×5factor=275曲线、共同1145源日；**source_budget_gate=false**。初步只有packet_only_D8有合格配置，其余输入的选项是fallback，不能称合格winner。T8 fallback joint/block/.5/f1.5的CRPS27.393631、覆盖96.0635%、最差85.6534%、宽210.552117、IS226.065190；native f1.25为26.694362/92.9545%/85.2273%/176.272555/202.247558。尚待独立实际audit（必须重fit全部63固定ET，而非仅接受有限δ+公式自洽），不提前签actualPASS/最终manifest。审计候选`results/native-refit-v21/audit_outputs.py` `3799d72a549538a64f8bfe0a28f16f4ffcc2c7030c16a1d68423cad6d4d290ac`由Pythonpeer独立写、code-reviewer正预审；不重跑producer或改冻结字节。

V21当前四pins：module `537d29653c85e4dd15a74d02472f6e67843a65bf1cb86b345054c65f687c16cd`；runner `70305891ac901f353d84f079ebef23d71e101abb7a176d8e7ce77e58bdb7830f`；tests `306f4575cfa580b58a70a9f759e2744ad0bc097a0b5b27e5dee2491369566473`；protocol `4b8688aecac18fb315612059729012543d501b1b736c99362287d9dcba2e4b5c`。该负结果只检验native-offset ET，不否定所有native特征或粗包信息；原V17/18是day-anchor offset，若要检验加强原强中心，仍需同anchor offset的joint ordinary control，不能混淆。

后续risk参考有严格嵌套要求：若H2教师S要模拟没有S小时fine的pseudo-target，inner point不仅不fit S的H1响应，训练教师U的native特征与offset也不能由包含S fine的R1双排除bank造。应另行预定三排除{outer,S,U}四源bank（35无序组合），S的H2 query用双排除五源bank；最终outer模型教师五源、query六源。经典nested geographic cross-fitting不是新机制；它解决间接S-fine路径但不消除4→5→6库规模分布差。当前V21无riskfit，不追溯阻断其合法point控制。未经新合同不直接实现复杂risk机制。

2026-10-26再次核原始[DGCFM WWW2026](https://wangdan.people.ust.hk/Publication/WWW26-CIF.pdf)：§4.2 Eq10在推理时使用历史CIF，§5.1.2的zero-data按不使用target dataset训练定义；不能直接据此认定无本地小时上下文也得到其性能。[CarbonX v2](https://arxiv.org/html/2510.01521v2)使用历史CIF序列，§4.2.3还用小时细残差滚动校准。我们的无本地小时上下文合同应区别无target finetune，但T8已含合法历史6h加权CIF，不能写无任何本地CIF，亦不能排除给TSFM相同粗上下文的适配强参考；上述区分不是宣称首次通用跨域预测。

已向用户发出非阻塞的信息渠道问题：目前在中国可取得仅月年燃料统计、另有小时G/负荷、还是另有机组可用/出力/调度记录。尚未收到答案；继续源实验，不把未回复视作取得数据、渠道认证或授权。原目标仍active，尚无论文机制突破或未知地区晋级依据。

### V21已完整审计封存，V23仅准备同anchor控制

V21首actualaudit98561真实exit1：63森林和主要全表已核后，auditor把背景`native_input_available`布尔列走数值减法而退出。原3799脚本保留`audit_outputs_attempt1.py`，原`audit.log`与`audit_attempt1.log`均hash `d89e1e3a9ccff21525ba556473b67f44b7e490c2d6b8ee877dc818c79b1b08e3`，没有擦成成功。只修auditor为bool精确比较并加mixedtable/boolflip回归，数值容差/producer/输出不变；root与独立generic复审后第二次4748完整重跑，真实exit0/PASS。新script `86a32a2cd6846bc8265c0d9c784dc8b4a57ba24c182d3bf7fdce2586545671cd`，successlog `03904061aee10f34b6d63d1488aba36fd5c86301fc0dc9671471129cc1794efa`，report `3bea0b9a7120fd106cb16fa4724ca1af420697d0daf50fe4d5e845a0f78b0cbb`。全部63固定ET重fit后δ最大差0；所有214072320arm单元/314875day/1925region/275macro/1260contrasts/2415refs和3850explicitpair行通过。

资格精确计数为native1/5，packet_only_D8为3/30（不是6），其余八个新输入各0/30。但matched joint−packet宏CRPS在D2/T8各30/30更低、G5为29/30、D8仅5/30；不能说native X无价值。固定day/.5/f1.25，joint_T8 CRPS26.691208接近native，但BPAT coverage83.0492%和IS203.077915均未形成合格收益；全选择T8 fallback仍不合格。文档[最终结果](NATIVE_REFIT_V21_RESULTS.md) `573a56a1ee4e477508ebc72810ef216011dd163d993463c4c4d31372416719c5`获科学/通用PASS，归因snapshot `results/native-refit-v21/diagnostic-options.md` `b7e0d9a497684c9701e8a439469b08b2409cf7b936a11831af5d303855bbfc45`是在actual审计完成前作的只读分析，不回填旧时间语义。

V21[最终manifest](evidence/native-refit-v21-verification.json) `d1ee8f68741a8ce3f5ab262e047c66f2ce014c1afc808e795a09ba3af59f0687`封69code/158input/43artifact（目录28常规文件+7NPZ/7JSON+主文1）。最终actualPASS，明确first1/final0及两次history、所有正式writer首0，source_budget_gatefalse与fullprobabilistic_controlfalse。独立最终manifest核对进行中，尚不提前称sidecar已PASS；不改此前任何sealed字节。

V23已授权codeagent准备四新文件`native_anchor_v23.py`/runner/tests/protocol，未正式prepare或读新fine。只补原强center的ordinary同anchor任务：native_plus_D2_anchor（mean/std/calendar54+合法D2A1），packet_only_T8/G5、joint_T8/G5，共五input，同H1行/weights/ET，拟合y−各packet原A并先clip绝对desired center，复用旧V17 PointModel('ET')。packet-only必须先精确复现14旧V17 ET中心及旧V18全部对应投影/alpha raw，native α0exact，失败不seal/score。共享native+5×3proj×2alpha=31raw×5factor=155curves，预计177475日/1085区/155宏。全部7seal后才2025sourcefine；不加weather/H2risk/新grid/新target。先三审且绑定V21finalPASSmanifest才formal。收益仍须对matchedpacket/native_A/G5及完整旧强背景，单胜弱V21任务不能称native提升原架构或原创机制。原目标继续active。

### 用户确认仅公开月度或年度统计：主路线回到真实信息预算

2026-10-26，用户回复本地可取得数据为“仅公开的月度或年度燃料统计”。这是后续中国主场景硬约束：不得依赖本地daily D2/D8、6h T8、Gmin/G²/hourG、小时CIF或小时尾部标签。V16–V22的较细预算结果保留为信息价值/源区诊断，不能转写为中国可部署输入。公开天气也不自动纳入主预算；原GFS previous-runs是历史NWP代理、未认证逐时发布版本。

V23在用户回复前只有module/runner/tests三个draft、12个合成测试通过，protocol尚未创建，绝无prepare/实际fit/predict/score。收到约束后停止扩写及formal；另加明确DRAFT/PAUSED与formal入口在IO前拒绝的状态修正，不靠缺协议文件偶然阻断。它不是已执行的阴性/阳性实验，也不是当前优先路线。此前“授权准备四新文件”描述的是当时任务，不能视作新预算下恢复formal的授权。

V21最终manifest独立核对已PASS：69code+158input+43artifact共270映射、267不同canonical paths，28常规文件+7NPZ/7JSON+main1库存，父R1全部225映射及V18全部293映射均核对；sidecar `results/native-refit-v21/manifest-review.md` SHA `2cc9cfe5421efd7051613b91105215118e91ec1070e56a375b862669361a7eae`。核心manifest `d1ee8f68741a8ce3f5ab262e047c66f2ce014c1afc808e795a09ba3af59f0687`及旧sealed字节未改。审计PASS不改变source gate=false，也不授权V23或未查看地区。

只读代码盘点发现：月度强参考是V9 harmonic_ridge+native recompose及matched additive anchor；V11–14同家族但proper从Jan25开始，V9从Jan1开始，两者不承诺逐字节相同，应保留完整H1 V9强背景。`weather_coarse_v14.legal_coarse_law`检查period_end/available_at不晚于origin，但输入月表是修订EIA代理及人为month_end+30d；尚无原始历史发布vintage认证。年度-only没有同合同入口，`annual_drift`只是月序列上年同月差，不是年度输入；不可将年度表复制为12个月或把份额直接等权平均作年组成。V11 `_load_training`会无条件读target月表/telemetry，后续source-only不可直接复用该loader。

下一项优先诊断从“增加细信息”改成“明确同频率字段”：燃料发电量/燃料耗用量、煤气油细分或thermal合并、覆盖总量/统计边界、期间/首次发布时间/修订。用户尚未指定省份和字段，因此不默认公开月表等价EIA八种发电份额。国家统计局[2026年9月能源生产报告](https://www.stats.gov.cn/sj/zxfb/202610/t20261019_1965613.html)示例公开火电/水核风光及规上范围，亦披露同比口径变化；该全国示例不认证任何省份全口径历史数据。仅燃料生产/消耗量没有转换效率和统计边界不能转发电份额。

拟先做M8月发电细分、Mgroup合并类别、Ygroup年度聚合的源区信息退化诊断；先锁字段映射与发布假设、缺失掩码，再看是否值得新建月年统一迁移机制。年度需要真实年度标签或完整同边界月发电总量汇总，不补缺月，不把预测月细分伪装成观测。最强native、matched简单锚、标准动态和同预算条件分布参考都应共享信息闭包。尚未运行该新诊断，尚未找到原创机制增量；不因原信息预算缩小就降低覆盖/效率或未见地区门槛。

补充数学澄清（纯推导、非新实验/原创定理）：固定正G、H及0≤C≤B的单块小时峰值sharp interval为 `[H/ΣG, min(B,H/minG)]`；常数路径取下界，将能量优先放到minG小时取上界，路径凸插值取得中间峰值。四独立块整日峰值区间取各块上下界的max。因此Gmin还可复现完整G的峰值识别区间，并非只复现K=0证书；仍不能识别实际峰时、实际峰值或全部K集合。没有Gmin的公开月年预算不能使用此额外约束。旧V20/V22sealed报告不回填。

### V24同频率字段诊断及首次独立实际审计均通过

按新硬预算新增`coarse_fields_v24.py`、synthetic测试与[运行前协议](COARSE_FIELDS_V24_PROTOCOL.md)。只字符串筛选已锁七源月表2022–2024后转数值，不开小时/telemetry/weather/目标。六固定schema是过去加权代理水平sharp界，不是新预测方法；包含原细分真值不是经验覆盖。年度只用完整12条月proxy按observed energy汇总，完整月条目不等于完整小时普查。

初代码15tests检查后Python发现freeze本体未绑定completion。正式前仅修manifest捕获/重核及receipt freeze_sha256，加mutation拒绝回归；三审APPROVE最终module `dbd4527c010697e5dbe2a69e4193ca1127000c1c402bb40d10cdd69de5650435` / tests `c82b0498debf7e32ef56d8385cecc6843d36fbf24113e91820d95f504aa5d1e6` / protocol `d3df952dd4343179bb2e7bd3e9227012954e163037f3b51e2796273b4a1e48a5`。最终V24 16tests与V23暂停15tests合并31PASS；没有正式失败或结果后改producer。

正式首次run真实exit0，`results/coarse-fields-v24`，独占日志`results/coarse-fields-v24-run.log`。freeze `9479d26fcb7aac0c129cba6a85a4496b53b9c11ed841c7adeed85d0f844273f3`；242源区月/2023共同84月，18完整proxy年、3不完整源年。2023等地区等月识别宽度：M8=0、CGO=295.870173、合并正EF=322.078613、CGO+coal=173.136356、+gas=12.775297、+oil=293.709921 g/kWh。+gas变化只相对保留other的CGO基线，不能说从合并全部正EF只加gas就得到12.8。固定EF和本批源条件下字段价值明显；不能推成中国各省最佳字段，亦未解决小时覆盖。

2023原月代理强度年内range等地区均91.170947 g/kWh，月值相对年加权值的等月绝对差均22.913403；只是观测季节variation，不是年度约束给出的sharp逐月界或未来可预测性。固定schema的年度上下界/宽度等于各月界的energy加权和，年/月宏宽差不能归因于频率收益；年度缺失的是年内分配信息。新方法或未知地区门槛均未触发。

独立actualaudit原进程87363首次真实exit0/PASS：242month/18completeproxyyear/21supportrows、六schema全部witness和五CSV全字段/匹配宏队列通过，3120独立LP最大目标差1.1369e−13，全表最大差3.4106e−13。首尾freeze/completion/五CSV/auditor自身8-hash向量一致；原producer completion的PENDING保持写入时原字节，不回填。auditor `651c74f60ad7eae1390f5375f5e7459cd1a449be20eaacbaec98705d1ea471c1`，独立实际report `d17a0910d2dfab97ce705d3b00a5ae278342fd6cb57efec8d0a39939e033b39f`，log `cd0975b0e09db86dbc10736ee747dd4741345a03c1c3e9a4c08cee4004c06490`。

新[结果主文](COARSE_FIELDS_V24_RESULTS.md) SHA `964a1a869fea43a9fe5cb3b04e2adfc7caba00855bf6bddfe389747b2ae3a4d7`；final[证据manifest](evidence/coarse-fields-v24-verification.json) `b2f98a68f8bebf64852c75c8a9eba6d373132bbb0a52648a4ab262b9538ecf98`封3producer/1auditor/1input/17artifact共22映射，outputdir7常规文件。最终科学数字稿审查PASS sidecar `results/coarse-fields-v24-results-review.md` SHA `b30ef5d8e8db84a7a0c7db6ece46a20bf9ef86682448de0330ecdaf82b76a767`；独立manifest全部22hash/库存/aliases/计数标志PASS，sidecar `results/coarse-fields-v24-manifest-review.md` SHA `c8742759b5521bedfdeef042ad7a80260f00f92b640e329058196837afd79332`。两sidecar均不嵌入其审查的manifest。科学结论仍是字段预算缺口值得优先研究，不是预测效率或原创算法达标；长期目标active。

V23最终三draft有DRAFT/PAUSED和三formal入口无条件IO前RuntimeError、无CLI绕开开关；15tests独立通过，暂停复核`results/native-anchor-v23-draft-review.md` SHA `8a7900693c0658890ac1274a74a03e46309cfeee76619f1df4a26de1ab2b4ead`。没有协议/freeze/predictions/seal/completion。月年路线见[独立科学设计审查](MONTHLY_ANNUAL_RESEARCH_REVIEW.md)；后续预测须另立M8/MG同预算控制，不从八燃料预报旁路泄露MG隐藏细分。

### V25：从字段诊断转入真实月度预测控制（实现完成，待预审后运行）

上一目标轮归类为progress：V24实际诊断/审计/封存及用户实际月年预算改变了后续路线。本轮保持长期目标，不将普通基线成功当作论文突破。

新增[运行前合同](MONTHLY_TRANSFER_V25_PROTOCOL.md)：source-only2025H1七outer，月M8八项与MG五组(POS+核水风光)独立谐波/prequential law，MG先删隐藏八项，不能merge八维forecast。每轨conditional加性锚/global重组/conditional重组/打乱q，MG另加conditional plugin比例，共9ordinary臂×5倍率=45曲线。MG隐藏比例按供体地区月mix与同q联合，zeroPOS donor共借j的比例+整组q；全zeroquery不误fail。shuffle只保留fine_draw/q输入多重集，不保证输出CIF边际或中心；M8是有限粒子配对负对照。plugin差异含比例随机性/依赖/非线性中心，不称纯风险机制。

R1并没有七份六源query bank，21份均双排除五源；从同outer两个pairbank按region/day去重拼六源，overlap q/rawCIF/mix/monthmean/support/month exact，再按旧query provenance排序。无新训练/预测fine解析；全部七档seal后才读七个已查看源2025H1fine评分。M8 native IDs必须exact、raw≤1e−9复现旧强native，失败不seal/score。禁止新target/未知区/daily/T8/hourG/weather/telemetry。年度-only及完整异方差/条件分位数强参考尚未完成。

root与codeagent分工已实现5新文件：module `5ee0dfd76f2ba6de1143c579acd3eadd0500cb55d1edde55eedf7793a328a0fc`，puretests `4b1bfc82b27a05e9959a36f1dba9340fdff41413c231592fbca9aa24e06f92fc`，runner `95a31c58f8338742e6d2e649987e7a7b8cd115e48660bcd856a8f7a40a1a3ed5`，runnertests `fa35c88e1aa9229a64cdb232beceb37badc8ed18bf00162ea0edb84da7015d3d`，protocol `e858b000d63c93fb4e05c0a8bebd8a4b0648523d1950510c3c7ddb14678a7c4a`。初23tests后codepeer合成证明check可自带空maps/换source通过，正式前修expected_contract固定父重建全键/值/aliases/flags/闭包、加入8篡改回归；root最终31tests PASS。generic独立APPROVE root三文件，module仍待独立Python/science最终预审；尚未formal，不称实际预测结果。

新增[文献核查](V25_LITERATURE_UPDATE.md)：实际下载并核Puffin2026-10-08 v1主文§1–4，PDF `80492dc3b0d4156c8ac97298af5a7c46191935aadefc1f9ce3a64a5743d8f1d5`。它用高分辨卫星covariate、同期精确aggregate条件化；§3粗残差gamma后还用validation fine确定κ，不能把training nofine扩大成calibration nofine。我们的滞后粗状态forecast/燃料分组/潜在比值不同，但“粗误差+细校准/概率解聚”并非新颖性本身。原始摘要/全文边界与本地收据均留档，未复现其分数。

### V25真实预测、独立实际审计与后验支持诊断已完成

V25三份正式前审查均通过，五个冻结producer文件保持上段hash，最终31合成tests通过。首次prepare/predict/score真实exit0；七区M8强参考raw和donor IDs完全复现，全部七区封存后才解析2025H1已查看源fine评分。首次独立实际audit真实exit0/PASS：两法则/union/RNG/比例与q/全35,030,016 raw重建，raw最大差2.05e−12，native差0；255项首尾hash未变。51,525日/315区/45宏/245匹配对比和选型均核，630预定行显式pair CRPS最大差4.26e−14。无新增地区，所有MG fallback实际计数0（回退边界只由合成验证）。

[最终主文](MONTHLY_TRANSFER_V25_RESULTS.md) SHA `4dfb6b7e648006db00673486b92937402294c3080159e4c7b2d6256ccd1473a8`，独立科学结果审查PASS `results/monthly-transfer-v25-results-review.md` SHA `2f46bf882962d48bb30d184760bd7fc16be48addb6485d6078467ee23ced19ca`。新[最终证据清单](evidence/monthly-transfer-v25-verification.json) SHA `740e8b1f18c93ec4bb63136da59cbb026ff24233587c17ca1569b2e6d90f5ad3`封70producer code/226formal input/1auditor/2独立posthoc input/40artifact=339映射，outputdir8根文件+14prediction文件；独立manifest核对进行中，尚不提前称sidecar PASS。

全部5个MG臂×5倍率不满足原门槛；M8只有3个合格配置。f1.25 MGconditional CRPS40.817/覆盖87.70%/最差38.58%/宽245.302，而M8为26.694/92.95%/85.23%/176.273。MG中心化shapeMAE19.608接近M8的19.955，日levelMAE51.866高于30.208。不能由此断言小时风险已解决。MISO MGconditional coverage38.58%、上漏61.42%、下漏0，levelMAE142.949，shapeMAE23.584；统一加宽导致其余多区过覆盖，原覆盖效率门槛不放宽。

评分后另用已审V24完整2023年度分项和原允许六源donor_mix做额外预算描述：MISO过去年POS内部能量加权EF649.564609，六源支持[400,542.948807]，差106.615802。仅是借用潜在比例及其EF函数的凸支持缺口，不是经q输出小时CIF的支持界，不是2025年真值、不证明年度细分普适必要，也未证明中国公开表具有这些字段。该posthoc输入与正式freeze严格分开；CSV `3b31b7a6849439249d9c05ace81730e38fd074eac9e0f31f072512d51c22f766`，复算note `414dd6a0df63387666afb3fd1bdd4df725b7459e3ea3d3a08d686931e0f48569`。有限六源库不含高煤电比例支持是另一解释，扩大合法源多样性或更强MG历史模型仍可能有效，不能把普通核先验失败升级为信息论不可能。

下一轮仅拟条件信息价值研究：若过去年度有细燃料发电分项，而月表只有合并组，先对原MG背景比较最近合法年度固定比例与普通源年度偏移残差，均给同预算加性锚/比值重组。尚无正式V26合同/运行；未把该假定字段当作用户已有中国渠道。年度2024不完整源不得补齐、release前不得提前用；source proper的年度年龄支持与query差异要显式报告，不将普通delta transfer冠以新机制。异方差/条件分位数完整强参考仍未完成，未触发未知地区确认或论文突破，长期目标继续active。

V25最终清单独立核对现已PASS：全部339映射、271不同canonical文件hash、exact8根/14pred库存、父级/收据与五评分表aliases一致。新清单外sidecar `results/monthly-transfer-v25-manifest-review.md` SHA `2a03853ce620b236945b1f008c50fe3b5e2a075bd7524de6af9c64a51b4ebf1b`；仅hash/元数据核对，没有重跑或新fine读取。原manifest及主文等sealed字节不改。

### 最新字段澄清：只有月度火电与可再生发电量，V26条件年度草稿未执行并暂停

用户补充“有火电可再生能源发电的每个月份的发电量”。因此已确认的是月度两类发电绝对量，未确认煤／气年度细分、可再生内部细分、核电、其它类别或全口径总量。年度细分不能作为实际主预算。也不能将火电+可再生相加默认全总量，或将EIA POS/zero映射自动认证为中国统计分类。统计边界与总量未确认时，发电侧CIF的分母覆盖仍须显式处理。

在回复前V26已准备五个新文件（普通年度固定比例与源比例delta、匹配加性/比值），37合成tests通过，但**没有任何prepare/predict/score，正式outputdir不存在，没有freeze/seal/completion或实际结果**。收到回复立即停止三预审及actualauditor计划；runner三formal入口现在任何IO前无条件RuntimeError，无CLI绕过，protocol标DRAFT/PAUSED。暂停状态合并39tests root exit0/0.79s；纯module/tests保留条件研究用途，绝不伪装成实际阴性试验。

下一主问题改为部分覆盖的月度可见绝对量：不能借用隐藏燃料份额、核电或小时G来正规化本地两类量。先严格定义“真实两类全覆盖”与“已观测两类、其余未观测”的不同预算，评估完整总量是否实际可推导，未知分母与火电内部燃料比例分别怎样产生不可识别风险。源区监督可学习潜变量先验，但这是条件迁移假设，不是月表识别事实。不得通过假定新增年度细分逃离当前用户预算。

用户随后进一步澄清“所有可再生能源发电量都有的，不可再生能源的也都有”。据此恢复为完整电源类别月发电量、可推同口径分母的主预算，不再执行缺核电／双通道部分分母分支；煤／气是否在火电内独立分列仍不默认。V26年度细分条件并未被确认，继续PAUSED。其独立草稿暂停review `results/annual-context-v26-draft-review.md` SHA `3b55ca448644ec76d1ea48d2c259170732e871a4695d813fe6d2812e073a5b17`；39tests与AST证明三入口IO前无条件拒绝，无formal目录/收据。之前部分分母建议只是用户最新补充前的条件解释，不能写成实际数据缺口。

拟V27转为同MG预算普通监督强控制：源region-month去重、地区等权，用可见五组预测隐藏POS四比例；Ridge允许投影后比例超供体支持，ET点只内插。严格源地区折外残差配同有效q/IDs，比较匹配加性锚/比值及Ridge插件，共7臂。目的检验有限供体重加权的支持限制是否只是方法限制；不能因V25失败就断言需要新增本地年度信息。正式前协议为GROUP_REGRESSION_V27_PROTOCOL.md，尚未运行。源真group条件学习加查询粗预测粒子积分仍需条件迁移与粗预测有效假设，不提供无条件小时覆盖。原创机制尚无证据，未知区继续关闭。

V27实现交审（尚未formal）：纯module `f214b648ac04df545d805e7186b181a05e980b046127ed6f66d4a88a455a682a`，runner `e68f195c929180b003af27ea1ddf6e034de7a01b31a0d44ed68fb313e144cc6c`，puretests `511b3276a4b3e20679c66f68c4e601328b873239697d1a021356f3d6f1d432f7`，runnertests `985b3745d81f52005df238add96f906e9eb9ce380208283e215b3caf776e7e70`，protocol `e814f5cccf2dd61fe4ee91bcc7f4d63f5ae24cbd751e6e46f5c8e9433b92b571`。root combined25synthetic exit0/1.64s，包括源held响应扰动不改变该fold预测、整月去重/权重、1ULP拒绝、fallbackj residual联动、形状支持与七臂配对。纯module初写dict语法导致早期runner收集exit2，正式前修复后重跑通过，不属于正式失败。仍待三预审，不提前称真实增量；query到source可见X最近行Euclidean距离只是描述，不是凸包距离/路由或新机制。

### V27正式运行与首次独立实际审计已完成：局部修复仍未达标

三预审最终code `f2c0198437fc9189ae0e3f3484f67ad94966c89ff28f43f9bb82fb1d1153a860`、Python `c25b1769e46f9deac080148670dc8822397b0ad388f80d14df7a3e3726a2398e`、science `8f270bfa9235e0df9c0bbdcd00fcd9ae238dd28461a820987b46e27374fa999b`均通过。原session21983 prepare/predict真实0/0，原9248 score真实0；5code/275input，七区各36正POS月，无zeroPOS或跳过fold。全七预测seal后才评分原1145源日。freeze `14d640eeb9600d5d6af2b2d9fde517472ce965c97a97a2b380879bf4b5a3fac2`、seal `00953a3514612f689b4985c85289fcb3d2419b70cd21ff884240dd63ad251674`、completion `274c3caff481c939efe069436a5e4a50b5f7f76fa2da0c6e751df61086d95829`。

独立auditor `d7b3c09a496734d1bacee8206a6bd44a43542cca3d3e8566f750378e70ac5985`经generic预审 `d896d1a82840c85178f90e0ec04d67f2f6a158a619eeaef642a8391791faa3c4`及selftest后首次actual原66205真实exit0/PASS。不导入V27producer，98final/LOSO模型真正独立重fit，全部querypoint/残差与fine/projection/support/visible-distance差0；27,245,568raw单元最大差5.68e−13，304首尾hash未变。40,075日/245区/35宏/210contrasts/选型全部通过，490预定Jan/Apr首valid日曲线显式pairCRPS最大差5.68e−14。独立report `2e8b1f7062aa0e2afc7230caf123f7ce97a5708700148e584913c6797cd09987`，log `b78ce0b0439102f9c8ffdc96877083dec7d0e6b506dddc337c7c0d61bf7cdb75`。

35curves全部不合格。f1.25 ETresidR相对MG：CRPS40.817→37.551（约−8%）、cover87.70%→92.90%、worst38.58%→73.37%、width245.302→264.919（约+8%）、IS344.650→295.593。MISOCRPS110.036→87.374、levelMAE142.949→118.040、uppermiss61.42%→26.63%，shape略变差；ERCOwidth333.203→443.432、CRPS/IS均恶化。99.13%宏CRPS改善来自MISO，其余六源均CRPS仅−.033、均IS+12.922，不称普遍迁移改进。ET f1.5虽worst88.63%，macro96.88%仍failed，不放宽门槛。

MISOETresid31.27%粒子组内EF越原源scalar范围，Ridgeplugin0%；这说明普通折外残差可部分突破有限库比例支持，不是新算法或完整4维hull测试。MISO可见MG到训练正POS最近行欧氏距离均.0997是七区最小，BPAT.6228最大，不能只凭这个可见OOD分数解释隐藏r失配。MISO unsupported模拟POS能量2.19%沿q=1是显式先验，未证是主因。

[独立科学复核/下一步](GROUP_REGRESSION_V27_SCIENTIFIC_REVIEW.md) SHA `1b2f2801f37b326e7ee980ff333a1981c009e5b82196c1fd04ce4a646a368f33`是在actualaudit完成前写，原pending时间语义不回填。它核V9/V14没有等价的跨源MG→隐藏r长期粗监督head。下一轮拟固定Ridge/ET、q/IDs、MG law和H1残差池，仅扩合法源粗分项监督时间队列；短/长用同粗表定义及同发布截止，不能把截断proper bank r自动当作完整月统计嵌套子集。更多源标签的收益先归信息利用，不能当新机制；尚未V28设计/运行，未打开新源地区或未知区。

root在科学最终措辞审核前先封了原主文153ea251…与315映射原证据manifest `6e5d941daab8925439692d1047c851c1906d0ccdaa8720a532b726d0b8dc68d2`。science随后指出“未使用小时CIF”应明确预测输入与允许proper源监督及seal后评分。保留原文件原字节，新增仅措辞澄清的`GROUP_REGRESSION_V27_RESULTS_R1.md`，当前candidate SHA `a67a1e831807ffaaea5fd5d03ae1a76c96e5aac30a89faf2072d8bc0261d643f`，没有producer/audit重跑或数值改变；等最终科学PASS再建立R1证据闭包，不回填旧manifest。长期目标仍active，暂无原创机制突破或未知区授权。

V27 R1最终科学PASS `results/group-regression-v27-results-review.md` SHA `b7b892f8cafc176bd7616a58617ee9398460c56bf1e8372bb416d1f800ea558e`绑定[当前结果稿](GROUP_REGRESSION_V27_RESULTS_R1.md) `a67a1e831807ffaaea5fd5d03ae1a76c96e5aac30a89faf2072d8bc0261d643f`。R1仅输入/源监督/评分措辞澄清；旧153稿/原6e清单不动，未修改代码、数据、评分或首审计。新[当前证据清单](evidence/group-regression-v27-r1-verification.json) `25f0df9f032d1d0d8ecd1169be6ed2ebc06700bf5e5ec26f3d3ce19de8f4b18e`封5code/276input（formal275+旧清单1）/1auditor/35artifact（原34+新主文1）=317映射，numerical_experiment_rerun=false。最终generic hash/库存核对进行中，未提前签该sidecar PASS。V28尚未执行，目标仍active。

V27 R1最终独立manifest核验已PASS：317映射/314不同路径hash、原315条目保留+恰2新增、正式freeze仍275inputs、exact8根+14prediction库存、父aliases/收据/五输出、首次audit98模型/0skip/0qualified/firstexit全部一致。当前R1清单25f0df9f…与当前R1主文a67a1e83…保持原字节；清单外sidecar `results/group-regression-v27-manifest-review.md` SHA `314a5e22efd0b18117efb0af81d3bcfcd895e26813f418dee581800f7380e621`。没有重跑、解析新fine/目标或sealed改动。当前轮属于有证据的progress，不是目标complete；下一source-only控制应优先验证同算法长粗监督，相同query信息预算、q/IDs、残差抽取池及粗预测法则全部保持，真正未查看地区仍关闭。

### V28进入同算法源粗监督时间窗控制（尚未正式运行）

上一目标轮是progress：V27正式四阶段及完整独立重fit审计/科学解释/封存均完成，结果改变下一步选择，目标未完成。本轮保持长期目标；固定q、H1 bank残差响应池、MG law与IDs，只训练source coarse supervision heads三个同表嵌套窗口。S24为2024Jan–Jun，L24为2022Jan–2024Jun，LR24为2022Jan–2024年底且period_end/release均≤部署2025Jan1；加入较早与较近期监督分别解释。outer/teacher全部粗历史排除；source月全日proxy标签与bank eligible-support月标签不同，S24相对旧V27不称纯长度收益。

协议COARSE_SUPERVISION_V28_PROTOCOL.md锁21arms：原MG/V27 RidgeResR/V27 ETResR exact背景3 + 三window×Ridge/ET×pluginR/residA/residR共18；105curves/120225日/735region/105macro/28pairs×35=980contrasts。新增294final/LOSO固定模型，旧98可hash继承不重fit。固定Ridge1/ET128leaf2与mean1源均衡权重，无调参；承认N改变有效正则和平滑，信息/近期样本收益不冒充新算法。无新增本地annual/fine/weather，未知区不开。

初实现31合成通过后root发现standalone query允许2024Jul但head已使用Jan2025截止数据；正式runner固定2025无实际泄漏，正式前已加query≥cutoff首check，Dec31 regression用所有headpredict哨兵，不能仅依旧registry guard。当前32combinedtests root真实exit0/2.95s，五pins：module `8cbc045a2dc36a5fba0301576952ad23a610ffa4c56347d21779471113c71718`，runner `d07c2cd5262b973fbfd95eb5172467a66c0f71793271d64b5c03ca7655c6c1bf`，puretests `def7d521a394cac8938902bdc1ebe3bebeb44910ad210408ab6ac4aeb356234b`，runnertests `4c1f1f7aa42a44f7e26d76f74aff5659e644a0c551d14a54ad608bbac15ecbba`，protocol `b7b4a11c82a2304d6d4e3c8827630e56ab3616c07c6a212ad68bc454fca1817d`。三预审进行中，无formal目录/收据/结果。

新增原始文献核查COARSE_SUPERVISION_V28_LITERATURE.md：JMLR2026 rainfall downscaling只核出版摘要；2610.19885v1及2602.11893v3核部分HTML方法/设定，标明预印本/workshop和阅读范围，不声称复现/超过。两阶段分边际和依赖、月到日生成或解耦上游forecast与fine模块已有研究，不直接作为创新。V28只是判断更多source coarse监督的价值，尚未找到超越ordinary完整强参考的原创机制。


### V28首次正式运行、独立重算与最终科学批准完成：长粗监督有效，原创机制尚未成立

首次prepare/predict原session17369真实0/0，score原81207真实0；actualaudit原79147真实0/PASS。5code/318input，三窗口每outer42模型，总294新fit，固定H1 bank36月；七档全部seal后评分共同1145源日。freeze024f6412…、sealcded0457…、completion0b68a56a…保持原字节。

独立auditor最终SHA2b776b013db50ba47060741e1d9069ca1aaa09e4a3f52ac85a0837c5d050619e，经独立预审和最终selftest后首次actual PASS；完整294新final/LOSO独立重fit，三旧背景八字段exact继承（不重复旧98fit）。全21臂81,736,704raw最大差5.69e−13，四表120225/735/105/980和选择均核，1470预定日曲线pairCRPS最大差5.69e−14；347首尾hash不变。报告b88efba03ab0cd9551504e15ef02cee09034917e82923edca642e5615c3223de，actual log f8f682722a0079e158d9854eba88d4fbaa16a6b0a50b17e0ed465c5f7e2ae804。原selftest日志保留旧候选hash，r1日志绑定最终审计脚本，无实际数值重跑或失败回填。

固定f1.25，L24ET残差重组对S24 CRPS37.414→33.340（−10.89%）、宽265.025→248.724（−6.15%）、IS298.348→261.880（−12.22%）。6/7地区CRPS改善、7/7IS改善；MISO贡献67.45%，其余六区平均CRPS−1.547。MISO自身覆盖71.84%→93.85%但宽285.578→343.473，形状MAE基本不变；宏缩宽不可当成MISO缩宽。LR24近期额外粗标签未产生同等ET宏增量，不能据此证明近期信息无价值或因果识别稀有状态支持。

全部105冻结曲线仍false：L24ET重组宏95.2697%超过95%上限0.2697pp，worstBPAT87.36%已不欠85%，不是总体欠覆盖；不后验插1.2或放宽门槛。没有合格离散倍率不代表连续或条件校准不可行，后者尚未独立检验。相对同预算L24ET简单加性锚，重组CRPS仅−1.22%、宽+2.98%、IS+0.69%，尚无专门方法增量；插件MISO覆盖60.04%，窄不能算成功。收益首先属于普通方法利用更多合法source粗标签，不是原创算法。

[最终主文](COARSE_SUPERVISION_V28_RESULTS.md)SHA6f815e2c676c86e97860d00f054ef6d5fda3b8049739e52c136d4ac40011fc6e，最终独立科学PASS sidecar cc9afc06cfc6b16833e7c8837371bd392ec51951219af1c658e4065089697f8d。科学批准后才创建[证据清单](evidence/coarse-supervision-v28-verification.json)，SHA18cbd15d32d76801c0467b43370916c582237b6b45014a11971f8918a32c2a51：5code/318input/1auditor/37artifact=361映射358不同文件，exact8root+14pred。独立清单核对正在进行，尚不提前签该sidecar。

用户最新确认所有可再生和不可再生月发电量均有，采用完整分类月量/同口径总量的MG保守预算；仍不默认火电内部煤气油细分，不新增本地小时/天气/年度细分。source粗监督可延长、q小时池保持短期，是已确认有效优化方向。下一轮先补同预算普通联合条件分布/异方差强参考，把隐藏燃料水平风险与小时依赖分开诊断；任何源粗校准不得冒称小时覆盖定理。2025开发响应不当新确认，未知区仍关闭，V26未执行，长期目标active。

V28最终清单独立核对已PASS：361映射/358不同文件、318父派生输入、36 consumed aliases、exact8root+14pred、原收据五输出与先后关系全部一致；实际首次audit报告/log scope与294fits/81736704raw/1470pair/347tracked相符。新清单外sidecar `results/coarse-supervision-v28-manifest-review.md` SHA `36009570b57548bf0ab359144e30bf43a190d0628d15908ff179221e20b58baf`，仅hash/JSON/报告核对，无模型重跑或新fine解析。主文6f815e2c…和manifest18cbd15d…保持封存；本轮为progress，不是目标complete。

新增下一轮[条件分布DRAFT](CONDITIONAL_LAW_V29_DESIGN.md)：同L24/五维X/MG/q/IDs/倍率，保留三个L24背景（含简单锚），拟新增Leaf整向量经验分布、全协方差Gaussian、q-paired位置尺度各配锚/重组；可选预注册q置换。源teacher树结构/叶池/scale均严格重新排除，49森林/窗。不增加total_mwh或coverage特征，不把粗概率校准当小时保证；这是结合两位Sol只读建议的草案，尚无实现/正式合同/运行/新结果。新增QRF及DRF原始方法、两篇2026预印本摘要的阅读范围明确，未声称完整复现。下一目标轮从此草案继续，当前目标仍active。


### V29实施组成联合分布／异方差强参考（尚未formal）

上一目标轮为progress：V28实际结果及完整独立核验封存，长粗监督有效但普通锚已接近、创新未成立。当前选L24开发窗口，原五维X/MG/q/IDs/128/五倍率不变，新10臂为三个L24ET exact背景(pluginR/residA/residR)、Leaf整r4经验分布A/R、同全局混合叶权重Helmert3D Gaussian A/R、qpaired Scale A/R及fullqshuffleR，共50曲线/49新ET森林；科学边界和旧目标/未知区关闭不变。

五最终pins：module4163feb191436b110cd13d2d8e0bfae0585bafedb9af915adc94cb5f3f7d8854，runnercf1a0c0a83eff3c33fb131639126f5af15c1016664cb8948e30e404690591fb2，puretestsc8d4a16202197614e4643ce6d092e2a58e10703e1ac2e65a73dffc9e3417dd84，runnertests1bd7a7273c595ddf256cf7e132d81cf5bbec8b3a8784c73dc70a1e737579ce3b，protocol34ea74eea5e7741047a6251d6fb9b95945b5a1f38e24a320352fa06f55ff6bf9。root最终43combined tests原55012真实exit0/7.18s。原作者首批test中Helmert单位数值容差1e-16过紧导致test-only失败，改2e-16，非数值机制或正式失败；补queryPOS0正例后17pure通过。generic指出predseal未核真实库存，正式前修exact14regular非symlink并新增四个beforetruth拒绝回归。

独立generic root三文件APPROVE `results/conditional-law-v29-code-review.md` SHA b29c0a0ed64611452e184083b25d1b85fdc8f742cb1d42fd04e4f19e177da0b2；其纯module/tests由该agent编写，不自批，仍待独立Python/math和science最终5文件审阅。当前没有formal产物/新实际结果，不提前称风险修复或原创性成立。


V29三最终预审现均通过：generic b29c0a0e…、Python/math a9c660d3d2cd0f755652d4def585083e06d3f7ef962802eed48cb6d3e73472c3（独立7ET/61440合成路径另核，maxraw2.274e−13）、science af0286979cc049b33e98b0df57ac97da6ddbea621ff05bd410c1c7e1ddb43eba。首次prepare/predict原21704真实0/0，score原14656真实0，5code/362input。freeze70d785a281d08ebb6528deb68b8877fea312029b29dba4677f91c921f939ff6b、seal dafa93724d15cc66ce0be474f4330f4a92186eee83b3eb0a528665c839ed4874、completion076cb7ddf4c75e8a8cce0cbcd39c12eef9438f568756ae8d65447e275b21db33。七预测全封存才读源2025评分，共57250/350/50/560行。

仅Scale-anchor/f1的1/50曲线合格：CRPS33.729310、macro92.6912%、worstFPL85.0559%、width223.260240、IS256.137630。FPL覆盖3654/4296小时的门槛余量等效2.4小时，少3小时即失败，不叫稳健确认。同f1对旧anchor CRPS+1.409%、width+15.037%、IS+3.891%，更高覆盖有成本；对旧f1.25 anchor width−7.563%、IS−1.519%、CRPS仅−.0629%是跨factor描述，且旧anchor宏95.51%原不合格。不触发≥5%CRPS自己的机制门槛或unknown。

Leaf/GaussianR f1.25的MISO覆盖75.18%/74.60%，仍显不足；Scale-anchor1 MISO覆盖98.54%、CRPS55.730、level64.372，ScaleR1覆盖99.47%、CRPS61.704、level73.595；R形状改善却其它风险不合格，无法把普通条件月散布等同地理失配。Scale无near-floor且regularizer最大约2.86e−10，MISO/BPAT投影率88.64%/94.35%，不是纯方差修正。独立科学快照 docs/CONDITIONAL_LAW_V29_SCIENTIFIC_REVIEW.md SHA88fc63ae34bc34481f396fe134c1c758f8b224fbb827be8fcb396bc691e84c1e已核这些数字，actualaudit仍pending。

源已存bank折外矩的posthoc252相关记录诊断：teacherMISO平均EF signederror+132.516、leafSD61.635，±1.64485SD仅55.56%；是Gaussian散布近似而非Leaf经验覆盖、非252独立点。提示长期粗标签下的空间偏移与同区月散布需分开；不能据此宣称新机制。已核V16有标准IPF/共同hour缩放不变控制（原本地lagG预算不同、default .001松支持），V18有细packet普通中心投影；不重新包装为原创或把额外信息带进本预算。主文CONDITIONAL_LAW_V29_RESULTS.md尚未封存，实际auditor候选在预审，无实际审计启动或失败。


V29独立actual原57407首次真实exit0/PASS。最终auditor199f72150ee7b0f5291c2fc9673ea881ecdc8c21374ae7db5fb1e6a700619029，generic preflight b9b0804bcf4771cdf7962a1a6f5f452d600b66dc37b7e036c1af2f69ef49df4a。报告e97a445e84df1b494ce7223ae2d7879e3f955c44801fa9fe3da0181ead8ba082、actual log2037b4cf5aeaf7d5bea997a3b656a2ba3e498e0add0db40eacd5567b3ca94de0。49新ET全独立重fit，391首尾hash不变，10臂38,922,240raw核验max1.592e−12，独立叶矩/尺度max2.776e−16；50曲线和57250/350/50/560表/选择全部核，700预定pairmax5.684e−14。背景三臂八字段exact，旧V28与银行不重训。

实际前候选9ff4038d…的近floor Scale独立trace1ULP经除1e−6会被放大，codepeer合成发现正确产物可能被误拒，修改auditor而非producer/算法/公差/产物，保留旧selftest logs；最终新候选加近floor与阈值舍入回归，root与codepeer独立selftest真实0。Gaussian和Scale均先独立核moments/scale/residual至1e−12，再用已核stored μ/cov/s/delta作非线性回放；delta定义、有效source lookup与scale ratio除法exact，报告明示不称从另一舍入矩端到端完全同路径。合成另23,168fallback的181日全路径通过。无actual失败或重跑。

根节点主文更新actual及ERCO损失/MISO收益同factor对照，最终候选cb3d985b89f3edf193011bce095c57a685488bb54742f855d5852cc088e7adee；当前等最终科学审定，尚未创建新manifest，不提前封主文。所有源/真实预算/unknown限制与目标active不变。

V29实际核验后，新增下一轮仅DRAFT COARSE_CONTEXT_V30_DESIGN.md：优先检验完整公开月量中未用的合法滞后组成/年龄/过去绝对量能否区分隐藏燃料水平，先源粗尺度地区排除并samecohort同学习器，禁止当期实际总量/隐藏coal细项/本地未确认coverage，第一阶段不新读小时响应；若有信号再进入固定q小时控制。无新实现/运行或效果宣称。标准IPF与中心projection保留为普通归因候选，不冒充原创；未知区仍不开。


V29最终科学PASS `results/conditional-law-v29-results-review.md` SHA be5a8a07ded76457f861de5916b2c2baafb06eaef59cf1570d887e85df6c9b5a绑定[最终主文](CONDITIONAL_LAW_V29_RESULTS.md) cb3d985b89f3edf193011bce095c57a685488bb54742f855d5852cc088e7adee。科学批准后新增[证据清单](evidence/conditional-law-v29-verification.json) SHA7512d5e67a21073e42150f7ffb3b379cdd0d03f1061bdcff18348a8c9253b35f：5code/362input/1auditor/42artifact=410映射/407不同文件，exact8root+14pred，保留6个preactual selftest历史log。最终独立manifest PASS `results/conditional-law-v29-manifest-review.md` SHA93af91e150100b8060587291be77a24253a8eead9ebcda2c74cf8048bf453a18：全部映射／362父派生输入／36消费aliases／收据五表与时间顺序／solequalification和staged范围一致，无新数组或fine读取，无重跑或sealed改动。

本目标轮为progress：新增普通联合条件分布和异方差的50真实曲线、首次actual49独立重fit/全路径评分复核、solequal但脆弱与ERCO代价的证据改变下一步。不是原创机制成功，不是未知地区确认，也不是长期目标complete。下一goal turn从V30 DRAFT合法月上下文可识别性实验实施起，先粗尺度samecohort强参考，不重复V29运行或仅重述状态。


### V30中断恢复与真实粗尺度上下文实验

此前被主动中断的V30启动轮只派发/确认设计，没有代码或正式产物；恢复时rg只有DRAFT且liveagent列表仅root，不把当次派发计作实质progress。本轮沿MLE技能并行lane，重新用Sol实现module/tests，root runner/protocol，独立code/math/science三预审。五最终pins：module70d945d1b43217df6d08f51c5bd6b3edb36ef2414d876da577e75128f2abcff8、runner0f5da71c19d5b29ffae0ed5a44bd2762564f3122858c444bed9993054c617d6c、puretests1e4b1f964c5f16eb5e8017bd53231d3cb6115abcc3b26b9b048fd6eb524c0e88、runnertestsfde7fc9f50ce7e1a594e72cc1502c43ecc5d8b9104682e4fa0436c40b5dbd9f4、protocol668b8a57662b4c3bdf87c6e43988eac4e7673f223d8f0b4b5b1e4761eae361c2。35combined tests原73529真实exit0/1.50s；正式前补tiny负POS补集roundoff和六源强约束两个真实边界，无实数据运行失败。

三最终独立报告code16227b32e915b30b9d2d06a23eea5c4a739ac33642c653907e32adae0fb8581c、math ef7b76ca26c12a77109007967766cc45b154a217b3d8acc843b9bcd1da381da3、science51b9db1472c8761529f25fa39184ae73562c0ce12897b48a179dfb4c63f0d86d；另独立全合成70fit、weighted Ridge正规方程、ETraw、feature/leap/context、1800类coarse指标验证通过。唯一formal原97445 prepare/predict/score真实0/0/0，一次观察beforedispatch aborted后只重poll原handle取得终态，没有restart。freeze f98553bfc804275c0a4f850ef96ea977b18badf631b6eeb09c2797c122db9e63、seal37e2b58d2edc7a7187b24da94d659423c0d040c737f92f5a783a4e3ae3d77c1a、completion0bb2de113afedfc3004d3bac81a4171e8c3fd5413112a2080c80a6c536f3848b。5code411input，180共同coarsecases（各26,FPL24），1800case/70region/10macro/133contrast；当前g为已实现publicG5的oracle逆向条件，非未来预报，无任何rawhourly数值读取。

Size-Ridge EFMAE58.639→27.326(−53.4001%)，SizeET64.671→50.309(−22.208%)；季节/历史/趋势未给同等收益。Ridge ERCO143.182→21.440、MISO160.826→117.759，5/7improve，CISO不变；但BPAT对Trend1.132→1.805(+.673,+59.41%)违反冻结relative10%规则。SizeET6/7improve但ERCO对G5+12.27%、对Trend+10.78%，亦fail。因此14signal全false、原规则不回填，不触发小时/unknown。MISO26月全部同向低估−117.759；r4L1只−10.16%，EF改善不等于燃料细分/小时风险识别。

首次独立actual原89756真实exit0/PASS，auditor85a649dc059d119ec12b3430402391ebe296d05bb71d5245f6f920963fd19e5c，经generic审计预审12cf08c42bee86658713bc70f14615c2fbe84f800db70361937bc97e631ca6a9；候选与root合成selftest真实0。独立70fits（35加权未罚intercept正规方程+35rawET），maxRidge2.70895e−14、ET0，439首尾hash不变，所有cohort/features/scalers/metadata和1800/70/10/133/14decision核过。rawrefit核后用已核stored输出重投影/评分保留近零，不称另一舍入bit-exact端到端。actuallog完整JSON c0d0b697dfb2bac3c9cfc9ebaf3bc3adc401e8ca7a4a0a76dd19fb7f2725301f，root摘要report e613b8aef3c2ec969a83414bea5b53576f726cce7bb1ec947b77b1c41625ffd7。checker不用任何V30producer或hourlyhelper。

后验两探针单独声明，不入原70fit审计或资格：固定SizeRidge系数k=.5/1/2系统复制假设stress21行，k2宏EFMAE27.326→85.432、预测均变化+70.852；这只测边界/规模代理敏感性，不是真观测或因果否定。直接clip(rawr·EF,400,1000)代原simplex后EF的35行负控全部family更差，Size27.326→39.009，ERCO21.440→92.794，否定“先保持rawEF就会更好”的简单想法。无新训练/原月表解析，真值由已封EF−signedscore恢复；科学review独立重算全部56行误差<=5.69e−14。

最终主文 COARSE_CONTEXT_V30_RESULTS.md SHA58e7f9a8177f71e4afd549993681ac7185a6bfa325322759516b8c0599973e54，后验说明f4e72ae355226f679b1c05a3545f23ea3d5c2bb0489087c9e8d9feb13f5419d6，最终科学PASS14551efdd57d2bcdebb4a9008c93d5bc526fe13d4efefc99f237d86784a7e1e1。科学后新增证据清单 coarse-context-v30-verification.json SHAb3590c85d46d99510403432a24c658f1df85137831a3c46005054f0df994cf4a：5code411input1auditor41artifact=458映射455不同文件，exact8root14pred，41含2probeCSV/6docs/2selftestlogs（原tmp候选日志原字节复制，不是重跑）。独立清单元数据复核正在进行，尚不提前称该sidecar完成。

新增FORWARD_CONTEXT_V31_DESIGN.md仅DRAFT：锁V30的G5/Trend/Size×Ridge/ET，2025Jan–Jun七源月初用原MG forecast粒子代当前g oracle，pastcontext按起点合法更新，不重训2025隐藏label，先评完整月发电侧代理强度与内部EF点误差，所有预测封后score；不读hourly、不把MG-only散布当完整概率区间、不称盲地区确认。当前V30已产生新实证并改变下一实验选择，属于progress，尚非原创风险机制/小时增量/目标complete；unknown仍关闭，V26仍暂停，目标active。

V30最终独立manifest核验已PASS：458映射/455唯一文件、仅三预审跨组重复、408父路径+3新review=411输入、exact8root14pred、收据五输出/全预测、actual439首尾当前hash/70fit/1800-70-10-133/14false及oracle/无hour/后验scope全部一致。新sidecar results/coarse-context-v30-manifest-review.md SHA4a97d75033b0d9e9cd3a49101b7df6cf1aa9179a4a79595ae2ad69a87cecdd38；只hash/JSON，无新数值解析/重跑/封存改动。下一goal轮直接落实V31合法未来g替代oracle的固定方法源月度验证，避免重复本轮已完成结果或只重述设计；目标仍active。


### V31：月初预测验证与写作规则

用户指定的 Humanizer／Humanizer-zh 已安装至 `~/.codex/skills/humanizer` 和 `~/.codex/skills/humanizer-zh`，已完整阅读并用于本轮报告。以后中文报告、英文论文稿都按对应规则减少空话和模板表达，保留事实、限定、归因和证据强度；不为语言修改旧封存档案。

[V31 主文](FORWARD_CONTEXT_V31_RESULTS.md)记录本轮正式实验：最终49项测试在 pandas2.3.3／3.0.3均通过，显式纳秒修正正式前发现的兼容错误；三预审后唯一原15836 prepare/predict/score真实0/0/0。5code／459input，42冻结head重现，不训练2025隐藏标签，42月初预测先封存后score。35共同合格月（6/6/6/6/1/5/5）、210case／42region／6macro／63contrast／42support。ISNE只有1月，不能称完整半年确认。

Size-Ridge CIFMAE38.735607→28.002090（−27.7097%），但ISNE30.558932→44.993572（+47.2354%）违反固定地区损失规则；对Trend单项通过，联合Size仍空。SizeET的ERCO／ISNE仍损失。MISO SizeRidge五个月CIF全部低估、内部EF signed−129.731080。普通合法上下文有价值，原创机制、可信小时风险与未知地区确认仍未成立；目标active，V26仍暂停。

首次独立actual原98270 exit0／PASS；auditor f226f88e817796edb1a6641f9695cf37c84565ecdc2b0e01c0da50e8ca417ac5，log0db3167322d0bec69d981b9047dccbbef41bb6431f85028df3134f129146b02d。独立42refits，Ridge raw最大差8.64309e−14、ET0，488首尾hash不变；staged storedraw／storedpoint回放范围见[审计摘要](../results/forward-context-v31-numerical-audit.md)，不称不同舍入次序逐位一致。无原始小时、天气、旧目标或未知区新数值读取。本轮有实际数据推进，不是仅复述计划；最终科学批准／清单状态见后续追加。

V31最终科学审核PASS，侧件SHA8f124e15d8464e8085c1b77b95d71d49245ebee2a0736b552847031cd43bf5a5。新[证据清单](evidence/forward-context-v31-verification.json) SHA f6544892fc7bf24256cadd5809b3b9fcdfde60c4e42bde9bfa13db41ea9b46df：5code／459input／1auditor／36artifact＝501映射、498唯一文件；本轮将最终science sidecar纳入artifact。独立manifest-review PASS，SHA3d12eff268005f198ef0f12646c82393fae68a613b7a497ec50da1ffb313f0c2：只hash／JSON，无新数值或重跑。主文0cc4c94b696ee8f77392a880886d774d68b6d2bd1d647d6cee593a1ff67bbe6b已封。下一轮从主文提出的源内排outer门控／回退强参考和MISO持续偏差定位继续，不能按35评分月挑模型或重放本轮当作进展；真正原创机制仍待发现，目标保持active。


### V32：长历史分层检验已封存

上轮为实质progress：新增九族18臂126fit，metadata潜在117最终也是117，统一12条合法历史队列；唯一formal原27180 prepare/predict/score0/0/0，5code502input，42forecastorigins先seal后score，exactV31同35labels/630case126region18macro161contrast42support。40合成tests两环境通过，3最终预审。JointShapeET MAE37.167→30.284，但LongTotalET已30.673，JointTrendET29.867更好，phase机制未获支持；Ridge31.368比LegacySize28.002差12.0204%。20信号仅3单项true，phase/practical都空，MISO5月仍低估、ISNE1月不用于地区专属调门控。

中断在actual启动前，恢复核无进程/无actual日志且原pins不变，首次actual原22697真实exit0/PASS，auditor dd0984743aee0bd14865a3d058106e3e4b42ed074d345e0514eaa878a753f1bf，log708c794583c3868e96de627e1404cca4afb6ad883a66221dc0d5227d31a31158。独立126fits，Ridge max2.75557e−13/ET0，531首尾hash不变，staged raw/point范围见审计摘要。最终科学review2a91d23d6e01553e78e63a04b1379ac4788a68e6dfa95543a6126ecdf5e976df批准主文ef6b88a123a922ba3d05fb929d3b0083d8109117f579a00b9346498a8cd7da2d。[证据清单](evidence/history-signature-v32-verification.json) SHA8e3cbfe354deab91e4f6f42e518f8340677c075339a88a55558791fe1d081a4a：5code502input1auditor37artifact=545映射542唯一，独立manifest-review133aa0a352240916a7c5d8e039e4421e44737e17f028a67e4d75c5fb2cd7f86d PASS。无sealed修改或actual重跑。

新V33仅开始官方装机数据获取与表头审计，合同FLEET_INFORMATION_V33_DATA_PROTOCOL.md。B0公开月报保留，B1额外公开装机必须独立比较，容量信息胜利不等于方法原创；先核2020–2023官方公开时间、当前ZIP修订proxy、BA映射/多燃料/状态及口径。当前不转换任何容量数值、不训练、不打开旧目标或未知区。EIA nameplate可低于net summer capacity，不能作认证未来出力上限。V26仍暂停，长期目标active；下一轮不能重复本轮模型或仅叙述计划。

V33获取／字段阶段已有实际进展：官方2020–2023四ZIP均已exclusive获取，字节与HTTP size一致且CRC通过；12个workbooks、28个sheet表头已核，四年所需BA/机组/主能源/容量/状态/多燃料字段存在且对应表头一致。schema-audit SHA ba38bb6ab5303e2662a76e85d394b77252532e95186de2dd8dead3aa1de45fe5；独立schema-review4552fa01733b2961caa2e650cf093dab627798c2ffdd7898e66d31e4882b58f4 PASS。未解析地区rows或容量数值，未训练B1模型。

[字段结果](FLEET_INFORMATION_V33_DATA_RESULTS.md) SHA8c6afc424a4e0c3410dc056f06cd7c88c5da2c0a09939156304ef25647dfc049；[获取证据](evidence/fleet-schema-v33-acquisition.json) SHAbfd20570f0b4c58e05108c93a1aaa41b0a95874574fd8e32e116626596ee3398，status仅METADATA_ONLY_VERIFIED，12artifacts，不冒称模型验证。独立metadata manifest-review2102224319325a82e04d284bd65d3266fa4d093aec1b8b157937f78ea5c7a681 PASS。当前检索ZIP的Last-Modified均在2025部署前，仍不等于独立认证原始vintage；当时公开记录在schema-review内保留标题/日期/正文年份，未存immutableHTML。

下一goal轮直接预注册并实现七源容量数值整理：先BA筛再容量转换、唯一机组、主燃料/状态/多燃料/缺失规则、origin时点选择可得年度；如用year−2须验证2022→2020/2023→2021/2024→2022/2025→2023各已公开。公开月报B0与额外装机B1分开，不把附加信息价值算算法创新；普通cap锚与同B1回归须保留。不得重复下载/字段检查或重跑V32当新进展，不得把nameplate当可用出力硬上限。旧目标/未知区继续关闭，V26仍暂停，长期目标active。

## 2026-10-26：V33装机数据整理通过，V34信息价值实验待执行

V33首次实际整理原session41026真实exit0；首次独立actual原session97658真实exit0/PASS。形成49519条跨年度非储能记录、1468条储能metadata排除、OP与Operable各28个源年。独立Decimal重建核对全部unit/排除/汇总/audit，573绑定文件首尾hash不变。独立结果文审阅通过，manifest为`docs/evidence/fleet-inventory-v33-verification.json`，SHA `f3649828c812a19bcc81371c3f1960239fc6c8572962afd61c6bf7be127aebe8`，585映射/582唯一文件；manifest metadata审阅另记sidecar。

本轮只是B1登记库存代理数据完成，未训练CIF模型，未改善覆盖或认证物理支持。未知地区与原小时标签仍关闭。V34已写预实验合同，以原V31的42查询/35评分月和六源训练cohort，比较11个臂：原Size两头、两容量锚、学习利用率、SizeCap/IntensiveCap/DirectEF各两头。计划49个新fit，只检验附加库存信息价值；实用筛选通过也不自动构成创新或授权新地区。本条写入时未执行V34真实fit，不预报结果。

## 2026-10-26：V34完成，装机锚最低CIF MAE仍未通过迁移筛选

首次正式session53701真实exit0，prepare/predict/score0/0/0；11臂、七源49个新fit，42查询和原35评分月，385/77/11/168行结果。两原Size臂逐值复制。独立actual原session38395真实exit0/PASS，不导入producer，独立49fit、全部粒子/端点、训练记录与所有评分筛选通过，616绑定文件首尾hash不变。未重跑正式模型或已通过实际审计。

Capacity-OP macro CIF MAE18.629763，优于原Size-Ridge28.002090（−33.47%），但EF MAE48.315235劣于36.461059（+32.51%）。MISO CIF MAE72.990792→22.7384，ERCO19.9210→26.6850，BPAT/NYIS也退化。九个新增臂均未同时通过两旧强基线筛选，B1_signal_arms为空；所有新增学习头仍未胜OP锚。该锚是附加信息普通基线，不是方法创新。

事后精确分解385条error=(gbar−gtrue)*e_true+gbar*(ebar−e_true)+Cov(g,e)，最大残差1.6342482922482304e−13；独立结果审核另行复算（不混入原actual616范围）。ISNE CIF改善含POS低估与EF高估抵消，EF绝对偏差反而50.5899→54.8270；MISO则有明确水平修正但仍偏低。不能仅依CIF宏均值认定恢复了真实运行组成。

结果/科学复核通过；manifest `docs/evidence/fleet-information-v34-verification.json` SHA `c9e6a6f4f91465407268d5dbe7a0ba56eedfaf8fecc1a5bad5c643ff4259437e`，630映射/627唯一。独立metadata sidecar `results/fleet-information-v34-manifest-review.md` SHA `1147ce2f46a12297260a5d03526df84cbed1b2fb9f526db686785574166cab57`。所有V34输出/代码/协议/报告及V33封存保持不变，未知地区和原小时仍关闭，长期目标active。

下一候选曾考虑“训练真G5、查询forecast粒子错配”。经独立数学反驳，若MG正确估计p(G|H)、head正确且可迁移，积分G_POS*E[EF|G,H]正是全期望，本身没有错误。不能把past-only或oracle收益当成该因果解释；将独立forecast粒子配给实现标签还可能抹掉条件关联。新的未执行设计见`docs/FORECAST_ALIGNMENT_V35_DESIGN.md`：先以既有35月做MG分布与固定head残差诊断，同时检查均值/中位数和MAE/MSE的目标差异，再决定是否做四臂past-only复杂度消融或专门机制。此时V35尚未开始，不预告创新或覆盖增量。

## 2026-10-26：V35零训练诊断完成，简单输入对齐假说优先级降低

首次actual原session30834真实exit0，零新fit。MG35/7/1、mean/median点摘要770/154/22、固定四Ridge真G条件EF140/28/4以及42support；formal重建原42查询EF粒子最大差0。独立只读公式复算全表和649项hash首尾通过，最大差2.728e-11<预锁1e-9，review使用不同NumPy/pandas版本已明确；未另建auditor或重跑actual。

MG POS90%宏包含94.7619%，35月pooled33/35；80%宏79.5238%，pooled27/35，不能认证联合/条件校准。OP锚median CIF MAE17.75035优于mean18.62976，其EF48.31524不变；全11臂中仅OP与Operable两容量锚的CIF median改善，其余九臂均未改善。四个Ridge给真G后EF宏MAE均不改善：Size36.46106→37.33854、SizeCap45.61131→46.07990、Intensive59.01020→59.41027、Direct47.05844→47.26320。MISO水平低估仍在。这不证明coarse q正确，也不单独因果识别head域shift，但不支持直接投入粒子替换真G的训练修复。

结果表两处展示舍入与标题经独立审阅修正，数据/代码没改。最终docs/CONDITIONAL_DIAGNOSTICS_V35_RESULTS.md SHA1d31932b1ac43156b0c14740d6a4527d3d054f36b3337bfcafff50a26855731c；results-review SHAa91bb3a9bb3791edac289e5cbbbec629ddb0a2f077202bddebc7bcefa0c006ef。manifest docs/evidence/conditional-diagnostics-v35-verification.json SHA7518b714451d50fc6404b5c784e67ab827c583dd27516d4cc531786935f6e5a9，654maps/652unique，19artifacts/14root；metadata sidecar SHAaff8417c7ab3d7f213580411bc45bda15df97960f4c5072a66680d150d2cf550。其review字段为independent_formula_review_status，不虚称另一个actualauditor。

V36已固定普通baselines的候选合同：5个固定global blends与5个训练容量范围fallback，均用原SizeRidge/CapacityOP粒子，零新fit，mean/median全报；目标只是建立后续专门机制必须战胜的同B1强对照，不称创新或统计支持识别。当前V36未actual计算。煤/油实际发电比例是EF的非零方向，但不能推断gas/other容量无关；少量旧source-year极值不是统计支持边界，路由可能源级恒定。未知地区/小时仍闭，长期目标active。

## 2026-10-26：V36普通门控有点预测增量，尚非论文创新

首次actual原session11011真实exit0，零新fit，420routes/7limits/700case/140region/20macro/140去单源敏感性/42support。全部42查询先固定路由后标签；两端对V34原粒子/mean逐值保留，median在混粒子后取。37synthetic tests及两独立code/math预审通过。

独立临时公式重算10343项、980 binary端点通过，671绑定文件首尾不变，最大差9.094947017729282e-13<1e-9。结果文/图/收据已独立核对；没有另一个actual/auditor或大harness。主结果docs/CAPACITY_ROUTING_V36_RESULTS.md SHA86ace5192f8a2859fa75aab67c5da48e1c894b737e27db5ff13a2ef13248fb85；results-review SHA970879a1d14c537b203205c4c2e31343a2623ea830fdb86f3858c9b6a4db65cf。manifest docs/evidence/capacity-routing-v36-verification.json SHAa7d9466339a40a3db9ffd80747530553e35c3ec2d34d68187041f26789c80aee，679maps/677unique；metadata sidecar SHAf24586aef9bb6763d28e003980f100219f5ad3a9bfbceff2b9a4ccb7d8fdcc29。原11root及图文件保持封存。

CoalOilUpper mean CIF/EF macro MAE16.396137/25.708227，对原28.002090/36.461059降低41.45%/29.49%；median15.495588/25.607305，对原median28.972536/36.431601。两摘要都优于本轮五档fixed blends各自最小CFI和EF，但不等于优于连续全局权重或已学习gate。普通Range4 mean17.778524/46.426833，median17.169889/46.418459。CoalUpper=AnchorEFUpper仅选MISO；OilUpper仅ISNE；CoalOilUpper两者，Range4还选BPAT/ERCO/FPL。CoalOnly EF更好25.102925但CIF20.823177，ISNE的切换仍是CIF改善/EF退化与单月误差抵消。去MISO后双指标优势消失；去BPAT后EF也不胜某些global blend。因此实用点优化空间明确，但仍为源级、反复查看后的探索性普通基线，不能冻结原创方法或打开未知区。

论文图 results/capacity-routing-v36-figures/ef-cif-tradeoff.png及PDF已保存，两个摘要面板同尺、Coal/AnchorEF重合明确标注。图为posthoc展示，非原runner输入；receipt保存原macroCSV SHA和Matplotlib版本/图hash。

下一步不是把gate包装成创新。先补同B1的状态相关利用率/offset-softmax强基线，再决定专门的风险机制。只读检索确认旧PublicGenerationBoosting/ProbabilisticDispatchNet已有容量×状态解码，但用高信息weather/load/mix/cost/hourly监督及epsilon/hardcap，不能偷渡到本预算；V34只有全局3β+MSE。当前B1代码尚无masked soft-label CCE的plain/offset-softmax。建议V37最小3头：PlainSoftmax36、OffsetSoftmax36、OffsetInterceptOnly，各相同6源keys/y/weights与SizeCap36输入（常数头只学截距）；q为已封4容量比例，gas reference、0初值，λ=1加权和CCE+全部自由参数二次惩罚、analytic gradient和固定SciPy L-BFGS配置。必须明确q0/y0合法保零，q0/y>0报告key并失败、不加epsilon不丢样本；offset输出零不认证物理缺席。共21新fit，原V36完整Pareto直接复制对照，mean/median及去MISO敏感性全保。此设计尚未写执行合同/实现或跑真实fit。

另审查过逐燃料portfolio复制增广候选，暂不实施：正a才是乘法群，a=0降支持不可逆；只共同倍乘POS不等于八燃料整体尺度不变。细r'=closure(a*r)、q'=closure(a*q)为已有Aitchison perturbation/容量归一化；coarse G5变换依赖未知r，动作不能一般降到观测商空间，不能据此声称部署等变或新信息识别。新方法至少还要胜状态相关普通capacity decoder、同augmentation普通模型及重复样本正则控制。来源与数学反驳已由Sol独立审查，未实现或评价其性能。长期目标active，未评价小时风险/区间效率，旧目标/未知区/原小时仍闭。


## 2026-10-27：V37封存，下一轮检验粗历史反演

V37正式21个新fit及35源月比较完成，三新头均未超过V36煤/油门控。最大训练梯度L2为9.721608785647101e−6，负结果不能归咎于未收敛。独立参数重算覆盖全部粒子与评分，700原参考行精确保留；结果主文见UTILIZATION_DECODER_V37_RESULTS.md。证据清单SHA639d54297999aad035bedad5f9ba0795d9448af79be4ddda264637690aae62ad；最终metadata sidecar SHA dbf6b903e9d093ec4206a677141b90edee9e2a0ac1672836a87dd6d4de4f67a6，720映射/718唯一文件通过。仍未证明方法创新、小时覆盖或未知区泛化。

写作规则已再次按用户指定核对Humanizer和Humanizer-zh，减少套话，保留结论的证据强度，不改封存历史。上一goal轮只完成该写作操作，对突破目标无新科学证据；本轮开始新的实际研究。

V38合同COARSE_INVERSE_V38_PROTOCOL.md固定小规模证伪：Constant/Month源端名义利用率profile，None/Common/FuelRidge三适配。源标签键不增，但显式新增相同键的源总量响应，六新臂共享。42查询元数据检查均有12条合法历史，不过存在缺月与ISNE陈旧记录；不称连续一年。尚无新fit或性能结果。旧目标、未知区、小时继续关闭，长期目标active。


## 2026-10-27：V38六个粗量反演对照完成，粗量改善与碳误差分离

首次actual原session32177真实exit0，prepare/predict/score0/0/0；14源profile、84Common/84FuelRidge及84None，原42query/35labels。53synthetic tests和两独立预审通过。910父case数值精确保留，新增420case，总1330/266/38/266/42；210粗量errors、1440裸profileerrors、84operator诊断。MonthNone CIF/EF31.469995/55.383624，Fuel32.802333/61.577771，差于None与V36门控16.396137/25.708227。Common粗量源均衡MAE3855.22→2071.17MW，EF/CIF逐值不变。去MISO后Fuel仍劣None，不能归因单源。

独立同正式NP1.26.4环境临时公式重建通过：14profile/42history/252已存γ目标-梯度/全部粒子和caseexact，聚合max3.637978807091713e−12<预锁1e−9rtol0；168优化解最大KKT1.3704315460216776e−16。53direct/code/formal+718父闭包union753首尾hash不变。另168行posthoc碳泛函方向敏感性2352数值max8.32667e−17；MISO MonthFuel梯度84.63%能量位于经验弱方向，不是目标operator误差界/有限EF范围/覆盖保证。最终主文COARSE_INVERSE_V38_RESULTS.md SHA024e3a4efe8c0236329dbf954d31d0dafc0cd1976078f537f4c4513675844da2。最终图用figures-r1，原第三panel把None识别NA漏点，已明确superseded，数据/代码未改。结果review/薄证据清单正在封存，不提前声称sidecar完成。

V39仅DRAFT，未训练：考虑嵌套source-only持续燃料误差分配的普通mu-only/isotropic/diagonal/fullcov控制，不能称Gaussian/GLS/MinT新。实际容量mask支持预查新得到：按2023-07分界，每fuel两半段都至少一个定义率时，outer BPAT/FPL/NYIS只有4个完整源向量，其余只有3个。因此不应直接上完整4D ratecov；先检验较低维碳方向持续性，或单列定义完整的G_f/C_POS贡献残差对照，禁止把无定义G_f/C_f填0。表SHAa32ea5660f21765397d82deaf2638bfd6839e570fbcbcc6533949e34a61c01d9。它是支持metadata预查，无新fine值/fit，不是方法增量。

新先例已核：COLT2023泛函识别现有JRSSB88(3)2026期刊版本；LatentCP2608.03607v2允许没有latentcalibrationlabels，但需knownforwardfamily，不能直接移植到跨区学得偏移operator；TaskUQ仍需真实任务输出校准及exchangeability。下一轮研究机会只在联合过去/未来算子误差、粗约束可收缩碳方向与小时未识别风险的机制增量，不能泛称latent反演或functional创新。小时覆盖/宽度、旧目标、未知地区仍未打开；长期目标active。本轮有实际新实证及支持约束改变后续选择，属于progress。

V38最终结果报告SHA62908c8f066d8b86346d14c4341c45bfc7f0d84025e6a4c39060f9187d0804e6，独立数值/文图PASS。薄证据清单docs/evidence/coarse-inverse-v38-verification.json SHA1a3b73183848a5a0b06bfbad1e00d9bc5a9e46f97dfefb42c6e2ec28bc042f36：5code/21directinput/39artifact共65maps65unique，显式引用父718闭包，合并765唯一，未继续复制历史闭包到新inputs。39artifact包括27formal、3logs、posthocCSV、主文、review、finalr1图3及superseded初图3；V39草案/支持预查均明确不入V38。最终只读metadata sidecar SHA3092d5bd0cb6dce902b0781dc11043acfd897f1128f95892b4181b7d74e128df PASS，20行，无重数值/再次遍历718。V38全部封存，后续从V39支持约束与嵌套残差持续性检查开始，不重跑V38当新进展。目标未complete、无阻塞，旧目标/unknown/hourly仍关闭。


## 2026-10-27：V39嵌套源偏差持续性完成，V40仅设计与元数据预查

上一goal轮V38实际六臂负结果及V39容量支持预查已改变路线，属于progress。本轮V39唯一actual原session19958真实exit0：42个五源Month profile，每outer排T、inner再排s，训练128/130例，N/(5ns)权重；1080个月残差/42源pair/7outer，原180key每个held6次。37synthetic及两独立预审通过。source-only贡献G_f/C_POS全定义，没有把G_f/C_f缺失填0；无新fine月、2025score/预测字段消费、小时或unknown。

EF偏差早晚相关.984570–.999390，早coarse d→lateEFbias相关.415771–.917040。Fine earlybias延续的源半段均值参考MAE5.2513–9.6737 vszero34.8131–57.7976，但不可部署、不是monthly forecastMAE或数学上界。NYIS在五outer正→负，FPL大粗残差但EF偏差近0，表明持久性不代表粗误差可直接分燃料。profile在2025截止训练含其他源late标签，回溯而非2023逐月asof。B1库存/同键绝对源G预算未扩大、不得冒称中国B0已可用。

独立正式NP1.26.4临时公式重建通过，42profile/5400weights/6888audit和1080月32400数值exact；42pair max2.84217e−14、7outermax5.68434e−14，恒等式max6.11067e−13<1e−9rtol0，24direct绑定首尾hash不变。主文RESIDUAL_PERSISTENCE_V39_RESULTS.md SHA54b877dd9f47255040dca5f2bf1a9d3f39ebaa6d62f6f36ca8e91646aa531512；最终results-review01c66da90ab43b4b03ea211503abf758aa5b25fa484205dcd4fe2d62bc662877。图数据未改；初receipt写入时误把自身空hash列输出，独立复核发现后exclusive receipt-r1 SHA8f904649f823496adad39ed61dd1ac3940516acfe318eb3e489625f983a3d660只绑PNG/PDF，原收据保留并明确superseded。

薄清单residual-persistence-v39-verification.json SHAd6dd3c74e2c3c094008dee1be78934f9ff2e6afc1cde6a5b764cbe83fe5ccb26：5code/11direct/15artifact=31unique，父继承765，union788；不重复抄父闭包。metadata sidecar正在核，尚不提前称其完成。本轮有新实际源证据，目标active，尚无新方法/未来增量/区间效率/小时或unknown资格。

V40仅草案COARSE_RESIDUAL_V40_DESIGN.md：复用V39六个五源component形成同训练规模ensemble；source pseudo2023-07-01与targetquery都按end/release选last12公开粗月；D=mean12d，S=mean12(EFpred−400)/600，late12平均EF true−pred作source级y，比较None/mean/D/S/D+S fixedRidge，暂不加interaction或k，以免引入未来分母。每outer仍六个source样本，fineearlyEF/latetruevol不进X，sourcecurve含后段他源标签是回溯校准，future单月只称持续平均偏差修正。metadata-only预查7source窗口全2022-06至2023-05，age2/span12，SHA99c09880b9705cfa75da8ab0418b43c541172919f575fca93ff4b2338f4904cb；未转发电/份额、未拟V40。先锁合同实现普通强控制，需D+S胜S-only且EF/CIF和逐源/删源稳定才支持coarse增量，不把ordinary residual calibration称原创。下一goal轮从V40固定合同/实现开始，不重跑V39。

V39最终薄manifest metadata sidecar已PASS，SHA8f6318af0687a50901cd74e21bb47e275e8f84a5824daeda0af5ae0d76d38102，20行，只hash31本轮文件和父manifestSHA，不重遍历765/numerics/tests/actual。原d6dd3c74...及31绑定不变，V39封存完成。目标仍active；本轮已有实际源持续性证据和V40合法窗口metadata，属于progress，无阻塞；下一轮直接实现V40固定五臂future开发验证，不重复V39。


## 2026-10-27：V40同预算低维纠偏失败，下一步改检验供体覆盖

上一goal轮V39源持续性实际诊断及metadata窗口预查属于progress。本轮V40首次actual原24503真实exit0，prepare/predict/score0/0/0；28小校准器，不重fit42profile，6源级样本/outer，33synthetic与两独立预审通过。全部42queries先seal后原35labels评分；1330旧case numericexact，新增350，总1680/336/48/336/42。None CIF/EF31.745721/55.808309；Mean30.808701/59.330059；D38.340809/63.904191；S39.412804/73.064202；DS48.013856/79.864856，未有coarse增量。DS vsS仅4/7dual改善，vsNone仅ERCO/NYIS；去MISO后DS仍差。Median完整保留，主文草稿5个median列值锁定前已按正式CSV更正并自动逐行assert，原数据/代码未改。

独立正式NP1.26.4临时公式重建通过：42source/252query feature及252base、28正规方程、1260raw/clip/210ensemble、26880EF与CIF粒子和1680case exact；aggregatesmax3.637978807091713e−12<1e−9rtol0，normal残差L2max2.7755575615628914e−17，46direct绑定首尾hash不变。实际24formal（10root14pred）/17directinput/5code；未重遍历父788或新fit。Posthoc252feature marginalminmax与35clipdiag单独CSV，FPLD36/36、MISO S30/36、BPATD30/36outside；不是joint支持/风险界，不据此临时gate回填。Clip None/Mean/D/S/DS=0/0/34/63/61每臂252component。ISNE范围内仍退化。

主文COARSE_RESIDUAL_V40_RESULTS.md SHA79b6568e818715e566230b70d6ab3a7b9619e9e9358df9b1fa86060d709e76f5，图两个panel各42点、receipt无self-reference。最终resultsreview/薄清单正在收尾，不提前称metadata侧件完成。无新方法/未来增量/interval/hourly/unknown资格，目标active。

新查UAI2026Huang证书+gated residualBayes head已存在：原文§3.2/4.2–4.3/A.5保留标签机制λ余项，Eq24输入surrogate省略该项，Eq25–26仍需预测后同响应y反馈；只有coarseGPOS不能替换targetCIF监督继承其界。CLeaR2026Proxy-guided只核abstract，proxy独立偏差结构未在本项目验证，普通gate/proxy/calibration不新。

V41仅设计DONOR_COVERAGE_V41_DESIGN.md，尚未新容量数值/扩展训练。worker只读metadata确认V4 SWPP/AVA/BANC/JEA/SC/NWMT与V5 SCL/WACM/SEC/SCEG十域均已查看（RISK_TRANSFER_V5_RESULTS.md:72），月表2022–2025、四年Plant同名BA存在；不读取新G/share/coverage或Generator容量行，不重新下载。当前17域候选训练池仅增加这些已查看开发供体，原七outer/35评分保持，PACE/LGEE和真正未查看域继续封存。先另锁十域库存/报告membership规则，源监督截止/质量/总权重质量及同预算强基线，然后比较原池vs扩展池整包价值，禁止把数量/类型/质量一起变动归因为纯N增益或算法创新。下一goal轮从扩展库存合同/实现开始，不重跑V40或堆六样本校准复杂度。

V40最终结果review SHA838095d96cd291d1fc22f3a5179caa3861e28f27dd6256c6f5e37f65a7e016e6 PASS；两个posthoc独立1652numeric/504flags exact、主文表/图/范围均核。薄证据coarse-residual-v40-verification.json SHAe4684d1c4f749ca3ccb79ab79058aa133be926597ebd66ef26e11335b47f8e7d：5code17direct34artifact=56unique，继承788、overlap13、union831。20行metadata sidecar SHA17c3b4f549fecbb006aabd5d13002af598d8bf0c2a6665a50ed3065e3fabf430 PASS，只hash56与父manifestSHA，不重遍历788或数值/tests/actual。V40已封存，无新方法/区间/unknown资格；本轮有28真实新fit及供体元数据角色检查改变后续路径，属于progress，无阻塞，目标active。下一轮从V41新十域库存/报告口径合同与实现开始，不重跑V40、不把增加供体的收益称算法创新。


## 2026-10-27：V41扩展登记库存实际完成，尚未做新监督/预测

上一goal轮V40实际28新fit负结果及已查看开发供体metadata盘点属于progress。本轮V41实际原94297 prepare/curate0/0、native终态exit0；39synthetic和两独立预审通过。新纯curate仅参数化old7/fixed17白名单，复用已封V33reader/码表/容量/聚合，不改/monkeypatch旧SOURCES。17之外status/fuel/capacity不访问，started先于新float，4年全成+旧4CSVbyteexact后才落表。合同初稿zero/multi计数误称annual audit已在actual前澄清为policy各source/year行，代码/算法不变，无需重复已过tests。

4年共58737非储能unit年度记录/1587storage metadata，两68policy，9formal，14directinputs5code。旧7子集op28/operable28/units49519/excluded1468四CSVbyteexact，新增9218/119不称跨年独立机组。所有68source-year OP/POS正，anchoreligible只是分母可算；SCL2022POS4.6/2023 7.6提示scope需核，尚无新G/fine数字、模型/forecast/interval结果。PACE/LGEE和unknown均未打开。

独立临时验证只共享原已核字符串decoder，另独立过滤/固定分类/Decimal容量与聚合，58737/1587/两68和账目均符；OPmaxdiff2.01692e−11、Operable2.02204e−11、unit1e−13<锁atol1e−9rtol1e−12，标识/分类/计数exact，旧4subsetbytesexact，28direct绑定首尾hash不变；未重遍历父831或读monthly新G。另posthoc候选capacity-support28行140ratio max1.81007e−16/56flags exact：旧7/28marginaloutside→候选17的2/28；MISOcoal.358565旧max.219757新max.847713；FPLgas.968985仍>.958789，ISNEoil.236893仍>.134357。仅候选库存范围，未有cohort资格/joint支持/风险覆盖/性能证据；不选域回填。CSV SHAe90bc6d1ef57d4ea703aa3bbe727ece00c0e62b9c8fe39da68783e5977518153。

主文DONOR_INVENTORY_V41_RESULTS.md SHAf2526b9fdc0b6589ce48366fbfd4d30fcbf2fc95249d3a9761599c19ec7ef0b0，10供体2023表20值root自动assert并独立核。最终results-review/薄清单收尾中，不提前声称metadata侧件完成。V41只证明数据整理，不是新方法或预测收益，长期目标active；本轮有实际新容量整理并收缩后续定义域，属于progress，无阻塞。

下一阶段DONOR_BUDGET_V42_DESIGN.md仅DRAFT：先旧180键保持+新十源2022–24Jun/Jan2025 cutoff同队列/质量/报告membership，区分Cf0/Gf>0率未定义与Gf/Cpos贡献、记录极小库存分母不造epsilon；再用现有强骨干Size/SizeCap/DirectEF×Ridge/ET比较扩展池，原MG冻结以隔离内部EF数据价值，原强锚/门控保留。总训练权重154(FPL156)固定且源均分，源级校准器以后重试总权重6，防止SUM正则混淆；ET叶计数等剩余差别披露。任何新质量处理改变原180键必须另列干预，不暗混进sourcegain。不从弱profile骨干加复杂度，不把增加数据称算法创新。下一goal轮从V42监督合同/实现及同预算强模型比较开始，不重跑V41。

V41最终独立结果review SHA6c95ac9a611760dc0a7a392a00b6f342e8692f89069578df5f4871fc524bc1ad PASS。薄证据donor-inventory-v41-verification.json SHAb796422578980f1855fa091ab11bc160890491b4ea46a9b305f61b48d1e85c9e：5code14direct14artifact=33unique，父831/overlap10/union854（V5角色证明doc新进入input）。20行metadata sidecar SHA166430a818ccae2780061b8335af49dd2ead191ac95bea742b7ad125cddeddde PASS，只hash本轮33+父SHA，无重遍历/数值/actual。V41已封，无新预测/coverage/方法/未知区资格。本轮实际新增10域库存并通过旧口径一致性和独立Decimal重建，属于progress；目标active，无阻塞。下一goal轮从V42源监督队列合同/实现开始，再同预算强模型扩展池比较，不重跑V41。


## 2026-10-27：V42扩展供体比较完成，部分树模型获益，尚无方法突破

应用humanizer与humanizer-zh写作要求，保留失败、数值、条件与结论边界。V42原prepare session77171因代码预审报告尚未落盘exit1，未创建OUT/freeze/started、未读新fine；原失败log保留。先exclusive固定ATTEMPT_R1说明并核两报告后，r1原session12158 prepare/predict/score0/0/0、native终态exit0，完成42新fit，旧模型未重训。30synthetic通过；正式五pins包括runner884fda72a377f2f944bb573b36cbb31ecd1f604b9c19a0a405a85089fdb0e0de，未改旧封存代码。

共同队列412去重源月=旧180+新增232，17候选中16有合格fine，SCL公开POS近零按原规则跳过30月，其余缺三合法上下文63条；实际每outer15source、386训练月，FPL388。总权重154/FPL156固定并源均分，原MG和查询保持，原1680参考numeric exact。42查询/35评分仍为七个已反复开发地区，ISNE只1月；无独立确认。26formal/2520case/504region/72macro/504去单源/42support全部保留。没有新增coverage过滤；412去重月coverage约.8667至1，四类零容量正发电矛盾均0，名义POS率约.0496至.8928，仅诊断不认证物理或报告成员边界。

Mean CIF/EF宏MAE：SizeCap-ET34.593384/71.654565→19.960158/55.339684；DirectEF-ET33.370695/67.160919→22.376286/53.230044。另四头均退化，Size-Ridge EF增加102.59%。MISO明显改善，同时BPAT/FPL负迁移；旧CoalOilUpper16.396137/25.708227仍优于全部扩展头。该旧门控范围未扩池重建，不能冒称完整expanded强基线已穷尽。结果与旧监督池覆盖不足相容，但不能锁定库存表示或特定供体因果作用，增加数据不等于算法创新。

独立临时公式验证session66821exit0，未导入producer或重派正式stage，21Ridge+21ET验证refit，旧180键/y/日期及Size31最大差0；Legacy EF/CIF/MG literal exact且meanpoint最大差0。EF/CIF粒子最大差2.27374e−13/1.70530e−13、全部case最大9.45875e−11<预锁1e−9/rtol0；54核心文件+5尝试记录初末hash不变，无旧854data重遍历。主文90表格数字和范围已核。main SHAabf726b5665d113930d6096f6e35d6c9ba2a1802e1b11f5a8628b3ac9d828316；results-review SHA6b8dc79a0a286c2755e3e3628cc9846cc56c73cfd2629356e51bf60d50bb8226。

薄manifest donor-budget-v42-verification.json SHA059b97ed4cc7013f6634ca053f31560276dda20ca2b55917c099cdd48316bc67：5code/23direct/35artifact=63unique，父854/overlap18/union899；next-steps备忘录SHA4ab32ac5d5dd0a430034d3ed1989e2b51d76e6be37ee8529206f3832529acd5d，尚非V43合同且未运行新blends。后续先固定普通混合与强锚/扩池规则对照，再检验合法粗历史/库存能否预测源外B1增益方向；若仍集中少数源或弱于强参考，停止当前B1纠偏分支。多源密度加权、错设下shift稳健化和proxy桥均已有原始文献，不为旧部件造新名。

本轮42实际新fit并改变对供体/模型交互的判断，属于progress。没有小时或区间覆盖/宽度结果、原创机制或方法冻结资格；PACE/LGEE、真正未查看地区、原小时及新增donor2025fine继续关闭。长期目标active、未完成，无阻塞。

V42最终薄清单metadata侧件已PASS：results/donor-budget-v42-manifest-review.md SHA80b2b814b92414465d08030a991dfc897606ef1ea7be7e063bbb4cf737ebbea3，20行，仅核63当轮hash/父SHA/计数/范围和两attempt账，无新数值、fit、tests或旧854data重遍历。V42已封存；下一goal轮先预锁普通混合和扩池强参考，未有V43合同/结果，不重跑V42。目标active。


## 2026-10-27：V43固定普通对照完成，CIF基线提高，但尚无双指标或方法突破

上一轮V42的42新fit与供体负迁移证据属于progress。本轮不重跑V42，先固定10对端点各.25/.5/.75权重和5expanded经验范围规则，共35新臂，全部EF逐粒子混后乘原MG；仍为7反复开发地区/42query/35评分月，PACE/LGEE/真正未知/原小时/新增donor2025fine不读。原strongB0 Size-Ridge和CoalOilUpper全保留，旧2520参考numeric exact。实现预审发现score异常后可能重入，已加exclusive score_started+completedhash和测试；两最终报告实际落盘/核SHA后才第一次启动。

首次原session19973（start886ed1/terminal2e9979）prepare/predict/score0/0/0、nativeexit0，zero fits。32synthetic通过；5code/34directdependencies，父链只metadata解析并核实际6helper pins，无旧899data遍历。27formal（含score_started）、1470routes、7limits、4970case/994region/142macro/994LOSO/42support完整，所有预测/范围先seal再评分。

35新臂4个meanCIF赢旧CoalOilUpper，0个EF赢，0双赢。最低新CIF固定Mix-Router-DirectEFET-w0.25 mean14.72192464/29.76414529（CIF/EF），旧16.39613741/25.70822689，CIF−10.21%而EF+15.78%；median14.1160/29.7086对旧15.4956/25.6073。6源CIF改善/5源EF改善，FPL两项退化，BPAT EF1.4988→60.9688；7种LOSO CIF均赢，EF只有去BPAT时赢。这是普通收缩对照的局部收益，不是原创算法。后续新机制要保留此更强CIF参照，不以退化Expanded SizeET作弱基线。

扩大合格15源q范围只会减少fallback，42query触发old→new：Range4 30→12、CoalUpper6→0、OilUpper6→6、CoalOilUpper12→6、AnchorEFUpper6→0，逐query单调exact。MISO原CoalOil锚六次→旧SizeRidge六次，CIF22.7384→72.9908、EF50.2241→129.7311。此处基模型固定，只改极值池，说明进入经验范围不足以证明旧预测可靠，不否定扩池训练本身收益。

事后70行error-components独立核对：BPAT粗项−4.647947不变，内部EF项+.093857→+4.071170，CIF改善含抵消；FPL粗项+9.331440不变，内部+2.882575→+11.111092同向恶化；MISO内部−31.007657→−22.898121有可见水平改善。恒等式最大残差1.19543e−13，分量有符号，不把绝对分量相加当MAE，不用于训练/权重。

独立临时公式（不importproducer或旧weightshelper、零fit/无stage/无旧tests）全部35×42weights/188160EF+188160CIF粒子/1470routes/7limits exact，旧router点exact，旧2520case numeric exact；新case/region/macro/LOSO max2.91038e−11/1.45519e−11/2.72848e−12/3.63798e−12<预锁1e−9rtol0。66核心+4logs首尾hash不变；70事后表单独追加验证三分量exact/其他差5.68434e−14；主文108表数字/所有范围均核，无异常控制字符。main SHA911bf60c9b79fff7ca57f22ad7de690e49ebd7f209417238eca291019010f7da；results-review SHAc6004eb37115f2b13c1163b5e4e17faa4246eeead2200346727723f59c60ec89。

薄manifest ordinary-controls-v43-verification.json SHAa98874cbccf7e1f9d916de4bce15bc27a1abe79d4bdcb86f505c7b2369644ab6：5code/34direct/34artifact=73unique，父899/overlap31/union941。root创建时校验73当前pins与父SHA并仅从父metadata核计数；不另造大型auditor或追加重复数据检查。V43已封存。

下一轮优先检验源外‘B1增益方向能否由合法输入预测’，尚无V44合同或实际结果：outer地区fine必须始终排除，inner源响应不能进产生其增益标签的基模型，meta参数固定；若另报meta跨源验证，须防其他行基模型含被评源标签的交叉泄漏。先锁输入时点，不能把source当前真G5的条件诊断冒称month-start预测；R0仅当时合法粗历史，R1才追加可得库存，两预算分开。中国目前仅确认完整粗月报B0，B1可得性未认证。保留匹配结构消融与strongB0/Coil/V43普通混合，不先堆gate或更多权重。

worker只读核对认为扩池V37 Plain36/Offset36/OffsetInterceptOnly可作为后续标准基线，但不是优先创新：旧三头已有充分收敛负结果，扩池尚未重做。V37 q0/ypositive会fail，query q0永远输出0，是硬类别支持而非软prior；412源诊断无矛盾不认证目标物理缺席，禁止epsilon/删行掩盖。若做须同386/FPL388、15源均衡、固定154/FPL156总权重和原lambda/优化参数，三个对照全保留。

本轮35实际固定对照与误差分解提高参照并改变对经验范围的判断，属于progress。仍无足够原创机制、interval覆盖/宽度或小时证据，不能冻结或开未查看地区。目标active、未完成，无阻塞；下一轮不重跑V43或继续搜混合权重。


## 2026-10-27：V44源外增益真实拟合完成；V45探针限定粗反馈能识别的风险

上一轮V43有35实际zero-fit对照和误差分解，属于progress。本轮从粗历史能否预示强端点增益开始。事前缓存潜力检查保留16行endpoint-potential：弱匹配ExpandedSizeET/SizeCapET硬路由oracleCIF16.2052仍弱于普通mix14.7219，拒绝该弱对照；强Coil/Direct单独oracleCIF11.2920与EF8.3514不可合称同时可达，EF挑共同actionCIF16.6547。该事后开发盘点不作独立确认、也不参与训练。

发现V25原MG是各地区ownpast harmonic/prequential law，无跨源参数；generic薄入口先metadata裁prefix，仅核水风光补POS，不用八fine入口或monkeypatch。412源月同draws跨fine折共享，合法prefix3..28，146不足12/266至少12，冷启动不删。实际sourceMG计算harmonic2083/fallback3423/prior1188，保留6153残差中zerohistory1072/below12history4336；原queryMG仍literal不变。

V44原session32476(start14f0a4/terminalecd509) prepare/teachers/predict/score0/0/0、同handle至nativeexit0。每outer排fine，15inner再排fine，base旧5/6source Ridge+Coil，candidate14source SizeCapET/DirectET，105折315细头；对合法MG积分mean后得EF/CIF/Joint三gain。meta HistoryForecast28=合法26suffix+MGposmean/std0；Cap33追加q4/过去压力。两view整体都是B1，不是B0。固定Mean及两view Ridge/ET：168meta sklearn+42Mean，加细头共483sklearn，源粗估计单列。所有source/query先seal、scorestarted防重入后才读取35旧评分labels。34synthetic、独立代码/Python预审均通过且报告实际落盘再启动，无新失败attempt。

新30臂中只有Gain-SizeCapET-CIF-HistoryForecast-ET mean同时超过旧Coil与普通mix：CIF/EF14.65224194/18.93546768，对Coil16.39613741/25.70822689下降10.64/26.34%，对mix14.72192464/29.76414529仅CIF−.47%而EF−36.38%。median15.2923/18.9257相对mix14.1160/29.7086，CIF反增8.33%。ERCO32.1551/38.6402差于mix19.5081/29.5803，BPAT EF38.9147仍远差Coil1.4988。相对Coil7种LOSO均双赢；相对mix7次EF均赢、CIF四赢三负。最低CIF HFC-ET14.2301/47.6727隐含更差EF，最低EF DirectCIFHFCET18.8892/13.7204另有CIF折中，30臂全部保留不作单一方法冻结。

CIF-HFET对SizeCapET匹配Mean的regret .003968对.009276下降57.22%，区域macro非劣74.76%/action69.05%，ERCO占regret80.28%。这些不是置信保证；源月不是独立域，teacher是Jan25快照回溯训练，不是2022–24当时真实前向回测；14→15和5/6→6 operator差异保留。普通crossfit/gain regression/gate不能称原创。

独立临时公式验证483fit+42Mean及sourceMG，原验证session2527 terminalexit0(chunk6e7358)。sourceMG数学/诊断exact，fine coef/scaler/ETimportance等exact，独立simplex算术导致teacherEF/CIF最多3.41061e−13，meta/Mean/全部actions/1260routes/端点literal/7070case与1414region/202macro/1414LOSO/42support、gain1050/210/30全差0，旧4970numeric exact。102核心+5logs首尾hash不变，无旧941data重遍历。main26bdc8d0cde352a7ad5429b2b9e4db9ece3819f2f2d0643e3397b38b61ca780d，resultsreview09dbae7e9c139b36e3b06aa04b8d0d9b8afdb8cf7a7bb884e613b3720846959d。V44薄manifest5d91faa3d8a25a94f8b6100bd1bd7fc462edb2c52254216b336e70f596f2f8fd：5code/50direct/56artifact=111unique，父941/overlap47/union1005，root创建时逐111当前pins核且父仅metadata计数。V44已封存，未来不重跑其stage。

中国B1公开资料memo已存docs/CHINA_B1_CAPACITY_AVAILABILITY_V44.md SHAa2e446ad1cb45a0be102d51774781b2c9057a99be73d0da918294e431760d656。GIPT官方省份/机组入口有容量/燃料/状态字段，当前Sept2026混合tracker更新；中国oil也有，不能误称仅coal/gas。但阈值、other/完整四POS分母、地理省份与报告membership、历史首发均未认证。NEA已核入口全国水火核风光不足省四分表，不推断其他官方资料不存在。只可称部分库存代理可得，不默认中国B0用户已有B1；未下载配对或用作模型输入。

在V44封后又做V45零fit识别率probe，先写协议f2be5791e58168c089d175083c1020104fdeccafa5e4df0de7bcb0ab77b2a753。固定因子/相同报告范围/无误差公开g下，monthly generation-weighted CIF t∈[400g,1000g]；两已封CIF点L/R的d(t)=|L−t|−|R−t|单调，端点给sharp界。数学review指出weak支配/neutral/跨0需分开；g>0内部EF比较界仍对称±|eL−eR|。若共同MG的gbar=真实g，则L/R均在可行truth区间，非同端点没有稳定严格优劣。故粗反馈可识别比较主要来自粗预测错位，可能奖励EF错误抵消，不能当fine伪标签。

V45共5408重复折内比较=2704×2，涉及412源月；每candidate453/2704稳定strict，总906(16.75%)，raw678candidateStrict/228baseStrict/4476crosszero(82.77%)/26equal，无weak。至少一次strict只涉及76去重月；所有5408内部EF预测非中性比较仍不识别。gbar控制稳定strict0，fine只验证gain包在界内，max1.13687e−13；release延迟58–61日是已保存假定规则，不是官方首发认证。fine不用于缩界，未识别不填0、不删。rawnumeric margin1e−9仅诊断非统计证书，不能推未来或小时保证。

原root probe JSON428994d42ea0907640bc7653ab2275ae91388f52d8ee615b43072eac30912592保留ROOT_PROBE_COMPLETE_NOT_INDEPENDENTLY_VERIFIED字节；独立临时公式全5408+32summary PASS、17inputs+3outputs20hash首尾不变、汇总差5.68434e−14，独立结果以results/coarse-comparison-v45-probe-review.md SHA1ba00136ea717fbed746e31ae7c96383805a44062ea242a8ffb85d8a90bccfce为准，未重V44验证或新fit。V45结果文档5341ca790b695f715d57cc66227e3f490f1376e286bff15591a98b8ca1faf249；针对性文献memo a3aba2fa07d61437f52144aacdf0704220be3a3dc2cbc5af8fca51e7c26e7f55。

新核primary：2610.19804v1（2026Oct14预印本）§1任务/§2.2/§3相关保证，要求realizability/globalobservability/nondegenerate，不能直接移植到同g不同EF的不可识别反馈；JMLR24(346)2023 LinearPM官方摘要表明IDS/contextual等早有先例（未通读定理）；2501.10117v4（2026Feb23）摘要+§4intervalscore/Thm4.1/局部说明已有partialID与区间conformal，iid/exchangeability不能自动赋予新域/小时。不是‘新gate/区间回归即原创’，未完成全面先例审查。g本身可直接监督粗层，未来不能用复杂partial比较掩盖一个普通coarse correction参照；要与该强参照比较。

只读小时集成检查：V29 run_conditional_law_v29.py:82/LeafContext.predict:160是B0 hourly现链，MG(181,128,5)、paths(10,181,128,24)，原同预算合格Scale-anchor/spread1；完整M8重组预算更高。recompose_paths(src/transcif/research/fuel_transport_v7.py:44)需mix8+q(...24,8)，EFscalar不能造r4，gas/other同EF也不意味着小时分母/shape相同。V29 anchor:206只依赖fine@EF，故可替换为scalar MG_POS*EF，再加donor raw paths，属于普通additive level控制，不是物理重组/月总量约束；clip会改形状/区间。仅6月初可与V44精确粒子耦合，不能把同s跨181天新seed硬配。monthly池扩16源并未扩old6小时shape bank。V29 NPZ有sourceOOF residual/scale/cov但scale是比例条件尺度非小时tailcalibrator；旧risk-transfer-v5-source2024/fit risk_v5-{0,30,90}.pkl有源2023日水平/中心化小时shape联合残差组件，但oldcoarse_risk还用M8、旧fit不能认证新center。未打开hourly数组。

下一步应回到可识别粗风险与未解释level/shape风险分开处理，并优先设计同6月初小时source开发对照（缓存Scalar-anchor接点与原Scale-anchor同预算），新增协议前不读hourly真值。不要再只搜月度gate/阈值；若做gain风险校准必须防其他teacher训练labels含校准源fine的交叉泄漏。Partial feedback不能虚构内部EF监督，也不能偷用当前release后coarse改善同月已发预测。新增B2局部数据价值如年度细分/同口径碳因子需另预算/先核可得性与口径，不默认用户有。

本轮483实际sklearnfit与42Mean、源粗估计、V45零fit识别率与前沿条件核查共同改变下一步，属于progress。仍无可信小时coverage/width增量、真正原创机制或冻结资格，PACE/LGEE/未知域与原小时在本轮均关闭，目标active、未完成，无阻塞。


## 2026-10-27：V46完成小时桥接开发评估；普通中心对照已能解释部分收益

本轮应用用户指定的Humanizer及Humanizer-zh，后续说明与结果文档减少铺垫、重复和夸大判断，保留数据、信息预算与不确定性；已有封存文本不改写。

V46协议固定六个月初的原128粒子、45个月度水平，构成Replace/Keep各45臂、三个预锁水平的CenterOnly/FlatKeep各3臂及3原参考，共99臂（B0 10/B1 88/M8 1）。零新拟合，不构造完整燃料比例，不声称物理重组或月能量约束。两个最终预审先落盘后运行，原CSV读取精度问题在启动前改为round_trip并加17位double回归，49个合成测试通过。

首次正式进程84888，起始ef9718、终止e000a5，prepare/predict/score均0。七区42个预测query全部先seal，再评分预锁32个UTC月初日：1月全部不可评分，ISNE仅2月/5月，其余六区2至6月。共15840 day、3465 region、495 macro、3465去地区、42支持和4032 raw诊断；不是完整半年或独立确认。只将这32天的小时generation转换为数值；PACE/LGEE、未知地区及新增donor2025细标签继续关闭。

Scale-anchor尺度1覆盖92.2619%、宽度225.391144。预锁V44 HFET水平的Keep为94.0476%/222.369062，CenterOnly为94.4048%/225.338630：Keep只少约1.32%宽度，覆盖反低0.3571个百分点，CRPS略差，不能把普通中心调整的收益称为新风险机制。Replace为84.5238%/140.212863，最差地区52.5%；删水平残差显著缩宽但少算风险。U−P均值不保证为零，两模式也不只是方差消融。FlatKeep覆盖85.7143%，形状MAE30.315749。

普通B0 Keep-Legacy-Size-ET达到93.5714%/221.890759；普通B1 Keep-Legacy-SizeCap-Ridge为96.3095%/222.189985、CRPS27.264774，比原锚低13.16%，其七种去地区汇总均覆盖更高、宽度更小、CRPS更低。该突出臂来自开发盘点，且没有其匹配CenterOnly，不能概括归因或选优冻结。V44门控不胜全部普通基线；去掉FPL后其覆盖比原锚低0.5556个百分点。CISO仍退化。全部五个预锁尺度已保留，M8高信息参考不能算中国B0可实现方法。

独立临时公式进程29682 exit0，不导入V46producer、不调formal或fit。45levels、全部raw/U/P/4032诊断差0，三参考raw及MG/keys/origins exact；评分最大差4.26326e−14，原320八指标与clip numeric exact。98核心与5日志首尾不变；主文48表行和叙述数字已核。raw非Flat形状变化最大2.27374e−13，代数invariant最大6.25278e−13；零POS粒子实际0，未验证该边界修复。ERCO继承残差mean66.835009/std114.341197按42query诊断，不与32评分队列混合。

主文docs/HOURLY_BRIDGE_V46_RESULTS.md SHA0b5bdb7b1d6b1e30a62005006205d89cb0fbdb171cf1d78d548b02cab9633e73；results-review SHA77e79afc9de4dbf06674790780c24840c2505e92801eecdfb08337747a0a7130。薄清单docs/evidence/hourly-bridge-v46-verification.json SHAe0597281ae5ce839df6588b0f7a7bb2af78cd8bb49d99ccb84ad2b7ccd78257d，5code+67direct+34artifact=106unique，父1005、overlap64、union1047；root创建时核当前106pins及父metadata，未重遍历旧1005数据。V46已封存，不重复其stage/测试/数值核验。

本轮提供了小时覆盖与宽度实证，属于progress，尚无足够原创机制或冻结资格。下一轮优先设计原开发地区更多前向日期的同预算实验，先解决每天合法粗输入、装机时点与粒子耦合，再检验粗风险与未解释水平/形状风险的分离。不能把月度CIF/EF代理成绩直接当小时目标，也不能继续只搜gate/尺度。若训练风险头，必须保持源外验证，防teacher间接含校准源fine。V47尚无协议或实现，未运行。长期目标active、未完成，无阻塞。


## 2026-10-27：V47扩展至1145日，普通中心收益保留，仍无新风险机制

上一轮V46提供32日小时证据，属于progress。本轮原7开发地区扩展到2025H1每日181query，共1267预测；有效评分支持仅由旧V29 metadata预锁BPAT176/CISO178/ERCO180/FPL179/ISNE89/MISO166/NYIS177，共1145日。未打开PACE/LGEE、真正未知或新增donor2025fine，仍不是独立确认。

V47重放Legacy/Expanded三族Ridge/ET共12水平，加Capacity/Coil/固定Mix共15。每水平全部配Replace/Keep/CenterOnly/FlatKeep，另3原参考，共63臂（B0 18/B1 44/M8 1）。每天仅更新完整发布时间合法的三月context和真实release age，月份相位/月龄保持旧定义；库存固定2023 OP。42ET确定性重建，42Ridge用旧coef无fit，无新训练标签/MG/shape bank/风险尺度。六个月初所有12头raw/EF/projection与15CIF及原features重放最大差0，不能将月度头daily调用说成daily监督或校准。

两实际预审和34synthetic通过后，首次session10555（start d6d12f、terminal d538df）prepare/predict/score均0。七预测archive全seal后才转换原七源2025H1有效小时。360675day/2205region/315macro/2205LOSO/1267support、76020raw诊断完整，旧11450八指标+clip重算差0后round_trip复制。一次事后打印regional表时误选macro字段导致只读分析命令KeyError，未修改正式文件、未重启stage；修正选择列后分析完成。

主尺度1原Scale-anchor覆盖92.6912%、最差85.0559%、宽223.260240、CRPS33.729310。B0 Keep-Legacy-Size-ET为93.4384%/220.268921/33.345906，宽−1.34%、CRPS仅−1.14%。B1 LegacySizeCapRidge Keep96.2716%/220.689913/30.398589，宽−1.15%、CRPS−9.87%，七LOSO相对原锚覆盖/宽度/CRPS均改善；其CenterOnly96.4746%/223.812974/30.383230，Keep仅多缩宽1.3954%，覆盖低.2031pp、CRPS高.015358。全部15匹配Keep宽14低、IS13低、CRPS7低、coverage5高，并非全面优于Center；这些相依臂不作独立检验。ExpandedSizeCapET Keep CRPS29.532509但宽223.183292仅−.0345%，有指标取舍，不事后挑优冻结。

Ridge Replace84.9149%/142.007169、最差65.7870%，删U−P少算风险且改均值；FlatKeep92.6754%但最差83.6610%，形状MAE25.509403。原anchor形状23.920756，recompose19.268498但覆盖91.7930%、最差79.6875%、宽232.339374；月度水平加常数几乎不改小时shape，值得下一步检验更强形状底座的完整匹配风险控制。普通Replace固定Mix factor1.5已有96.0968%/最差87.4074%/宽213.143198/CRPS28.743797，需作为全局尺度强参照，但此点未独立训练选择，不算新校准或冻结资格。五固定尺度全保留，未插值/拟合。nominal+efficient尺度1共11含原锚自身1，新10=B0 2/B1 8；条件不要求覆盖>=原锚，不能泛称全部提高覆盖。

CISO Ridge CRPS32.069844→35.639560，ISNE23.536951→28.138240，仍负迁移。全1267query继承残差ERCO均值66.996782/平均std114.758803、MISO27.521809/78.132787；与1145评分队列分清。BPAT2025-05-31唯一一次有3个MG_POS0粒子，RidgeKeep raw非零72小时/clip正45；只诊断未按分数修补。raw越界最高42.96875%，非Flat raw居中shape最大3.410605e−13，不是clip后或物理约束保证。

独立临时进程6703 terminalexit0（a10907），不导入V47producer、不调stage/旧tests或旧1047data。42验证ET+42Ridge，全181context/features差0，raw最大3.41061e−13、EF2.27374e−13、level1.70530e−13、diag2.27374e−13、invariants7.95808e−13；360675day最大4.49063e−12，聚合低于2.27374e−13，coverage/path/tails/clip/flags exact，旧11450numeric exact。126核心+4logs首尾不变，exec另核；主文62表行与全部pp/percent/诊断核对通过。main SHA2fd6e52342711fa415dfc1037fa1e75f29c0b157fc7e3f712a6852ea5d7b4c4a，resultsreview ed5375a7ccbc39e41643e8b5ef41f126a95f5690f136930ad77bb69e66286a41。

薄manifest docs/evidence/daily-bridge-v47-verification.json SHA4e5a3cfd29b6c6db424f734e89bbe05f7e7ba253b7dea652b4a1b994075df87c：5code+95direct+34artifact=134unique，父1047/overlap92/union1089；root创建时核当前134pins，父只metadata，不重旧全数据。V47封存，不再重跑。新增文献短记docs/DAILY_BRIDGE_V47_LITERATURE_NOTE.md：NeurIPS2024 Ge/Mukherjee/Fan已有无标签目标域区间效率优化；只核摘要/引言/§3设定与Assumption3.1，条件响应不变和bounded density ratio不能由粗月报相似认证，另一对齐分支未通读证明。不以区间混合/密度权重命名作原创。

本轮改变了稀疏日期与完整日期的判断，属于progress。下一轮优先在原Scale-recompose小时底座做同预算的完整匹配架构消融，可复用V47已封15levels、原每日粒子，无新finehead fit；先锁公式/registry/评分，不重训旧头。所有60新控制与原anchor架构及全局factor强参照一起比较，不能只挑Ridge或最佳scale。若更强底座仍无可观增量，再形成严格outer排除的源小时风险数据集，比较全局校准/异方差/条件分位数及可识别粗风险与未识别level/shape的分离；不要先堆gate或偷用校准源fine。新center的风险支持与小时shape比继续选月度头更值得投入。V48尚无合同或实现，未运行。方法创新、冻结资格、完整外部数据与真正未知确认仍未满足，长期目标active、未完成，无阻塞。


## 2026-10-27：V48发现日均漂移与形状的混杂，普通重组参照可显著缩宽

上一轮V47有1145日完整开发评估，属于progress。本轮不新fit/头/MG/bank，复用15个已封daily水平，原V29 Scale-recompose为R、anchor为A，F与MG/日期/粒子逐值配对。先锁四模式×15=60；两首版预审与7测试完成，但首次formal/数值读取前发现R/A还混有日均变化，遂另加15 AlignedKeep=C−P+S，C=R−mean_hR+mean_hA。它保留旧Keep-A逐粒子raw日均/配对并换R的居中shape；新增两恒等式，另直接核父实际Keep[18:33]日均。首版报告/testlog保留为prelaunch记录，非失败formal；最终两-r1报告落盘及7扩展synthetic通过后才首次启动。

新75+旧63共138臂，B0 38/B1 99/M8 1。0fits，不读训练审计数字。1267query全部先seal，再按原七源1145有效UTC日评分；新429375+旧360675=790050day、4830region、690macro、4830LOSO、1267support、95025新diag。首次session18798（starta5d559、terminaleb1735）prepare/predict/score均0，无重启。PACE/LGEE、真正未知与新增donor2025fine不读。

直接R-Keep LegacySizeCapRidge coverage95.3165%/worst82.9309%/width229.608163/CRPS29.948537；旧Keep-A96.2716/88.3427/220.689913/30.398589。AlignedKeep则97.0292/91.2453/222.285339/29.350791，shape23.946589→19.249290（−19.6157%），宽仍+.7229%、IS稍升，不全面支配。B0 LegacySizeET Aligned95.1824/worst90.5787/width221.980987/CRPS32.146882，对旧Keep覆盖+1.7441pp/CRPS−3.5957%/shape−19.5036%，宽+.7773%。全部15匹配Aligned vs旧Keep cov/CRPS/shape15改善、width0降低、IS12降低；3个重点控制7LOSO CRPS/shape均好，但cov不降且width不增仅2/7。不是15次独立检验或逐地区一致，MISO RidgeAligned CRPS40.255549→41.659927、shape21.885326→23.788357。

BPAT Ridge直接R-Keep coverage82.9309%/width47.568094，Aligned98.1061/73.180874；保留A的日均分布恢复了部分覆盖。该干预同时保留level中心与散布，不能只因果归于缺方差或具体燃料调度。全部新primary nominal+efficient共4（皆Aligned：B0 LegacySizeET；B1 LegacySizeCapRidge/ET、LegacyDirectEFET），总15含父旧11，仍不冻结。raw差异与clip交互须分清。

普通R-Replace-Ridge固定factor1.25覆盖94.0989%、worst87.1449%、width187.220093、CRPS27.529858；对原anchor1（92.6912/85.0559/223.260240/33.729310）宽−16.1427%、CRPS−18.38%、cov+1.4078pp。R-Replace固定Mix1.25为93.7419/worst86.9318/width190.037304/CRPS26.570703。它们提高了普通参照，显示效率空间，但factor只是预锁曲线事后查看点，并非源端独立校准，更不是原创。690宏曲线点全保留。后续必须面对这些全局scale强参照，不能只比原anchor1。

BPAT2025-05-31仍唯一3个MGpos0粒子，Ridge RKeep raw非零0、CenterOnly72、Aligned72；clip正分别0/72/48。对齐A日均也继承了物理不一致，不作边界修复。maxrawclip37.7604167%，非Flatraw居中shape最大3.410605e−13；不声称新路径仍物理重组/月能量一致或风险已校准。

独立临时公式首次自身oldcopy检查把已作index的factor又选作值列触发KeyError，初始脚本/log保留；只修索引值列后进程66579 terminalexit0（5a3119）全量PASS，未重formal/fit/tests或旧1089数据。75raw/U/P/95025diag差0，MG/S/fineA-R/key/date exact，8invariants最大9.09495e−13，直接父Keepmean最大3.41061e−13；新429375及全790050day max8.52651e−14，region/macro/LOSO max1.42109e−14，其余coverage/path/tail/clip/flags exact；旧360675全部numeric exact。70核心+5logs及exec共6附加首尾不变；main64表行与全部pp/percent/零POS/LOSO核。主文SHAcb5f1c1d4f00370ec98c9d6a6f565702a0e0ae78d2bf12be4cb71f365d565d85，resultsreview6584f2ca8c8f20fbd2bc7133df2a5931f3e4517b76f635f8085b09900270f036。

薄manifest docs/evidence/recompose-bridge-v48-verification.json SHA2790acfbf96e24b37d1baa2f45a6431ccf247a8e8f71c482d0eab114cea59b64：5code+39direct+37artifact=81unique，父1089/overlap36/union1134，root创建时核81当前pins与父metadata，不重旧全部数据。V48封存，不重复stage/测试/数值核验。新增文献短记docs/RECOMPOSE_BRIDGE_V48_LITERATURE_NOTE.md：已核Nature官方ScientificReports 2026-07-19 SP-MinT摘要/引言/基本协调公式，已有历史星期型shape先验与方差取舍，未复现/通读证明。无目标小时历史的预算不同，但Aligned是普通仿射控制，不称首创shape保留或协调定理；CIF日算术均值与月能量加权均值不同，不能乱套能量coherence。

本轮的实证线索是日均风险与更好的日内形状必须分别检验；有明显普通curve效率空间，尚无专门原创机制或冻结资格。下一轮先形成严格outer排除的源小时风险数据集，再比较global/component calibration、异方差与条件分位数；避免继续挑月度head/gate或按2025score调scale。新模型创新必须解释粗统计可识别的部分，并保留未观测内部燃料/调度风险，两个scale本身不算原创。

下一步只读源码线索（未打开新的bank/训练数字）：V25已有21个pair-excluded银行，bank proper期常量START=2024-01-25/CUTOFF=2024-07-01，并非2023；union_bank是两个5源pair banks拼6源outer视图。源风险teacher需直接用排outer+held的5源bank，而不是原6源bank。V29 fit_leaf_controls的history L24来自2022-01至2024-06、Jan25快照；其现有outer入口只排outer，必须同时在monthly rows先排held，donor residual folds还要排donor，不可只换bank。V25 coarse_law接受任意有限日期且只见MG列，可生成合法2024H2 own-past粗预测；风险数据候选2024-07至12共184日，必须另锁valid metadata/发布时点，H1 bank不含H2。V31当前daily guard限2025，历史daily feature要新入口/同公式，不monkeypatch旧常量。Fine heads与bank均排outer+held，校准标签仅held2024H2，2025查询前全risk fit冻结；解释为Jan25训练快照的源外回溯风险数据，不伪装2024实时as-of回测。若做源级风险模型验证，要防其它teacher间接含被验证源fine的交叉泄漏，必要时再排第三源，不继承conformal新域保证。V49尚无合同、实现或实际运行。长期目标active、未完成，无阻塞。


## 2026-10-27：V49完成源区风险数据；持续地区偏差与校准源分离成为下一步重点

上一轮主要接入Humanizer/Humanizer-zh并安排预审，未提供新的科研结论。本轮实际完成168次正式拟合、42方向全量预测和源风险标签，新增了决定下一步的数据，属于progress。两项写作规则继续应用，已封历史文本不改。

V49在原七开发源中作21个无序双排除：5源final/4源donor OOF，每pair共8fit、两个方向共享。固定2024Jan25–Dec31的342query，共14364预测日；全部预测先seal，再按预锁metadata支持计算2100 unique源日、12600有向有效日、100800 base日，保存8base及level/zero-mean shape误差。H1/H2源日1100/1000分别报告，不能当2024真实在线；fine头为Jan25快照，own-MG/context仍逐origin合法。小时label release policy=null/首发未认证，后续fit另锁lag。

首次正式session57759 start13d2d2、terminal9a7197，prepare/predict/label均0，未重启。43producer synthetic与13诊断synthetic通过。预审发现ISO us/ns比较问题，显式ns归一修复后再启动；冻结7pins未再改。零fit事后分解native26cb03 exit0，672task/112held/16macro按先MSE均权再sqrt/ratio；H=L+Q最大4.14389e−11在其预锁容差内。重复outer/日期不当独立样本。

下半年B0 SizeET-ReplaceR覆盖74.43948%、宽156.38732、CRPS38.86676；84.65091%的小时点MSE来自level。匹配AlignedKeep覆盖88.93406%、宽258.09988、CRPS42.29697；原anchor87.33846%/258.73096/42.96005。B1 ReplaceR79.78170%/155.95368/34.55053，Aligned93.76573%/257.59490/37.81703，B1在中国同口径可得仍未核实。四组Aligned相对KeepA coverage/CRPS/shape均改善，但width四组均增；普通仿射控制不构成新机制。与V48的2025六源曲线日期、支持和教师规模不同，不能单独因果归于季节或5→6算子。

MISO下半年B0 ReplaceR六outer bias均正82.1287至158.3888，FPL均负−74.0761至−28.7068；逐task时间平均bias先平方再平均/除levelMSE为88.50872%/86.28144%。总体bias仅+.8166，是地区间抵消，不能视为近无偏或可部署去均值收益。标题审阅时已纠正点MSE≠coverage失效归因，后者还受width/粒子配对影响。CISO B1 ReplaceR仅32.3706%点MSE来自level，不能统一说所有地区由level主导。ERCO69/ISNE26个H2标签及Q3零支持保留，不默认Q3/Q4拆分。未封存的描述性Spearman只作探索，不进主文/方法选择。

独立全量复建首次9870因误把自身活动log纳入不可变集合主动TERM，真实exit143/def4c0，确认≥24验证fit、额外启动数未知；原script/log/hashsnapshot保留。仅修日志集合后63575 start8f69d0、terminal09aeed真实exit0，完整168验证fit另列（147ET/21Ridge）。21pair/42方向/全部342日MG/context/条件叶/OOF/Scale/中心头/raw8、2100truth和risk arrays差0；100800case CRPSmax5.68434e−14、672region2.84217e−14、112outer1.42109e−14，其余列差0。252core+6固定附加首尾不变。独立零fit诊断另exit0，672/112/16及bias²核，68绑定与core重叠。

只读time/region/valid的外10旧开发源盘点也完成：SWPP/AVA/BANC/JEA/SC/NWMT/SCL/WACM/SEC/SCEG，共H1/H2=1575/1812；Oct1/Nov1/Dec1排他累计支持2478/2787/3080，H2=903/1212/1505。SCL保留；未转换其generation或coarse，未解析2025值。JSONa97117445e444f70439a4e506f36c7cd6c78e187dc91a20e0d803e43055d1e71，独立metadata review b88a189c0edf33faaa7ffb9354416d361c5b2013024278d6d0130ee2390873d8逐月/5cutoff/11直input核。它们未进当前原池教师，但已用于其他开发（九域细月监督进V42），不是新确认、实际label release可用性或交换性认证。

针对primary文献新核：JCGS2026曲线registration+conformal仅摘要/设定（全文403），ECI ICLR2025的revealed-score反馈条件，2609.33868v1依赖coverage推断设定；HCP作者v4§2.2式6/Theorem1及hierarchical jackknife+已有组权重/留组预测，六cal组正无穷质量1/7使该普通90%规则阈值∞，不推广为所有方法不可能；StatisticalScience2025 multi-environment只核摘要/引言、GHCP2608.15500v1只核初始响应设定；JMLR2024弱监督Alg1/Theorem9是集相交弱覆盖，不能代替小时真值强覆盖。文献备忘fe52e336e0209bc4855cb1a37e79519424874a4db654546afc5db6cfa1ee9021，不认证全面新颖性。

主文docs/SOURCE_RISK_V49_RESULTS.md SHA802d59962e3f19f678cd28a5e07acb505bf786f1cf1cca23137d7381b9346357；formula md1b2de65e5cc7f176eb87f1b72d0b630bed7afae0a4c27d35f296586e68dcf7c3、JSON5c9fa9e4af1d6ba31f9e491e0c638c0d79d73e7d3b24c910b2cadd04671f1d99；results-review6b5fa3afad7c5421b6530bdda191ba533fbf8c8ec25878a5309e0f0dddcb01de，31表行140格通过。独立临时核验与metadata源码/收据7份原字节归档results/source-risk-v49-verification-sources，不运行副本。薄manifest docs/evidence/source-risk-v49-verification.json SHA8269659f1f9e5d4d6c06fca3a9b3b74c2299d9c6c3fd2242db5bdd0475335530：9producer+76direct+4auditor+210artifact=299unique，父1134/overlap63/union1370；root仅核当前299文件与父metadata，未重旧全数据。V49已封存，不重跑其fits/suite/全量核验。

下一轮先锁标签delay/时间split与严格源分离的数据流，再做同预算global/component/异方差/CQR比较。可用原6proper源风险拟合、外10旧开发源校准的设计检查现有held间共享训练材料问题，且固定同一部署point/risk函数后再calibrate；不把已有(o,j)表随意拆成独立proper/cal，也要控制5→6算子变化。源校准的数据量不等于地区/时间交换性已满足，不能直接借用HCP等理论保证。后续应区分粗统计相容、真实小时覆盖和兼容世界稳健性，评估统一benchmark/信息边界是否比继续加gate更有论文价值；尚未运行V50或信息等价类新压力测试。当前仍无专门创新、冻结资格或真正未知确认，PACE/LGEE/未知与新增donor2025fine关闭，长期目标active，未完成、无阻塞。

## 2026-10-27：V50正式运行中，尚无覆盖或效率结论

V50九个producer/test/protocol pins与整体code/math预审已冻结，首次顺序launcher session60204（start f9f06c）仍运行，严禁重启。prepare exit0（34de65）：212 direct inputs、9 code，固定Jan1快照的90/30/7天延迟分别支持922/1514/1743个外部校准源日，十源每种情景均非空。predict PID49744；本条记录时已完整完成BPAT/CISO/ERCO各11查询档案和64拟合，接着顺序处理FPL/ISNE/MISO/NYIS。外十源响应尚未开启，未运行calibrate/score；PACE/LGEE、未知域、外十源2025 fine继续关闭。native观察超时不是失败，不启动第二个正式进程。

本轮固定5点底座×6标准风险法×3lag×2rule=180配置；点63fits、风险385HGB，总448fits。proper是原六源2024H1、校准是外十源2024H2元数据合法窗；cal/target共享同一六源点函数。五源proper教师到六源部署差异、既有开发历史和未认证交换性保留。不能把这些标准基线当新算法。所有预测和缓存parity先seal、再开启外部响应；全部q/final界再seal、才评分旧七源2025的1145日。

独立零fit核验脚本已盲准备/tmp/v50-independent-L4wRFM47/check.py，SHA e576fe06b2014fa4d758e7cfa5ae866f3ad786373942d1c20761e1b9cb337b54，v32只读静态审阅后补注册路径/完整唯一head名单门禁，语法检查通过但未运行。必须正式completed且root授权后再运行，报告/活动log放formal OUT之外，receipt仅显式稳定终态文件，禁止活动日志自glob。可重放175个保存最终风险头；210临时OOF头只能核审计、保存bias与scale标签代数，不声称385头均独立重拟合。点模块作者身份和共用旧底层库要披露。

两份非冻结补充短记已完成定点来源审阅及措辞修正：docs/DISJOINT_CALIBRATION_V50_LITERATURE_NOTE.md SHA2e0cb5dc6a07e2d961ac45828c2e41e02618c2f44d461e389d864c216795d5fa；docs/CHINA_ANNUAL_FACTOR_AVAILABILITY_V50.md SHAfacdd87a2c0c119463ccb8f7564044394c5fc60abc706c61b33aa1ea7b65340e。CarbonBalance实际年度、Yan年度到月度分解有本地真实月度碳反馈，不等同当前无目标小时CIF设定；仅差异不证明首创。UAI2025聚合监督和ICLR2026凸分区粗高斯均值只核正式摘要，不移植定理；V15/V19已有小时可行界/尾部反例，不能重复包装新突破。

新核现实线索是公开年度省级碳因子：2022官网日期2024-12-26，2023官网日期2025-12-31；后者不能提前用于2025Jan1。2021官方方法式7含省际/境外/区域净送入，2022/23公告称方法一致，不能把含调入因子硬当当前本地发电×固定EF锚。若后续需要，这是另立B2的可得性线索，尚未入模、未认证历史vintage；所有简单/强基线都应同预算适配。本轮不更改B0，不据此提前拟合下一方法。长期目标active，正式运行未完成、尚无新覆盖或论文突破结论。

补记：正式结果现已完整，不应沿用上一段“正式运行未完成”的状态描述。session60204于2026-10-27 07:42 UTC自然结束，prepare/predict/calibrate/score与launcher terminal均0，未重启。`completed.json`状态COMPLETE，freeze/prediction/calibration seals链接一致；day/region/macro/LOSO/raw/support/calibration行数分别为206100/1260/180/1260/28625/1267/1260（文件行数含header），原七个外层×全部77预测包完成，外十2024校准支持与目标真值已按合同先后开启；原七2025仍1145有效源日。PACE/LGEE和未查看地区值仍未打开。数据全是反复开发过的源与目标，summary明确independent_confirmation=false。

正式目标评分的关键反证是校准带来极宽区间，且强异方差/HGB/CQR未在同覆盖下胜过冻结简单基线。主情景90d+equal-region：B1AlignedKeep+MCStd宏覆盖99.93%、宽431.16、IS431.24，七源最低覆盖99.75%；同base未校准factor2参照覆盖99.95%、宽431.50、IS431.56，七源最低覆盖99.77%。两者实质相同，固定factor2反比校准版窄0.34且覆盖高约0.02pp。更收缩的同base简单factor1.5覆盖99.55%、宽330.30、IS332.23；coverage最低98.29%（CISO）；factor1覆盖97.03%、宽222.29。以上是已看开发目标的描述值，非独立统计验证；不能挑factor后称新方法。

B0AlignedKeep的简单factor2也达到99.99%/436.37/436.38；90d equal-region MCStd为100%/523.22/523.22，30/7日同样超过factor2宽度且覆盖相同或略低；HCP更加宽：B0 MCStd 90d 100%/646.22，B1AlignedKeep 100%/571.96。GlobalABS校准等宽方式、风险HGB/location scale/CQR大多更宽或出现封顶至1000，均没有清楚的覆盖-宽度增量。逐小时覆盖接近100%代表过覆盖，不是成功证明；逐日路径覆盖也只是诊断。正式主要比较、全部180宏配置/1260地区行、lag sensitivity、q/尾部及区域差异待独立公式全量零拟合复核完成后写结果报告。

独立复核agent v33已获准以唯一脚本SHAe576fe06b2014fa4d758e7cfa5ae866f3ad786373942d1c20761e1b9cb337b54运行；native session34836仍运行，首轮212输入+9 code+210formal hash、17 role MG/context/cap重建通过，BPAT自预测和首个外SWPP包已复放，当前没有报告全量PASS。任务只重放175个保存风险头并代数审计210个临时OOF头，不声称448头都独立重拟合；日志JSON在OUT外且receipts固定列表。目标源fine通过规则读取属于该正式评分；尚无外区confirmation。

这轮标准模型实验没有形成方法突破，也暂不满足既定“有清楚增量后冻结方法”的解封条件，不能打开PACE/LGEE或真正未知源。它改变了优先级：应先查为何可用源风险校准向已看目标转移时把q推至使区间宽度近factor2，而预测尺度HGB仅在部分臂有效；拆区校准支持/区域偏差可作为诊断，不能按已看目标调q后宣称泛化。鉴于V15/V19已有结构可行界与额外信息oracle，不能重复这类LP反例；下一步更有价值的研究路线可能是低频官方年度省级电力因子（只作新增B2预算、需诚实处理其消费边界/调入电与发电侧代理差异）能否减少可预测水平偏差，同时用独立来源评估小时形状风险。这个方向必须先设立相同信息给简单/强基线的合同、发电/消费估计目标明确及真正未见区域验证；不预设效果、也不改本轮已冻结的V50。长期目标active、未完成。

## 2026-10-27：V50外部校准源影响复算完成，发现显著区域风险异质性

按`docs/V50_CALIBRATION_SOURCE_DIAGNOSTIC_PROTOCOL.md`完成只读诊断。脚本仅打开70个outer×外区预测包与10个外区校准标签包；未打开outer目标档案、目标真值评分文件、proper档案、PACE/LGEE或未知源，拟合0次。独立重建1260个官方q最大差0，输出12600条source×configuration行，详见`docs/V50_CALIBRATION_SOURCE_DIAGNOSTIC_RESULTS.md`及`results/disjoint-calibration-v50-source-diagnostic/`。

Equal-region留一源q通常只变约0.5标准化分，但去掉SC/NWMT/WACM时更常使阈值下降，表明少数源提供较多尾部质量；不据此删源。外部源局部p99的配置平均值从SCL 2.46/BANC39.84到SC246.51/NWMT223.60/WACM220.70，且官方全局q下各源局部超限率差异明显（SC/NWMT约15.9/15.7%，AVA/BANC/SCL低于0.2%）。HCP在删除一源后有限level升至1、q等于余源最大分，故其中位上升约58–67不能误读为某个源抬高全样本q；这是有限源校正变化。

结果仍不是算法创新或新域验证，但将研究问题收敛到“能否用合法公开低频发电构成预报小时风险异质性”。若进入下一轮，应以地区为独立单位做源留出，比较简单锚定和强风险基线同一信息预算；10地区不足以把小时样本当独立重复，也不能按已看尾部删源或调阈值。没出现明确增量前不冻结方法、不解封未看地区，长期目标继续active。

## 2026-10-27：单一月度POS特征不能解释跨源小时尾风险

按`docs/V50_COARSE_MIX_RISK_PREDICTABILITY_PROTOCOL.md`进行地区级LOO信息价值探针，使用合法已发布三个月窗口中POS份额均值，预测固定Scale-anchor+MCStd/B0/90d下的地区归一小时p90分数；同源7个outer结果先聚合成一个源响应。零点/风险重拟合，仅拟合预注册的单特征地区LOO诊断回归。70外区预测包、10外区标签包；目标档案/目标分数、PACE/LGEE和未看地区未打开。

月度POS单特征不能预测残余尾风险胜过训练区均值：p90 LOO MAE/RMSE分别为2.188/2.347，对比常数基线2.009/2.154；p99 MAE/RMSE为2.941/3.117，对比2.682/2.831。10折斜率方向虽均为正，地区相关仅0.26且误差更差，不能据此缩窄区间。结论只限10个已重复开发源、1个预注册特征：不能外推为全部月度信息无价值。

结果压缩了低信息解法的空间：当前POS均值无法解释源间风险超限差异；下阶段若继续，要独立检验低频燃料组分的时间变化或可得时间局部运行/计划支持信息能否在source-held-out比较中优于简单锚定及强条件风险基线。已知的更细燃料比例/天气等方案不可作为月报免费信息，必须明确新的信息预算、发布时点与口径。无可复现增量前不冻结、不解封真正未查看地区，目标保持active。

## 2026-10-27：月度完整燃料构成对跨源小时风险有初步信号，尚未过方法门槛

在独立协议`docs/V50_MONTHLY_FULL_MIX_RISK_PROTOCOL.md`下，仅使用V50冻结的十外源月度燃料数据与已封存source-level风险响应。POS条件因子实验依预设support guard失败：某些全三个月窗口POS mass为零，故该条件均值未定义，不做填补。重新预注册全燃料月度EF proxy（固定项目EF×8燃料占比）以定义零POS行；每个query严格采用当时可得的三条已发布月份，核对v4/v5输入原哈希。目标/未查看数据均未开，生产模型fit0。

十源LOO中，月度混合EF proxy将p90 source-risk的MAE/RMSE从常数基线2.009/2.154降为1.498/1.767，p99从2.682/2.831降为2.064/2.414；7/10留出源绝对误差下降，斜率方向10折均正。相比月度POS单量（p90 MAE2.188），分燃料月构成有初步价值。不过NWMT/WACM高尾仍被低估，SEC被明显高估；仅10个旧开发域，风险分数预后改善不等同区间覆盖/效率改善，且该proxy系固定EF而非官方中国碳核算。故只能作为下一轮方法假说，不可宣称突破或直接冻结。

下一关应在相同月度信息预算下，预注册构造一个source-held-out、粗混合强度条件化的尾部风险映射，并与简单锚定、全局equal-region校准、标准异方差及条件分位数强基线比较；报告每源覆盖、区间分数与宽度，不得用目标小时标签校准目标。先查条件/局部校准类文献避免把常规方法包装成创新。若在十已看源源外验证仍有稳定增量，再考虑完全未看源。长期目标继续active。

## 2026-10-27：月度混合条件阈值在整源留出中失去覆盖，不作为算法路线

按`docs/V50_COARSE_MIX_CONDITIONAL_INTERVAL_PROTOCOL.md`完成十地区LOO×七outer的小时区间测试。每个held source的所有小时标签完全不进入q或特征回归；同budget B0 `Scale-anchor+MCStd`作为主arm，比较九源equal-region q、月度全燃料mix单特征线性p90映射、九源全局HETScale，以及简单raw anchor×2。70外部包、10标签输入，生产点/风险重拟合0。所有source留出指标在`docs/V50_COARSE_MIX_CONDITIONAL_INTERVAL_RESULTS.md`及输出CSV。

单特征条件阈值的source-macro覆盖77.13%、宽493.6、interval score892.1；九源全局MCStd是86.86%/623.2/833.7，HETScale 87.30%/641.6/884.0；简单raw×2达到88.50%/528.5/712.7。条件法尤其在NWMT/WACM仅29.7/28.6%覆盖，显示月均风险排序不足以预测上尾；把它缩成local q反而覆盖失守。这里所有覆盖都是已开发外十源source-held-out描述，不是新域保证。

文学边界：localized conformal、条件/群体覆盖与空间局部QR+conformal均已存在（Guan 2023、Gibbs等2023、Jiang & Xie ICML2026；见`V50_CONDITIONAL_RISK_LITERATURE_NOTE.md`），通用按特征调q不能包装为原创机制。该条件q线路停止作为候选创新。本项目当前更有价值的贡献可能是明确本地信息预算与风险可识别性benchmark，比较合法月报、时间局部信息和小时响应需求，并记录信息不足导致的覆盖—效率边界。无稳定新法增量前不冻结、不解封真正未看源，长期目标继续active。

## 2026-10-27：逐日合法月度构成校正保留平均增益，但源区风险仍不合格

此前窗口均值探针把整个7–12月的月报特征用于窗口内每一天，早期日期有潜在未来信息泄漏；该数值保留为探索性诊断，不计作有效结果。按`docs/V50_DAILY_ASOF_MIX_LEVEL_PROTOCOL.md`重建逐查询日特征后重新做十源LOO×七outer。只打开70外源预测包和10标签包，未打开target/PACE/LGEE/未知源，点/风险refit=0。结果见`docs/V50_DAILY_ASOF_MIX_LEVEL_RESULTS.md`及`results/disjoint-calibration-v50-daily-asof-level/`。

严格as-of的三月八燃料固定EF proxy线性日水平修正在宏小时CRPS上为71.43，优于不校正125.14及等源全局bias shift112.80；factor2区间的宏覆盖95.27%、宽478.64、interval score510.42，对比无修正88.50%/527.98/712.16；日源偏差MAE由138.41/全局136.56降至88.93。相比简单全局修正，mix法在8/10源上小时CRPS与区间分数更好；相比未修正anchor，小时CRPS只在6/10源改善。

该宏收益不足以作为突破：SEC held coverage仅75.18%；SCL有88.88%粒子小时被[0,1000]边界裁剪，令该源表现不可放心解读；校正还使AVA/BANC/JEA/SCL的小时CRPS变差。十源都不是未知确认域，V50外部月表的时间戳不等于独立认证的原始公开发布时间；proxy与固定EF也没有中国口径验证。当前仅支持“月度构成可能帮助部分地区的小时预测日均水平校正”这一假说。仍不冻结方法或解封未知源。下一步应在冻结强同预算基线下检查固定EF/特征敏感性和每源尾部/裁剪稳健性；若仍有增量，先冻结严格确认合同，再测试此前真正未查看的地区。

## 2026-10-27：日历季节不能解释月度构成增益，待解决校正异质性

按`docs/V50_DAILY_ASOF_SEASONAL_CONTROL_PROTOCOL.md`补充固定annual sine/cosine强对照，比较无修正、全局bias、季节-only、mix-only、mix+season五种日水平修正。结果在`docs/V50_DAILY_ASOF_SEASONAL_CONTROL_RESULTS.md`与输出目录。七outer×十source上hourly CRPS分别125.14/112.80/113.28/71.43/71.62；mix-only在8/10源胜季节-only，日level MAE 93.47 vs142.33。简单日历季节性解释不了月度燃料构成信号，加入season项也几乎不改变mix结果。

增量仍不够冻结：SEC覆盖75.18%，SCL剪裁问题保留，mix-only小时CRPS仍有4/10地区劣于原始anchor，source macro改善有明显异质性。故当前结论是“月度燃料结构在十个旧开发源上含有超越日历基线的跨源日均偏差信息”，不是可信未见区方法。下一轮需仅用训练源选择校正强度并报告全 shrinkage 曲线，以压低SEC过度校正及SCL裁剪；不允许按held-source结果挑最优系数，也未打开未知源。

## 2026-10-27：收缩系数与硬适用域门控仍未解决极端组成迁移

按`docs/V50_MIX_OFFSET_SHRINKAGE_PROTOCOL.md`完整报告λ=.25/.50/.75/1曲线，没有从留出结果宣称独立最优。λ=.75在十源开发LOO中宏覆盖96.82%、最差NWMT 86.64%/WACM89.10%、CRPS74.58、宽497.69、IS526.64，优于raw anchor的88.50%/125.14/527.98/712.16；但SCL裁剪率仍86.1%(raw56.5%)，SEC在λ1回落到75.18%，所以不冻结。

再按`docs/V50_MIX_SUPPORT_GUARD_PROTOCOL.md`用训练源特征min/max做适用域回退。它将剪裁率从8.97%降到5.97%，macroCRPS92.78/coverage94.39%/width508.56/IS577.53仍优于raw，但SC与SCL均因处于训练特征范围外而100%回退；SC覆盖从无门控λ.75的99.29%降至75.06%，SCL回到raw路径。门控不能区分“有害外推”(SCL)和“有价值外推”(SC)，故否决作通用迁移规则。未知高燃料强度地区的外推依据仍不足，暂不冻结/开未看域。

当前核心研究缺口更具体：仅靠九源学习出的单一EF—水平误差斜率，无法安全处理特征支持域之外异质地区；硬回退会丢掉强区增益，完全外推则产生SEC/SCL风险。若继续，应优先识别新增的最小公开月度信息是否能区分源区的运行结构/基荷与边际火电角色，而不是再调全局λ。论文潜力仍更像“月度统计下CIF迁移的可识别性与失效边界”，算法创新未达标。

## 2026-10-27：月总发电规模没有补足燃料构成的迁移缺口

按`docs/V50_MONTHLY_ACTIVITY_SCALE_PROTOCOL.md`测试V50月表里此前未使用的`total_mwh/expected_hours`三月均值log1p活动规模，和三月燃料EF proxy作整源LOO。活动规模单独修正使CRPS155.79、覆盖75.25%、IS1407.56，劣于raw锚；加到mix模型后CRPS92.95、覆盖88.64%、宽466.95、IS647.56，也劣于mix-only的71.43/95.27%/478.64/510.42。SEC与SWPP在combined的覆盖分别72.2%/36.1%，提示绝对规模主要带来非迁移的source identity信号。数据与限制见`docs/V50_MONTHLY_ACTIVITY_SCALE_RESULTS.md`。

因此当前月度代理里，燃料构成比系统总规模更有信息，但尚不足支持安全预测小时风险，且固定EF proxy不能冒充中国核算口径。没有方法冻结，也没有打开未看区。研究问题继续收敛到：如何在有限月度统计下区分可迁移的组成驱动水平误差，与目标源特有的边际调度/小时尾部结构；单一强度标量、统一收缩和硬OOD回退都不足。

## 2026-10-27：低维燃料角色拆分改善水平分数，但区间增量未胜EF强基线

按`docs/V50_MONTHLY_COMPOSITION_ROLE_PROTOCOL.md`把低频组成压成fossil share、wind+solar、nuclear(以hydro为参照)，与固定EF proxy作同源LOO。EF+fossil将hourly CRPS 71.43降到64.53、日level MAE93.47降至72.38，却使覆盖从95.27%跌到86.34%，SC覆盖仅61.2%；更多组成系数的group-only模型不稳、裁剪21.0%。因此不是直接突破。结果见`docs/V50_MONTHLY_COMPOSITION_ROLE_RESULTS.md`。

固定fossil增量的λ路径中λ=.25宏CRPS63.35、覆盖96.14%、IS500.15、宽482.64，显得优于EF-only，但系数在同一开发路径后选，故再做训练源内嵌套留源选择。`docs/V50_NESTED_FOSSIL_ROLE_PROTOCOL.md`结果：70 outer folds中内层选择λ=0.25共33、0共22、0.5共13、0.75共2、1为0。嵌套结果CRPS66.62、覆盖94.87%、最差SEC85.48%、宽481.31、IS517.98；相对EF-only CRPS改善约6.7%，裁剪降低，但宽与IS略变差。虽优于raw factor2锚点，不足以证明强基线上的区间效率增量，且十域皆旧开发源；未冻结、未开未看域。

当前可支持的机制线索更具体：月度fossil share含独立于固定EF proxy的日均误差线索，但仅靠水平增量不能安全锐化小时区间。下一轮应把同一嵌套训练合同用于比较强风险校准臂（异方差、条件分位数/保形适配）在增量纠偏前后是否能改善同时覆盖和效率；若仍无增量，则论文贡献应收敛到有限月度信息下的分解基准和不可识别/迁移失败边界，算法突破未达到。

## 2026-10-27：嵌套fossil增量未胜V50 B0/B1强点臂的区间效率

为排除Scale-anchor特异补偿，按`docs/V50_NESTED_FOSSIL_STRONG_BASELINES_PROTOCOL.md`将相同内层整源λ选择用于全部5个封存点预测臂、7 outer×10 held source。结果见`docs/V50_NESTED_FOSSIL_STRONG_BASELINES_RESULTS.md`。月度EF-only在aligned B0路径上把CRPS/coverage/width/IS从raw 138.11/86.02%/516.52/763.54改善为67.93/95.43%/468.95/502.10；在aligned B1上改善CRPS和IS但coverage从91.92%到89.70%。这些结果说明低频月度组成可校正点水平误差，但最差源覆盖依然分别仅74.61%/43.44%。

嵌套fossil角色增量对EF-only的CRPS只有小幅改善：B0 aligned 67.93→66.20，B1 aligned70.94→70.91；对应区间评分都变差(502.10→515.11；569.97→574.26)，宽度基本不变。B0/B1 ReplaceR的因子2区间极度失覆盖，内层规则70/70次都回退λ=0。故燃料角色创新仍未形成coverage-preserving效率增量；确认域继续封闭。生产文件完成写盘后因最终print引用旧summary key以exit1结束，摘要及四CSV已完整保存且哈希对得上实际producer；报告已披露该无损的写后控制台错误，后续重跑应先修正并变更输出目录。

## 2026-10-27：月度水平修正后分开校准上下尾有开发集增量，算法创新尚未成立

按`docs/V50_CROSSFIT_ASYMMETRIC_RISK_PROTOCOL.md`交叉拟合EF-only日水平修正，再对B0-SizeET-AlignedKeep+MCStd比较90%对称与两个95%边际尾分位数。结果及逐源指标见`docs/V50_CROSSFIT_ASYMMETRIC_RISK_RESULTS.md`。该实验使用V50冻结的70预测包、10标签源和7个旧outer，没开target/PACE/LGEE/未知源，也没有重拟合点头或风险头。EF-only separate-tail相对EF-only symmetric宏coverage为93.82% vs95.47%、宽415.57 vs475.15、区间分数447.47 vs512.83；最差源coverage从WACM 83.08%提高至SWPP 86.52%。排除SCL只作敏感性诊断后，宽和IS增益仍约12.6%/12.9%，最差源86.52%，说明并非完全由SCL极窄区间驱动。但SWPP/SEC等源coverage仍低于90%，且配置在3×3开发网格看过后选出；十源都是重复开发源。

文献边界更明确：非对称保形、异方差分位数和skew-adaptive conformal已有直接工作，故不能把两个尾分开校准宣称为新算法（Romano等2019 CQR；Marques & Graziadei 2026 skew-adaptive conformal）。当前只得到“EF水平修正可能与已知的非对称区间规则互补”的开发信号。`external-v6-data`的AZPS/DUK/LGEE/PACE/PSCO/SRP均为此前研究过的目标，不是未知地区；但V6先前冻结时的资格审计把TVA定为第七合格、仅因固定前六入选而保留，独立库存未发现其历史预测或评分文件，只有完整性元数据被看过。因此依照原先“增量明确后冻结并测未见区”的授权，已经先冻结TVA 2026H1方案：`docs/V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md`。预测已对七个外层的181天网格封存后才评分；首个评分器因漏掉V6原始快照提供的2026-01-01边界日，在产生任何指标前失败。按既定V6审计登记边界修复后，TVA确认结果为负，详情见docs/V50_TVA_CONFIRMATION_BOUNDARY_AMENDMENT.md与docs/V50_TVA_ASYMMETRIC_CONFIRMATION_RESULTS.md。该单区结果否定这次候选的迁移主张，但不代表所有新地区都会失败，也不支持中国迁移结论。
## 2026-10-27：TVA新地区确认否定分尾校准迁移

对十个已用开发源交叉拟合EF-only水平修正后，预注册候选“B0 MCStd + 上下尾各自95%分位数”相对对称版本在开发集有宏效率增量，遂按原授权冻结并选择V6确定性质量顺序中未评分的第七候选TVA。先验协议和结果见docs/V50_TVA_ASYMMETRIC_CONFIRMATION_PROTOCOL.md与docs/V50_TVA_ASYMMETRIC_CONFIRMATION_RESULTS.md。

七个V50外层函数×181个TVA日预测先封存。首次计分因只读2026文件漏掉由2025原始快照提供的2026-01-01边界日，在写任何指标之前支持门禁停止；没有产生目标比较。按V6审计预先指定的native snapshot 7+8规则登记边界修正，再运行冻结计分器。错误和修正公开记录在docs/V50_TVA_CONFIRMATION_BOUNDARY_AMENDMENT.md。目标标签在预测封存后才用于计分，且需披露首次失败的计算访问。

TVA单区结果为：候选coverage77.67%、width191.77、IS294.46，下尾miss22.33%、上尾0%；相同EF修正的MCStd对称强对照coverage95.30%、width207.63、IS221.83；简单B0 factor2锚点coverage90.72%、IS269.85。候选coverage不足、且IS比强对称方法差32.7%、比简单锚差9.1%；其4/6月coverage降至63.89%/55.28%。完整密封指标在results/disjoint-calibration-v50-tva-confirmation/score_seal.json。不得在TVA上改q或换方法后重计确认。

结论：开发区上的分尾重分配没有迁移至新区域，开发时改善WACM上尾不意味着TVA可安全缩短下尾。保留此负确认，拒绝该候选的迁移/算法贡献主张；单一地区失败也不能证明普遍失败或中国域表现。V50/V6冻结源数据未改，未打开其他目标。近期下一步是在可用开发数据上判断公开月度特征是否能在测试前识别尾向风险迁移；若不能，论文应集中在月度低频信息预算下的迁移基准、覆盖—效率边界和不可识别性，而不是继续调通用分位数规则。

## 2026-10-27：月度燃料特征不能可靠预测留出源的尾向失衡

按docs/V50_TAIL_DIRECTION_INFORMATION_PROTOCOL.md预注册，仅复用十个旧开发源的source-macro symmetric EF+MCStd尾部miss，不重开小时标签或预测包。把每源七个outer聚合成一个响应，用90日合法支持上的三月平均EF、化石份额、风光份额分别做单特征整源LOO，和训练源均值基准比较。结果见docs/V50_TAIL_DIRECTION_INFORMATION_RESULTS.md。EF特征MAE仅0.05394→0.05289，RMSE从0.07011微升到0.07083，尾向符号准确率与均值基线相同为4/10；化石/风光MAE与RMSE均恶化。

因此，已经测试的低频燃料构成不足以提前判断目标地区下尾还是上尾风险更强；用它设置源区特异尾部q没有证据。此诊断n=10、且都是重复开发源，不证明月度信息全部无价值。当前破局问题进一步收敛为：预测前什么最小本地小时风险信息能约束尾失衡；没有该信息时，何种保守不确定性保留能诚实给出覆盖—效率边界。下一步应事前固定具体信息预算做本地风险测量价值实验，对比月度基线与强风险方法；未见地区继续关闭，算法贡献仍未满足。

## 2026-10-27：四个本地小时CIF标签仍不足以解决超阈日识别

按`docs/LOCAL_HOURLY_QUERY_BUDGET_V23_PROTOCOL.md`和两份边界修订，模拟同一固定源区历史日上b=0/1/2/4小时CIF查询，提供方同时免费得到同期T8 H/G和全部G24。这是极宽松的源历史oracle预算，明显超过中国当前只有月/年公开燃料统计的可部署输入；新增小时CIF属于本地碳标签。七个已开发源取V20继承队列首末两日，共98源区日；主阈值300，100为敏感性；精确标签与10 g/kWh分箱均保留。R1/R2合计896行，98个源区日不是98个独立区域。R2严格使用floor分箱左闭右开，只让最末分箱含1000；两轮汇总相同。错误首跑及数值行访问和R1分箱边界问题均保留在修订说明中。

主阈值300下，b=0源宏未决率50.0%。b=4精确标签时，固定时钟、最宽可行CIF和一步minimax把未决率降至35.7%，随机查询仍50.0%；但最佳自适应查询相对最佳固定策略增益0个百分点。10g/kWh标签下所有策略仍为50.0%。`K>=6`未决率仍71.4%，K跨度约12小时/24小时；所有已认证事件正确且真实K始终在可行集。100阈值下最宽CIF策略b=4未决率从14.3%降到7.1%，只对应小样本内有限几日，不能推广。

这一少标签预算未达预注册门槛（b=4自适应策略须在精确和量化模式都比最佳非自适应至少降低10pp），且本身不构成原创主动学习方法。它只说明在非常富的同期H/G+G24信息上，再添四个本地小时CIF点仍不能稳定确定过去是否存在C>300；月/年统计预算下更不能把该实验作为部署机制。未开其他目标或未查看区；研究缺口依然是找到真实可得、发布时点合规且能排除目标小时燃料替代/尾风险的最小本地信息，或诚实保留不可识别部分。详见`docs/LOCAL_HOURLY_QUERY_BUDGET_V23_RESULTS.md`。

## 2026-10-27：月度组成变化度没有解释额外小时尾风险

按`docs/V50_MONTHLY_MIX_VOLATILITY_PROTOCOL.md`，对合法三月混合强度序列计算相邻绝对变化率，并在十个已用源上做固定特征集合的整源LOO。全燃料均值level对p90的MAE/RMSE为1.498/1.767；变化度单特征为1.861/2.315；level+变化度为1.601/1.754。p99上level/变化度/联合RMSE依次2.414/3.377/2.727。单纯level仍最佳；联合模型仅在p90 RMSE上有极小改善，同时MAE和p99退化。故暂不把变化度加入条件风险模型，也不宣称它能解释调度不稳定。结果见`docs/V50_MONTHLY_MIX_VOLATILITY_RESULTS.md`。

## 2026-10-27：月度局部阈值的嵌套收缩全部退回全局校准

按`docs/V50_NESTED_MIX_Q_SHRINKAGE_PROTOCOL.md`在每个七outer×十整源留出折中，以9个训练源内部LOO选择月度条件q与全局equal-source MCStd q的凸组合；内层要求宏覆盖90–95%、最差源≥85%，再在合格项中最小化区间分数。70/70折没有任何λ通过，严格回退到λ=0。外层结果与全局MCStd逐项完全相同：coverage86.86%、width623.18、IS833.66；直接局部q仍为77.13%/493.57/892.10，简单factor2锚点为88.50%/528.53/712.72。数据、source-macro结果及哈希见`docs/V50_NESTED_MIX_Q_SHRINKAGE_RESULTS.md`。

这说明训练源内嵌套选择能识别出当前月度特征支持不足，但没有把它转成新区域收益。均值proxy的风险相关性目前无法经校准成为可信的局部区间；继续增加同类门控/收缩没有证据，且这些都是十个反复开发源。仍未找到算法突破。下一优先级应回到目标函数和信息合同：用用户真实可取得的月／年表定义粗强度的sharp区间，再在不偷用小时G/标签的条件下拆分迁移模型输出为“统计已识别”和“源假设补充”两部分，且与普通预测基线在相同信息预算下评分。未开真正未知区。

## 2026-10-27：V51官方月报字段审计确认分类与边界风险

按`docs/CHINA_MONTHLY_GENERATION_FIELD_AUDIT_V51.md`核对国家统计局2026-09能源生产公报、指标定义、统计问答和河南能源监管办2026-05全口径发电月表。常见公开月报确有火/水/核/风/光等分类；河南样例另列储能，但未将火电拆成煤/气/油。国家统计局定义的火电还包含生物质与余热、余压、余气。国家统计局月度公报是规上工业统计，河南此表为全口径，因此两种来源不能直接并表。

这支持用户月度可再生/不可再生字段对粗组成诊断有价值，但尚不等于煤/气份额，也不等于项目固定EF的正EF组。正式方法须先核用户原表字段和发布时点；月/年度总量在物理上仍不决定小时调度、小时排序、G/净交换或尾部风险。V24历史粗水平集合、V50开发集月度均值相关与小样本尾风险失败都需保留在各自统计边界内。

本项没有读取用户表或目标小时标签，也没做模型数值实验。当前最有价值的下一步是依实际 schema 明确粗/细月预算及年度-only预算，分别把“硬统计识别集合”与“源模型概率补充”作为独立输出，与同预算锚定、标准动态重组、异方差和条件分位数基线比较；旧源必须出现可归因覆盖—效率增量后，才冻结并确认真正未查看地区。没有算法突破或冻结决定。

## 2026-10-27：12月组成均值是略稳的源风险描述量，但不能校准小时尾风险

按`docs/V51_MIX_LOOKBACK_RISK_PROTOCOL.md`预注册，比较最近1/3/6/12个合法已发布月的整燃料固定EF proxy，在10个旧源的风险分数p90/p99整源LOO。12月窗MAE/RMSE最低：p90 1.425/1.649、p99 1.875/2.288；已用3月窗为1.498/1.767及2.064/2.414。四个窗口均有10/10正斜率，但均仅7/10地区优于空均值。改进温和、不可作为独立确认；更长期平均可能刻画稳定地区组成差异，而非易变小时调度。

按`docs/V51_12M_NESTED_Q_PROTOCOL.md`把该单一12月特征送入原7×10整源嵌套阈值选择。70/70折没有任何λ满足内层90–95%宏覆盖和最差源85%门槛，全部选择0并逐行等同全局MCStd。直接local q覆盖72.11%、宽494.28、IS810.26，失覆盖且劣于简单raw×2锚定的88.50%/528.53/712.72。SCL local区间平均宽仅3.07、覆盖6.65%。数据与逐源结果见`docs/V51_12M_NESTED_Q_RESULTS.md`。

结论是：长窗燃料proxy对地区平均风险排序有弱信息，但当前十源证据不能将它转成可信小时局部尺度；不再沿同一标量扫窗口或收缩。这一负结果压缩了“用静态月mix局部调q”的路线空间。用户本地字段尚未按原始表头核定，中国边界/因子和真实发布日期未验证，未知地区未开；算法创新仍未达到晋级门槛。下一步需要改研究对象（将可识别的月度能源加权强度与不可识别的小时尾部分开评测）或寻找真正能揭示小时风险自由度的合法特征，而非继续调现有月mix阈值。

## 2026-10-27：组成×季节水平交互未修复小时区间风险

针对TVA对称月度中心校正春季coverage下降，新增一次只在旧开发源上的线性交互诊断：月EF＋sin/cos＋EF×sin/cos，严格和已有EF-only/EF+season相同V50袋、90d标签、整源LOO、同factor2 spread匹配。结果见`docs/V51_SEASONAL_INTERACTION_LEVEL_RESULTS.md`。交互相对加性季节的hourly CRPS71.620→71.885、level MAE93.762→94.252，皆恶化；区间分数仅510.168→510.028微变、宽度478.890→478.244，覆盖94.92%，逐源方向异质。没有达到预设5%增益，且非独立源集。

因此不再通过增添时间交互拟合V50 proxy；TVA春季失效不能用这些可见源的简单季节斜率变化解释。当前相对有希望的仍是月组成作中心修正、小时shape风险单独保留，但其开发源最差覆盖83.08%、TVA 4月覆盖82.08%，且分尾候选在TVA失败。尚无方法突破/冻结/未知区测试。下一方向应改为识别集合＋概率形状补充的明确定义，或获取可观测小时风险自由度的新信息，而不继续扩展相同标量回归。

## 2026-10-27：日水平与日内形状分量给出有限开发增量，未达到突破

按`docs/V51_LEVEL_SHAPE_INTERVAL_PROTOCOL.md`对每一天 128 条封存小时预测样本拆出日均水平尺度与去日均形状尺度，再按两尺度之和归一化逐小时误差，源平衡校准90%阈值。十个旧开发源上的探索结果见`docs/V51_LEVEL_SHAPE_INTERVAL_RESULTS.md`。EF交叉拟合双尺度候选宏覆盖94.79%、宽437.98、区间分数483.29；同修正MCStd是95.47%/475.15/512.83，简单×2锚定是95.47%/470.58/503.60。它相对MCStd分数下降5.8%，但覆盖少0.68pp；相对简单锚仅6/10源区分数更优，WACM覆盖83.58%。

EF置零消融分数在十源上全部更差（宏分数836.98、覆盖87.14%），提示开发收益主要依赖月度中心修正；不构成独立信息/算法贡献证明。两个其他试过的构造分别以双95%分量相加或日最坏小时校准，前者过覆盖99.95%、宽735.98，后者覆盖98.11%、宽530.57，均更差。逐小时双尺度候选是几种探索定义中事后筛选出的结果，故作为探索信号报告，不按预注册确认或新颖算法主张。没有真正未见源验证、没有中国 EF 适用性证据，也没有找到足以冻结的突破。下一步先把该公式单独冻结并定义源级覆盖晋级线；如果数据库存无法证明存在未参与探索的地区，不以旧源重测冒充确认。

## 2026-10-27：严格小时可行范围显示，小时总量不够，排放正组小时量有信息价值

用户目前可取得月度各发电类别统计。已知的数学识别对象是燃料 EF 明确且口径完整时的发电量加权月 CIF；它不识别普通小时均值、小时排序、尾部概率或未来小时 risk。既有`source-aggregation-bias-v4`反例甚至固定了每小时总发电量和月燃料总量，仍得到不同时间均值。

本轮没有重做反例，而是对旧开发源203个完整月、148,104小时计算每小时sharp可行边界。仅月总量的平均区间宽838 g/kWh；再给小时总发电量，仍宽765，收紧约8.8%。再给每小时正项目EF燃料组发电量，边界平均宽降到220、月内p90宽度加权均值265，覆盖观测代理值100%；小时CIF实际月内标准差按小时数加权平均为37。正组oracle信息带来清晰物理边界增量，但各源差异大，ERCO/MISO边界仍宽，且这是已看过的美国源数据和固定项目EF代理。

详细定义、算法和研究边界见`docs/V51_PARTIAL_IDENTIFICATION_PROTOCOL.md`及`docs/V51_PARTIAL_IDENTIFICATION_RESULTS.md`。该正组小时序列在中国是否公开未知，也不能拿官方“火电”替代（其可含煤、油、气、生物质、垃圾及余热等混合口径）；月表类别和发布边界仍须逐列核实。文献中小时平均mix计算通常需要小时燃料发电量，而边际EF又需小时系统变化数据；两种estimand不能混称。本轮支持把论文聚焦为**低频信息合同下的识别边界、最小本地信息价值和迁移预测基准**，但连续背包/transportation bounds本身是自然优化构造，不构成新算法主张。未确认中国可取得的额外小时列，因此没有冻结方法或开启未见地区。尚无突破。

## 2026-10-27：小时组信息阶梯显示风光时序有用但排放正组更关键

按`docs/V51_INFORMATION_TIER_PROTOCOL.md`在同一完整月、同一月燃料总量和小时总发电量下，逐个加入风光、水风光、煤气油、项目正 EF 燃料组或煤的小时合计，计算sharp可行范围。结果见`docs/V51_INFORMATION_TIER_RESULTS.md`。平均边界宽：只加小时G 764.68，风光635.55，水风光547.14，煤气油414.32，完整正 EF 组220.11 g/kWh；每一组都覆盖真实代理值100%，这是物理界定义验证。结果说明小时组别提供的信息价值差异很大，排放正组量比天气相关可再生总量更直接约束代理CIF。

所有分组量都是已查看旧源小时真值，属于 oracle 信息预算；中国是否公开得到对应小时组未证实。尤其官方火电分类不等同项目正EF组。下一步若追天气路线，应外层留源预测风光/正EF组时序，再传播预测不确定性进CIF边界，不能把天气点预测当实测约束或借此宣称100%覆盖。锐界本身是常规优化，论文增量仍需来自信息预算基准的跨域检验或有稳定收益的可迁移组量预测机制。当前没有达到突破，也未开未查看地区。

## 2026-10-27：本地辐照度塑形不能迁移改善 as-of 小时 CIF

先以相同合法 as-of 月总量（上一年同月、发布可得时点早于目标月）比较跨源 2024 月×小时气候形状和本地归档辐照度替换太阳能小时形状。37个有效源月上，气候形状宏 MAE/RMSE 为39.55/48.71，天气替换为47.08/57.93 g/kWh；总量预测误差完全相同。CISO、ERCO、FPL明显恶化，其余来源近乎不变或轻微改善。然后以其他来源选择混合权重，做7次外层留一源校准，alpha网格{0,.25,.5,.75,1}。七次均选alpha=0，源平衡宏 MAE/RMSE为39.93/49.42；完全天气形状为46.50/57.45。说明 oracle 月总量下风光 shape 指标的正向表现没有转化为合法月总量预测下的小时 CIF收益，当前 weather→solar shape 机制否证。

协议、结果、边界见`docs/V51_WEATHER_SHRINKAGE_PROTOCOL.md`与`docs/V51_WEATHER_SHRINKAGE_RESULTS.md`；输入/输出哈希及逐源数据在`results/v51-weather-shrinkage-calibration/summary.json`和同目录 CSV。七源及2025H1标签早已用于开发，档案天气预报发行版本亦未完全核实；所以这只是探索性源外校准诊断，不是未见地区确认。此天气机制不冻结、不再宣传为候选创新。下一步转向合法月度类别总量的时效预测误差如何传播至小时 CIF 和区间风险，并按用户实际表格字段合同限制信息预算。仍无突破。

同轮合法月度类别总量预测对比中，上一年同月优于最新发布月份外推和3月滚动均值：总量误差5.80% vs 10.81%/11.25%，燃料向量误差15.69% vs 20.96%/21.71%，固定EF月度能量加权 proxy MAE 15.29 vs 27.48/27.20 g/kWh。之后把天气太阳能形状测试改为目标月太阳能总量 oracle，天气仍比气候形状差：39.03 vs 46.90 g/kWh。说明近期单月动态与月总量太阳能估计均不足以解释天气塑形失败。完整限制和结果见`docs/V51_MONTHLY_ASOF_BASELINES_RESULTS.md`。这些都在已查看源上、部分代理EF和归档天气条件下；没有新颖方法收益，也没有未见地区确认。

## 2026-10-27：五类月量条件下火电组内有效 EF 是更关键的 oracle 信息

把代理月报折叠为thermal(煤+气+油+other)、核、水、风、光，做37个已查看源区月的合法as-of小时CIF诊断。上一年同月五类量＋跨源火电组EF的region-month加权MAE/RMSE为57.53/66.52；即使用目标月五类总量oracle，仅为56.74/65.82。保持合法五类总量，仅把火电组内有效EF换成目标月oracle，MAE/RMSE降至34.78/43.20；目标区上一年同月EF合法可得，MAE/RMSE为34.93/43.61；类别总量与火电EF均用oracle时为31.18/38.80。先按来源平均再七源等权的六个MAE依次为57.91、35.29、35.13、57.45、32.83、31.38。MISO源级MAE由跨源EF的153.60降至本地滞后EF的51.61、目标月EF oracle的46.02。等源外层CV在{0,.25,.5,.75,1}收缩网格七次均选本地滞后EF全权重，跨源收缩未提高结果。这是更具体的正向信号：目标区合法历史火电内部水平明显比借源均值有用，但依赖火电燃料细分且只在已看代理源上的点预测。

协议、结果见`docs/V51_COARSE5_THERMAL_EF_VALUE_PROTOCOL.md`和`docs/V51_COARSE5_THERMAL_EF_VALUE_RESULTS.md`；oracle逐月产物在`results/v51-thermal-factor-value/summary.json`，合法滞后与收缩产物在`results/v51-lagged-thermal-factor/summary.json`和`results/v51-thermal-ef-shrinkage/summary.json`。组映射与固定EF只属EIA类代理，非中国核验。下一步必须把用户实际月报列映射到可计算的组内scalar：若火电细分可得，测试as-of前瞻估计与简单/概率基线；若只有火电合计，该维度仍不可识别，不能以代理oracle收益声称当前预算已经解决。此次没有覆盖或区间效率证据，算法新颖性和未查看地区确认仍未达标。

滞后因子外层收缩也跑完：每次只用其他六源选择alpha∈{0,.25,.5,.75,1}，七个外层均选alpha=1。等源MAE donor-only 57.91、合法本地滞后EF 35.29，RMSE 66.65→44.31；目标月火电EF oracle为35.13。V51的逐燃料基线同配对来源等权39.93/49.42，本组形状＋滞后EF为35.29/44.31，但这改变小时模型，不能归因成单一scalar消融。当前方法仍只是合法历史锚点，没有风险覆盖结果。文献核查提示月量约束参考 profile 的小时分解、source-specific hourly forecast→CIF层叠都已有相邻研究，因此原创空间必须落在组内水平状态与小时尾风险不确定性分离、并证实跨区覆盖效率；详见`docs/V51_THERMAL_EF_SHRINKAGE_RESULTS.md`。目标仍未完成。

## 2026-10-27：本地滞后火电 EF 改善点值，但来源风险校准无法缩窄区间

对使用合法 lag-EF 锚的小时点预测器，以其他六源的2024H1残差校准2025留出源。源等权 pooled 对称90%覆盖90.59%、宽177.14、区间分数232.95，但ERCO/MISO覆盖69.40%/82.14%；分尾区间coverage仅87.41%。每个校准源q90取max可到96.30%宏coverage、宽246.04，ERCO仍83.03%；max-source q95升至98.10%、宽285.02、IS293.07、最差ERCO89.94%。因此失覆盖只能靠显著过覆盖和加宽缓解，没有在保持区域覆盖时缩窄区间的进展。结果与逐源哈希见`docs/V51_THERMAL_EF_INTERVAL_RESULTS.md`及`results/v51-thermal-factor-intervals*/summary.json`。这一轮保持点误差35.29不变，仅比较区间；所有源均已查看，不是独立地区确认，仍未形成算法突破。
## 2026-10-27：月度组成条件风险尺度跨半年有宏观收益，但未解决最差来源

对前一轮月度风险尺度特征消融不再调参，原样把校准窗由2024H1→2025H1延伸到2024H2→2025H2。最接近用户现有月报预算的`mix_only`（热组占比、风光占比）在H1/H2宏观区间分数为218.22/200.35，均好于各自pooled q90的232.95/215.07；H2宽度也略窄（174.70 vs176.83）。但最差源仍为ERCO，覆盖只有78.75%/81.66%，未达到90%；所以宏观分数改善不能算维持区域覆盖下的区间效率增量。

加入滞后热组EF的`level_mix`在H1分数较低但ERCO覆盖79.22%，H2覆盖跌至74.00%；`all5` H1最低源覆盖85.67%、H2仅80.03%。这说明扩展协变量没有稳定修复源间迁移尾部；同时用户已知的中国月表字段不能确认煤/气/油细分，代理实验中滞后热组EF不可直接视为可用输入。H2仅六个来源有可用支持月，月份不齐。两个半年都来自反复开发的美国代理源，仍不是独立新地区确认。

完整表、信息合同和局限见`docs/V51_CONDITIONAL_THERMAL_RISK_RESULTS.md`及`results/v51-conditional-thermal-risk-ablations/`、`results/v51-conditional-thermal-risk-h2-forward/`。目前可保留的结论是月度组成对平均风险分配有弱信息，但不能识别目标区小时尾风险；不冻结该异方差尺度方法。下一步转为月度统计识别量与小时风险未识别部分的显式基准，不继续增加相似风险回归特征。仍无突破、未开未看地区。
## 2026-10-27：严格月度类别预算无法恢复区域水平或小时尾部

前一轮条件风险 `mix_only` 的点头仍使用目标区滞后 thermal EF，因此不符合用户当前确认的月报信息预算。按`docs/V51_MONTHLY_ONLY_RISK_PROTOCOL.md`去掉本地EF和本地小时形状，仅用目标区合法滞后五类月量重组留源先验，风险头用热组/风光份额条件化。H1 pooled/conditional宏覆盖86.97%/88.75%、区间宽278.04/271.30、分数430.10/394.51，最差MISO覆盖26.57%/32.28%；H2分别83.19%/85.14%、宽275.63/270.48、分数491.86/459.79，最差MISO仅11.97%/15.32%。条件方案降分缩宽是以低覆盖为代价，明确未过可信区间门槛。

本地热组因子的旧同源消融显示MISO点MAE可由donor 153.60降至本地滞后EF 51.61，但仍没有修复另一高风险源ERCO区间。故现在分出两个未解决自由度：本地类别月量不足以识别热组内部排放水平；即便补充月度组内EF，仍不足以识别小时调度和尾风险。完整结果与严格信息合同见`docs/V51_MONTHLY_ONLY_RISK_RESULTS.md`。这仍是已看美国代理源，不是中国或未查看地区确认；没有冻结方法或形成突破。
同一H2来源/月份的跨预算配对给出更具体的信息价值：目标五类月量＋跨源热组EF donor点头的MAE 61.65、pooled区间覆盖83.19%/宽275.63/分数491.86；额外加入目标区合法滞后热组EF后，MAE 31.52、覆盖92.26%/宽176.83/分数215.07。最差覆盖却从MISO 11.97%变为ERCO 70.16%，仍不可信。该本地scalar显著解决平均热组水平，不解决小时尾部。结果和信息预算限制并入`docs/V51_MONTHLY_ONLY_RISK_RESULTS.md`；中国数据字段未核实，仍无突破。
为排除UTC日内轮廓相位错配，又在严格月度预算下只替换UTC与地区本地钟点分箱。H1源等权MAE/RMSE 57.91/66.63→56.51/64.35，H2 61.65/68.65→61.28/67.59；收益主要集中CISO H1，MISO无改善且保持约+154至+158 g/kWh有符号偏差。时区对齐保留为标准基线，不能解释主要地区水平偏移、也不构成创新。结果、假设及MISO多时区局限见`docs/V51_TIMEZONE_PROFILE_RESULTS.md`，复现脚本`scripts/v51_timezone_shape_probe.py`。
按计划对严格月度预算的24格本地钟点、48格weekday/weekend×小时和168格星期几×小时做源留出对照。H1最优48格仅将MAE/RMSE从56.515/64.351微降至56.413/64.201；H2反而由24格最好，星期几×小时两期均无收益。MISO偏差不变（MAE约154–158），说明增加日历形状细度没有处理热组水平错误。详见`docs/V51_DAYTYPE_SHAPE_RESULTS.md`。不冻结日类型机制；下一轮转向目标区燃料内部水平的可识别范围和年度/本地信息价值，而不是继续加同类小时特征。
## 2026-10-27：年度热组 EF 保留同月锚的大部分点误差收益

固定同一五类月量和源外上一年本地钟点shape，只改变热组EF；年报发布延迟显式取30/60/90/120天。目标区上一年年报（d90）的H1来源等权MAE/RMSE为32.60/40.93，H2为30.33/37.37；同月滞后为32.93/41.40及30.16/37.22；跨源donor为56.51/64.35及61.28/67.59。年报相对donor显著改善，并在H1/H2分别距同月MAE约−1.0%/+0.6%，是目前“年度本地水平信息是否可替代月度细分”的强正向点误差信号。最近12月估计支持稀疏，仅6/3个来源，不晋级。

此结果仍是已看美国代理数据上的固定EF点预测，年报发布时间是假设；未测覆盖，年报也未证明中国火电边界可计算。详见`docs/V51_ANNUAL_THERMAL_EF_RESULTS.md`。下一步在同一前推窗测试年度EF点头的区间覆盖与区间效率，再决定是否保留为最小本地信息候选。尚无算法突破或未看区确认。
继续对年度 EF 做2024H1/H2→2025H1/H2源留出区间前推，并与同月滞后 EF、pooled q90、月度组成条件尺度配对。年度EF的宏覆盖/区间分数保持在约90.97–94.26%/199.75–227.75，但最差覆盖仍为ERCO的64.76–79.06%，全部不达85%门槛。H1条件尺度年度宽185.13，反而宽于同月EF的174.43；H2年度宽166.84略窄于同月169.99，但最差覆盖73.86低于76.01。年度 EF 解决MISO水平偏差后，ERCO小时尾部成为瓶颈。结果记录在`docs/V51_ANNUAL_THERMAL_EF_RESULTS.md`和`results/v51-annual-thermal-ef-intervals/`；目前仍无可信覆盖突破。

随后按精确恒等式分解年度d90点误差，并用真实小时热组份额替换donor份额作不可部署oracle。ERCO形状项MAE/RMSE为H1 66.88/79.83、H2 55.14/67.04，远高于内部EF项14.56/17.96、10.90/13.50；负交叉项显著，不能把分项RMSE相加。小时份额oracle的剩余误差降至14.56/17.96和10.90/13.50，指出最主要的ERCO机制缺口是小时热组份额轮廓迁移。已有相似日、静态/动态形状及季节×相似日组合探索，因此再做常规donor加权缺乏新颖空间。对中国仅月报预算，小时份额不可识别；下一步应将已知月度总量下的小时分配风险做成显式部分识别/最小本地摘要价值实验，而不能声称年均EF消除了小时风险。详见`docs/V51_ANNUAL_THERMAL_EF_RESULTS.md`及`results/v51-annual-thermal-ef-probe/source_macro.csv`。这是机制定位，不是方法增益或未查看地区确认。

继续量化最小本地信息的时间分辨率：对已查看七区2022–2024 H1共90,168小时，只增加逐小时总发电量及块级正EF组总量（组内正因子400–1000），计算该简化信息集下的逐小时sharp bounds。块级正组量从月、24h、12h、6h、3h到1h时，宏观平均边界宽度为1000、977、957、918、854、298 g/kWh；6h到1h之间才明显收紧。6h下CISO、ERCO、FPL、ISNE、MISO、NYIS均约999–1000，BPAT为431；逐小时正组量下地区均值43–422，但宏观平均宽仍约298，是小时CIF标准差125的2.4倍。信息值输入是回顾性真值，区间不含逐燃料月库存约束，不能当覆盖率/预测成果或中国可得性证明。结果见`docs/V51_COARSE_POSITIVE_GROUP_INFO_RESULTS.md`、`results/v51-coarse-positive-group-info/`及`scripts/v51_coarse_positive_group_info.py`。该结果将“最小额外观测”收窄到接近小时热组序列；尚无算法突破。下一步须合并现有月度细类别库存与块级总量约束，检查是否改变时间分辨率结论，并对照强预测区间而非只报物理边界。

继续将逐燃料月库存与六小时热组汇总纳入同一有限维LP：2024 H1七源每月中旬各一完整日，共1,008小时逐小时求上下界。相比只用小时总量+块级正组总量，真实月度八燃料边际把总体均宽从917降到682（−25.6%）；BPAT/FPL分别收窄60%，CISO 20%，ERCO/MISO完全没有收窄。联合锐界在ERCO/MISO仍为约998/1000 g/kWh，平均宽约682仍是实际小时CIF标准差120的5.7倍。说明月表细类别与粗时段信息有互补，但无法保证跨区小时识别；LP为成熟方法，不能主张算法原创。脚本、样本协议、源别结果和输入哈希在`scripts/v51_joint_month_block_bounds.py`、`docs/V51_JOINT_MONTH_BLOCK_BOUNDS_RESULTS.md`及`results/v51-joint-month-block-bounds/`。下一步同合同比较1h新增组量与6h新增组量的联合锐界，并考虑模型预测集合是否有用；仍无算法突破。

完成配对联合界：同样月库存和样本，只将正EF总量从6小时块细化为1小时；在每来源月四个固定时点、共168小时上，总体联合界均宽682.17→178.44（−73.8%）。1小时正组量加月库存后BPAT/FPL样本点宽度归零；CISO/ISNE/NYIS仍163/193/181，ERCO/MISO仍331/381 g/kWh。于是需要两类本地信号：小时热组量约束组份额时间形状，小时内燃料组成约束正组EF跨度；一个小时“火电总量”字段依然不足以在所有地区消除内部燃料不可识别性。该样本仍是2024 H1已查看美国代理区、事后真实月库存/热组oracle、每月四钟点，不能等同预测效果或中国实证。结果见`docs/V51_JOINT_RESOLUTION_BOUNDS_RESULTS.md`、`scripts/v51_joint_resolution_bounds.py`及`results/v51-joint-month-hour-bounds/`。这是分层信息边界证据，尚未产生/验证算法突破。

核对WWW'26 DGCFM全文后明确：其zero-data是训练排除目标区，但推理TSFM输入目标区历史CIF，因此不同于TransCIF目标的“零目标小时历史、只给滞后月报”。同时CarbonMonitor-Power中国记录是用月度CE C类别电量和八省日耗煤估计拆出的日频序列，中国本身无该论文所称的全国小时高频原始数据；它不构成中国省级小时标签。该边界可支撑一个更严谨的跨域信息合同 benchmark，但基准和锐界尚不足以冒充算法创新。详见`docs/V51_RELATED_WORK_ZERO_TARGET_HISTORY_GAP.md`；对DGCFM的复现/同数据协议基线仍需做。

另将预测对象降到日均：同一来源月真实逐燃料总量+真实小时总发电量，对3,757个完整日做每日平均CIF线性运输锐界。整体均宽627 g/kWh、p90为1000；ERCO/MISO平均宽仍1000，远大于样本日均值跨日SD64/60。此条件给了未来目标月最终燃料库存，是日均预测的事后oracle；结果表明仅把小时输出改成日均也不消除月内日际燃料重新排序的不确定性。锐界100%包含真值是优化保证，不是预测覆盖。见`docs/V51_DAILY_AGGREGATION_BOUNDS_RESULTS.md`和`results/v51-daily-aggregation-bounds/`。所以当前月度-only数据无法支撑可信窄小时或日均集合；唯一由月表直接约束的是发布后同口径月度发电加权强度。需继续以模型强基线和外层跨区回测检验边界内能否预测，不能把物理可行集合直接当概率预测。

把上述歧义结构改成真实预测输入特征：滞后月度热总量、donor小时总发电预测以及donor热组EF导出的月内thermal-reallocation宽度(mean/p90)，在原mix-only两特征之外训练source-balanced Ridge风险尺度，点头逐行相同。2024→2025 H1宏覆盖/宽/IS从pooled 86.97/278.04/430.10、mix-only 88.75/271.30/394.51变为91.83/275.74/335.11；H2从83.19/275.63/491.86、85.14/270.48/459.79到93.38/322.69/355.26。宏概率分数和覆盖存在稳定正增量，但H1最差MISO只有56.66%，H2最差FPL只有75.21%。max-source归一化校准将最差提高到77.20/87.24，却扩大宽度至363.66/494.46，且H1过覆盖96.45%、H2分数506.79。见`docs/V51_IDENTIFICATION_WIDTH_RISK_RESULTS.md`和`results/v51-identification-width-risk-r2-h1/`、`r2-h2/`。这是目前最接近的风险方法正向增量，但尚未达到85%最差源覆盖或保持效率的门槛；保留候选、不冻结、不开未查看地区。

对风险特征做post-hoc支持域门控：只在训练源4D标准化特征包络外换成max-source q90。宏覆盖H1/H2为95.40/97.28%，但均宽312.81/451.21；最差仍是MISO 77.20%和FPL 87.24%。BPAT多次包络外且校准良好，H2 MISO全域外但覆盖100%，H1 MISO域外同时严重失覆盖，因此轴对齐特征域外不能判定迁移失败。结果原样保留于`results/v51-identification-width-risk-support-gate/`。风险特征确有宏层新信号，但还无法可靠区分最差区情形，当前不是过关的算法突破。

再加donor CIF小时profile分歧均值/p90作为风险特征，点预测不变。它在H1把歧义宽度头宏覆盖91.83→89.23%、宽276→285、IS335→403，MISO覆盖仅32.76%；H2宏覆盖93.38→96.79%，但宽323→393、IS355→409，最差FPL 88.61%。标准max-source版本更宽，亦未修H1 MISO。反向半年结果拒绝了简单source disagreement路由，不堆叠该特征。见`docs/V51_DONOR_DISAGREEMENT_RISK_RESULTS.md`、`scripts/v51_shape_disagreement_risk.py`和`results/v51-shape-disagreement-risk-h1/h2/`。风险候选“月度热组歧义宽度”仍有宏层正结果但过不了稳定最差源覆盖；还未找到可信区间突破。

再对前一候选用nearest-source normalized q90代替pooled q90。H1宏指标改善为94.22/290.30/320.22，胜过pooled候选但最差MISO仍77.20%；H2则90.00/313.95/366.85，最差FPL仅58.37%，区间分数比pooled候选更差。源特征近邻不能可靠选择目标的尾部分布，判为失败路由。详见`docs/V51_NEAREST_RISK_SOURCE_RESULTS.md`和`results/v51-nearest-source-risk-h1/h2/`。目前多源风险因子对宏平均有信息，却未能预测何时目标源严重负迁移；可信覆盖门槛仍未达到。

## 2026-10-27：显式核对“新地区覆盖”的有限样本单位

按地区部署时，覆盖校准的独立单位应是地区而不是地区内小时。标准组级 split-conformal 的有限样本秩为`ceil((K+1)(1-alpha))`：目标90%覆盖至少要9个校准组才能得到有限秩；现有单个外层折只有5–6个校准源，无法在该目标下构造有限阈值。名义85%在K=6时也只能退化到最大组分数，K=5仍无有限阈值。已有max-source结果只是经验保守对照，不是严格组级新域保证：H1/H2宏覆盖96.45/97.65%，但平均宽363.66/494.46，最差留出源覆盖仍77.20/87.24%。这说明按小时汇总出的高覆盖不能当作90%新地区覆盖。

文献对照：层级共形预测支持新组预测的前提是层级可交换；分布式/联邦共形文献也指出站点异质可能破坏标准交换性。故创新路径转为“明示新地区信息合同、地区级有限样本分辨率、物理部分识别与概率预测分层评估”，并检验可观测粗结构能否支撑地区级迁移。该方向可能成为可信benchmark设计，但尚无算法增量、不能声称覆盖突破或普适新颖性。详情见`docs/V51_UNSEEN_REGION_COVERAGE_LIMIT.md`；文献为Mallick et al. 2026 HCP预印本与Lu et al. ICML 2023 FCP。

## 2026-10-27：以整半年度地区最大残差构造 80% 组级同时覆盖探针

按地区将整个半年所有月份/小时的归一化残差取最大值为一个地区分数，K=6/5时取有限样本名义80%秩（阶数6/5），测试新源半年度“所有小时同时在区间内”的经验效果。H1目标覆盖7/7，均宽558.67、IS558.67；H2目标覆盖5/6，均宽729.90、IS729.91，FPL仍有0.11%小时漏出。相比同一风险头pooled q90，均宽增加约103%/126%；原方案整段覆盖仅1/7和2/6。该结果是很宽区间换来的高源级覆盖，并非可信窄区间突破。90%组覆盖在K=5–6时有限样本阶数超过K，规则只能返回无限阈值。

脚本与逐区结果在`scripts/v51_group_conformal_risk.py`、`results/v51-group-conformal-risk-h1/h2/`；完整限制和结论见`docs/V51_GROUP_CONFORMAL_RISK_RESULTS.md`。实现利用交叉拟合风险尺度，未满足严格已证明的HCP算法条件，故只报告经验诊断。下一步不能再靠增加小时行数声称提高新地区保证；需扩大独立地区数、检验明示层级交换假设，或把无法支持的覆盖率作为拒答/信息预算结果。无方法冻结或新地区开启。
# 2026-10-27：官方年度因子条件化小时形状没有源外增量

在30省×5个CISPO 2025情景压力测试中，本地MEE 2022年度因子平铺优于全国平均常数（宏MAE约158 vs 225 gCO₂/kWh），但不能复现小时变化。对留省水平映射加入本地2021/2022因子条件化的小时形状回归后，五情景MAE全部略增（+0.18至+0.48），RMSE仅小幅下降（−0.45至−0.75）；150个省×情景配对仅75个RMSE改善。该形状模型使用同年其他省份的模拟小时结果，是oracle压力测试，不是可部署迁移。该专门机制不保留为创新候选。

数据为2026年发表的模型投影而非小时实测；MEE年因子为消费侧estimand，与TransCIF发电侧主轨不等同。完整协议、各情景表和边界见`docs/CHINA_ANNUAL_OFFICIAL_FACTOR_VALUE_V52.md`。本轮只支持“合法滞后本地年因子具有空间水平信息价值”的窄结论，不支持小时形状识别、区间覆盖或真实新地区增益。算法突破仍未找到；继续以真实月度字段合同和小时风险可识别性为中心，不开启未见区。
# 2026-10-27：原始EIA快照存在更大的地区级校准候选池

只按原生时间戳、完整燃料行及发电—净发电账差筛查，没有查看任何新地区CIF误差。在66个有记录BA中，25个通过2022–2025各全年及2026H1双重80%覆盖门槛；其中一些只有水电等退化支持。进一步要求每期至少两个正EF活跃燃料组和一个零EF类别后，剩17个候选，含6个此前未评分区（FMPP、GRID、IID、LDWP、PSEI、TEC）。完整原生列、逐年完整率和文件哈希见`docs/V52_EIA_REGION_POOL_QUALITY_CENSUS.md`及`results/eia-ba-quality-census-v52-r1/quality_census.json`。

这是重要的实验机会：足够多的名义区域数可让地区级90%组分数有有限秩；不等于独立交换组充分、覆盖成立，也不证明中国外推。下一步必须先在开发集冻结区域校准与同预算基线，处理EIA BA的互联市场相关性及异质字段；只有有明确开发增量后，再封存并评测未评分区。此次没有读取未评分区CIF结果，算法突破仍未确认。
# 2026-10-27：10区校准池仍未解决最差源尾部风险

在11个已看过的严格月度-only区域做嵌套留区开发比较，每个外层目标用10个校准地区。2025H1/H2，pooled q90宏覆盖84.38/85.06%、最差SC覆盖3.29/13.33%、均宽448/488；月度组成条件尺度宏覆盖85.11/86.25%、最差12.22/21.59%，没有可信收益。地区最大q90将宏覆盖推至97.29/98.76%，均宽增至767/746，但H1最差SC仍74.56%。故名义地区数够做90%有限秩不等于地区尾部可交换，也无法靠校准解决点头水平漂移。

定位到SC的H1正偏差约+299 gCO₂/kWh、目标残差q90约366而校准最大q90约338；H2同一SC偏差约+272但最大q90约382，覆盖恢复至96%。NWMT连续两期正偏差约+231/+256；H2用最大q90宽度836才达到90.67%。这显示区域水平偏差时间变化比月度组成风险尺度更关键。完整结果见`docs/V52_REGION_POOL_RISK_DEV_RESULTS.md`与`results/v52-residual-center-dev-r1/`。全部均为旧开发区，不是新地区确认；六个封存区未读取。没有算法突破，后续开发聚焦粗信息能否预测残差中心/漂移，且需和合法锚、强动态点基线比较。
# 2026-10-27：滞后月度组成没有迁移残差中心

针对SC/NWMT开发误差的百级正偏差，按协议做同月源平均中心与月度thermal/VRE share Ridge中心；中心模型对校准源整区交叉拟合，中心化误差重新pooled校准。H1两种中心法点MAE从112.06升至113.42/118.59、Winkler从795.54升至841.35/864.96；H2季节中心略改善MAE/分数（118.01→117.09，779.67→750.49），但宽度升约7%、最差覆盖仅19.71%。Ridge中心两半年均未降低点MAE，最差覆盖未改善到可用水平。结果见`docs/V52_RESIDUAL_CENTER_DEV_RESULTS.md`和`results/v52-residual-center-dev-r1/`。基于月度组成的中心—尺度分离不构成增量；新区域仍封存，暂无突破。
# 2026-10-27：完整五类组成也不能预测开发区残差中心

为追查中心修正失败是否由两比例特征压缩过强，加入完整thermal/nuclear/hydro/wind/solar份额与月份sin/cos，使用source-balanced Ridge并整源交叉拟合。2025H1/H2点MAE由基线112.06/118.01升至123.21/128.84；区间宏覆盖仅85.29/87.40%，最差源0.93/21.59%。H1分数恶化，H2略降分但平均区间宽从488增至555、点误差更大。与R1一致，月度组成不能预测地区级小时误差中心，暂停继续扩展相似风险特征。详情见`docs/V52_FULL_MIX_CENTER_DEV_RESULTS.md`、`results/v52-residual-center-dev-r2/`。封存区未打开；仍无突破。

## 2026-10-27：V62最新月全燃料形状有点增量、区间仍失效

新增V62只使用预测起点前最新发布的目标八燃料构成，再乘来源区上一年同历月逐燃料小时形状；17个已打开EIA代理区以2025H1来源校准、2026H1前推。MAE/pooled覆盖/最差区覆盖/宽度/Winkler为61.54/92.43%/64.12%/324.15/401.71；V57型滞后同比重组为70.32/85.65%/51.85%/293.77/458.96。相对本地最新热组EF强锚MAE仅改善1.3%，Winkler改善6.7%但宽度增加17.9%。V62保留了“最新本地组成优于陈旧组成”的信息价值，未达区间晋级门槛。[V62协议与结果](V62_RECENT_FULL_MIX_SHAPE_TRANSFER_PROTOCOL.md)。

## 2026-10-27：V63月度组成近邻加权形状未改善迁移

用预测时可见的目标/源区月度组成Bray–Curtis距离做来源逐燃料形状核加权，带宽在2024整区留出上选择、2025校准、2026H1评分。相对V62等权形状，点MAE只从61.54降至61.44，Winkler从401.71微升至402.11，最差覆盖从64.12%降至63.54%；17区仅10区点误差改善。最大来源q90宽约512时最差覆盖87.36%。关闭此支路，不把常规近邻权重改名为原创方法。[V63协议与结果](V63_COMPOSITION_WEIGHTED_SHAPE_POOL_PROTOCOL.md)。

## 2026-10-27：V64中国公开小时负荷仍是重建代理

审计新近发布的31省2015–2024小时负荷Figshare数据与2018 Zenodo负荷曲线。前者基于NDRC 2018典型日、日高低负荷、年度用电量和天气重建；后者按NDRC典型日内形状逐日仿射缩放到高低值。省级小时文件不等于小时发电/CIF实测。2016–2024与Ember总发电匹配的30省3240月，负荷/发电比中位1.025、P10/P90 0.662/1.457，每省月比值CV中位9.95%；不支持将负荷不加不确定度地用作发电分母。它可另设模型化负荷辅助预算做敏感性分析，但不能验证CIF覆盖。[V64审计](CHINA_HOURLY_LOAD_PROXY_AUDIT_V64.md)。

当前候选的实质性破局点仍不是模型堆叠：已确认本地热组成对冷启动点水平有跨留出价值，但小时风险不可由月构成/负荷重建曲线可信校准。下一步应检验“月度燃料边际 + 模型负荷形状”作为显式额外预算时，代理误差不确定性如何影响CIF识别集合；然后决定它能否支撑新的可靠预测机制，不能把代理曲线硬塞进物理约束。

## 2026-10-27：V65负荷形状条件性收窄，不是可验证收益

V65实际运行后，30省3240省月的负荷形状臂把月边际only锐界均宽从918.8收至54.0 g/kWh；然而强制均匀24槽总量时，等槽均值直接由月燃料边际确定、宽度为零。因而代理形状没有优于平坦时段约束，收窄不能解释为曲线准确。2018年同年模型重建曲线与Zenodo模板的宽度仅差4.6 g/kWh。停止把模型负荷作为中国冷启动的预测特征；仅可作为标记清楚的假设敏感性。现阶段论文破局要靠真实小时数据的分级信息预算/部分识别 benchmark，明确月度中国数据只能支持的主张。[V65协议与结果](V65_LOAD_SHAPE_INFORMATION_VALUE_PROTOCOL.md)。

## 2026-10-27：V66直接量化中国公开月度边际不能识别小时尾计数

在3348省月、四个严格阈值的理想化sharp分析中，月边际可认证K=0（没有任何高于阈值的燃料）或K≥1（月燃料加权强度高于阈值）；多数省月仍允许K从0到T−1。τ=600下193个月可认证K=0、2118个月保证K≥1、其余1037个月小时计数全未决；τ=900下未决3001个月。V66从负荷代理转回用户实际可取得的月度燃料信息，结论是低频水平可锚、小时风险需额外本地时序或支持信息。它仍是经典识别边界而非预测算法新颖性。后续价值要落在真实小时区域的信息遮蔽基准、真实额外本地摘要与同预算基线的未来风险评分。[V66协议与结果](V66_MONTHLY_MARGIN_TAIL_IDENTIFIABILITY_RESULTS.md)。


## 2026-10-27：V67发现可合作的中国高频计量验证渠道

网页检索发现现行T/SSM 32-2026及山东15分钟试点、宁夏T−2分区/节点碳监测平台。它们证明高频计量链条在真实业务中存在，却没有提供本研究可直接复用的历史数据。新机会是把TransCIF定位为**稀疏公开月度信息下的冷启动预测/拒绝系统**，再用获授权的真实计量层检验coverage和效率；需分别核发电侧与含网间电力流入的消费/节点因子。未拿到授权数据前，只能作为具体数据合作路径与研究设计，不能写成实验确认。[V67渠道审计](CHINA_HOURLY_CIF_VALIDATION_CHANNELS_V67.md)。


## 2026-10-27：V68出现非线性尾频率预测信号，来源尚未归因

固定EF小时代理尾频率在17个已开EIA区的2026H1留区中，可被全构成ExtraTrees跨区预测；相较季节均值，τ=300 MAE 0.380→0.139、τ=600 0.212→0.086，地区聚类差值区间均低于零。τ=900仅4个正尾样本。预定的全构成Ridge在τ=300虽改善33.6%，聚类区间仍跨零，故不晋级线性候选。此结果不能分清收益来自最新热组EF、全燃料比例、三月动态还是树模型自由度；所有区已开，小时标签真实首发时点也未审计。下一轮仅做相同ExtraTrees预算下分层特征消融，并与最新EF树基线比较；若组成增量可复现，才冻结并用封存区测试。


## 2026-10-27：V69关闭全燃料组成与近三月动态的树模型增量

同一100棵树、相同留区和同一地区月标签下，τ=300最新全构成仅较热组EF基线降低5.3% MAE，地区聚类区间跨零且未达10%门槛；τ=600构成臂恶化14.5%，再加三月动态仍较EF臂差8.1%。因此V68 ExtraTrees优势不能归因于稳定的组成交互/动态迁移，本方向不冻结也不打开封存区。下步检验可公开月发电总规模是否含额外类型信息；若没有，回归“月度边际可识别均值、小时尾部不可识别”的部分识别基准/拒答创新。详见[V69结果](V69_COMPOSITION_TREE_ABLATION_RESULTS.md)。


## 2026-10-27：V70找到阈值特定的月规模交互候选，暂不能确认

热组EF+月份树加入log月总发电量，在τ=600下100个地区月MAE下降13.5%，按地区聚类差值95%区间完全低于零；τ=300反而恶化16.2%，全构成臂也没有复现规模收益。预设单阈值门槛虽过，但并非通用增益。检查发现六个严格“未评分”区已被V54评过2025并进入V68的2026H1行级数据，不能作为新盲测。本候选冻结为待确认假设；在新的未查看地区或新时段到来前，不可称突破。论文机会转为透明字段预算下的阈值特异迁移/拒答benchmark，特征工程自身不足以构成算法创新。[V70协议与结果](V70_MONTHLY_SCALE_INFORMATION_PROTOCOL.md)。


## 2026-10-27：V71真正未见GVL区否定月规模候选

按V70冻结模型在未参与V54/V68的GVL预测2026H1 τ=600。六个月真实超阈频率全零；仅热组EF树MAE 0.0612，加入合法月规模后0.0732，恶化19.6%。这是单区描述性确认，不能推断总体，但足以不把开发区单阈值增量称为可迁移方法。关闭EF×规模树机制。当前算法破局尚未找到；下一论文主线转为低频信息合同下可识别小时均值与未识别尾部的锐界/拒答benchmark，并等待新标签验证。[V71结果](V71_GVL_UNSEEN_REGION_CONFIRMATION_RESULTS.md)。


## 2026-10-28：V72量化月度边际的决策分辨率

从V66精确K锐界构造风险目标选择/弃权前沿。对于“至少一次超阈”，τ=100/300有93.7%/87.0%省月可认证；但对于“≥1%小时超阈”，四个τ下没有任何正事件省月可认证，τ=600/900仅5.76%省月可认证无事件，余者必须弃权。平均尾频率界宽94–100%。这不是算法突破或实测风险预测，但揭示了关键估计目标差异：月度边际可能支持存在性告警，不支持持续/频率风险。下一步评估合法时间摘要预算能否压缩该决策前沿，重点是风险导向信息而非继续增加普通月度特征。[V72结果](V72_MONTHLY_ONLY_SELECTIVE_RISK_RESULTS.md)。


## 2026-10-28：V73同预算最小G摘要仅在特定来源零事件识别上有利

四个6h块min(G)相对四个ΣG²，在V20源日τ=300多认证33个K=0案例（89 vs56），但全部33例来自BPAT；整体K span略差。τ=600/900只微弱收窄。故特征目标匹配有价值，但无通用四摘要突破；当前来源异质与信息预算目标必须显式分层。[V73报告](V73_RISK_TARGETED_MIN_G_RESULTS.md)。


## 2026-10-28：V74出现风险定向G摘要候选，待鲁棒与外域确认

四个6h块的S₂（两小时最小G和）在τ=900、目标K≤4/24h时认证42.97%来源宏日，接近完整24h G的43.36%，并高于同四数ΣG²外界36.91%；分布于多个源。L₁在τ=100、K≥8也接近完整G。此候选的潜在创新是按风险门槛解析设计低维局部摘要；实际中国月/年公开数据缺块H/G和小时G，因此是测量/基准机制，不是现成预测器。继续先做测量误差和源外复现，不急着称突破。[V74结果](V74_ORDER_STATISTIC_RISK_MEASUREMENT_RESULTS.md)。


## 2026-10-28：V75初步通过小误差压力，但依赖精度合同

四块S₂对τ=900、K≤4的认证率在独立最坏±1%误差盒下仍为40.68%（无误差42.97%），±5%下降到29.90%。该敏感性不是实测传感精度，支持继续复现但不支持部署保证。下轮按预定2025H1/H2时间窗复核，避免仅凭2024H2开发信号命名机制。[V75结果](V75_ORDER_STATISTIC_MEASUREMENT_ROBUSTNESS_RESULTS.md)。


## 2026-10-28：V76风险定向摘要在三源三时段与完整G证书完全匹配

共同质量合格的BPAT/MISO/NYIS在2024H2、2025H1、2025H2中，S₂对τ=900/K≤4、L₁对τ=100/K≥8、minG对τ=300/K=0均逐日精确复制完整24hG的证书集合，零误证/漏证；四个风险摘要依赖同一六小时H/G基线。与V74相比这是匹配区域的时间复现，尚非真正外域确认；但形成了实质方法候选：按目标阈值/count-tail设计次序统计摘要，代替通用二阶矩。下一步用未评分EPE/TIDC异质区域冻结确认，再决定原创性。[V76报告](V76_ORDER_STATISTIC_TIME_REPLICATION_RESULTS.md)。

## 2026-10-28：V77外域确认与V78测量误差审计——候选成立但边界很窄

V77在两个此前未评分、燃料结构不同的EPE/TIDC区域上发现：对τ=100且目标为K≥8/24h，四个6h块L₁摘要在六个半年片段逐日复制完整24h G锐集，零假证；与四块G²必要外界相比每片段多认证15–48日。τ=300以及低尾/无尾对应目标均未产生有用证书，不能包装成普适risk模型。V78再对H/G/L₁施加最坏独立误差：ε≤2%时保留完整G证书91.0%，较G²多124日；ε=5%时已经不如G²，显示计量精度是方法的实质条件，不是附带细节。由此可继续探索的原创空间收敛为“在给定分块碳能量/总发电基线和精度合同下，为特定尾计数构造最小次序统计量证书，并刻画其紧致性/区域燃料支持边界”。要形成论文创新，还需证明理论非平凡、测量成本和真实误差可接受、更多独立地区/电力系统复现以及独立小时CIF标签；现有数据不支持中国月度-only小时预测主张。[V77结果](V77_UNSEEN_EPE_TIDC_RISK_SKETCH_RESULTS.md)、[V78结果](V78_UNSEEN_ORDER_STATISTIC_ROBUSTNESS_RESULTS.md)。

## 2026-10-28：V79把候选从L₁统计量转向月报燃料支持自适应

在已开EPE/TIDC用同月活跃燃料EF上界M替代全局1000。τ=300/K≥8时，完整G证书从0/1081升至858/1081，简单G²外界829/1081，L₁仅466/1081；τ=100时局部完整G为1066/1081、G² 1062、L₁ 1059。局部支持边界带来大幅识别收益，L₁不构成必要增量。下一步冻结支持自适应与G²基线，在质量普查中此前未评分的CHPD/DOPD/GCPD/TPWR确认；仍是同期EIA代理及高频摘要测量研究，不能解释为中国月度only预测。[V79报告](V79_MONTHLY_SUPPORT_ENVELOPE_RESULTS.md)。

## 2026-10-28：V80确认队列质量达标，但风险目标退化

CHPD/DOPD/GCPD/TPWR四个未评分区在固定三个半年窗口完整日比例均≥98.3%，但月度活跃支持最大EF均为0；τ=100/300的K≥8物理上不可能，所有方法证书自然为零。不能将它们算作支持自适应方法的有效独立复现。仅据质量普查和活跃燃料类别，下一冻结名单改为未评分且含天然气支持的AVRN/DEAA；保留地区与支持筛选规则，不读取未评分区CIF/尾计数。[V80报告](V80_SUPPORT_ADAPTIVE_UNSEEN_CONFIRMATION_RESULTS.md)。

## 2026-10-28：V81在两个未评分气电区复现支持上界增益，L₁未胜过G²

AVRN/DEAA由质量与年度燃料列筛选。有效窗均显示本地活跃EF上界M=400；全局M=1000的完整G在τ=300/K≥8下全无证书，而本地M恢复识别。AVRN三期局部完整G证书为22.5%、26.2%、19.0%，G²为20.9%、22.0%、15.8%；DEAA两个合格期本地完整G/G²/L₁都达到100%，呈天花板。局部L₁不是普遍优于G²的方法，技术增量应归因于支持上界信息。两区仍属相似天然气结构、EIA同期代理且H/G需小时遥测；结果不足以证明可用中国月报预测小时CIF。[V81报告](V81_GAS_SUPPORT_UNSEEN_CONFIRMATION_RESULTS.md)。

## 阶段性论文判断：有问题价值，尚无强算法突破

V81外域结果支持继续研究“低频燃料支持信息如何收紧小时尾风险集合”，但V79/V81均显示G²基线吸收了大部分增益，L₁未成为稳定原创方法。聚合数据下的部分识别/sharp bounds已是成熟方法域；CIF相关研究也有年到月Denton/SVM及依赖小时数据的月/小时排放估计。因此不能把“聚合数据不唯一”“附加约束收紧”或“小时CIF有价值”单独宣称新颖。最有希望的论文形态是有冻结信息合同、预算受限风险目标、精确锐界与弃权效率指标的统一基准，加可解释的最低新增遥测信息价值；但需证明跨燃料/跨区、计量误差、发布时效及中国独立高频验证。阶段审查和相关文献入口见[V82评估](V82_NOVELTY_AND_PAPER_POSITIONING.md)。

## 2026-10-28：V83测出日内摘要粒度的风险效率曲线

四个开发区同月度支持M下，K≥8证书率随H/G/Q块从24h缩短至12h、6h、3h而增加；τ=300的G²宏率44.6→60.3→66.1→73.9%，τ=100为79.6→82.1→86.2→92.0%。τ=300宏目标60%对应本队列最低12h，但AVRN明显低于宏值，DEAA接近天花板。此处有可复现的信息成本-识别收益曲线，候选论文贡献转为“尾风险目标×发布粒度×区域燃料支持”的最小数据基准，而非新次序统计公式。当前4区均已开、固定EF同期代理、Q/H的内部采集需要也未计价；不作跨网普适或中国月度-only预测结论。[V83结果](V83_TEMPORAL_INFORMATION_BUDGET_RESULTS.md)。

## 2026-10-28：V84将时间预算曲线扩展到25区普查队列

V52全时期质量合格的25区中，风险目标K≥8的G²正证书率τ=100/300分别由24h 46.3%/17.5%增至6h 62.0%/30.8%、3h 74.6%/46.1%，1h时75.3%/59.3%。6h/3h的G²与固定G锐集较接近，矩摘要优于同粒度H/G粗界。全队列含零化石支持区，正证书率不体现其正确“无事件”判断；这是本轮指标缺口，下一步补完整正/负/弃权决策前沿。仍为相关BA固定EF代理，不是中国预测结论。[V84结果](V84_CENSUS_SCALE_TEMPORAL_BUDGET_RESULTS.md)。

## V85：总决策价值提高，但低频月表的根本不识别仍未突破

把V84正向证书补成正/负/弃权后，加Q可大幅降低宏平均弃权，尤其24h/6h；20个百分点负认证来自五个M=0支持区，活跃支持组的弃权仍显著，不能用全队列数字夸大。严格证书零错是条件识别保证，不是预测精度。由此可支持风险目标明确、选择性输出、字段预算透明的benchmark论文方向，但“Q摘要收紧界限”仍是应用成熟部分识别工具，未达到新算法突破。中国公开月报可提供燃料月边际，不能直接提供Q/H/G的小时内摘要；V85本身因此不是目标部署方案。下一轮应将中国月度-only基准（V72）与EIA信息增强曲线分开展示，并具体定义在公开月度数据之上哪一个最小、可发布的额外摘要能识别目标；若无合法数据和独立小时标签，算法创新主张必须收缩为benchmark/识别价值。

## V86：确认时间分辨率的风险目标依赖，尚未形成可部署创新

同样本EIA月→日消融中，1%尾频目标由日H/G已大幅改善；5%/10%目标则在日H/G仍大量弃权，需额外日Q才显著缩窄。由此可将新问题正式收敛到“风险目标条件下的最小可观测摘要与选择性决策前沿”，比统一插值或小波重构更可证伪。但日Q现为小时数据回聚、非公开月报；H/G也是同期oracle聚合；五个M=0区形成简单负证书；样本与标签不独立。因此这是研究方向的明确化和信息增量证据，不是算法突破。下一步应定义中国月报能合法支持的最小摘要合同（尤其总发电分母和分燃料口径），并验证Q是否可由某个可公开日统计替代；若没有额外时间信息和独立标签，论文应定位为识别边界benchmark。

## V87：证明“更细的月度燃料分类”不是小时尾风险破局点

V66的13,392行与目标阈值相关的支持侧/超额符号公式完全一致，精确逐燃料边际没有超出`D=Σ(EF−τ)M`与跨阈值支持的信息。这排除了在同一兼容类中靠月报再细分燃料类别恢复小时尾频率的路线。必须引入真实约束（如逐时容量、日边际、负荷或可信物理/气象形状）。此定理本身可做benchmark/识别论文理论支柱，但不够成为单独算法创新。[V87](V87_FUEL_MARGIN_SUFFICIENCY_RESULTS.md)。

## V88–V89：年容量约束是首个明显超过简单锚定/统计摘要的候选增量

在EIA月燃料边际锐界上加入BA×燃料每小时装机上界，解析容量约束集合使τ=100、≥1%目标正证书从5.4%提高到75.8%，滞后一年容量仍复现；界宽约缩窄三分之一。这相对前序Q摘要更具机制意义，因为是外部年度物理上界约束。但368/589（同年）、326/589（滞后）通过容量对齐核验；筛选本身基于同时点小时标签，且容量/BA燃料桶存在口径不一致。视为首个值得冻结的候选，不视为已证突破。必须在全部分母保留、事前质量/映射筛选、未看尾标签外区、容量误差压力与中国省级能力数据上复核。[V88–V89结果](V89_LAGGED_CAPACITY_IDENTIFICATION_RESULTS.md)。

## V90：粗技术组容量保留增量，但尚非突破

将滞后年容量从逐燃料放宽为thermal/nuclear/hydro/wind/solar五组共享上界，并用统一的组别极值判据重算逐燃料比较。粗组审计通过358/589地区月，逐燃料为326/589；在共同326月上，τ=100平均K宽为月边际542、逐燃料362、粗组419小时，粗组相对月边际缩窄约23%，τ=300约19%。粗组精度较低但增加32个回溯可用地区月；τ=300、≥10%目标粗组正认证30.7%，逐燃料40.9%。这是“粗年信息也有物理部分识别价值”的候选，不是预测算法、标签隐藏覆盖或中国字段匹配证据。下一关应先核实中国实际省级年度装机口径、月度分类、边界和发布时间，再决定是否冻结；同时需要推导带正发电约束的锐群组界。[V90协议与结果](V90_COARSE_CAPACITY_IDENTIFICATION_RESULTS.md)。

## V91：中国公开源字段尚未满足接入合同

官方页面支持全国层面的粗技术年度容量和月度信息入口，但当前检索未确立省级、历史可得且与用户月度发电表同边界的容量面板；火电类别本身混有多种燃料。工作区没有中国原始字段文件可匹配。因此下一关不是把EIA容量代理移植到中国，而是拿到具体省级原表并核实时间戳、统计边界、单位、分母与技术拆分。若合同不成立，V90收缩为跨系统识别信息预算候选。[V91审计](V91_CHINA_CAPACITY_RELEASE_CONTRACT.md)。

## V92：补集正发电锐化理论成立，但当前数据无新增收益

把事件外零发电的边界排除后，推得共享粗组容量下的精确K可行性判据。在358个冻结地区月、2,148个目标行中，V92锐界下端点与V90粗组外界逐行相同，覆盖K的检查全部通过；因此前轮约19–23%的区间缩窄不是由零发电松弛虚构的，但锐化也没改善结果。算法本身是短小的组内线性极值，单独的论文创新度有限。下一步不继续包装这个公式，转向中国月表同预算消融：隐去EIA煤/气/油/other边际，仅留火/水/核/风/光月度组总量和年粗容量，检验V90的增益是否仍在。[V92报告](V92_SHARP_COARSE_CAPACITY_RESULTS.md)。

## V93：粗月表预算下的容量增量仍保留，但效率有阈值依赖

在V90固定358个月内隐藏煤/气/油/other月边际，只保留热/核/水/风/光月总量和滞后年技术组容量。τ=100界宽425h、τ=300宽488h，相对粗月边际alone的544–545h仍有收缩；相对详细月燃料+容量，τ=100只宽6h，τ=300多47h，≥300尾事件认证在≥1%目标下降约10个百分点。所有识别集端点包含和观测K检查通过。它是更贴近中国常见表头的、可证伪的信息价值候选；但字段来自EIA小时回聚且评估样本通过目标小时容量审计，不能写成中国部署或前瞻预测。下一步针对火电内燃料EF不确定性和容量口径误差做鲁棒集，随后才考虑外区确认。[V93协议与结果](V93_CHINA_BUDGET_COARSENING_RESULTS.md)。

## V94：高阈值下“火电总量 + 年容量”仍不能正认证

在τ=600/900下，火电EF区间跨越目标阈值。358个月粗字段加粗容量虽缩小K界，但尾风险正认证仍为0、弃权约78–81%；详细月燃料信息可大幅恢复负证书，逐燃料容量更进一步。该结果把突破问题从“火电总量够不够”改写为“高阈值所需的最少本地燃料组成字段是什么”。当前尚无单煤电月量/热组EF的一字段消融，也无中国真实月报与小时CIF标签；因此是方法设计与数据采集方向，不是可投稿的算法突破。[V94协议与结果](V94_HIGH_THRESHOLD_COARSENING_RESULTS.md)。

## V95：阈值以上的燃料月总量是潜在最小字段

只增加煤电月MWh，在τ=900下即恢复详细月燃料的负尾识别（宽582→213h，负认证19→70.3%）；τ=600时因油电同样位于高因子侧，煤单字段仍不够，容量加煤字段的弃权仍约74–77%。这有更具体的创新假设：按目标τ把燃料月电量聚合为EF高/低两侧的风险定向统计量，并为容量和未知子组EF推导锐尾频率集合。现在直接测试τ=600的coal+oil与τ=900的coal聚合，若能逐行复制详细边际并跨数据集验证，才可能形成有价值的可迁移信息预算机制；当前仍是EIA oracle coarsening。[V95结果](V95_SINGLE_MONTHLY_COAL_FIELD_RESULTS.md)。

## V96：阈值侧月电量几乎足以替代详细燃料边际

风险阈值以上的月发电量聚合（τ=600煤+油，τ=900煤）在2,148个行级决策中有2,088个与详细燃料月边际端点完全一致，余下仅宽1h；加粗年容量后区间宽214/142h，细燃料容量交集185/100h。此结果将“最低本地信息”具体化为随τ变化的风险侧能源统计，而非全部子燃料份额或小时形状，是当前最像可投稿方法机制的方向。仍需同粗容量下精确配对详细燃料、误差压力、独立地区/系统和真实CIF标签验证；目前不能称突破已确认。[V96报告](V96_THRESHOLD_SIDE_ENERGY_RESULTS.md)。

## V97：同粗容量配对仍几乎复制详细月边际

在完全相同的358区月和同一粗组年度容量上，阈值侧统计界宽比详细月燃料界仅多3h（τ=600）/5h（τ=900）；无容量行级端点2088/2148完全一致，同容量行级1797/2148一致，其余平均只宽4.9h且无一行更窄于完整信息集。这个结果消除了先前容量粒度对照的主要混淆，形成实质候选创新：按目标风险阈值压缩本地月统计，再解析传播到小时超阈计数界。接下来重点做摘要测量误差±1/2/5/10%、和简单月均锚及标准线性规划界比较；只有增量保留且跨域/真实CIF验证成功，才冻结为突破。[V97结果](V97_SIDE_VS_DETAIL_SAME_CAPACITY_RESULTS.md)。

## V98：压缩字段误差稳健，但没有形成优于详细数据的预测机制

在风险侧月量±10%误差压力下，同粗年容量集合平均宽度相对零误差增加2.4–4.3小时；τ=600的1%尾正认证下降1.7个百分点，τ=900没有正认证。误差集合正确包含零误差集合和实际K，但全部来自固定EF代理与EIA回聚。当前候选贡献仍应定位为“阈值条件的最小统计字段及选择性识别边界”，尚未显示优于详细燃料统计的算法收益。下一实验应压力测试EF集合/类别边界错配；若风险侧划分强依赖固定排放因子，论文主张需改成数据合同/字段预算分析，而不能称通用迁移方法。[V98报告](V98_SIDE_ENERGY_MEASUREMENT_ERROR_RESULTS.md)。

## V99：中国月表支持风险侧字段，但EF阈值跨越会让字段失去稳定定义

V60原始中国省月快照上，按敏感性因子区间评估分组：τ=600只有煤稳定处于高侧，τ=900下煤区间跨过阈值，受影响电量占全样本62.1%。这把创新边界限定得更清楚：可迁移机制必须联合输出因子支持/估计对象、阈值侧字段和部分识别弃权；不能假设一套固定风险侧 fuel mapping 对所有地区/阈值有效。敏感性区间并非官方中国因子置信区间，数据是修订型二次估计且没有小时标签，因此只能作为真实月度字段的稳定性诊断。[V99报告](V99_CHINA_FACTOR_THRESHOLD_STABILITY_RESULTS.md)。

## V100：容量机制通过不依赖小时峰值的全候选重审

取消V90的目标峰值筛选后，513个事前月总量与滞后粗容量相容的地区月中，包含小时峰值超容量的155月仍达到K集合100%经验包含、零证书错误；τ=100平均宽423.5h，τ=300为439.7h，仍优于同样本月边际553.5h。76/589月在只看月总量时容量就不相容，必须计入12.9%输入不可用/弃权；若把它们作弃权，≥1%正认证约66.3%/42.5%。这是当前更可信的机制增量，不是前瞻预测：输入含目标月聚合量，标签为已开EIA区固定EF代理。下一步价值不在继续重复已开区，而是冻结信息合同，在真正未查看区/独立时间段确认，并将不可用地区、结果时点与小时真值条件写清。[V100结果](V100_NO_PEAK_SCREEN_CAPACITY_RESULTS.md)。

## V101：机制跨时间窗仍收窄，但容量合同不可用率不可忽略

不调参把V100机制应用于2026H1和滞后2025容量，在124可行省月的K集合覆盖100%、证书错误0，宽度τ=100/300为445.6/440.5小时，月边际为563.7小时。24/148候选月容量不相容，含CHPD/FMPP/GCPD/GRID四个整半年度失败区；计弃权后全候选正认证为65.3%/35.2%。说明机制价值并非仅局限于2024–25运行窗口，但所有2026H1目标此前已用于其他方法实验，不能称盲确认；要形成可信“新地区”论文结论，尚需真正未评分的外部标签或新时间块。[V101报告](V101_2026H1_CAPACITY_TEMPORAL_PROBE_RESULTS.md)。

## V102：容量风险集合可证明为精确，而非宽松近似

把有理数精确可行性判据扩展到V100全部513个月后，严格锐界和宽松外界的完整整数集合在3,078行全部相等，实际K全包含； peak-screen失败155月也没有集合漏失。故目前最强候选是“以月燃料边际和滞后粗技术容量识别小时阈值超限频率的闭式sharp set”，而非泛化小时预测算法。一般聚合数据部分识别与时间分解都已有成熟文献，创新性需证明具体K尾风险命题、闭式结构与跨系统前沿确有领域新增，且不能把EIA fixed-EF proxy当中国验证。[V102结果](V102_FULL_COHORT_SHARP_CAPACITY_RESULTS.md)。

## V103：高阈值EF不确定性未完全吞掉粗容量的识别收益

在513月、固定滞后粗组容量中，把四类火电因子同时扩展到基准±10%，鲁棒K端点仍精确且实际fixed-EF代理K全包含。区间宽仅增加τ=600/900的11.4/16.2小时；τ=900鲁棒集合仍排除67.1%的≥1%高尾目标，但完全不给正证书。故可将论文假设从固定点EF推进到显式EF-box敏感性，并把正尾能力缺失如实报告。该因子盒只是压力设定，不是中国官方/经验置信集；需要基于燃料热值、机组效率与电厂实测排放支持标定后，才能作可信跨区主张。[V103报告](V103_FACTOR_UNCERTAINTY_SHARP_SET_RESULTS.md)。

## V104：外区集合覆盖确认，实际判别增量有限

五个未开区CPLE/FPC/IPCO/PNM/SOCO共30个2026H1月全部可用，±10%因子鲁棒集合对fixed-EF代理K 100%包含；但τ=600弃权100%，τ=900弃权约77–80%，没有正认证。故仅能确认集合机制的域外一致性，不能称有用小时风险预测突破。下一步做同样本的容量信息配对消融；若不能相对月燃料边际显著缩窄并增加正确证书，停止把容量作为主创新叙事。[V104结果](V104_UNSEEN_REGION_ROBUST_CONFIRMATION_RESULTS.md)。

## V105：容量信息显著缩界，但决策前沿增量不足

同样本配对的月边际 vs 月边际+粗容量鲁棒宽度在τ=600/900分别收窄39.9%/53.3%，证明区间收益确来自外部物理容量。但正证书仍为零，负证书只在τ=900、10%尾率从20%升23.3%，多数仍弃权。当前“缩窄集合”机制复制成立，“实用预测/风险预警”创新点未成立。应进入最小额外信息识别实验；除非合法信息显著抬升选择性风险前沿，否则将贡献限于可迁移部分识别基准/不可辨识边界。[V105结果](V105_UNSEEN_REGION_CAPACITY_ABLATION_RESULTS.md)。

## V106–V107：公开粗燃料信息的风险前沿不足，预测新颖性门槛更高

未开地区粗火电+非化石月量加滞后组容量仍能缩界，但τ=900/10%目标只有6.7%负认证、93.3%弃权；较详细燃料信息的同源敏感性结果更有用，但EF盒不同，不能作纯消融。文献核查还发现IEA月/季平均因子构建及德国小时生成型CIF短期预测已存在，因此小时重建/预测的宽泛主题不新。优先工作应是证明最小合法本地字段对K-tail sharp set的边界，再找真实中国小时平均CIF许可验证；达不到时应把论文主张限为信息充分性基准和不可辨识边界，不宣称预测算法突破。[V106](V106_UNSEEN_CHINA_STYLE_COARSE_INFO_RESULTS.md)、[V107](V107_HOURLY_CIF_NOVELTY_BOUNDARY_LITERATURE.md)。

## V108：当前可辩护命题是sharp partial identification，未达到预测论文门槛

已把尾计数、共同组容量、事件/补集严格不等式和EF盒单调端点写为方法命题。下一步不能只靠增加变体实验：需要独立实现/LP验证完整K集、与标准低频重建和概率区间基线同预算对比，并建立实际中国小时平均CIF数据合同和未来窗。缺少真实标签或正/负前沿实质改善时，结论应是合法低频统计的信息不足界，而不是新CIF预测算法。[V108门槛](V108_SHARP_SET_METHOD_AND_PAPER_GATE.md)。

## V109：独立LP复现为sharp集合提供数值证据

180组完整K集合（30月×2阈值×3因子向量）逐行与独立LP一致。这强化了集合算法可信度，但仍是数值交叉核验，不等于论文可直接声称解析定理已完整证明，也不能弥补真实CIF验证缺口。[V109](V109_INDEPENDENT_LP_SHARP_SET_AUDIT_RESULTS.md)。

## V110：名义月均锚挑战部分识别方法的应用价值

在同一30月外区，名义月均CIF锚点90%经验区间宽45.8h/0h且标签覆盖100%，明显优于宽数百小时、弃权多数的鲁棒集合；τ=900标签全为K=0，形成近乎天花板的易样本。该基线尚未承受相同EF不确定性，不能据此判定锚点已胜出，但必须做同不确定性比较与未来盲窗。若锚点保持锋利，真实创新点可能不是更宽的鲁棒集合，而应寻找何时简单锚会失效、何种信息使得风险排序可迁移。[V110](V110_UNSEEN_REGION_MONTHLY_ANCHOR_BASELINE_RESULTS.md)。

## V111–V112：稳健校准与尾部判别率形成前沿，但没有赢家

同EF盒池化锚点更窄但漏掉FPC两月；以最坏源区q90校准后覆盖恢复至100%，区间宽仍略窄于容量集，但正/负风险证书全为零。容量集稍宽却在τ=900保留少量负证书。证据支持“信息预算×覆盖×区间效率×选择率”基准价值，未出现支配基线的新方法。真正突破需用强尾事件的新时间窗/中国小时标签证明一个方法同时守覆盖、缩区间并提高正负判别。[V111](V111_FACTOR_ROBUST_MONTHLY_ANCHOR_RESULTS.md)、[V112](V112_WORST_REGION_ROBUST_ANCHOR_RESULTS.md)。

## V113：首个未来季度复现机制，但仍属低难度尾部样本

在协议冻结后下载的EIA 2026Q3新时段上，15/15地区月完整且全部方法的EF盒集合覆盖通过。窄锚仍全弃权；容量集合更宽但对IPCO提供20%负证书，τ=600全弃权，τ=900实际K全为0。首个真实时间迁移没有反转前沿，但样本只一个季度、无高阈值事件，不能定论预测泛化。探索锚点×容量集合交集并继续追踪后续时间，是当前更有价值的方向。[V113](V113_2026Q3_FUTURE_TIME_CONFIRMATION_RESULTS.md)。

## V114–V115：校准形状包络 ∩ 物理可行集成为首个空间留出有增量候选

在四个新地区SPA/TAL/TEPC/WALC的Q3空间留出中，11个质量合格地区月上，锚点与容量集合的交集对EF盒代理K端点覆盖100%，保留容量法75%负认证/25%弃权，同时宽度降到τ=600/900的85/66小时；单独容量为161/126h、锚点338/260h。增益不由全易例造成：TEPC有τ=600 K=22–69的实际代理尾时段，交集缩窄但仍弃权；三块全零尾区给负证书。当前这是值得冻结的“物理约束+跨区校准包络”候选创新，但区数、时间跨度小且锚点部分经验校准，尚未证明统计保证、正尾识别或中国真实CIF迁移。下一关必须固定全部超参/协议，去更难新季度复现，并寻找有真实小时标签的中国合作数据。[V115](V115_UNSEEN_REGION_Q3_INTERSECTION_CONFIRMATION_RESULTS.md)。
## V116：区域分组校准有明显的阈值依赖收益，但未新增可判别性

以源区跨月最大误差构造25个区域级conformal分数，α=10%、rank=24，V115重评分中锚点∩容量集在τ=900从65.5h缩至20.1h，在τ=600从85.3h增至97.2h；因子盒代理端点仍22/22包含。高阈值收益来自V112经验半径被单个极端源区抬高。此缩界没有改变任何证书/弃权决策，且目标数据已用于此前V115评价，故只说明校准器值得冻结，不算突破验证。下一步将V116完整冻结并在新的地区×时间窗口盲测，重点保证存在非零高阈值事件；同时测试源区交换性和目标质量筛选的影响。[V116结果](V116_GROUP_CONFORMAL_CALIBRATION_RESULTS.md)。
## V117：季节匹配半径缩窄过头，未覆盖非零尾部难例

把校准与目标限定在同一季度后，V115的τ=600交集宽51h但TEPC-9漏覆盖，τ=900宽仅0.17h且目标全零；这个“全负证书”没有高阈值尾事件支撑。季节匹配版本拒绝，继续用全年组分数做更广留区审计。[V117结果](V117_Q3_MATCHED_GROUP_CONFORMAL_RESULTS.md)。
## V118：广域留区诊断推翻当前交集法的稳健覆盖主张

## V119：支持冲突是可观测的失败信号，但最大值报警失灵

## V120：源区交叉拟合的风险支持回退形成高阈值候选增量，尚需独立确认

## V121：新季度单区确认区间缩窄，但没有非零高阈值验证

## V122：首次冻结后新时段验证中，支持门控修复尾部集合截断

## V123：将突破候选与真正的文献创新分开

## V124：新空间验证被当前数据覆盖性限制，未重用旧目标冒充留出

## V125：支持门控应对校准零半径失效，机制更清楚但目标仍事后打开

历史October源分数给τ=900的所有目标校准半径为0，普通锚点在JEA/SC漏掉非零尾端点。冻结q20支持差门控回退精确容量集，37/37覆盖，仍比容量集窄66%；比全年校准门控更窄。这个结果把候选机制从“再调一个分位数”推进为明确的保险门：当已知月度构成给出的物理尾部支持远大于锚点所显示的小时风险时，使用锐物理集。这是复用V122标签的事后敏感性分析，需用未来区块重新确认。[V125](V125_OCTOBER_MATCHED_CALIBRATION_DIAGNOSTIC_RESULTS.md)。

低历史覆盖地区的2026年10月月内完整度筛选仅有NWMT达到80%，而它已被V121/V122评分。当前EIA快照没有剩余真正未评分的低历史空间目标，因此停止扩张同一窗的“新地区”集合；维持V122为冻结后时间确认，等待新地区小时数据/完整未来季度来检验空间迁移。[V124审计](V124_LOW_HISTORY_OCTOBER_AVAILABILITY_RESULTS.md)。

条件/局部conformal、协变量偏移校准、conformal物理安全集，以及IEA小时级平均强度产品都已存在。故目前不能把分组校准、集合交集或低频到小时重建包装成新方法。保留的机会是：针对燃料/容量低频信息合同的**严格小时尾风险锐部分识别+风险支持门控回退+信息预算决策前沿**，并以非零尾事件、同域边际基线、真实中国标签证明价值。V122证明了一个新时段代理增量；还没证明统计方法创新、可迁移到新电网或中国有效。[V123边界](V123_NOVELTY_BOUNDARY_AND_CLAIM.md)。

在未用作设计/调参的2026年10月月内窗口，τ=900纯交集在37个容量可行区月中漏掉SC代理上界；q20支持差门控检测到冲突并回退容量集，覆盖恢复到37/37，区间仍比容量集窄51%。三个非零高阈值支持月(JEA/LGEE/SC)均被包含；JEA/SC回退，另对零尾PACE保守回退；LGEE因锚点已足够宽而保留交集。它展示了明确机制：不要一律加宽，而是用本地月度粗统计+容量可行支持检测锚点何时不可信，再局部切换到物理集。当前可称“突破候选”，仍受部分月份、同一地区池、仅3个非零高阈值月和美国代理限制；下一步需要完整未来季度及真正空间留出。[V122结果](V122_OCTOBER_MTD_FROZEN_RULE_CONFIRMATION_RESULTS.md)。

冻结的q20%支持门控在NWMT 2026Q3三个未评分月上把τ=900容量集均宽504h压至80h，仍覆盖固定EF盒代理K；但三个τ=900目标全部为零尾时段，所有决策均弃权。NWMT历史数据在源校准池中（目标折已剔除），所以是新季度而非陌生BA身份确认；τ=600仍靠宽容量集、均宽735h。结果确认区间效率在一处时间目标迁移，但没有验证最关键的高阈值非零风险条件。[V121结果](V121_NWMT_Q3_FROZEN_SUPPORT_FALLBACK_RESULTS.md)。

25个源区逐地区留一、589可行区月/阈值的精确CV中，按源区最大月冲突的10%支持尾边界做锚∩容量/物理容量回退，τ=900代理端点覆盖由97.0%升至99.83%，平均宽70.5h，仍比容量集窄47%，回退7.17%；其余已预注册分位形成可见风险-效率前沿。τ=600只到92.8%，不能称全阈值解决。这个机制比盲目缩窄更有论文潜力，因为它用局部粗统计发现“锚不可信”后让物理集合接管；但当前数据相关、时窗单一、目标季度已开，也未选定运行点。暂定突破候选是“区域级风险支持校准门控的锐部分识别集”，需要冻结运行点并在真正新的地区×时间、含非零尾事件的样本验证。[V120结果](V120_SUPPORT_GAP_REGION_LOO_CV_RESULTS.md)。

## V131：日度阈值侧摘要提供月度统计之外的大部分可识别增量

在V129冻结的358个EIA区月上，同样本无容量消融中，三项日度UTC统计（热电总量、阈值高侧电量、清洁电量）把τ=600的尾计数集合均宽从316.08h降至225.16h（−28.8%），τ=900从213.50h降至192.76h（−9.7%）。逐日详细燃料边际 oracle 分别为222.28/189.42h，日度风险侧摘要只宽2.88/3.34h；区间嵌套审计零失败。≥1%目标下τ=600有0.71%正证书、弃权43.28%，月度风险侧为0%正证书、弃权43.99%；τ=900弃权无变化。初步说明额外价值主要来自分时段而非更细日度燃料拆分，但收益取决于阈值、τ=900主要是缩界、仍有约190h级残余不确定。它是EIA fixed-factor回溯遮蔽，不是中国预测或CIF真标签；还需周度/多日粒度效率曲线、测量误差压力、真实报告时间可用性及真实标签迁移。候选论文贡献更聚焦为“阈值特定信息合同×时间粒度”的风险部分识别基准，尚未证明新算法优于预测基线。[V131协议与结果](V131_DAILY_RISK_SIDE_INFORMATION_PROTOCOL.md)、[V131结果报告](V131_DAILY_RISK_SIDE_INFORMATION_RESULTS.md)。

## V132：三日风险侧摘要以约三分之一字段量保留四分之三日度收益

在相同358区月、两阈值、四种1/3/7/14日窗口上，48个合成格点交叉核对闭式无容量风险侧集合与V129通用求解器相同，8,592行全量嵌套零失败。τ=600/900的3日区间均宽为246.10/197.95h，相对月度316.08/213.50h已缩窄22.1%/7.3%；每月约31.7个标量，较日度90.4个减少65%。三日合同保留约77%/75%的日度区间宽度收益，并与相同分箱的详细燃料oracle只差1.00/1.18h。τ=600的≥1%正证书仍仅0.48%、弃权43.52%，τ=900选择率完全不变。因此候选是信息效率提升，不是可靠尾部预测。三日运行点在已打开的V129 cohort事后选出，必须冻结后测试新时间块；美国EIA代理结果不能解释为中国可得字段或真实CIF验证。[V132协议与结果](V132_MULTISCALE_RISK_SIDE_PROTOCOL.md)、[V132结果报告](V132_MULTISCALE_RISK_SIDE_RESULTS.md)。

新增前沿检索确认AAAI-26已发表多频率CIF预测并使用小波卷积核及动态跨变量关系；因此“多尺度CIF模型”本身不能作为创新点。当前候选必须紧扣其与现有预测任务的差别：训练/预测目标区缺少逐时燃料混合与小时CIF历史，仅有合法粗统计；贡献是风险尾计数的sharp/outer集合和最小信息合同，不是换一个wavelet/网络结构。[AAAI-26论文](https://ojs.aaai.org/index.php/AAAI/article/download/41310/45271)。

## V133：±10%风险侧测量误差几乎不扩集合，但让稀有正证书失效

V129同一358区月、1/3/7日合同、0–10%风险侧MWh误差共32,220行，8个小型合成核对与V130通用误差求解一致，真实 cohort 名义K包含、细燃料外包、零误差V132一致和误差单调扩张全零失败。三日宽度在±10%误差下τ=600只增加0.117h，τ=900增加0.009h；然而τ=600的≥1%正证书在±2%时仅0.48%，到±5%即归零，弃权回到43.99%。所以三日合同的区间效率对这项误差压力稳健，正尾识别并不稳健。热电总量和清洁总量被假定精确，误差水平不是中国实测发布误差。方法价值仍为风险导向识别集合/信息成本前沿，不是正尾预测器。[V133协议与结果](V133_MULTIDAY_MEASUREMENT_ERROR_PROTOCOL.md)、[V133报告](V133_MULTIDAY_MEASUREMENT_ERROR_RESULTS.md)。

## V134：冻结三日规则迁移到2026Q3五个区，缩界复现但高尾仍弃权

用V113同一份2026Q3 EIA源数据重新计算CPLE/FPC/IPCO/PNM/SOCO共15区月，小时覆盖100%、V113名义K复算一致、三日集合对nominal K和逐日详细oracle全包含，90行零失败。τ=600月度→三日均宽735.0→599.6h (−18.4%)，τ=900为588.0→561.3h (−4.5%)；三日风险侧界与日内详细燃料oracle仅差0.4/0h，但≥1%尾率的正证书仍为0%，高阈值仍80%弃权。三日信息压缩的方向迁移，但新时间段的绝对区间更宽，不能支撑实用高尾预测。该季度此前已用于其他方法，故不是盲测，仅为冻结规则方法迁移探针；需真实未查看未来窗/地区和中国小时标签确认。[V134协议与结果](V134_FROZEN_THREE_DAY_Q3_TRANSFER_PROTOCOL.md)、[V134报告](V134_FROZEN_THREE_DAY_Q3_TRANSFER_RESULTS.md)。

## V135：跨日共享排放因子不再收紧区间

在相同358区月与三日信息预算下，把全月未知燃料因子统一固定为同一个全低/全高情景，再聚合各块锐界；与每个块独立放宽EF盒的V132比较。两个阈值的三日平均集合端点完全一致：τ=600宽246.10h、τ=900宽197.95h；详细燃料oracle仅窄约1.00/1.18h，所有2,148行、包含审计零失败。说明当前宽度瓶颈不是跨块EF不一致，而是每个时间块内未观测的小时调度/形状。此机制方向不构成增量；下一轮应测试同口径粗容量或少量块内形状摘要是否减少这部分不可辨识风险。[V135协议与结果](V135_COHERENT_FACTOR_MULTIDAY_PROTOCOL.md)、[V135报告](V135_COHERENT_FACTOR_MULTIDAY_RESULTS.md)。

## V136：加入粗容量后，三日风险侧界在τ=600超过月度细燃料强基线

在V129同一358区月上，将滞后年度热/核/水/风/光装机容量作用于每个三日块，3,782个块容量均可行，无回退，2,148行代理K/详细oracle包含审计零失败。τ=600下“三日风险侧+容量”均宽191.15h，比“月度详细燃料+同一容量”222.14h窄14.0%，且离三日详细oracle 9.23h；τ=900两者宽度几乎相同148.60/148.57h。≥1%阈值决策存在权衡：τ=600正证书下降4.52%→0.71%，负证书56.01%→58.04%，弃权39.47%→41.25%；不能说支配所有前沿指标。更重要地，这只是EIA容量代理和事后容量筛选cohort，不能证明中国省级容量合法可得。它是目前值得冻结的具体信息合同候选，下一步应用于V113 2026Q3地区×时间cohort做规则迁移探针（此前标签已被其它TransCIF实验打开，非盲测）。[V136协议与结果](V136_THREE_DAY_CAPACITY_COMBINATION_PROTOCOL.md)、[V136报告](V136_THREE_DAY_CAPACITY_COMBINATION_RESULTS.md)。

## V137：冻结的三日+容量合同在Q3五区保留小幅缩界，仍不识别正尾

不调参应用V136规则到V113 CPLE/FPC/IPCO/PNM/SOCO 2026Q3的15区月；小时数与V113复核一致，160块容量全可行，90行代理K/详细集合包含审计零失败。相对月度详细燃料+容量，三日风险侧+容量均宽τ=600 423.20 vs 455.47h（−7.1%），τ=900 260.93 vs277.60h（−6.0%）；和三日详细燃料oracle差23.13/3.13h。≥1%高尾正证书仍为零，τ=600全弃权，τ=900仍20%负证书/80%弃权；τ=600/10%目标有13.3%负证书。结果给出方向迁移但效用弱于2024–25 cohort。Q3已被其它TransCIF方法打开，非盲确认；同样不能替代中国省级容量合同/真实小时CIF标签。[V137协议与结果](V137_FROZEN_THREE_DAY_CAPACITY_Q3_PROTOCOL.md)、[V137报告](V137_FROZEN_THREE_DAY_CAPACITY_Q3_RESULTS.md)。

月均锚上界与容量物理上界之间的归一化冲突在V118错判案例较大，显示它有望作为局部风险支持指标；但源区历史最大冲突(τ=600/900为0.480/0.978)太宽，未触发任何回退。事后把阈值降到源区风险差80分位，在同批数据恢复全覆盖并保留明显缩界，但该分位经过目标结果敏感性挑选，不能证明泛化。下一关改为源区留一交叉验证，预先比较风险支持分位与宽度/覆盖前沿，再冻结规则做全新时空确认。[V119结果](V119_SUPPORT_GAP_FALLBACK_RESULTS.md)。

43个V52质量合格区的2026Q3审计中，物理容量锐集在可行的115区月全部覆盖因子盒代理K；交集平均缩窄，但τ=600漏1行、τ=900漏6行。高阈值失败集中于非零尾支持的JEA/LGEE/SC；PACE-9是中阈值失败。当前主要破局点不是进一步压缩区间，而是识别哪些粗月信息能预警“经验锚可能截断真实尾部”，再有条件地拒绝交集/放宽锚区间。需要对条件规则做新地区、未来时间验证；现有证据不支持突破结论。[V118结果](V118_REGION_LOO_BROAD_COHORT_RESULTS.md)。

## V138：三日容量合同缩界耐受字段误差，正尾识别仍脆弱

358区月、10,740行中零误差端点复现V136，误差单调扩张和包含审计零失败。±10%风险侧电量误差使τ=600/900均宽从191.15/148.60h增至193.96/150.24h，负证书稳定，但τ=600的≥1%正证书从0.71%降至零。用户已确认全部月度发电类别可得，因此三日摘要只能作为额外信息价值实验，主预算应回到逐类月度边际。[V138结果](V138_THREE_DAY_CAPACITY_FIELD_ERROR_RESULTS.md)。

## V139：修正容量不可用处理后，物理归一化有增量但高尾条件覆盖不足

输入审计发现并修复V92的全月可行性漏检，保留589区月并对76容量不可用月回退[0,T]；513可行月的1,026组端点与V103逐对一致。全25区留一、17,670行比较中，alpha=0.10时物理宽度归一化相对简单绝对误差锚将τ=600/900宽度201.99/33.60h降至180.05/27.30h，经验覆盖同为99.47%/97.17%。但τ=900非零可行尾月份覆盖只有78.96%，失效集中JEA/SC；总体高覆盖掩盖稀少风险。方向余量和宽度归一化所有端点相同，不能算两个创新。下一轮需要用完整月度组成辨别高尾调度机制，并与同预算条件分位数强基线比较。归一化conformal本身已有文献，尚未突破。[V139报告](V139_PHYSICAL_RESERVE_CALIBRATION_RESULTS.md)。

## V140：完整月度组成下的条件模型没有稳定胜过锚，源区高煤电支持明显不足

两次独立源角色划分、25区留一、两预算六方法共56,544行审计零失败。月度完整分类主预算下，物理坐标CQR在τ=600宽235.65/243.37h、覆盖97.67%/99.33%，简单锚宽198.10/198.89h、覆盖99.30%/99.14%；τ=900也无稳定联合增量。附加容量的归一化锚仍优于坐标模型。后验输入支持审计显示，唯一平均煤电份额>60%的SC有20/24个月超出其余24源区的煤电份额支持；其留出失败严重。下一步应以已经查看的地区补足高煤电源历史，保留相同目标和校准角色做数据覆盖消融，再判断隐藏调度机制；源样本丰富本身也不是算法创新。没有晋级未查看地区确认，小时预测原始gate仍未满足。[V140结果](V140_MONTHLY_CONDITIONAL_RISK_RESULTS.md)。

V139修正的后续影响已部分收完：只用容量输入相容的源月重算V121/V122支持阈值，V121端点不变；V122只改变PACE τ=900一行[0,508]→[0,68]，已保存代理端点仍包含。属于对旧目标行的事后校准修正，不能追认为新盲确认；V126/V127支持校准仍待重审。[影响范围](CAPACITY_CONTRACT_CORRECTION.md)。

## V141：风险相近源区带来真实增量，普通CQR已取得收益

固定V140原589目标区月和两套12区校准角色，只向fit增加已开地区历史。19候选中18区/347月合格，WALC仅11月不合格；煤电平均份额≥0.4组为LGEE/PACE，两区低煤电对照数量相同。80项精确计数核对、70,680评分行审计零失败，原预测完全复现。全新增源的普通CQR在τ=900两seed覆盖99.50%/99.83%、宽33.90/62.36h，相对不变锚98.00%/98.50%、50.13/86.83h有联合增量。SC覆盖41.67%→95.83%，补两低煤电区无此改善；但只补高煤电两区的direct-all对全部非零尾区的条件覆盖仅72%，不能只报SC成功。物理坐标法无独立优势，归一化作为主创新退役。增量归因是源数据支持；旧锚校准不变是因果消融设计，不能据此认为已赢过充分使用新增源预算的所有锚。[V141结果](V141_SOURCE_DIVERSITY_ABLATION_RESULTS.md)。

## V142：充分使用源预算的锚仍强，LGEE未被CQR解决

PACE18月/LGEE24月作已开历史开发目标，完全排除目标身份，30fit+12cal，两seed/两alpha/七法共2,352行零断言失败。加入全42源校准锚与12邻近源局部锚。LGEE/900普通CQR仅12.5%/45.83%覆盖，active控制12.5%/62.5%；局部锚100%但宽707.38h。LGEE/600 CQR宽720h，全部源锚100%覆盖宽381.17h。PACE/900普通CQR94.44%/100%覆盖宽17.50/69.39h；全源锚94.44%覆盖宽14.94h，局部锚100%但398.78h。新源池没有普遍解决困难地区，未打开真正未查看确认。[V142报告](V142_PACE_LGEE_MONTHLY_HOLDOUT_RESULTS.md)。

## V143：LGEE高阈值的主宽度来自EF合同，中阈值才有巨大可约减空间

43区936月、73,032条既有预测的oracle因子宽度审计完成。给定已观测小时配置及当前假设EF盒，任何覆盖所有因子响应端点的区间必须至少宽F=K_high−K_low；F仅是评估oracle，禁止当目标特征。LGEE/900均F=651.42h，占小时数90.4%；完全覆盖的局部锚707.38h只多55.96h，100%覆盖同样本的最大平均收窄空间约7.9%。LGEE/600 F=0，而CQR仍宽720h；PACE/600 F=199.78h，PACE/900仅2.39h。LGEE/900所有direct-CQR漏失行都因宽度小于自己的F，不能靠平移救回。应分别处理因子合同不确定性与调度迁移误差，不再把所有宽区间当同一种校准问题。[V143结果](V143_FACTOR_FLOOR_DIAGNOSTIC_RESULTS.md)。

### V143 后的计划（已由 V144–V148 实施，以下保留原计划）

优先在LGEE/600与PACE/600评估完整月度组成到尾部风险的响应机制。先重用并正面对比已有V7/V25燃料重组思想（不要重新命名为创新），以源月完整fuel profile均值归一，在严格fit/cal/target地区排除下重算同月组成的风险响应。通用CQR必须允许使用完全相同的源小时信息/派生响应特征。可探索保持源profile的组成扰动曲线/临界煤电份额表示来修复树模型在高煤电区域的外推；这只是待验证机制，合成响应依赖profile不变假设，不是真实干预标签，也不能增加校准独立地区数。物理增强、重组、conformal本身不算创新，必须超出普通重组和同特征CQR。高阈值单列EF范围驱动的不可约宽度，不用oracle F作预测输入。

另需在下一次源小时profile缓存时审计EF常数表示：V120–V143为兼容冻结结果保留Python浮点乘法后的十进制有理因子，部分高因子如440带极小浮点尾数；严格超阈边界可能敏感。未来合同应显式写为有理9/10与11/10，并对照旧标签差异，保留旧输出，禁止静默替换或将浮点边界差异当方法收益。当前V143的F是旧冻结合同下的精确会计，不是中国实际因子置信区间。

## V144–V145：规范数值合同与完整月度重组强基线

43区936月完成规范整数EF缓存，旧3,744端点全部复现后只有6个端点各少1小时；3,744自身组成响应与96跨区完整有理数核验通过。全部方法统一使用规范合同，数值差异不计为方法收益。V145完成5,808行同预算评分：LGEE/600普通条件重组覆盖100%/100%、宽204.71/140.29h，残差CQR宽173.96/123.79h，优于全部源锚381.17h；LGEE/900残差CQR覆盖仅66.67%/87.50%。普通重组与残差化不是新方法。V148又发现直接CQR的排序敏感性，不能用其单次较差实现夸大优势。[V144](V144_CANONICAL_PROFILE_RESPONSE_RESULTS.md)、[V145](V145_RECOMPOSED_RISK_BASELINES_RESULTS.md)。

## V146：等效煤占比的联合风险校准未胜过普通扩张

固定查询组成处的30源区权重，以虚拟煤占比位移构造嵌套曲线，共同校准两阈值、两EF端点；对照使用相同评分的纵向物理余量扩张。1,828条查询曲线、438,720项源响应恒等核验、1,056目标行通过。alpha=.10水平法全部目标覆盖，但LGEE/900回到720.96h物理全宽，SC第二划分也明显扩宽。GRID在两族各4/6折驱动最大位移，不能按目标效果剔除。没有独立增量。[V146](V146_CRITICAL_COMPOSITION_RESULTS.md)。

## V147：条件尺度未改变目标区间，零堆积需单独处理

3,788个proper-fit月排除自身地区后构造逆位移标签，878,816源响应核验通过。中位数/q90尺度四法的2,112目标预测与V146对应原族上下端点完全一致。每折仅17–33个源月需要非零修正，全部目标尺度落至1/256下限。这是该拟合器的负结果；V148显示默认分位数优化器在零堆积下可能退化，不能由此断言合法月度信息不可学。[V147](V147_CONDITIONAL_CURVE_SCALE_RESULTS.md)。

## V148：强基线数值诊断完成，仍未形成原创方法突破

PACE/900 seed41在相同数据和参数下，仅改变拟合顺序就得到17.5/599.94h。默认pinball零残差子梯度使首轮梯度全同，约1e-14浮点伪增益导致任意树分裂。预注册默认、1e-10分裂保护、严格正残差子梯度+保护三种控制，同等作用于三种CQR、canonical/reverse两顺序。r2完成9,504行、432梯度块/272,736值；1,584原canonical基线全部复现，79项输入/产物哈希与端点审计通过。

PACE/900 seed41严格控制的公共特征直接CQR在两顺序同为100%覆盖、122.78h；seed83同为70.22h。其它折仍有排序差异，LGEE/900缺覆盖未解决。r1为API语义/梯度保存修正前的预检，r2与其全部目标端点相同，不算独立实验。冻结helper仅用于gradient_out=None的研究路径，共享输入/输出buffer的边界调用未支持。[诊断](V148_QUANTILE_NUMERICAL_DIAGNOSIS.md)、[结果](V148_QUANTILE_STABILITY_RESULTS.md)、[审计](../results/v148-development-audit-r2/summary.json)。

### 当前下一步

先用源区留一验证重组失效的发生条件，把稀少失效发生与非零误差幅度分开建模，并与修正后的条件分位数、普通hurdle和正误差回归同预算比较；模型选择只在proper源区完成。不能将逆校准、条件尺度或数值修正包装成原创。出现稳定可归因增量后，再回到小时CIF分布及发布滞后合同，检验原始小时覆盖、最差地区、CRPS和区间效率gate；月度尾数中间任务不替代它。当前不冻结专门创新方法、不打开新的确认地区。[机制与文献边界](V148_MECHANISM_AND_NOVELTY_UPDATE.md)。


## V149：正误差幅度改善LGEE中阈值，发生概率未泛化

复用严格地区排除的源逆位移与特征，八法完成4,224行/96半径，独立重建全部输出与哈希零失败。水平正幅度Ridge在LGEE/600两轮100%覆盖、115.46/76.54h，相比V148规范顺序严格tie残差CQR174.50/120.67h有局部缩界（联合与逐阈值校准口径不同）。但SC/600仍222.96/476.75h、SC/900为330.54/628.88h，PACE无稳定优势；LGEE/900近物理全宽。纵向正幅度法在SC有91.67%阈值行回退全物理界。普通hurdle未取得稳定增量。

六个源区角色划分的发生概率AP高于校准发生率，但Brier全部差于拟合常数发生率：存在排序线索，尚无可信概率门控。两个曲线族共用相同失效事件，不把其重复指标当独立复现。保持“正误差幅度”作为强基线，下一轮检查其支持范围，并在独立源角色下检验概率模型泛化；不得复用conformal校准标签调概率后再宣称独立保证。仍未满足原创方法及小时预测gate，不冻结、不打开新地区。[V149完整结果](V149_HURDLE_RISK_RESULTS.md)。


## V150：幅度缩放的LGEE改善依赖单个源区，不晋级

五种幅度控制×两族完成5,280行，全特征模型复现V149全部1,056行。56维降到9维几何特征后仍无统一优势；独立校准源正误差对数MAE描述均值常数0.651，优于几何0.721及全特征1.563。

固定教师的事后拟合影响诊断进一步发现：LGEE/600第一划分从幅度训练中仅去掉AZPS一个正误差月，原115.46h变为720h全宽；几何模型也如此。两划分全特征逐源删除宽度分别112.04–720/70.96–130.33h；这是影响诊断，不是完全排除该源区的外部验证。V149局部增量应降级，条件幅度缩放只保留为基线。下一轮回到小时分布与月度约束的风险支持迁移，仍以原始小时覆盖/最差地区/CRPS/效率及发布时间gate为完成条件，不打开新地区。[V150结果](V150_SEVERITY_SUPPORT_RESULTS.md)。


## V151：恢复小时CIF评估，时效性与形状失配需要分开

36个2025开发地区月、1,152条评分，源/校准固定为2024年的30/12不同地区，精确加权经验CRPS核验通过。完整同月统计+max-region校准的普通条件重组宏小时覆盖93.89%/93.97%、最差89.35%/89.00%，CRPS45.40/45.46；相对简单水平残差下降9.71%/8.60%，相对等权重组仅1.22%/0.60%。此常规强基线表现不够支撑原创方法。

模拟月末+30天发布后，pooled覆盖94.85%/88.50%、最差92.23%/82.83%，第二划分失败；max-region虽近全覆盖，宽632.71/546.96 g/kWh，相比同月313.91/316.58明显膨胀。原始CRPS53.87/54.03仍差，校准不改变CRPS。PACE即便完整同月预算，pooled仅81%左右覆盖；NWMT则是滞后预算下最大校准误差的重要来源。因此后续分别研究当月形状失配与统计年龄风险，不再以月度尾数覆盖替代小时预测证据。全部地区已查看，源可用性与发布延迟仍是受控回顾性假设，未开新确认。[V151](V151_HOURLY_LAW_RESULTS.md)。


## V152：识别出月内恒定CDF的评分上限，调整主优化空间

1,440条oracle诊断、288条V151端点/CRPS复现、均值恒等式及20组分段CDF积分核验完成。当前36目标月中，所有月内恒定CDF的经验CRPS下界为43.7403；完整同月预算的强重组仅剩3.65%/3.78%（条件）或4.82%/4.35%（等权）改善余量，无法在该模型类内再达5%CRPS门槛。此为已观察样本的模型类限制，不是小时条件预测的不可约噪声。

PACE同月预算的均值oracle几乎无效，标准差oracle将CRPS约82.44降至80.69，覆盖约81%升至90%，宽度随之增加；定位为低估波动，但没有可部署机制。滞后预算距相同边际下界仍有约19%余量，均值oracle提供较大改善。完整月分支转向小时/日历条件分布；滞后分支保留未来组成/水平预测的源区学习问题。所有oracle仅评估，不当特征，不打开新地区，原始gate保持。[V152结果](V152_HOURLY_MOMENT_DIAGNOSTIC_RESULTS.md)。


## V153：小时条件化产生稳定开发收益，特殊边际机制无增量

恢复UTC时间并逐行复现677,928条发电数组；六法完成1,728评分，V151整月基线288行全部复现，295,560期望边际核验最大误差1.11e-15。普通小时条件池相对整月CDF，同月CRPS下降17.84%/17.31%、宽度下降9.84%/10.51%，宏覆盖90.69%/91.24%、最差87.55%/87.36%；asof预算CRPS下降11.77%/11.34%、宽度下降4.78%/6.98%，覆盖96.09%/90.94%、最差93.09%/88.86%。月份配对描述区间均支持负CRPS差异。

增量来自标准小时条件化，逐燃料期望边际重组几乎无贡献，周末细分也未进一步改善，+7小时相位负对照明显恶化。PACE日历oracle自评分50.37，整月79.94、同大小随机分组77.49，显示日内结构机会；oracle不部署。正式gate仍需同预算小时条件回归/模板等强基线，而不是仅胜过不看小时的旧分布。下一轮升级强基线，再考察源区日历信息强度和合法地理时间对齐；未建立原创方法，不打开新地区。[V153](V153_CALENDAR_LAW_RESULTS.md)。


## V154：同预算强条件模型已启动，当前为运行中观察

冻结四类控制：小时残差模板、分布式ExtraTrees、整区五折残差训练的异方差HGB正态模型、HGB条件分位数。使用相同20维合法输入；每proper源最多4,096个2024小时，所有模型共用抽样；CQR仅报区间/点指标，CRPS留空。分布森林相同原子合并是精确优化，正态分布明确使用1,025点离散。

数值核对：20个合成查询中森林经验分布均值与点模型最大误差7.11e-15；60组裁剪正态积分对照中离散CRPS最大误差0.01542 g/kWh。这些是算法数值核验，不是目标性能。

本轮末实际观察进程PID98543仍活跃，exec session8081，日志/tmp/transcif-v154-r1.log；完成PACE/41两个预算（122,880/118,608源小时），共2/12训练预算块。结果尚未全部产生，不能宣称胜负。下一目标轮先核对同一session/PID，继续等待，勿仅凭收据重启。执行收据见results/v154-run-progress-r1.json；收据本身不是活性证据。首次未合并重复原子的试运行已确认exit143终止，当前为精确合并后的同协议运行。[V154协议](V154_HOURLY_STRONG_BASELINES_PROTOCOL.md)。


## V154–V155：直接条件模型比较收完，下一步补锚定残差与数据量控制

V154已确认exit0，12/12预算块、1,152条目标评分完成；360源抽样逐项复现、180整区OOF角色通过，比较与月份配对审计完成。V153小时重组在CRPS上优于本轮小时残差模板、直接分布森林和异方差HGB，但部分直接模型LGEE失效极重。输入支持审计：第一划分proper源锚上限672.66，LGEE同月十二个输入831.12–907.83全部超出；源煤份额最大54.69%，同样全部超出。不能以此差距声称原创机制，下一步需已知水平锚+条件残差模型。

V155严格pinball零残差控制完成288区间记录，360抽样与V154完全相同。修正能解除合成零标签q05退化，未修复真实跨区覆盖；保留默认与严格实现，不按目标选择。CQR没有CRPS。

另外，学习器每源最多4,096小时，而经验重组读取选中源月全部小时：合法数据类别相同，但源记录量未完全匹配。下一轮应补全量源学习或匹配源小时的重组控制，再判断是否存在可归因方法增量。原先的运行收据已标COMPLETED_CONFIRMED，无V154/V155进程待恢复。[V154](V154_HOURLY_STRONG_BASELINES_RESULTS.md)、[V155](V155_STRICT_HISTOGRAM_QUANTILE_RESULTS.md)。


## V156–V157：锚定与来源数量控制完成，剩余机会转向联合燃料监督

V156 sampled/full各864行、两进程exit0；V157固定供体月全量/抽样576行，三进程均完成。源索引、角色、分母、哈希审计通过。显式物理锚大幅修复直接模型，例如第一划分同月森林CRPS87.79→40.05，因此旧巨大差距不作算法创新证据。

严格匹配来源后，普通小时重组相对锚定森林CRPS仍约低6%–15%，full也未消除差距。但宽度/覆盖未统一支配：full第一划分滞后重组宽360.76、森林324.88且两者过覆盖门槛；sampled第二划分滞后重组覆盖89.77%未过90%。纯小时数量变化CRPS仅0.16%–0.39%，源暖启动月份选择另列。

下一步不再只改标量CIF校准，检验来源多燃料联合小时监督是否提供可迁移的物理形状信息，并与同数据的联合燃料学习器比较。此处记录V156/V157结束时的计划；该机制随后由V158实现并评价，见下文。原始gate与新地区确认条件不变。[V156](V156_ANCHORED_HOURLY_RESIDUAL_RESULTS.md)、[V157](V157_MATCHED_SOURCE_HOURS_RESULTS.md)。

## V158：联合份额残差和响应字典负结果，下一步检查迁移坐标

12/12组合、1440评分完成；主进程与审计exit0，288条标量记录复现V156五指标，360源抽样及三套归档哈希通过。普通联合线性臂对标量有局部收益，但截负归一使四组CRPS恶化4.55%–10.51%；响应字典只在一组略胜普通联合支持臂，所有组都输小时重组，CRPS差6.96%–23.04%。PACE同月响应覆盖仅75.64%/68.86%；LGEE滞后CRPS49.18/56.82，重组28.50/28.27。没有新地区晋级。

proper源诊断显示约77%跨组成加性份额创新需负分量修复。两小时构造进一步证明：即使源/目标共享完全相同的乘性燃料发电形状、月能源边际都精确，加性份额差也会因总发电分母改变而不迁移。下一轮先以乘性发电形状坐标建立同监督强参考，在proper源内部选规则，再评价物理响应学习增量；不能把已有重组坐标或有限特征核森林改名算创新。V158没有发现足够方法突破，持久目标仍active。[结果](V158_JOINT_FUEL_RESPONSE_RESULTS.md)、[查新边界](V158_METHOD_NOVELTY_BOUNDARY.md)。

## V159：29源活动坐标筛查完成，动态增量来自普通参考

6折×2预算、9088评分，进程/审计exit0，逐方法分母与归档通过。asof条件动态活动重组CRPS32.813，比形状持平34.934下降约6.07%，但宽度+9.63%、最差覆盖70.44%；更强标量锚定CRPS31.735。三种活动森林当月未形成CRPS增量，asof均输普通参考，所有联合gate为false。零总量定义域修正的增量约0.000015%，不作突破。5个固定EF源的简单月份抬高宏覆盖，其余24源仍有CPLE/AVRN等形状或水平支持不足。

下一项只在源区检验碳CDF风险嵌入是否比条件均值分裂有用，并同时比较一般MMD与标量CDF森林；这是待检验假说，不从本轮负结果倒推出必然有效。未复测当前2025目标，未打开新地区。[V159结果](V159_FUEL_ACTIVITY_SOURCE_PILOT_RESULTS.md)。

## V160：CDF/MMD分裂源筛完成，未支持专门碳机制

12区块4544新评分与审计exit0，482条asof均值控制完全复现V159。专门碳CDF当月CRPS22.732，原始活动22.519；asof36.225，标量锚31.735。一般MMD asof覆盖94.03%、最差85.11%，但CRPS37.117、宽238.70，没有效率增量。专门臂虽胜较弱标量CDF，但未胜完整强参考，所有新增联合gatefalse；不以换分裂目标声称创新。

下一步先检查本轮20维输入未使用的合法月度历史轨迹是否有粗层预测价值，补同数据合同的持平/趋势/harmonic及条件强参考，再决定小时机制；旧V14较长历史结果不是本轮同合同证明。没有打开目标2025或新地区。[V160结果](V160_CARBON_CDF_SOURCE_RESULTS.md)。

## V161：月报历史粗层5.29%表观增量，经简单参考后降为1.92%

粗层九方法、241月2169预测及审计exit0；只用已发布月度字段，未调用小时真值。历史森林MAE25.780，比基础27.219下降5.29%，但约65.6%绝对收益来自过去相同EF支持的5个简单源。事后增加合法过去锚点预测回退后，基础森林25.725，已略优于未回退历史森林；双方都回退后25.230对25.725，改善1.92%、月份描述区间跨0。历史没有形成相对完整简单参考的明确增量，不能推进为小时创新或完成目标。

接下来先做固定地区相位的源区oracle潜力诊断，分离可解释时钟错位与无法从月报识别的随机风险；有潜力后才核查合法元数据、实际模型与同预算强参考。诊断未执行，不把oracle当实际提升。[V161结果](V161_MONTHLY_HISTORY_VALUE_RESULTS.md)。

## V162–V163：固定相位潜力不足，仿射矩匹配会破坏概率峰值

V162完成54,528条相位诊断，V163完成20,960条水平/离散程度诊断，两轮均复现2,272条baseline。最佳固定相位同月/滞后CRPS改善4.17%/2.64%，最差覆盖仍约71%/72%。真实均值方差仿射校正虽改善CRPS，却可把滞后条件活动最差覆盖降至37.71%；TAL原有400峰值被移动，是分布形状失真证据。CPLE等多数漏盖仍在原CDF完整数值范围内，不能笼统归因为缺少极端取值。

没有可部署增量，所有oracle不晋级。下一轮冻结V164，仅用合法月报预测调整供体月情景概率，和普通预测组成重组、简单锚定、标准熵/概率更新及既有强参考比较；不把熵更新本身当原创。[V162](V162_FIXED_PHASE_DIAGNOSTIC_RESULTS.md)、[V163](V163_LEVEL_SPREAD_DIAGNOSTIC_RESULTS.md)、[V164协议](V164_SCENARIO_REWEIGHTING_PROTOCOL.md)。

## V164：情景更新改善平均效率，最差覆盖仍未恢复

七臂3374评分完成，12未发布统计扰动检查通过；r2全部重建评分与校准、月度OOF及宏汇总通过。Gaussian对条件动态CRPS改善7.35%、宽度−15.55%、宏覆盖90.61%，但最差63.64%；对标量强参考仅4.20%且描述区间跨0，对普通组成预测仅2.48%。熵max_region最差84.48%仍未过门槛。所有新臂joint gate false，无未查看地区晋级，无足够原创算法证据。

下一轮V165已冻结完整粗层分布到世界情景的概率传播、局部合法粗回测残差及同预算QRF/异方差控制；当前尚未评分，不预判收益。[V164结果](V164_SCENARIO_REWEIGHTING_RESULTS.md)、[V165协议](V165_COARSE_DISTRIBUTION_LIFTING_PROTOCOL.md)。

## V165：完整粗分布未过gate，停止把概率投影当创新

九臂4338评分、r2完整重建与36条宏汇总核验完成；964原重组+482 Gaussian评分复现。标准历史异方差CRPS30.262，宏覆盖92.97%、最差76.92%、宽211.85；对Gaussian仅0.46%CRPS增量但宽22.93%，对标量森林改善4.64%且宽9.09%。本地粗回测残差混合输同投影全局控制1.39%。全部316联合比较false，没有未查看地区晋级。

粗层宏覆盖90.50%仍掩盖CPLE/JEA仅55.56%；AVRN粗层88.89%但小时76.92%，说明失败需要区分。下一项先做固定样本内的理想粗信息oracle与零燃料/缺失形状约定归因，核查q=1填补和ε增长迁移的贡献，再决定方法；诊断尚未执行，不能预先认定原因。目标保持active，不以标准投影/残差混合改名宣布原创。[V165结果](V165_COARSE_DISTRIBUTION_LIFTING_RESULTS.md)。

## V166：完整粗组成oracle仍欠覆盖，载体强度变化没有整体收益

6748评分及完整重建完成，964原参考复现。真实八类组成oracle CRPS约22.14/22.17、宽约131.8，但pooled重校准最差70.17%/70.82%；不是可部署收益或所有方法上界。载体.001/.1 CRPS32.897/33.024均不胜32.813，源过去0→当前正放大代理宏仅0.0704%，不再以小分母作为主要假说。

源两月均0而目标过去为正的事后能源份额代理约5.97%，与上述出现事件不同，不能直接解释为误差贡献。V167将其视为未观测增长响应，比较中性份额/月能源增长、普通支持惩罚以及同Ahat/σ Gaussian控制；仍保留q=1，先看简单控制有无增量。V167尚未评分，不预判成功。[V166结果](V166_COARSE_ORACLE_ZERO_AUDIT_RESULTS.md)、[V167协议](V167_MASKED_GROWTH_PROTOCOL.md)。

## V167：中性增长收益被普通筛选与Gaussian解释，未支持新方法

6266评分与完整重建完成，964+482参考复现，12完整CDF未来月扰动检查通过。中性份额CRPS30.825，比原条件动态32.813好约6.06%，但对普通供体惩罚30.941仅0.37%且地区描述区间跨0；中性能源+Gaussian30.4023≈原Gaussian30.4008，宏覆盖仅89.43%、最差67.45%。所有352数值联合比较false，没有未查看地区晋级。

不继续把中性增长或支持惩罚当创新。下一项先做proper源内燃料形状掩蔽价值实验，核对q=1标签是否把未观测响应当真值，并比较常数、条件均值、独立与条件联合补全。只有条件形状确有可学习增量才开发更复杂模型；当前未做，不能据阴性结果预判有效。目标继续active。[V167结果](V167_MASKED_GROWTH_RESULTS.md)。

## V168：条件形状补全有用，联合结构未优于同边际独立控制

639源世界mask案例、5112评分完成。全部身份/192训练来源/24macro审计通过，192森林重新拟合，首尾93案例744评分重算最大差2.27e−13；其余case评分未逐条重算。双燃料24源联合CRPS17.117，独立17.060，联合宽度反增2.27%；联合对粗输入控制仅2.77%改善且描述区间跨0。去伪标签q评分更好但CIF未更好，不把联合补全当原创增量。

人工mask当前粗量与未掩蔽源小时值已知，不是zero-hour预测，也不能识别自然零月量反事实。下一V169协议及可行重排数值预检已准备，尚无主runner/性能评分：固定燃料边际/线性期望与世界概率，只扰动保持G>0的源耦合，在完整合法asof任务检验相关性迁移。有限交换不等于精确独立/最大熵；无增量就不发展复杂copula。目标active，未打开新地区。[V168结果](V168_MASKED_SHAPE_VALUE_RESULTS.md)、[V169协议](V169_FEASIBLE_COPULA_PROTOCOL.md)。

## V169：固定边际耦合扰动只有约0.03%收益，不发展复杂copula路线

10方法8676评分、全部评分重建及46272源单元列多重集合审计完成，最大评分差3.64e−12；12粗量/6小时输入干预通过。Gaussian50%重排两个种子CRPS30.3906–30.3908，相对30.4008仅改善约0.033%，地区描述范围跨0；pooled最差64.09–64.23%，max_region仍未到85%。相对标量森林改善4.24%，对普通异方差反差0.43%。全部512数值比较false，无未查看地区晋级。

不推论所有依赖无价值或边际已被证实是唯一根因。SOCC等已有评分优化耦合研究，普通重排或学习混合权重不能改名成原创。下一V170已写逐燃料边际协议/runner，先检查平均量、零概率、尾部，以及oracle月均值对齐能解释多少边际误差；尚未评分，不预认定可部署或会改善CIF。目标active。[V169结果](V169_FEASIBLE_COPULA_RESULTS.md)、[V170协议](V170_FUEL_MARGINAL_DIAGNOSIS_PROTOCOL.md)。

## V170：源月零与正月内小时零混合，形成新的可检验缺口

7712边际评分及全量重建完成，768独立单点CDF积分检查最大差5.55e−15。oracle当月均值对齐改善Gaussian煤/气CRPS25.63%/29.02%，但CPLE煤边际覆盖仍67.41%、AVRN风73.78%；不是CIF增量或可部署结果。

事后按预测时已知input月该燃料正量分层，核电p0预测27.70%/实际0.484%，其中27.55pp来自源月该燃料总量0的情景；CPLE煤却预测3.99%/实际16.15%，仅0.0441pp来自正源月内部零小时。单独压低全月零或统一加宽均不足以同时解释这些失配。该补充分层不是预注册验证，不把current-positive事后分层误当无条件校准检查。

下一步先测试同预算直接零分类、普通两级hurdle、持平与V167支持控制，区分合法月报可核验的月事件概率、无法由月报识别的正月内小时风险；普通hurdle不是创新。尚未开发或评分新的CIF方法，无未查看地区晋级，目标active。[V170结果](V170_FUEL_MARGINAL_DIAGNOSIS_RESULTS.md)。

## V171：直接零森林改善Brier，两级架构没有普遍独立增量

240森林、15424小时/9640月/2272条件分量评分完成；全模型重拟合、482查询概率逐项一致，独立展开Brier误差1.02e−15。input-positive核/风direct43 Brier0.02747/0.06740，旧Gaussian0.17079/0.14335；但核持平×条件森林0.00564更强，煤两级相对direct43仅微小差且描述范围跨0，气/风两级更差。CPLE煤直接仍低估、两级反而高估零概率，不能宣称困难地区已修复。

普通分类/hurdle不构成原创，且Brier不等于CIF。V172已固定四种零控制×两赔率强度×两原基础law，18方法、16388预期CIF评分；只改物理原子权重，保留固定/重校准半径和强基线比较。当前只完成预检，尚未正式CIF评分。目标active，未查看地区不晋级。[V171结果](V171_TWO_SCALE_ZERO_RESULTS.md)、[V172协议](V172_ZERO_ODDS_CIF_PROTOCOL.md)。

## V172：有相对旧重组/MMD的区间分配收益，仍未胜全部强参考

16388评分及全量重建完成，370176物理支持核验、12未发布粗量干预通过，最大评分差1.46e−11。条件direct20全强度max_region最差86.14%、宽244.23，比旧条件79.00%/259.39更好；对MMD同max_region CRPS−13.43%、宽−4.89%。但其CRPS32.133仍输Gaussian30.401、标量31.735、异方差30.262。1664比较8项仅过正重组/MMD，未达完整gate，未查看地区不晋级。

所有覆盖≥90%/最差≥85%的4候选都依赖重校准；没有固定半径候选达标。direct20候选在CPLE/AVRN宽度约+47%/+26%，总体缩窄包含地区重分配，不能宣称困难地区效率已修复。燃料均值/能源锚也明显漂移；更接近零分类器目标并不代表CIF更准。

下一V173仅写设计，尚无实现或评分：同零预测/原子下比较保护G/E两矩与八燃料均值的标准KL投影，检验损失来自水平破坏还是可独立改善的形状风险。普通投影不是创新，真正增量仍须完整CIF及强控制证明。目标active。[V172结果](V172_ZERO_ODDS_CIF_RESULTS.md)、[V173设计](V173_MOMENT_PROTECTED_RISK_DESIGN.md)。

## V173：预测矩保护避免退化，但没有形成CIF风险突破

18方法16,388评分及全量重建完成，370,176支持/证书核验、7,712固定半径匹配、12粗量干预通过。最佳Gaussian/fuel8/direct20半强度CRPS30.307，相对旧Gaussian30.401仅改善0.308%、宽+0.215%，最差覆盖62.89%；地区描述范围跨0。所有新候选连覆盖双条件都未过，1,792数值联合比较全false。没有未查看地区晋级。

原未保护Gaussian/direct43全强度45.005经GE/fuel8保护降到33.086/30.615，说明限制漂移有保护作用，不能据此唯一归因原收益或退化来自水平。2,002/370,176求解显式回退prior（0.541%），JEA特定测试方法达20.14%，须保留对地区解释的限制。

下一轮直接检验CIF分布误差的可学习性，先补普通全局/条件PIT再校准强对照。PIT映射、CRPS再校准已有2025论文，不当原创；只有同预算完整CIF和强参考有明确增量后才设计专门机制。此处为后续方向，尚无V174评分，目标active。[V173结果](V173_MOMENT_PROTECTED_RISK_RESULTS.md)。

## V174：直接CIF再校准改善平均评分，最差覆盖仍不足

30个内部地区留出物理模型、458048跨折累计固定源采样行的OOF归档与全部重拟合完成，1832192个PIT边界逐项一致。14方法12532 CIF评分及全量重建完成；5784映射、277632支持核验、12未发布粗量干预通过，964原参考复现，评分最大差3.64e−12。

最佳Gaussian/43维/half CRPS29.820，比旧Gaussian30.401好1.911%、20/29地区改善，但宽+5.215%、最差覆盖64.55%；相对同强度全局与20维的1.530%/1.560%收益描述范围均跨0。24非固定EF源宏覆盖仅89.74%。EPE覆盖89.70%→72.15%伴随区间缩窄，存在明显负迁移；不能以JEA等变宽后的覆盖提升代表统一修复。所有候选覆盖双条件不通过，960数值联合比较全部false，无新地区晋级。

普通PIT及输入相关CRPS映射已有研究，不构成原创。下一V175仅写待实现设计：固定相同OOF/输入/森林容量，直接在CIF横轴优化单调映射的CRPS二次目标，必要时加入标准地区稳健控制；当前无实现或评分。若仍不能覆盖保持地改善效率，停止单纯再校准支线，回到可识别信息与结构假设。目标active。[V174结果](V174_CIF_RECALIBRATION_RESULTS.md)、[V175准备](V175_DIRECT_CRPS_MAP_DESIGN.md)。

## V175：直接评分和地区max均未解决迁移，停止纯再校准扫描

18方法16,388评分及完整重建完成；30内部模型、916096组源二次系数全量独立积分误差2.00e−15，185136独立QP检查/370176证书应用零回退，原964参考复现，12未来粗量干预通过。1,504描述性比较（含重复标签）全false，没有覆盖双门槛达标。

最佳新全局half CRPS30.252，比旧Gaussian30.401仅好0.489%、宽+9.538%，最差67.83%；不胜V174最佳29.820。直接43维half30.511，比对应PIT差2.318%、宽+5.433%；八个条件直接评分臂均未胜对应PIT。Gaussian地区max half33.382、宽248.09（比原+43.955%）、最差75.05%。全新候选max_region最高最差也仅82.85%。源核优化下降不等于held收益，不唯一归因过拟合或地区偏移。

按事前规则停止扫描再校准网格/α/地区例外，不把普通QP/minimax改名为原创。下一转向聚合结构价值，目标仍active，未查看地区不晋级。[V175结果](V175_DIRECT_CRPS_MAP_RESULTS.md)。

## V176准备：原小时预算内可构造同步组合，尚无模型或CIF效果

只读profile index/时间轴/原采样与proper名单，完成12982元数据候选。二源9526候选中5338满足完整UTC当月/同月合法完整输入且24时钟齐；原采样交集中位288小时。四源固定64组/折，3456候选中1064合法，951全24时钟，交集中位119小时。没有读取generation/M/CIF或增加唯一源小时。

全候选key/布尔/汇总及hash复核一致；完整日历/原交集独立重建只抽每折每k共12例，不能称全量独立重建。虚拟组合并非独立地区或已核验真实地理合并，月发电量不能唯一识别机组数/相关结构；普通聚合增强已有研究，不构成原创。

下一步尚待实施：相同父样本的UTC同步、时钟内置换、比例缩放和原供体库扩张控制，严格父源外/内隔离、同信息/派生样本/容量比较，在真实held源评分；先检验结构价值，再决定专门机制。没有新地区晋级，目标active。[V176库存](V176_AGGREGATION_INVENTORY_RESULTS.md)。

## V177：同步增量过小，不发展复杂聚合相关模型

14方法12532评分与全量重建完成；5338源world、640560独立源单元、138816预测单元和115680原half支持/质量检查通过，12未发布粗量干预、964参考复现，评分最大差3.64e−12。1216描述性数值比较全部false，没有完整gate或新地区晋级。

条件同步31.882较旧32.813好2.838%，但对置换仅0.008–0.010%且范围跨0，仍输Gaussian/标量。Gaussian同步30.399≈旧30.401，对两置换仅0.0232/0.0227%（描述范围虽负侧，幅度极小），未胜同矩父混合的配对范围或selected-parent。最佳新selected-parent30.299只是普通选样/重权，较旧Gaussian仅0.334%、范围跨0，最差65.48%。

父混合固定pooled也能最差86.23%，但宽270.88较旧条件204.06多约32.7%，CRPS33.016输强参考；固定校准半径不代表law未变宽。不将同步、扩宽或一阶矩保护当原创，停止复杂同步/四源/混合强度扫描。

下一V178仅写待实现设计：原过去八维核和点组成已存在，但本轮粗不确定性经一维EF锚传播；补相同世界库/容量下联合粗燃料分布→world概率的普通强对照。V9/V14/V25等历史联合残差不抹掉，不声称首次联合或已验证创新。目标active，未查看地区不晋级。[V177结果](V177_PORTFOLIO_VALUE_RESULTS.md)、[V178准备](V178_JOINT_COARSE_WORLD_DESIGN.md)。

## V178：直接世界核未胜强参考，不否定所有联合粗预测

14方法12,532小时评分及全量重建完成，60月度森林重拟合、3,856粗/30,848燃料评分、6,748世界概率全部复现。最佳新森林历史logratio8 CRPS35.574，比原Gaussian30.401差17.02%、宽+48.99%。1,448带标签比较有9项仅通过新森林间弱参考，没有完整强gate或新地区晋级。

本轮部署的乘性世界支持与直接查询基点ΔS预测不同。V179固定概率只诊断粗支持，避免据此否定全部联合预测；缺失响应问题已见V166/V167，不称新发现。[V178结果](V178_JOINT_COARSE_WORLD_RESULTS.md)。

## V179：粗支持修正有价值，复杂联合标签尚无优势

3,856/30,848/3,374条粗/燃料/代理记录和九表完整重建，独立二分核验47,850个投影支持、最大差4.44e−16，独立损失误差1.24e−12，12未来月量干预通过。没有新增小时预测。

加性历史scalar8粗ES0.070261，全月Gaussian相同支持0.070380，联合MMD0.071685；前两差异范围跨0，不能将对原乘性森林的大幅改善当联合创新。下一V180固定普通粗组成/小时形状分开重组及持平、点预测、重算Gaussian控制，尚未有新小时评分；目标active，未查看地区不晋级。[V179结果](V179_INCREMENT_SUPPORT_DIAGNOSTIC_RESULTS.md)、[V180协议](V180_REBASED_COARSE_LIFT_PROTOCOL.md)。

## V180：普通标量粗核改善基线，仍未达覆盖和效率门槛

11方法9,640评分、4,338诊断全量重建完成；23,136源q单元、104,112物理/独立合成CRPS积分、12query输入干预、964原参考复现通过。补充检查所有暴露标量及排除数组有限，八个当前归档hash一致，正式代码未改。

最佳历史scalar8 CRPS29.930，相对旧标量改善5.69%、宽−4.28%；对原Gaussian仅改善1.55%、范围跨0且宽+7.86%，仍输既有PIT29.820。联合MMD31.403未胜普通标量。全部all29候选连覆盖双条件都未过，1,224数值比较全false；CPLE70.56%、AVRN83.61%仍失覆盖。没有原创方法或新地区晋级。

下一V181仅写待实现设计：固定粗与q预测边际，比较对角/精确独立配对，记录G>0条件化后的边际漂移，区分配对与边际价值。尚未实现或评分，不将COFE粗细分离、普通copula或输运当原创。目标active。[V180结果](V180_REBASED_COARSE_LIFT_RESULTS.md)、[V181准备](V181_COARSE_SHAPE_COUPLING_DESIGN.md)。

## V181进行中：精确乘积已实现并预检，正式结果尚待完成

库存482个跨折上下文，每product法5,155,658,936组合原子，只合并完全相同粗向量，不抽样/截概率。12完整预测未来月量/raw不可读检查通过，未合并toy和压缩law最大误差1.11e−16，峰值RSS约2.59GB（macOS bytes）。四方法3374正式评分正在六折执行，完成折的独立审计以原子part归档跟进；此处尚不报完整性能或宣称突破。

正在运行的主脚本`v181_coarse_shape_coupling.py`与审计`v181_audit_coarse_shape.py --follow-producer-pid`均保留。后续须先核验实际进程/终止状态，不能仅凭日志或本段文字判断仍运行，不能因观察超时重启。完整source/audit summary出现并通过hash/范围核验后才能报结果。目标active，未查看地区不晋级。[V181冻结协议](V181_COARSE_SHAPE_COUPLING_PROTOCOL.md)。

## V182：历史时间轴可覆盖23个完整月，尚无新增燃料监督

在V181运行期间，只盘点原29源2022/2023区域和时间列。696地区月均非空，667完整；不完整者全为2022年1月，缺5–8小时。标准库datetime/calendar/set独立重建507892元数据行和全部696网格，一致。r1的pandas单位错误已原样保留并标无效，只有r2及审计是有效证据。

没有解析燃料数值、导出新月量、增加源小时训练或改变V181。时间齐全不等于燃料质量/合法发布时间。未来可检验“更多粗监督、相同小时预算”的价值，但强基线必须同预算，尚无性能增量或原创结论。目标active。[V182结果](V182_COARSE_HISTORY_INVENTORY_RESULTS.md)。

## V181主评分完成，完整独立审计仍在运行

主程序3,374评分六折已终止成功。初步独立配对CRPS29.654，对角29.930，改善约0.92%；原Gaussian30.401。pooled自身重校准宏92.79%、最差69.21%、宽187.74，仍未达覆盖/效率门槛；fixed max_region最差84.26%、宽270.81，也不能晋级。暂不把这项小幅普通独立收益当原创，不发展更复杂copula。

完整分块审计当前继续；必须等全部part及最终root汇总/强比较审计通过后才报最终独立复核结论。此前“进行中”段为历史状态，不作为当前存活证明，后续先读实际句柄。目标active，未查看地区不晋级。

## V183：661历史月量已隔离导出并复核，准备同预算信息价值实验

只用原29源及2022–24六个旧文件，661个2022/23月量通过既有proxy质量规则；不把667个时间完整月当661个燃料可导出月。327个2024原cohort/2616燃料margin和hours精确复现。全部988月记录、7904项Decimal精确聚合、87源年度质量报告重建通过；原生映射/质量helper共享，未作public-vintage或独立质量认证。没有拟合新模型、计算CIF或增加小时训练样本。

下一V184只写待实现设计：short / long_fixed / long_all三区分更长输入前缀与更多粗监督；所有基线同预算，九种普通分布基线加公共点持平，先测粗价值再测小时。设计不是结果，更多数据不当原创，原gate不变。[V183结果](V183_COARSE_HISTORY_EXPORT_RESULTS.md)、[V184准备](V184_COARSE_HISTORY_VALUE_DESIGN.md)。

## V184输入预检：每折可用粗标签增至585–628，尚无模型结果

新增monthly-only容器保留原936个id/M/shares，append661历史且不暴露raw/小时offset。6fold short/long_fixed标签数171/161/151/151/160/170，long_all为597/592/585/588/626/628。964原source base/ΔS及241 test base精确复现；所有source合法prefix检查、24次feature-only未来M干预通过。

这是容器/特征及名单准备，不是九基线的模型预检或性能验证。下一仍须实现同预算粗价值实验；不以数据增多称创新，不改变原小时gate。V181分块审计继续，最终root汇总尚待验证。目标active。[V184输入预检](V184_MONTHLY_CONTEXT_PREFLIGHT_RESULTS.md)。

## V181完整审计结束：约0.92%同边际配对收益仍不足

3,374正式评分、1,446诊断全重建；23,136源q单元、34,704独立后验边际单元、48真实cell未合并乘积/独立积分检查通过，12输入干预，144数值比较全false。独立29.654相对对角29.930改善0.920%、宽+1.000%；对原Gaussian仅2.456%、范围跨0、宽+8.935%，最差仍69.21%。固定max_region也仅84.26%，不四舍五入称过85%。不发展复杂copula、没有新地区晋级，目标active。[V181最终结果](V181_COARSE_SHAPE_COUPLING_RESULTS.md)。

## V184九基线helper已实现，仅fold0小范围smoke通过

三信息预算、两个query、8个base law exact负控制及54未来M law检查通过；未做粗或小时性能评分。初版seasonal年份选择在正式评分前按设计纠正，r1退役保留源码，只有r2是当前smoke。下一仍须完整正式预检、runner和全六折粗价值评分，不能将新增数据或有限smoke当原创结果。目标active。[V184模型smoke](V184_MONTHLY_MODEL_SMOKE_RESULTS.md)。

## V184完整粗信息实验结束：新增监督局部改善，尚无覆盖保持的效率收益

28方法6,748粗/53,984燃料评分全量重建，18预算折重拟合、90线性OOF、90锚森林OOF检查和324未来月量law干预通过。更多粗监督使base scalar8 ES改善3.536%，all29配对范围负侧但nonconstant24跨零，锚宽+13.98%；仅拉长history输入退步3.586%。long_all最佳ES0.069448对同预算Gaussian的改善1.675%范围跨零、锚宽+46.61%。不放大至小时，不声称全部粗信息无效，不把更多数据当原创；原gate与未查看地区隔离不变，目标active。[V184结果](V184_COARSE_HISTORY_VALUE_RESULTS.md)。

## V185预检通过，开始粗响应/支持强对照

固定相同long_all预算与ET容量，scalar8/delta8/level8×base/history权重交叉increment/absolute支持，加H/G两支持，共16法。全六折12query/192未来M law干预通过，原模型指纹/负控制精确；没有新增小时输入或评分，预检未读held truth评分。正式及独立重建尚待完成，不将普通level响应或经验叶分布当原创。[V185协议](V185_COARSE_REPRESENTATION_PROTOCOL.md)。

## V185完整审计结束：绝对组成对照无增量，停止表示分支

16方法3,856粗/30,848燃料评分全重建，12level森林、192未来M law干预和1,205/9,640原参考复现通过，独立loss差2.63e−12。最佳仍原base scalar8 ES0.069448；新history level8/increment恶化5.418%、锚宽+17.46%；base level8改absolute support恶化74.02%、宽+57.68%。不向小时扩大，不将绝对支持内对弱响应改善当原创。audit首次bool差类型错误保留后仅改比较cast，正式模型/结果未改。原gate未过、未查看地区未晋级，目标active。[V185结果](V185_COARSE_REPRESENTATION_RESULTS.md)。

## V186当前仅输入准备：另查月度历史与小时调度风险的关系

六折964source+241test context、57,840状态行base identity、1205独立波动特征重建（8.33e−17）和24容器/96矩阵未来干预通过。有效preflight-r2；r1笼统无模型拟合措辞已退役，因为冻结context内部仍有既有harmonic Ridge。未加载generation、未拟合新的小时风险模型或评分；四控制20/short43/long43/long84待正式协议。先验假设/特征不算原创，也不能把未来零事件Brier代替CIFgate。[V186输入准备](V186_MONTHLY_RISK_FEATURE_PREFLIGHT_RESULTS.md)。

## V186正式及全量审计结束：部分燃料收益，尚无普遍风险增量

同原小时预算、192森林、482查询概率与19,280评分全量重建，原15,424评分及direct20/short43概率精确复现，独立Brier展开最大差3.33e−16。12个完整未来M/raw不可读检查通过，448项描述性配对比较保留。旧六world/hurdle控制仍用短历史冻结概率，未获extended信息；不能把它们冒称长历史预算下已优化控制。

input-positive核电long84 Brier0.007314对short43的0.027473有改善，但旧持平×条件控制0.005640更强；水电0.087783仍输该控制0.075358。风电long84为0.106567，比short43的0.067403更差。长43维只在部分燃料改善，不能把辅助Brier或个别CI当CIF突破。没有新增CIF预测或未查看地区晋级，目标active。[V186完整结果](V186_MONTHLY_HISTORY_ZERO_RISK_RESULTS.md)。

下一仅提出可证伪解释：长历史主要提供月燃料状态持续性，未提供正月内调度风险的普遍迁移信息。先补同extended预算的普通hurdle强控制，不继续堆分类特征；current-positive只能作条件任务诊断，不能作可部署筛选。若仍无稳定信息价值，停止该特征支线，回查数据合同和识别边界。尚无这项后续实现/评分，普通hurdle本身不构成原创。

## V187同长历史hurdle完成：局部月状态收益，未见普遍条件风险增量

六折240森林、482查询概率及40,488小时/15,424月状态/4,544正月条件评分全量审计通过，独立Brier最大差4.44e−16；原19,280/9,640/2,272记录复现，12完整law未来M/raw guard通过。1760比较仅反复源开发描述。最终预检/正式/审计输入输出hash全部通过，三个新增脚本AST通过。

input-positive水电long_month_volatility×clock Brier0.063072，相对同长历史history×clock改善42.31%，描述区间负侧；相对旧持平×条件0.075358和long84直接森林0.087783的区间均跨零。核电新持平×long_cond84为0.005185，对旧控制0.005640的区间跨零。条件任务long_cond84水电反而恶化11.66%，long_cond43风电恶化10.20%；other存在相对旧条件森林的局部改善，不能抹掉例外。真实current-positive仅条件任务筛选，不能作部署筛选；旧控制仍为冻结短历史概率。

同extended预算强控制已补齐，不继续堆分类特征。结果与部分月状态持续性解释相容，但分量Brier不能相加且不能唯一因果归因。下一待检验方向应先明确月表可约束的粗状态风险与未识别调度残差及其迁移合同；尚无后续实现或评分，不预设残差跨地区不变。不把普通hurdle称原创，无新CIF或未查看地区晋级，目标active。[V187结果](V187_EXTENDED_TWO_SCALE_ZERO_RESULTS.md)。

## V188受约束前向切分完成：普遍迁移未支持，保留油电例外

固定2024-07-01截点、原抽样时效子集，336森林和158 H2 test月/27地区，17,696小时/5,056月状态/2,172条件记录完整审计；8848经验叶均值、724真实正月query/fuel、12完整未来/raw guards通过，最大Brier差2.22e−16。有效源小时实际仅Apr–May；SPA部分fold无合法fine样本，未补样，粗/小时源覆盖明确。Q3 78月26区/Q4 80月27区，共同地区才作跨期配对；已查看2024、回顾性版本限制保留，不称真正线上确认。

input-positive油电条件cond43对clock H2改善27.084%，描述范围[−.085384,−.007555]；共同9地区Q3/Q4优势均为负，不能说所有调度风险不可迁移。但部署总体direct43 .093481仍优于持平×cond43 .098322，条件任务收益未自动变部署收益。核电六折正月训练小时零事件均为0，三个条件头全零一致是退化，不是稳定或真实低风险证据。原1584比较与查看H2部分点值后增补1152强direct比较分列；全部仅开发描述，无CIF/新地区晋级。

全部四归档input/output hash及四脚本AST通过。停止无依据堆特征；下一只提出固定源事件支持后用同预算source-block删留检验局部油电收益是否由少数源主导，尚无该后续实现/评分，不称普通诊断为原创。普遍迁移强假设未获支持，但保留燃料例外、不可唯一归因，原gate不变，目标active。[V188结果](V188_FORWARD_ZERO_RISK_RESULTS.md)。

## V189油电源块压力测试完成：保留局部收益，但两类别支持敏感

预锁全部29源删除，不补小时；122拟合块488森林，30场景仍共用原158查询。r2正式23,700部署/5,490条件评分及全量audit通过，最大Brier差1.67e−16，244完整未来/raw guards通过。r1正式五折后因fold5真实油电正月任务为空的schema写出错误退出；旧源码/log/五折保留，r2完整重跑；全部122预检指纹及旧五折预测exact。有效及退役快照映射hash均通过，不伪造空条件评分。

只有13源提供油电正月小时；16源删除对条件概率结构性exact，不充稳健证据。全样本、fine-affected、support-exposed分开：348/348后两者身份中336/156可估计，空集合status/None。CISO exposed同44月9区cond43−clock由−.032544到−.004660，变化+.027884 [+.004281,+.065688]；原全查询input-positive为56月11区，不混分母。该源baseline各proper折提供67.0–88.4%正月非零小时加权类别支持，却仅0.95–1.64%零事件类别支持；这些按折定义，不是容量、总生成或跨折独立贡献。IPCO删除cond84的H2/Q4点差反转但变化范围跨零，保留不确定性。

全样本含baseline复用会稀释源影响；源删除同时改训练量、特征覆盖和地区权重，不作唯一因果归因。cond43仍有各单删场景的局部点收益，不能说全部迁移无效；部署baseline持平×cond43 .098322仍输direct43 .093481，不能以条件任务收益替代强部署比较。没有新增CIF/未查看地区晋级，不把普通删源或分类特征当原创。停止无依据堆零风险特征，后续先核新的合法信息合同及事件支持，尚无该后续实现/评分，原gate不变，目标active。[V189结果](V189_SOURCE_BLOCK_OIL_RESULTS.md)。

## V190完整CIF支持稳定化结束：覆盖过门，强基线效率失败

先核QRF/weighted CP/DRO/类别条件Wasserstein与库内V119/V140/V159–175，未将类别重权或W1身份当原创。固定July cutoff、proper原Apr–May小时子集和55早期独立cal月，158 H2 test/27区；July回算cal不是cal逐月实时前向，也无新域交换性保证。16个完整CIF law含风险影响选源事件层、仅集中度、全局源均衡/冻结kernel删源、ESS/spread、强direct/锚/异方差及宽区间控制。

有效r2正式54风险模型、7900评分/163584 law哈希和层诊断全量审计通过，独立真实CDF² CRPS最大差5.57e−11、2112合成积分差2.45e−10；12完整未来/raw guards通过。r1正式因阈值±0原始字节指纹误差在评分前拒绝；三fresh证据精确定位num_threshold零符号，r2仅规范化该字段的零表示并每fold重复拟合、全部source/query预测strict bytes一致，预检实际108风险fit。首audit仅列序错误保留源码/log后完整重跑，生产未改；所有有效及退役快照映射hash/四脚本AST通过。

H2 full CRPS44.9361/宏98.1109%/最差87.1603%/宽435.453；对physical_bg只改善1.357%且宽+9.143%、Winkler+6.015%，描述CRPS差范围跨0。对strong direct84 CRPS差14.905%，对普通ESS/spread差区间正侧；对event_mass仅约1%点改善，未支持额外CIF几何收益。40/336逐参考通过（H2仅Uniform10/全范围/弱全局源均衡），全部强参考联合gate=0。保留对Uniform10 CRPS−5.855%/宽−13.851%的有限正例，不冒充统一突破；Q4最差80.2536%，不能用H2合并值称时间稳定。

核电“源正月内小时零”层仍absent，不等于全部核电零atom缺失或真实零风险为0；候选只重权已有支持，未识别目标新事件质量/shift预算。停止本支线扫描alpha/类别/地区，后续若需新信息先核真实合法可得性，不将普通支持修复称原创。无新增2025评价/未查看地区晋级，原完整gate不变，目标active。[V190结果](V190_SOURCE_EVENT_CIF_RESULTS.md)。

## V191延迟月观测隐状态查验：未知地区κ偏置留下gauge，候选停线

先排重V9 Kalman、V39–40粗误差/持续偏置、V152/V163能源—时间均值差、V186–188月历史风险，以及既有sharp sets/支持/校准。候选x=(a,κ)、μ=a−κ、月报只观测a；允许未知地区迁移b时，κ→κ+d、b→b+(I−A)eκ d保持全部粗观测条件似然，却改变时间均值。Fraction例子已知b rank2、未知b增广rank3；更多同类月边际仍不解除，新增μ/κ横截观测使该例rank4。V15物理例子明确复用，不称新定理/任意模型不可能性。

真实锁定source支持仅220月统计/108转移，六折19/17/17/17/18/20，每区至多一转移且自由region intercept时A额外rank为0，完整Fraction设计/null方向、原sample角色/时效/转移绑定审计通过。κ为抽样proxy，非完整月协方差。canonical另有27源五个早期月、TEC四月、SPA两月metadata，不虚称不存在数据；本轮未启用其额外fine监督。延长source序列可能帮助参数估计，但不能独自识别未知目标offset。

没有训练模型或CIF评分、不重跑常规过滤器当创新。解除一维gauge需要可靠本地时间均值CIF/κ联合矩，或独立可检验的目标offset约束；仅更多同类月量/完整G不够，少量小时点不认证整月μ。这些不是已取得新数据或完整CDF识别保证。当前候选停止、未查看地区不开、原gate不变，目标active。[V191结果](V191_DELAYED_STATE_RESULTS.md)。

V191补充核验：有效Fraction audit-r4另显式证明物理witness逐燃料月合计均为[1,2]，并从原v153 timestamps按V188矩阵raw_id核July cutoff；r1–r3与对应源码证据保留。未改producer或任何预测模型。

V191唯一有效最终审计为`results/v191-delayed-state-observability-audit-r4/summary.json`。r1因未显式核完整source Fraction rank退役；r2补rank但缺物理逐燃料列和断言退役；r3补列和但缺原timestamp独立cutoff核验退役；r4两项均补齐。冻结协议与producer未改，50项输入/输出hash及两脚本AST通过。

## V192 跨月重分配fine标签的κ预测：月均开发完成，未触发概率扩展

只在V188 fold0的19个proper源内部逐区holdout，Apr/May/Jun三origin、57query，旧cal/test其他10区、H2目标truth与2025标签不进入评估。固定八个普通控制（能源锚/季节、direct Ridge/ET、a−κ分解Ridge/ET、pooled/random-intercept VAR/Kalman）×spread/recent，912月均记录完成。每case16预测先封存后评分；source/fine发布时点、目标region排除及fresh未来M/raw干预预检通过。

预算范围须保留：单origin每分配19源池17262，LOO fit16320–16388；跨Apr–Jun去重标签recent34532、spread27317，不是跨三期全局唯一标签预算相同。完整metadata库存验证。spread最佳分解ET μMAE43.9254，虽胜direct ET50.0531，仍输最新能源锚37.0847；相对recent只改善0.569%，地区paired差CI跨0。κ ET MAE3.98898 vsκ=0的4.09295，仅小点值差；动态a误差大，不把失败唯一归因κ不可迁移。单June同预算端点的分解ET46.2427亦输锚42.8560。四个预锁概率触发均false，停止本候选、不调阈值或开新地区，不宣称所有SSM失败/新算法。

查看主表后固定的posthoc“最新能源锚−四κ头”八组合不重训；clip合法点均未胜37.0847，spread ET38.71995、pooled37.94847/random37.89609。raw仅用于有符号误差恒等式，不能把κ校正风险与未来a变化绝对误差直接相加，更不能据κ MAE微降推出CIF分布增量。

完整audit-r2 terminal exit0：57query/114 refit、5130 source-profile及1864296小时重复出现记录、912损失/32比较重建；maxscore7.28e−12、独立线代/叶均值prediction差2.52e−11。首audit因JSON排序与producer列序不同失败，源码/log保留；仅audit完整核键后恢复列序，正式源/模型/协议未改。五归档1093项hash与五脚本AST通过。没有新增小时CDF的CRPS/90%覆盖/宽度/Winkler，原完整gate不变，目标active。[V192结果](V192_REALLOCATED_KAPPA_RESULTS.md)。

## V198 时间外推确认：供体分歧有区间效率增量，地区覆盖仍失效

V197 固定 k=15 尺度化方法在2055/2060投影年份按预封存协议完成时间留出：目标缺小时整月排除156条，每轨3,642 query；没有供体预算失败，哈希/置换隔离通过。相对 pooled LOO，same/cross 区间宽度分别约−27.1%/−29.6%、Winkler约−30.5%/−30.2%，覆盖差+0.87/+0.41pp；31省聚类配对bootstrap下覆盖差区间跨零，宽度和W差区间为负。相对全部供体经验粒子，覆盖提高5.02/5.54pp且更窄，三个配对指标区间均不跨零；相对 k15 粒子，V197更宽且改善主要来自恢复覆盖。

这是迄今较强的未来投影时间外推机制信号，但不是未见地区或真实数据验证。省级覆盖极不均：最低同情景41.2%、跨情景47.0%，仅23/31和21/31省达到90%，京/宁/晋/津/疆低于80%。中心仍使用事后月均oracle，标签仍来自同一模拟器；归一化conformal也不是新通用算法。因此只支持“模拟CIF形状迁移的局部风险信号值得独立验证”，不能写成中国运行区间已可信。

不得在2055/2060继续调参。下一步优先落地用户可获得的逐类月发电量信息预算：需核对真实表头、单位、统计边界、分类和发布时间，特别确认“火电”是否有煤/气/油内部月度分项；用户描述了所有发电大类，但原始表头尚未进入项目，不能据此假定可直接算燃料EF。然后冻结粗月信息与模拟真值拆分的公平实验，目标是量化该公开预算究竟约束月水平、小时形状还是仅帮助风险先验。若无真实高频标签，论文结论须明确限于模型世界基准。

详见[V198结果](V198_FROZEN_DISPERSION_TIME_HOLDOUT_RESULTS.md)与[冻结协议](V198_FROZEN_DISPERSION_METHOD_TIME_HOLDOUT_PROTOCOL.md)。目标保持active，仍未找到可宣称的论文突破或获得未查看地区晋级依据。

## V199 中国月度八类组成探针：小幅形状信号，未达可靠区间增量

用户确认能取得各类月度发电量后，项目内Ember中国省月表作为公开估算代理已实际核实含煤/气/其他化石/生物质/风/光/水/核八类。V199只用2025八类份额按Bray–Curtis选k15形状供体，目标月均仍为oracle并固定；五个CEF情景2025目标早已在V194评分，因此结果明确为回顾开发，不是盲测。组成近邻相对全30等权MAE下降3.0% same、1.8% cross；same簇bootstrap差[-.00345,-.00029]，cross[-.00275,.00046]跨零。各情景方向一致，但仅20/31和18/31省改善，若干省退步；不能晋级或声称新方法，且Bray–Curtis迁移已是常规基线（V63此前在EIA代理上也未带来Winkler优势）。

Ember月发电侧强度与CEF投影月均的水平相关0.819，但换成相同单位后平均高160.8 g/kWh、MAE166.4；这是不同估计对象/边界的描述比较，不是任一方准确性证明，不能直接当预测锚。V199 r1错误地未换算g/kWh与t/MWh的数值，保留退役；r2已换算，形状结果不受影响。详见[V199结果](V199_CHINA_MONTHLY_MIX_SHAPE_RESULTS.md)。下一轮固定这个供体规则测试LOO校准区间是否保覆盖、降Winkler；仍是同一已查看模拟集上的开发探针，不能外推为实测中国确认。目标保持active。

## V200 组成条件区间：没有转化为覆盖保持的效率增量

V200固定V199 Bray–Curtis k15，在五情景2025目标小时做外层省留一，并对伪目标省跨情景profile严格排除，按省簇bootstrap。组成k15尺度区间宏覆盖同/跨90.03%/89.95%，但最低省覆盖仅57.6%/59.4%，≥90%省仅20/31、18/31。相对 oracle CIF 月均近邻 V197型尺度区间，Winkler差+0.00045/−0.00054且簇CI跨零；相对全30 pooled，same虽点上改善但CI跨零，cross宽度反而显著增加约0.0115。组成k15 pooled比尺度化更窄、跨情景Winkler更低。组成相似度方法不胜强基线，也不解决局部覆盖；停止调k/核宽/分位数支线。[V200结果](V200_MIX_CONDITIONED_INTERVAL_PROBE_RESULTS.md)。两轮使用Ember current snapshot与同一CEF模拟器/已查看小时标签，只能作机制开发，不可当真实中国确认。

接下来的破局空间不在继续包装近邻/归一化校准，而在严格的信息价值/识别界：用用户原始月表合同确认能精确推到哪种月级estimand，计算此预算下小时均值/尾部仍允许的最宽/最窄世界，再确定认证剩余风险所需的最小新增本地观测。真实小时标签仍是中国算法有效性的主要外部验证缺口。目标active。

## V201 稀疏逐日燃料边际：日度总量的小时尾部识别价值有限

在25个EIA平衡区、449区月上，用固定燃料EF代理精确比较月度八类燃料总量、随机1/3/7/14天逐日八类总量和全月逐日总量所允许的小时超阈值计数集合。对`τ=300`、5%门槛，14天报告只将宏平均区间宽度从月度的0.7589降至0.7435，无判定率从76.0%降至75.0%；全月逐日也仅降至0.7271。罕见1%门槛的正证书率会明显上升（月度4.0%、随机14天53.4%、全日57.2%），但该指标是部分识别证书，不是预测覆盖或概率准确率，且区间仍宽。V201的原始producer输出对每个稀疏方案有两份结果：确定性重复一致，随机是两批50次抽样；报告指标在批次级审计合并后得出，来源尚未从执行记录定位。限制见[V201结果](V201_SPARSE_DAILY_FUEL_MARGIN_IDENTIFICATION_RESULTS.md)。这不是算法突破或中国外部验证。当前数据边界启示是：增加同类月边际难以识别小时排序；后续要先核实是否有合法可用的日内分时/峰谷燃料构成信息，再评估该形状信息的价值。目标仍active，未触发未查看地区验证。

## V202 日内分辨率信息前沿：分时形状有效，但需高报送预算

对同一月度边际增加少数日1/2/4段（全天/12h/6h）八类燃料量。τ=300、5%门槛下，同为112字段，7天×12h方案的宏区间宽度0.7412、无判定72.4%，略优于14天×日总量0.7435、无判定75.0%；效应有限。14天×6h需448字段，宏区间宽度降至0.6888、正证书51.5%；全月×6h端点为0.6117。宏结果有强结构混合：6/25区月度边际已证明无超阈值小时，其余19区平均月度区间几乎全宽0.9986；在这19个post hoc困难区，14天×6h使区间平均窄9.22个百分点、19/19地区变窄，而112字段以内大约只窄2.0–2.3个百分点。信息粒度比单纯增加日数更关键，但显著收益目前需高字段成本，且是固定EF EIA代理上的确定性部分识别，不是CIF概率覆盖/中国预测验证。见[V202结果](V202_INTRADAY_FUEL_MARGIN_INFORMATION_FRONTIER_RESULTS.md)。暂未达论文算法突破；有价值的基准贡献方向是量化公开信息粒度与地区可识别性前沿，下一步先把中国真实公开字段、发布时间/边界和独立小时真值落地，再判断该基准能否支撑论文。未查看地区仍未开启，目标active。

创新性审查补充：EDGAR已发布年度/行业排放的高分辨率时间profile，深圳研究也用典型负荷曲线把月度工业排放分摊到小时；2026 Applied Energy进一步比较年、季节、时段和月-时段电网EF。因此不能把“更高时间分辨率有用”写作创新。可能的差异只在月度燃料边际约束下给小时尾部锐界、并建立统计字段预算—可识别性—跨地区异质性的系统前沿；仍需系统综述和中国外部验证。[EDGAR](https://doi.org/10.1038/s41597-020-0462-2)、[深圳研究](https://pubmed.ncbi.nlm.nih.gov/29414363/)、[Applied Energy 2026](https://doi.org/10.1016/j.apenergy.2026.127655)。

**V201/V202与旧分支重叠审计：**V90已经评估年度装机容量；V131–133已有阈值侧日信息、多尺度日期预算与报送误差；V136–139又加入粗装机及约束校准。故当前无需重做年度装机或一般日统计实验。V201/V202沿用同一EIA数据家族，分别以八类日边际和八类分时边际作代理变体，不是独立样本复制。V202的区段尾部集合按V87公式只依赖H/G/M；八类边际可压缩为3个风险侧段统计，而这一类阈值侧合同本身已由V131/V132探索。新结果应视为粒度敏感性旁证，不算独立创新晋级。真正未解决的是：用户所持中国原始月表的具体字段/发布时点、授权中国小时CIF标签，以及这些可用输入下相对强基线的可信外测；在这三项缺失时，继续合成EIA式信息粒度扫描的边际价值很低。

## V203 文献近邻审计：已有中国月燃料＋天气→日CEF预印本

新增检索发现2025-12 SSRN预印本已提出用月度分燃料发电量与天气下推每日发电，以CarbonMonitor-China日CO₂排放构造/训练中国省级日CEF；比较RF、GB、XGBoost、LightGBM，摘要展示内蒙古、江苏、云南三省时序验证与情景Monte Carlo。[预印本页面](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5960745)。因此“月度燃料+气象生成中国高频因子”不能作为TransCIF首次创新。本轮尚未取得全文，不能判断其目标构造、泄漏、源外评测或代码复现细节。

CarbonMonitor-China官网/API当前提供至2026-06的省级日排放**估算**，并明确属于会修订的活数据；它不是小时CIF真值或历史首次发布样本。[数据页](https://cn.carbonmonitor.org/)。新颖差别只能落在严格小时省份冷启动、信息as-of、风险区间和真实标签外测，但这些都仍未通过。EIA固定EF与中国模型世界实验不能充当对该预印本的胜出证据。V196已补充这一文献与数据边界；项目现阶段不能称已破局，优先动作是取得用户原月表字段/示例和一个可授权中国小时标签源，而不是再更名跑一个树模型。

另有[GHGbench 2026](https://arxiv.org/abs/2605.13743)在企业/建筑排放预测上做跨实体/跨区域split，故“unified benchmark + leave-one-region-out”本身也不足以构成TransCIF创新。电力CIF专属的物理标签合同、粗月报信息预算、小时风险和逐区域条件覆盖必须成为实质内容；目前由于小时中国标签缺失，这条贡献还只是设计，未实证。

## V204 用户确认月报含全部发电种类：更新信息边界，不改变小时结论

用户进一步说明其可得月报包含所有发电种类的月度发电量（此前对是否有内部非可再生分项的疑问已得到肯定答复）。原始字段表头、单位、统计边界、首发时点与修订机制尚未进入仓库，故目前只把它作为“全类别月边际可用”的预算条件，不自行补造字段。此条件比常见仅“火/水/核/风/光”粗分类更强，可在类别 EF 确定且边界匹配时识别发电量加权月平均强度；仍不识别等小时月均、小时排序/尾部或消费侧交换碳流。形式化边界与已有结果映射见[V204决策记录](V204_MONTHLY_FUEL_TABLE_IDENTIFIABILITY_DECISION.md)。

这条用户信息不要求再跑V87的重复月份边际实验；V87已对完整燃料月边际的尾计数锐界做过全样本复核。下一步的增量应是审计真实字段后，按同一月报预算比较识别集与迁移预测，并清楚分开生产侧与消费侧estimand。当前仍未找到已证实算法突破；真正的中国小时盲测仍受合法小时标签与发布版本缺失限制。

## V206 独立CISPO-China投影压力测试：形状信号保留、区间迁移失败

首次评分前依据发布JSON纠正闰年协议：PlanetaryEnergy 2040/2060也为8760小时，Li源工作簿为8784小时；源侧移除2月29日并用标准年月份切片。V1在评分前停止，V2封存后完成30省×6情景×7年份×12月共15,120查询、75,600条方法结果，prediction manifest哈希与重算预测一致。

在目标月均CEF事后oracle条件下，k15形状供体使点MAE由48.352降至45.343 g/kWh（−3.009；省簇bootstrap 95% CI [−5.111,−1.011]），在6/6情景改善，但2060年反而略差。区间上V197尺度化比同预算pooled区间窄6.724 g/kWh，却覆盖率低8.55个百分点（CI [−10.11,−7.04]），Winkler更差30.79（[21.08,40.56]）；pooled本身仅70.34%宏覆盖且仅1/30省达到90%，天津尺度区间覆盖44.4%。停止跨模拟器通用尺度风险机制主张，不在已评分B上继续调参。

同一已打开目标的探索性形状控制显示：目标μ匹配后的全部30省等权shape均值MAE 46.915，进一步只聚合最近k15为45.343（−1.572；省簇CI [−2.237,−0.983]）；单最近供体53.522，不看μ的源区月度气候57.971。这说明近邻聚合及目标月均匹配提供可测点形状价值，但算法本身是常规均值近邻，不能包装成新创新，也没有改善迁移区间覆盖。

V206只说明两套规划模型之间存在有限月内形状点预测信号；目标月均是oracle，目标B是进口调整的规划模拟器，不是实测中国小时CEF，不能据此声称用户逐类月发电报表已可部署，也没有产生新算法突破。论文候选仍应定位为严格粗信息合同下的迁移/可识别性基准，并必须把形状误差和风险区间分开；若无法取得授权小时标签或多个独立真实CIF来源，外部有效性仍不成立。详见[V206结果](V206_CROSS_SIMULATOR_TRANSFER_RESULTS.md)及[冻结协议](V206_CROSS_SIMULATOR_TRANSFER_PROTOCOL.md)。

## V205 月边际×因子不确定性：EF区间不是主要小时风险瓶颈

为避免把V66固定EF结论直接外推，在Ember中国八类月边际上对每类EF用低/高端点计算稳健K端点包络，并与V99的类别跨阈值诊断、V103/V129含容量的EF稳健研究对齐。宽压力盒下平均K宽度相对名义EF仅增加τ=100/300的0.01小时、τ=600的0.17小时、τ=900的0.24小时；端点嵌套检查0失败。尾率1/5/10%三档均无正证书，τ=100/300全弃答，τ=600/900约5.76%负证书、94.24%弃答。名义月度集合原已宽至整月的94–100%，因而这不表示EF可忽略，而是本设定里**小时燃料分配歧义远大于测试的EF区间效应**。这是稳健识别结果，不是预测覆盖。

结果闭合了一处月度边际单独约束下的敏感性缺口，但EF端点包络是简单单调扩展，且与V103/V129邻近，不能当作算法创新。停止继续调EF盒。下一步应测试本地时间分辨信息/可验证运行约束的边际价值，或优先获取合法中国小时标签。细节见[V205结果](V205_MONTHLY_MARGIN_FACTOR_ROBUST_HULL_RESULTS.md)、[协议](V205_MONTHLY_MARGIN_FACTOR_ROBUST_HULL_PROTOCOL.md)。使用Ember历史估算数据，不是用户原始表；用户表头、口径和as-of时点仍待核实。

## V208 江苏运行派生曲线：共享形状有效，近邻单体和地区风险仍失败

从中国工程院期刊公开江苏案例的四季、13地市小时CEF矢量图做探索性数字化；春季连云港、苏州、江苏全省共9个表图校验点最大差0.4 g/kWh。13市留一回测使用目标日均CEF事后oracle，作为比用户月报更强的信息上界。全部12个供体的中心化小时形状均值把点MAE从34.296降到26.323 g/kWh（−7.973；市级bootstrap CI [−12.260,−3.565]；11/13城市胜出；四季方向一致）；只选均值最近的单市反而为37.018。源市LOCO内选k在6–8，外层MAE 25.735，城市簇CI [−13.406,−3.553]。这支持“共享形状＋本地水平”机制假设，并表明广覆盖池比单一近邻更稳，但都是常规均值/交叉验证选择，尚非算法创新。

使用同一目标日均oracle，形状区间覆盖88.22%、宽120.26、Winkler90=175.59；相较锚定残差区间88.46%、宽158.96、Winkler=219.79，宽度和Winkler改善，但盐城/南通/宿迁/苏州覆盖仅54–71%。因此没有可信地区风险保证；样本也只来自江苏一个电网的4个论文典型日，不是独立小时观测。曲线未提交至仓库或分发，公开复用许可待查。该结果增强“小时形状可迁移”的真实运行派生案例证据，但仍不足以称论文破局。下一轮需换为用户可得月报信息合同、增加授权的独立省网/年份标签，并预先冻结对强层级及动态基线的对比。详见[V208探索结果](V208_JIANGSU_OPERATIONAL_CURVE_PILOT.md)。

**文献新颖性边界**：DACF已有六区CIF预测，但用历史小时电源结构和日前天气；Zhang等2026预印本已把小波多尺度卷积用于CIF多变量预测。[DACF](https://lass.cs.umass.edu/papers/pdf/eEnergy2022-dacf.pdf)，[wavelet CIF preprint](https://arxiv.org/abs/2601.06530)。因此小波、多尺度或普通跨区均值模型均不能单独算原创。TransCIF仍需把目标无小时历史、仅月度类别边际的信息合同做成实质问题；V208用了日均oracle，不能当成该合同下的确认实验。

## V209 目标月报水平锚＋独立形状模块：点误差有显著增量，风险覆盖仍崩

将Ember估算的2025八类发电份额作为月报代理，在五情景2025投影CEF上做留一省开发回测，目标省全部标签均排除。V209第一次不再用目标CIF月均oracle：按项目名义EF把八类份额压成透明粗强度，再用其他省份普通OLS拟合“粗强度＋月份”。月均水平MAE为104.49 g/kWh，优于月气候均值189.93、未校准固定EF代理167.28和八类组成直接OLS124.40；因子低/名义/高端点下经源区校准MAE约104.3–104.7。这个结果提示物理标量代理加源校准比高维份额拟合稳，但不是新算法，目标表与投影标签口径也不匹配。

在该水平锚上加入Bray–Curtis k15供体形状，小时MAE从129.71降到119.42；相对全30形状121.23再改善1.81（31省簇CI [−2.97,−0.73]）。相对气候均值＋全30形状的差−82.01，[−123.16,−42.20]。真实目标月均oracle＋k15为59.01，显示月报替代锚后仍有很大差距。

严格双重地区排除的90%残差区间宏覆盖89.6%，但均宽约482 g/kWh；k15区间在重庆仅24.7%、北京47.5%、吉林60.9%，月分层校准没有修复。宽度约为V208日均oracle区间的4倍。故该信息预算下的点预测有增量，但尚不能提供可信未见省份风险；总覆盖近90%掩盖地区灾难性失效。V209是同模型情景、Ember估算输入和已打开标签的开发回测，不是用户表实证。下一步冻结特征后检查这些高风险地区能否由预测前可得的源支持指标识别，并比较选择性弃答/物理集合回退；否则应承认月报不能识别该风险。详见[V209结果](V209_MONTHLY_MIX_LEVEL_AND_SHAPE_CV_RESULTS.md)。

追加的风险支持诊断表明：省平均最近Bray–Curtis距离与覆盖Spearman仅−0.061，最差月距离为−0.208；重庆和吉林虽然燃料组成仍在常见支持范围，小时区间覆盖仍严重失败。源LOCO距离90分位拒答只排3/31省、接受集最差覆盖24.7%；75分位拒答16省后最差接受覆盖仍60.7%。这是posthoc诊断，不能称有效的支持门控。V209具体暴露了关键盲点：月燃料组成可指示协变量远外推，却不足以识别相似组成背后的本地水平/调度风险。该风险需要本地时序摘要、运行/交换信息或更保守的部分识别范围；目前月度字段单独无法为窄而可信的区间背书。

## V210–V212 月度统计扩展和风险预测复核：当前没有晋级机制

V210把用户说明可公开取得的逐类月度绝对发电量纳入相同留省开发协议。粗EF加权代理＋月份校准仍最好：水平MAE104.49、小时MAE119.42；加总发电量后分别109.42/123.98，加火电量111.23/125.84，八类份额＋总量130.96/144.49，逐类绝对量139.05/151.04。V211对高维类别斜率施加非负约束也没有改善（小时MAE137.20）；代理标量的非负斜率约束与原模型完全相同。当前结果不支持“把全部字段送入高维回归”或“加简单物理符号约束”作为增量。

V212做了目标省排除式第二阶段尾部预测：以中心模型的每个省×情景×月份小时绝对误差q90为风险标签，输入当月/此前类别份额、轨迹波动与变化、月总量和供体形状分歧；训练残差来自其他省的一阶段留省预测。训练省合并q90基准宏覆盖89.58%、宽481.65、Winkler639.05，最差省覆盖24.22%。岭风险模型收窄到宽368.08，但覆盖跌至80.34%、Winkler升至664.36、最差覆盖14.68%；梯度提升q90覆盖92.63%，但区间变宽至550.59、Winkler650.87、风险q90排序Spearman仅0.029、最差覆盖38.06%。动态统计与供体分歧未可靠识别目标小时尾部；并不构成对所有算法的普遍不可能证明。

研究立意进一步收敛为**低频公开信息下小时CIF的识别边界和最小信息价值曲线**：把月度统计、日级摘要、小时本地锚/可再生出力等信息预算逐级增加，用严格留省构造的标签比较水平、形状、尾部和区间效率；如果新信息也不能同时维持覆盖并缩窄区间，就明确报告未识别/拒答，而不是再调整中心模型。时间反分解文献已讨论聚合损失，条件覆盖文献也说明边际覆盖不能推成地区条件保证；潜在原创性必须来自电力CIF任务的合法信息合同、可复现价值曲线和独立真实数据支撑，而非普通时间分解、波形模块或conformal包装。[时间反分解与信息损失](https://www.tandfonline.com/doi/abs/10.1080/07350015.2014.995797)，[条件覆盖保证](https://arxiv.org/abs/2305.12616)。

V210–V212仍使用Ember估算和Li等同一模型家族投影标签，不能算真实中国验证；未锁定可用方法增量，故不开放未查看省份。详见[V210–V212结果](V210_V212_MONTHLY_INFORMATION_RISK_LIMITS.md)。

## V213–V214 信息价值上界：少量跨天本地CIF锚比单个典型日有效

固定V209留省预测中心，随机给目标省每个情景月额外提供1–168个小时CIF标签，用本地残差均值校正中心、用抽样误差经验q90估计半径，剩余小时评分；每预算重复20次。48个均匀随机锚宏覆盖89.10%、区间宽234.99、小时MAE59.60、最差省覆盖88.74%，相对零本地锚的覆盖89.58%、宽481.66、MAE119.42、最差24.22%有明显风险效益。168锚后宽度几乎不再下降。单点经验区间退化为零半径，3–12点覆盖不稳定。

重要的是布点而不只是数量：24个小时锚若集中为一个连续典型日，覆盖68.70%、最差省42.98%；24个锚分散到不同天并覆盖24个小时-of-day，覆盖87.38%、最差86.17%。48个锚分散到跨天、每小时-of-day两次时，覆盖89.58%、宽236.73、Winkler293.88、小时MAE59.34、最差省覆盖88.96%；48个锚集中为两个连续日的最差省覆盖仅58.16%。这为“整月风险不能由单一代表日可靠校准”提供了更具体的信息采集假设。

目前最接近的破局候选是**目标侧信息采样设计/信息价值曲线**，而不是新的CIF预测器。该实验直接使用目标CIF小时标签作锚，仍为Li同模型投影回测、每查询20次抽样，经验q90不具小样本有限样本保证；48点不是中国真实系统的已验证最小量。下一步必须换独立运行曲线/授权小时标签，并测小时燃料或可再生出力是否可代替CIF锚；验证前不宣称算法原创或部署可靠，也不开未查看省份。结果见[V210–V212报告的V213–V214补充](V210_V212_MONTHLY_INFORMATION_RISK_LIMITS.md)。

## V215 EIA 2026H1时间留出：小时锚布点信号跨到运行发电结构代理

为避免只在Li投影标签里重复检验，使用项目EIA-930小时燃料发电量，按V202固定EF计算小时CIF代理。2025全年训练水平校准、形状供体和嵌套源残差；2026H1整区留出，24个可用平衡区、135区月。目标整区各年度数据排除于模型、供体和校准；源残差另嵌套留出源区。样本月若低于小时完整性门槛则剔除。

零目标锚时，小时MAE38.43，区间宏覆盖85.37%、宽165.28、Winkler267.47，最差区覆盖35.13%。48个小时锚若按每个UTC小时段跨日取2个，覆盖90.04%、宽141.85、Winkler175.44、MAE38.23，最差覆盖89.06%。对比24小时连续日，48分层锚的覆盖增加22.3个百分点、Winkler降低211.55（地区簇CI [−303.08,−128.93]）。48分层相对48随机仅有小幅增量：覆盖+0.4pp [0.3,0.6]，Winkler−0.96 [−1.48,−0.50]，点MAE−0.06 [−0.11,−0.03]。相对源校准基准的Winkler改善−92.04 [−126.97,−60.65]，但覆盖差与宽度差CI均跨零。最明确收益是避免典型连续日采样造成的风险盲区；分层相对均匀随机的额外收益较小。

该结果跨过Li投影标签，支持稀疏本地观测布点作为数据采样候选，但不能视作独立实测CIF复制：目标标签直接由EIA小时燃料估算和固定因子构造；EIA文件与项目之前分析有数据家族重叠；2026H1也是回顾性开发窗口。实际中国月报仍只有月度发电类别，48个小时锚并非已取得的信息，更没有有限样本保证。详见[V215协议](V215_EIA_SPARSE_LOCAL_ANCHOR_PROTOCOL.md)、[结果](V215_EIA_SPARSE_LOCAL_ANCHOR_RESULTS.md)、[逐区结果](../results/v215-eia2026-sparse-anchor-frontier-r1/region_metrics.csv)和[可复现脚本](../scripts/v215_eia2026_sparse_anchor_frontier.py)。

## V216 把48小时锚拆成独立拟合/校准：覆盖稳住，效率回退

在V215相同EIA 2026H1目标留区上，每区月固定48个随机小时锚，把一部分用于本地残差均值修正，剩余部分独立做split-conformal校准；比较12/36、24/24、36/12。24/24得到92.52%宏覆盖、159.13宽、Winkler183.02、MAE38.57、最差地区91.80%。相较源区q90基线，覆盖提高7.2pp [0.9,14.6]，Winkler下降84.45 [−118.58,−52.66]；宽度和点MAE的地区簇bootstrap区间都跨零。相较同48点重复拟合中心和经验q90的描述性方法，split CP多覆盖3pp，但区间宽17.48、Winkler多6.53，效率显著回退。24/24是三种分割当前Winkler点值最低者。

随机目标小时索引下，有限样本split-conformal秩可以解释为该区月随机未观测小时的边际覆盖；不能解释为固定抽样实现的同时整月覆盖或任意地区条件覆盖。标准split CP不是新算法；2026已有split比例优化理论，非交换时间序列的split CP也有专门结果，稀疏随机时间点上的连续轨迹conformal同样是新近研究。[最优split比例](https://arxiv.org/abs/2606.31600)、[非交换数据split CP](https://jmlr.org/beta/papers/v25/23-1553.html)、[随机稀疏采样的轨迹预测区间](https://arxiv.org/abs/2608.29559)。因此当前贡献仍是低频月报＋极少目标侧小时锚这个电力信息合同及其风险价值曲线，算法原创性很弱。

这轮让候选更具体，也揭示代价：要得到可解释的目标区月边际风险控制，需要48个本地小时类别发电记录（等价导出代理CIF），其中约24个估中心、24个校准；只有月度全部类别总量仍不够。用户现有公开信息没有这48个小时观测。EIA标签是固定EF对燃料份额的机械换算，数据家族与旧实验重叠，且2026H1回顾性开发。此项只能指导下一步信息获取和真实数据设计，不能作为已部署中国CIF突破。详见[V216协议](V216_EIA_SPLIT_CONFORMAL_ANCHOR_PROTOCOL.md)、[结果](V216_EIA_SPLIT_CONFORMAL_ANCHOR_RESULTS.md)、[脚本](../scripts/v216_eia_sparse_anchor_split_conformal.py)。

## V217–V218 收敛到本地风险锚的最小预算

V217把48小时目标锚改成18/24/30/36/48等分的中心拟合/校准样本。18点（9+9）是这组等分中首个有限90% conformal半径：覆盖90.49%、宽162.75、Winkler196.13、最差区89.48%。24–36点因校准q90阶次仍取最大误差而过宽；48点达到24/24时才转为23/24阶，Winkler降至183.13。有限样本区间的预算曲线因此不单调。

V218再固定9个独立校准锚，把目标中心拟合锚降到0、1、3、5、7、9、15。表现最好的是**0拟合＋9校准**：只保留月报＋跨区点中心，9个本地小时燃料结构观测只更新对称风险半径；宏覆盖90.57%、宽154.94、Winkler187.56、点MAE38.43、最差地区89.64%。相较零锚源区半径，Winkler下降79.92（95%地区簇CI [−115.83,−47.01]）；覆盖差的bootstrap区间仍跨零。只用1点修正本地中心会严重过拟合：小时MAE50.39、宽208.95、Winkler244.29。48点24/24 CP较9点覆盖高1.9pp、Winkler点值低4.35，但该Winkler差区间[−0.52,8.96]跨零；9点显著节省目标信息预算。

**目前最有价值的突破候选**因此更具体：跨区模型负责月报下的点中心，本地少量小时燃料结构只校准目标区风险尺度；先采风险锚，不拿1–15个稀疏锚移动中心。对于非随机化90% split-conformal秩，9个校准样本是取得有限半径的秩下限；它不是已证明的中国最小充分统计量。9点结果来自EIA燃料估算按固定EF机械换算的CIF代理，同一项目数据家族和回顾性2026H1窗口；真实公开中国月报尚无这9个小时快照。故现在得到的是可操作的观测预算假设，不是算法原创或中国部署突破。

方法新颖性还要面对现有工作：split conformal、分割比例优化、非交换序列校准都已研究；稀疏随机采样轨迹的conformal在2026也有新结果。因此投稿不能把“使用conformal”或“调出24/24最佳”当新方法。更可守的差异是CIF零小时历史地区的新信息合同、目标区风险锚如何采集、月水平/形状/尾部是否能分开，以及真实新地区的价值曲线。[最优split比例](https://arxiv.org/abs/2606.31600)、[非交换数据split CP](https://jmlr.org/beta/papers/v25/23-1553.html)、[稀疏随机轨迹conformal](https://arxiv.org/abs/2608.29559)。

详见[V217预算前沿](V217_EIA_ANCHOR_BUDGET_RESULTS.md)、[V218最小校准预算](V218_EIA_MINIMUM_CALIBRATION_RESULTS.md)及脚本/逐区产物（各报告链接）。下一步应找一份独立授权中国运行曲线，测试目标月公开报告外增补9个小时燃料发电快照能否计算有效CIF锚；并用未查看省份/年份冻结确认。若拿不到任何此类本地小时信息，就应转向月报自身无法识别风险的部分识别基准，而不能宣称9点方法可部署。

## V219 月度边际约束形状迁移：八类燃料改善点值，但区间没有晋级

在同一EIA 2025源/2026H1目标回测上，测试从近邻供体迁移星期×小时的分燃料出力形状，再按目标月边际逐类缩放；对照现有月水平＋小时形状、月度常数锚和常见五类边际。目标区整年历史、目标小时发电量、目标小时燃料结构均不参与预测。其目标月边际在真实应用中要月末发布后才能取得，所以此处是月后小时重建，不是月初预测。

八类煤/气/油/其他化石/核/水/风/光边际方案小时MAE 32.16，对跨区形状基线38.43，配对地区差−6.28 [95% CI −12.80, −1.01]；平均区间宽158.43、Winkler260.55，较基线仅点值略好，Winkler差−2.91的CI [−41.11, 28.25]跨零。覆盖87.93%且最差地区只有45.69%，覆盖改善CI也跨零，因此不满足“维持覆盖同时缩窄区间”的门槛。常见五类火/水/核/风/光边际更差：MAE80.14、宽378、Winkler543、最差区覆盖7.88%，显示只知道火电合计时，供体补煤气比例可能严重失配。

这是一条有用的**信息分辨率诊断**：若目标月八类燃料实际可得，跨区供体形状可改善点值；火电内部无法细分时，模型风险显著上升。但约束时间分解与近邻形状迁移均属常见思路；小时标签还是同一EIA燃料代理、2026H1已参与开发。r1曾误以目标真实月均值作为源区基线，明确退役；只引用严格修正水平预测后的r2。没有算法突破或中国真实验证，不能冻结方法或开放未查看地区。详见[V219协议](V219_MONTHLY_MARGIN_SHAPE_TRANSFER_PROTOCOL.md)和[结果](V219_MONTHLY_MARGIN_SHAPE_TRANSFER_RESULTS.md)。

## V220 供体预算优化：没有区间门槛增量

在八类燃料边际条件下扫描K={1,3,5,10,15,23}。宏覆盖始终87.76%–88.10%，Winkler最低仍为K=10的260.55，最差区覆盖最低43.66%（K=15）。K=15虽然宽度比K=10少0.59，但Winkler差的地区簇区间跨零且最差区更差；K=10与V219 r2完全重现。K网格不是改善风险覆盖的杠杆，停止继续调K或在已查看地区选优。r1重复加月水平导致失真，退役；仅r2有效。[V220结果](V220_MARGIN_SHAPE_DONOR_BUDGET_RESULTS.md)。

## V221 燃料字段粒度前沿：煤电分项是点值信息关口，小时风险依旧未识别

固定K=10、校准与供体模板，只把目标月边际从五类“火/水/核/风/光”逐步拆为“煤＋其他火电”、再拆气、最后八类。五类时MAE80.14、宽378、Winkler543；煤电单列后MAE32.47、宽160、Winkler264。点值改善47.67 [地区簇95% CI −72.00, −27.88]，宽减少218，但最差区覆盖仍45.51%。再将气电单列或补到八类，和煤电六组相比MAE/Winkler差的区间均跨零，未观察到可靠额外增量。说明在EIA代理内，煤电月量可解释火电合计隐藏的主要组成差异；公开中国统计表是否有此细分仍须原表验证。各粒度覆盖均未达到90%且最差区严重失效；结果是信息合同诊断，不是新算法或中国确认。[V221结果](V221_FUEL_CATEGORY_RESOLUTION_RESULTS.md)。

## V222 Carbon Monitor 留国日度重建：供体周内形状与历史季节匹配均未胜过月锚

引入公开 Carbon Monitor Power 全球日度燃料发电序列，做75国/聚合区留出，2024–2025作为目标期，共1,800完整国家月。每个测试月向预测器提供八类目标月总发电量，模型不接触目标国的日度历史，评价的是月后日度重建而非月初预测。目标标签仍由同一估算数据和固定EF机械计算，故不是独立CIF真值。

月度固定EF常数锚宏国家MAE为14.983 g/kWh。迁移10个当月燃料构成近邻的CIF周内曲线，MAE升到15.103（相对锚+0.120，国家簇95% CI [0.080,0.168]）；迁移并精确满足八类目标月总量的燃料周内曲线，升到15.200（+0.217 [0.146,0.293]）。用目标国2019–2023月报季节结构匹配供体也没有改善（15.211；相对当月成分匹配+0.011 [−0.021,0.045]）。中国24个留出月中，锚MAE3.70，CIF曲线3.88，燃料曲线4.85，季节匹配4.81 g/kWh。

结论是当前跨国日历形状迁移假设被这个估算基准否定，继续调K没有足够依据。已知八类月边际可支撑generation-weighted 月锚，却不意味着能恢复日变化；月度历史构成签名也未识别可迁移日形状。研究路线更应聚焦“低频边际识别了什么、剩余风险需要哪种最小本地时间信息”，并允许模型返回月锚或拒绝伪造窄日区间。下一步若继续该数据家族，应直接测稀疏目标日燃料快照的信息价值与区间覆盖/宽度，再找独立运行来源复制；不得把Carbon Monitor标签当独立真值。协议和结果见[V222协议](V222_CARBON_MONITOR_COUNTRY_COLDSTART_PROTOCOL.md)、[V222结果](V222_CARBON_MONITOR_COUNTRY_COLDSTART_RESULTS.md)。

## V223 随机目标日燃料快照：出现信息预算增量，但只是测量设计候选

在V222同一75国家留出、2024–2025目标标签上，保持由目标月八类总量得到的generation-weighted固定EF锚不变，只用随机抽到的目标日完整八类发电结构校准锚残差的split-conformal 90%区间。源端半径按留一国家排除目标国所有历史后估计。留国源半径宏覆盖90.94%、宽80.71、Winkler132.81 g/kWh。7/8个本地日样本的90%有限样本秩要求半径无穷；9点首次有限，覆盖90.32%、宽62.14、Winkler75.60，但75国中34国的平均覆盖低于90%。12点覆盖92.56%、宽65.83、Winkler75.07；相对源半径Winkler改善57.62 [地区簇95% CI 45.18–74.53]，覆盖差1.62pp的CI仍跨零；对24个月取平均时没有国家低于90%。这只表示随机抽样设计下该估算基准的经验/边际风险结果，不是固定样本或任意地区的条件覆盖保证。

中国子集24个月中，源半径覆盖100%但宽81.36；12个本地日快照覆盖92.31%、宽26.72、Winkler36.27。Carbon Monitor中国日数据是估算产品，其论文描述日度燃料发电以月度CEC发电量和沿海省日耗煤曲线分解，故这不是独立中国运行真值。9点是conformal有限秩阈值，12点仅是当前测试网格中第一个宏覆盖超90%且国家平均都不低于90%的预算，不能称中国“最小充分信息”。

该项与V222构成清晰方向：月度边际本身适合作为水平锚，迁移日形状没有点值增益；若额外取得稀疏的本地日燃料结构观测，专用于风险尺度可能比补复杂形状模型有效。算法新颖性很有限：design-based conformal已覆盖有限总体抽样设计，[2026稀疏随机轨迹conformal预印本](https://arxiv.org/abs/2608.29559)也处理随机时点观测。论文价值只能来自CIF信息合同下的特定字段/采样预算价值曲线和独立运行真值，不应把split conformal包装成新算法。下一关是独立直接运行数据复制以及覆盖/效率冻结确认；如果拿不到，应把结论降为次级估算数据的测量设计假设。详见[V223协议](V223_SPARSE_DAILY_RISK_SNAPSHOT_PROTOCOL.md)、[结果](V223_SPARSE_DAILY_RISK_SNAPSHOT_RESULTS.md)。

## V224 EIA小时代理复核：按日最大误差校准，9日有限、12日区域覆盖更稳

为避免把一天内24个相关小时当作24个独立样本，在EIA 2026H1目标区上把完整本地日曲线作为一个校准单元，分数取该日相对目标月generation-weighted锚的小时最大绝对误差；源q使用2025同月其他区域，排除目标区。24区、135区月留出里，源q小时覆盖95.64%、宽267.75、Winkler295.89，但随机未观测日的24小时同时覆盖只有83.58%，最差NWMT仅22.9%。随机抽9个完整本地日，有限半径首次出现；小时覆盖98.64%、日内同时覆盖92.58%、宽213.18、Winkler218.47。相对源q，日同时覆盖+9.01pp [地区bootstrap 95% CI 1.41–17.99]，Winkler改善77.42 [24.32–133.32]，宽度差CI仍跨零。9日方案仍有9区平均日覆盖低于90%。抽12日后，24区平均同时日覆盖都超过90%（最差TEC 91.78%），宏覆盖94.32%、宽222.37、Winkler225.82；Winkler改善显著，宽度差CI仍跨零。

这在EIA小时燃料生成代理上复现了“本地时间样本只校准尾部风险”的信息价值，不代表独立排放真值，而且与项目V215–V218同一数据家族。12日方案需目标区月后提供12×24×8燃料小时曲线，当前中国公开月报不含此项；这不是可部署方法。通用算法新颖性也不足：design-based conformal和2026随机稀疏轨迹conformal已有直接关联。当前可能的论文贡献是**月度边际下水平识别与小时风险识别的信息预算前沿**，要站住必须有独立运行数据和现实中国数据采集路径。r2由于微秒/纳秒日期键错误被作废，所有结论只引用r3。[V224协议](V224_EIA_SPARSE_DAILY_MAXRISK_PROTOCOL.md)、[结果](V224_EIA_SPARSE_DAILY_MAXRISK_RESULTS.md)。

## V225 粗字段的物理可行界：四条小时通道接近完整八类区间

在EIA V224开发区上，把目标日每小时只观测到的聚合字段与目标月八类边际组成一个燃料可行集。对每小时，在总量约束下按月边际推每类燃料可行上下界；在已知组总量下对线性排放因子做精确最小/最大分配。一个抽样日的分数取24小时中最坏的可行CIF偏差上界，再对随机抽到的日做90% conformal秩校准。该上界支配真实日最大误差，因此半径覆盖真日内最大误差的随机日边际事件；不保证固定样本或全月同时覆盖。

12日预算下，四条逐小时通道（总发电量、火电合计、煤电、气电）区间宽223.34、Winkler226.64、宏整日覆盖94.52%；完整八类本地曲线是221.74、225.22和94.32%。配对地区CI显示四通道宽度多1.60 [0.02,3.72]，Winkler多1.41 [0.01,3.34]，覆盖高0.20pp [0.004,0.52]。24区每区平均整日覆盖均超过90%。三通道（总量、火电、煤电）仍每区超过90%，但Winkler较完整信息多37.22 [11.95,68.36]。仅总量＋火电也覆盖，但区间明显更宽；仅总量或总量＋煤电过于保守。

这是截至目前最有潜力的破局：**月度燃料守恒边际＋少量本地聚合小时观测，可形成物理可行的部分信息风险分数；小时总量、火电、煤、气四通道几乎保留完整八燃料观测的区间效率。**不过这不是仅凭月报就可用，仍需12日×24小时×4项本地数据。一般的部分标签/稳健上界 conformal 已有文献，所以创新不能泛化成新的conformal理论；可能贡献是电力CIF的守恒可行集、小时风险目标与字段预算前沿。[部分标签 conformal](https://proceedings.mlr.press/v204/javanmardi23a.html)、[稳健上界校准 conformal](https://papers.neurips.cc/paper_files/paper/2024/file/0814a342597d65e0832fc7ec9b42c317-Paper-Conference.pdf)。详见[V225](V225_PARTIAL_SNAPSHOT_SHARP_RISK_RESULTS.md)。

## V226 冻结的四字段方案在未参与开发的区域与时间窗通过

在预先冻结四字段、12日预算、随机抽样、整日最大误差分数及门槛后，用2026年7–9月的新EIA文件确认。排除V224/V225的24个开发BA，按数据完整性筛出26个其他BA、74个区月；不使用目标BA的历史日曲线；数据完整性、源月供体可用性以外不按标签筛样。四字段方法宏小时覆盖99.09%、随机整日24小时同时覆盖93.82%、宽272.49、Winkler275.59；精确八类本地对照宽263.05、Winkler266.32、日覆盖93.32%；源区留BA校准是宽299.49、Winkler342.89、日覆盖82.72%。四字段Winkler相对源区改善67.30 [BA bootstrap 95% CI 18.48,117.00]；区间宽差为−27.00，CI跨零。对精确八类本地对照，宽与Winkler分别只高3.6%和3.5%，通过冻结的5%门槛；26个BA平均整日覆盖全超过90%，最差CPLE 91.18%。

这构成真实增量：**风险覆盖改善没有靠整体放宽换取**，四字段方法相对源区基线平均宽度更窄、Winkler显著更低，同时与完整八类本地观测几乎等效。不过部分区仍很宽（PNM宽665、PACE宽561），而目标侧数据需求依旧是12×24×4小时观测；EIA固定EF生成代理也不是实测排放。部分BA在旧TransCIF项目实验中出现过，所以这里是相对V224/V225新方法和2026H2新时间窗的外层确认，不宣称这些BA从未被任何项目分析接触。中国现有月报本身仍不能直接运行。协议和复现记录见[V226冻结协议](V226_FOUR_FIELD_PARTIAL_RISK_FROZEN_PROTOCOL.md)、[确认结果](V226_FOUR_FIELD_PARTIAL_RISK_CONFIRMATION_RESULTS.md)、[runner](../scripts/v226_frozen_four_field_confirmation.py)。

## V227 NESO GB 外部时间复核：覆盖复制，效率收益未复制

为检验V226是否只适用于EIA燃料生成文件，在NESO 2009年以来的公开半小时GB国内发电数据上冻结协议，取2023全年及2024年1–3月15个目标月；12个本地随机日校准、其余日评分。响应是由NESO燃料发电结构和同一固定EF算出的国内发电侧代理，不是NESO消费侧强度，也不是独立排放真值。小时整日覆盖从历史源校准的88.04%升到四字段方案的92.34%，但平均宽度增加17.40（6.4%），Winkler增加15.82，故覆盖改善通过扩宽实现，冻结的效率门槛失败。相对完整八类本地观测，四字段效率完全一致。

这组数据的排放因子只有煤、400因子组和零因子组；给定小时火电合计与煤量，剩余火电排放已知，气通道可省。因此“字段预算”应按因子可区分结构自适应压缩，不能把四个具名字段当成跨系统固定方案。V227削弱了“V226区间效率可跨数据源复制”的主张；它支持机制的代数可移植性，尚不支持真实CIF或中国小时预测。详见[V227冻结协议](V227_NESO_GB_EXTERNAL_TEMPORAL_PROTOCOL.md)、[结果](V227_NESO_GB_PARTIAL_RISK_EXTERNAL_RESULTS.md)。

## V228 因子行空间驱动的最小小时字段：精确重构且压缩局地通道

根据V227识别出的因子结构，冻结规则：只在月度边际为正的燃料支持上，寻找能使因子向量落入测量组字段行空间的最少聚合通道；总发电量保留为CIF分母。对V226已打开的26 BA、74区月做方法特定预登记重分析，故不是新的盲测。EIA的固定EF下，典型恒等式为`排放=400×thermal+600×coal+400×oil`；油为零时可省油通道。

57/74区月（77.0%）少于固定四字段：22个仅需总量+1个分组，35个需总量+2个，17个使用总量+3个。重构小时CIF与完整八类结果的最大误差<1e−8 g/kWh，且整日分数逐项相同。宏小时/整日覆盖99.03%/93.49%、宽263.48、Winkler266.61，等于full-8本地校准。相对固定`total+thermal+coal+gas`，宽少9.36、W少9.22，覆盖低0.48pp但每BA均高于90%；相对源区方法，日覆盖高10.72pp、W少75.72，宽度差区间跨零。结果说明字段名称和数量不应预设，应该按目标因子与月度活跃支持做测量设计。

这使**CIF固定因子可识别性的最小观测合同**成为较具体的原创候选，而不是conformal校准创新。行空间测量/最优传感本身属于成熟线性逆问题思想，因此论文价值须来自燃料活跃支持、小时CIF风险以及信息预算前沿的结合。该实验仍需月后边际和小时本地样本，固定EF代理而非真实排放，不能解决用户月度-only中国部署问题。[V228协议](V228_FACTOR_AWARE_MINIMUM_CHANNEL_PROTOCOL.md)、[结果](V228_FACTOR_AWARE_MINIMUM_CHANNEL_RESULTS.md)。

## V230 EF 压力的逐小时 LP 锐界：V229 保守界明显过宽

针对 V229 ±10% EF 下过宽的三角外界，在同一 26 BA、74 BA-month 的已打开 EIA 队列，对每小时解受月度燃料边际、小时总量和观测组字段约束的线性规划。零压力时因子行空间字段精确重构，最大误差 `4.55e-13 g/kWh`。±10% EF 压力下，因子行空间方案 Winkler 为344.82、区间宽344.35、日内同时覆盖99.13%；完整八类分别为344.82、344.35、99.13%；固定四字段为356.64、356.21、99.17%。相对V229保守界，行空间方案Winkler约从520收至345，证明保守月度上限明显夸大了EF不确定性损失。行空间方案对固定四字段的宽度/Winkler优势约11.9，BA bootstrap区间未跨零；对完整八类的差异为数值量级，整日鲁棒集合包含率仅低0.023个百分点且CI触零，故应表述为近似等效，不能称优于全信息。

此处±10%是压力情景。LP只对逐小时、月度边际诱导的可行多面体给锐界，未联立全月小时约束；分数校准仍用12个本地完整日。数据仍为同源固定EF EIA排放代理，且该批次已在开发分析中打开。V230把候选机制从“名义EF精确可识别”推进到“统一相对EF压力下仍近似复现完整八类集合”，但不能解决月度-only条件下的小时风险：还需量化没有任何目标小时读数时的可行集合和稀疏本地通道的信息增益，并取得独立标签。[V230协议](V230_SHARP_FACTOR_UNCERTAINTY_LP_PROTOCOL.md)、[结果](V230_SHARP_FACTOR_UNCERTAINTY_LP_RESULTS.md)。

## V231 月度-only 小时可识别性：类别齐全仍不足以确定小时混合

根据用户确认的中国月表可覆盖全部发电类别，先问月度燃料边际本身对小时CIF约束到什么程度。为偏向月报方案，额外给定小时总发电量；每小时只受总量、非负和当月燃料边际约束。26个BA、74区月里，逐小时可行CIF集合的平均宽度737.46 g/kWh，中位宽度达到1000；所有名义代理标签均在集合内。若再给目标小时V228最小因子行空间通道，集合收缩到点，重构误差`4.55e-13`；每小时本地数值字段含总量共2–4项，完整八类也是点识别。

这是**信息不足的构造性证据**，不是预测误差比较；本地小时总量对月度-only用户本身都不可得，因此737.46只是偏乐观代理基准，不能作为中国误差量级。已观测所有小时的最小通道产生代数精确识别，也不等于只给月报就能预测；少数本地日校准未观测日风险的证据来自V224–V226。由此论文命题可收敛为“月度燃料边际锚定水平＋因子行空间最小观测识别小时CIF＋稀疏本地日校准尾部风险”，但尚缺中国独立小时真值与可部署小时字段，方法本身也依赖成熟线性可识别性/部分信息集合思想。[V231协议](V231_MONTHLY_ONLY_HOURLY_IDENTIFIABILITY_PROTOCOL.md)、[结果](V231_MONTHLY_ONLY_HOURLY_IDENTIFIABILITY_RESULTS.md)。

### V232 解析加固：逐小时界已是聚合守恒约束下的全月锐界

原计划再做全月联立 LP；在没有结果文件生成前发现，无需重复求解：满足当前小时总量且不超过燃料月边际的燃料向量，其余各燃料月供给与其余小时需求总量相等，因此一定存在非负运输矩阵完成剩余月份分配。V231逐小时可行集与全月联立约束在该小时的投影相同；单小时界不是因忽略跨小时守恒而偏宽。装机/可用率上限、爬坡、调度、储能、进口和天气出力等额外约束才可能进一步收紧。重复LP在该等价得到证明后停止，没有生成或声称V232求解结果。[证明与条件](V232_TRANSPORT_POLYTOPE_MARGIN_IDENTITY.md)。

## V233 Carbon Monitor China日标签生成审计：省份留出不含独立月内形状

对2026-10-29本机快照的`Power`日CO₂序列（2019-01至2026-06，29省、79,402行）按省月归一化日形状。90个年-月块里，省际形状相关系数均值为0.999999999，最低0.999999954；平均归一化两两L1约1.4×10⁻⁶；“省月水平×共同月内日曲线”的相对RMSE约1.3×10⁻⁶。2026-10-30从官方CSV API复下载后SHA-256与本机快照一致（`723709ff6cc610e121d502f0c0aa9adb55aa6531a616cb145109de4e12b0ae52`），复跑r2结果重现。内蒙古和黑龙江在该快照没有Power行。CarbonMonitor-Power原始方法论文已说明中国日序列由月度各源发电量乘以沿海八省日耗煤月内比例生成，因此这个共享日形状是已知构造后果，V233是当前版本的量化复核，不是首次发现。[方法论文](https://www.nature.com/articles/s41597-023-02094-2)。当前版本不能作为跨省独立日形状迁移的目标真值；也不能据此复核其他论文所用历史版本的目标覆盖。

这是**对已发表构造机制的版本化数据审计**，不是算法增量或首次发现。官方说明数据可修订；方法论文说明中国日发电序列由月量按日耗煤曲线拆分。审计仅限当前文件，不能外推到历史vintage或别的修订版本。潜在论文价值在于把标签生成过程、地区覆盖、目标边界与独立小时信息纳入CIF基准有效性门槛，并在真正独立高频数据上做留省风险评估。[V233报告](V233_CARBONMONITOR_CHINA_DAILY_SHAPE_AUDIT_RESULTS.md)、[Carbon Monitor官方数据页](https://cn.carbonmonitor.org/)、[CarbonMonitor-Power方法论文](https://www.nature.com/articles/s41597-023-02094-2)。

## V234 月度-only 小时 CIF 锐识别集：活跃 EF 的凸包

在只有非负小时燃料分配和各类别月总量、没有小时总量或出力约束时，任一指定小时的 CIF 精确识别集等于当月所有正月量类别 EF 的 `[min,max]`。证明为：小时 CIF 是 EF 的凸组合；区间内任一点可用足够小的目标小时电量混合端点类别达到，剩余各类月电量可放入其余时段。月报仍精确给出能量加权月锚，但单小时集合宽度不会因同一活跃支持内增加更多月度细项而自动缩小。V234是解析信息边界，不是预测实验或算法增量；全月联合尾部风险需另做受约束优化，不能把边际区间独立拼接。[V234推导](V234_MONTHLY_ONLY_HOURLY_CIF_EXACT_IDENTIFIED_SET.md)。

## V235 中国逐小时负荷代理：单小时 CIF 锐界缩减为零

新找到并下载覆盖31省2015–2024的公开逐小时负荷重建数据。将其与Ember八源月发电边际结合，并把省月负荷形状按月发电量归一后，在2018–2024的2,604个省月中逐一计算月度燃料边际＋小时总量的单小时精确CIF界；另用每日负荷均分为小时作对照。两条件全部小时均保留月报凸包的完整宽度，平均宽961.29 g/kWh、平均缩减0%。原因是单小时总量至多占月总量0.286%，活跃极端EF类别的月量足以独立供给该小时。数据集是天气和有限NDRC曲线重建的需求侧估计，不是发电实测；本轮是信息集合情景，不是预测或标签验证。[V235结果](V235_CHINA_HOURLY_LOAD_PROXY_INFORMATION_RESULTS.md)。

该结果否定“只补小时负荷总量就能识别小时燃料混合”，但不排除负荷形状收窄**联合尾部频率**的可能。下一实验只测试这一联合风险量，不回头优化点态模型。

## V236 负荷代理与全月联合风险：超阈小时数集合明显收窄

在六个燃料结构差异省份、2024年1月/7月和500/900 g/kWh阈值上，给月度八源边际增加逐小时负荷曲线（归一为当月发电总量）后，用全月联立MILP计算超阈小时数`K`锐界。93/96端点最优、45/48省月阈值条件的双端点均证明。对完整条件，月报-only的平均集合宽743小时；500阈值下总量约束使其收窄到347–380小时，平均缩窄46.6%–48.9%；900阈值缩窄约41.0%–43.3%。云南7月900阈值从0–743小时缩到0–71小时。这表明小时总量曲线可能通过全月守恒显著约束**联合尾风险**，尽管V235证明它不能缩窄任一单小时CIF区间。

限制：只有六省两个代表月；发电总量是天气重建负荷代理，EF/Ember边际也非真实标签，因此是识别集情景，不是预测增益/覆盖率。另一个需要纠正的设计点：r3里的“日平坦”是把每日负荷平均成逐小时总量，和完整小时曲线不构成嵌套信息；完整小时与日平坦的计数集合宽度不能解释为小时细节增量。下一轮改为真正的三层约束：月边际、仅每日总量、逐小时总量，并验证嵌套集合与小时信息的增量收窄。[V236结果](V236_JOINT_TAIL_COUNT_LOAD_PROXY_RESULTS.md)。

## V237 嵌套信息前沿：小时总量对尾数界贡献远大于日总量

把V236对照改为真正嵌套约束：月燃料边际→每日总量但日内小时自由→逐小时总量。六省、2024年冬夏、两个阈值共24 query，96端点中93个最优，22个query的日/小时两层双端点均证明，嵌套集合检查通过。500阈值下月报only宽743小时，日度only平均728小时（收窄2.0%），逐小时总量396.7小时（收窄46.6%）；900阈值下日度only收窄1.7%，逐小时层收窄41.0%。22个完整配对中，小时层比日度层平均再窄315.6小时。候选机制因而从“粗总量有用”具体化为“小时总量的时间分布对整月尾频率识别关键”。[V237结果](V237_NESTED_DAILY_HOURLY_RISK_INFORMATION_RESULTS.md)。

但小时总量仍由中国天气驱动需求模型重建，且被当作发电量使用；没有真实CIF标签或覆盖率。六省两月也不足全国确认。因此这是当前**最强的可检验方法突破候选**，还不是已证实的中国预测算法突破。下一步将使用真实EIA/GB逐小时发电总量测试几何机制，再评估需要的中国本地生成侧时序字段与负荷—发电误差敏感性。

## V238 EIA真实小时发电总量复制：阈值500下六个非平凡BA的风险界进一步收窄

用EIA实际逐小时分燃料发电量聚合出小时总量，以月燃料边际、日总量和小时总量形成嵌套识别集；固定EF标签仅作兼容性见证。8个完整BA-month、500/900阈值共64端点，56个最优（87.5%），所有实际派生尾数落在求得的可行界中。500阈值下八地均完整证明；排除两个结构性零尾数地区，六个非平凡地区从日总量到小时总量的尾数界平均再缩窄140.3小时（736.5小时平均日界的19.1%），范围57–242小时。仅月→日的增量通常2–11小时。900阈值配对最优仅2/8且均为平凡零风险支持，因此不作增益结论。

这比V236–V237的负荷重建代理更强地支持信息几何机制，但只验证EIA燃料结构乘固定EF形成的代理，既无独立CO₂标签，也无预测覆盖验证；且仅2026年8月、已查看地区。方法工具是标准MILP，单独不足以创新。可能贡献限于CIF特定目标下的月→日→小时识别/信息价值基准与系统验证。下一步扩至多季节多年份，并检验EF类别/数值不确定性，再决定基准论文还是算法论文。[V238协议](V238_EIA_NESTED_RISK_INFORMATION_CONFIRMATION_PROTOCOL.md)、[V238报告](V238_EIA_NESTED_RISK_INFORMATION_CONFIRMATION_RESULTS.md)。

## V239 2025跨季节复制：16/16完整案例小时总量使尾数界严格变窄

在2025年4、7、10月冻结16个完整BA-month，统一500阈值，64/64端点最优、三层嵌套均通过。小时总量相较日总量平均缩窄165.6小时（726.4小时日界的22.8%），16/16均变窄，范围21–381小时；月边际到日总量只平均缩窄7.6小时。4、7、10月平均额外收窄依次200.7、137.4、151.8小时，方向跨季节一致。CPLE和ERCO最大（约307、262小时），PACE最小（36小时），存在明显地区异质性。

这从单月/重建负荷代理推进为真实EIA小时总发电量的跨季节结构复制，当前支持“小时总量是全月CIF尾数识别的关键信息层”这一科学假设。它仍不是CIF概率预测、覆盖保证或算法创新：固定EF标签是代理，MILP是标准工具，且没有调度容量/爬坡约束和中国实测小时生成数据。研究价值开始更像**字段价值/识别基准论文**，不是已成熟的预测算法稿。下一步测调度约束、燃料/EF不确定性对增量的影响，并验证中国公开可得小时总量字段是否真能用。[V239协议](V239_EIA_SEASONAL_NESTED_RISK_PROTOCOL.md)、[V239结果](V239_EIA_SEASONAL_NESTED_RISK_RESULTS.md)。

## V240 EF不确定性压力测试：发现阈值侧不识别关口

沿用V239的16个BA-month，允许三个正EF组各在名义值±10%内独立变化，并为保守包络允许EF逐小时变化。64/64端点最优；相较日总量，小时总量仍在12/16案例缩窄尾数界，平均126.8小时（17.4%）；名义EF时为165.6小时。收窄消失的四例恰为PACE 4/7/10月与AZPS 10月。解析检查显示这四例的月均EF不确定区间跨过500阈值；其余12例不跨且均有额外收窄。

得到一条可证明的**阈值侧不识别判据**：在已知任何正小时总量曲线和月燃料边际时，比例配置`x[h,f]=G[h]M[f]/M`始终可行，且每小时CIF恒等于EF加权月均值。若允许的EF集合中既有月均值低于阈值、也有月均值高于等于阈值的参数，则可行集同时包含`K=0`和`K=H`，所以超阈小时数的尖锐最小—最大区间必为`[0,H]`；小时总量无法收窄。外包络压力测试把该解析判据在当前16例上完全复现：4/4跨阈值例界为全月，12/12未跨阈值例均收窄。这指向最低本地信息需求：先把有效EF/月均锚定在目标阈值一侧，再讨论小时总量对尾部风险的价值。

这目前是最清楚、可分析证明的研究增量，但还不是全新优化算法：它结合简单比例可行构造和CIF业务阈值，可能支持一篇“最低信息预算＋锐识别边界”的方法/基准稿。±10%只是压力盒，不是EF真实测量区间；小时变化外包络可能过宽，尚需中国EF依据、共享EF和源类别不确定性验证。参见[V240协议](V240_EF_UNCERTAINTY_TEMPORAL_RISK_PROTOCOL.md)、[V240结果与证明](V240_EF_UNCERTAINTY_TEMPORAL_RISK_RESULTS.md)、[逐案例证书](../results/v240-ef-uncertainty-temporal-risk-r1/threshold_side_certificate.csv)。

V234–V240共同形成三层目标量信息前沿：月平均CIF由月边际确定；单小时CIF在活跃EF凸包内仍不识别；整月尾数可受小时总量约束，但阈值侧EF不确定性可使尾界恢复全宽。当前论文候选与投稿门槛综合见[V241](V241_CIF_INFORMATION_FRONTIER_SYNTHESIS.md)。

## V242/V243 阈值邻域与MILP尺度检查：400稳健，600部分证明，归一化无明显帮助

在同一16个2025跨季节BA-month，名义EF阈值400下63/64端点最优、15/16配对完整，15/15配对小时总量均收窄（平均107.8小时）；600下绝对MWh版本53/64最优、8/16完整配对，所有已证明配对收窄（159.9小时），但仅半数样本可用。把所有能量量纲改成月总量份额后，600端点最优率仅增1个（54/64），配对仍为8/16；两版本所有双方已证明的53个端点完全一致。因此数值尺度不构成创新或充分求解改进，需另行探索MILP强化界；600结论仍是初步。[V242阈值报告](V242_THRESHOLD_FRONTIER_RESULTS.md)、[V243协议](V243_SCALED_MILP_THRESHOLD600_PROTOCOL.md)。

## V244 解析筛查剪枝：全宽案例跳过MILP且逐项精确相同

把V240阈值侧命题变成求解前筛查：月均EF盒跨目标阈值的四个BA-month由比例构造解析给出`[0,H]`；其余12例照常跑MILP。64个最终端点与完整MILP的V240逐项完全一致，全部实际固定EF见证包含，16配对全完成；解析分支跳过16/64（25%）个MILP端点，余48端点平均1.76秒，总MILP耗时84.5秒，数据加载到结果输出112.2秒。完整V240未记同精度wall time，故不声称相对加速百分比。该剪枝是判据驱动的安全预处理，尚非独立新优化理论，但让“何时小时数据没有信息价值”成为方法流程的一部分。[V244协议](V244_THRESHOLD_SIDE_SCREENED_SOLVER_PROTOCOL.md)、[V244结果](V244_THRESHOLD_SIDE_SCREENED_SOLVER_RESULTS.md)。

## V245 CEMS 实测因子审计：差异主要来自机组群转换效率

为给V240的±10% EF压力盒补上经验校验，下载并校验2025 EPA CEMS季度档案（Zenodo PUDL归档），仅保留measurement indicator为`Measured`/`MEASURE`、正运行时长/毛负荷/CO₂质量的小时。得到10,589,566个有效机组小时和31,681机组-月-燃料记录。直接以CEMS小时 CO₂ rate / gross load rate计算小时强度，月均按运行时间加权。煤电小时相对本机组本月均值偏离>10%的发电量约5.0–5.5%；管道气约2.8–4.3%。但相对全体同燃料季度均值，煤约28–32%、管道气约56–68%的毛发电量偏离>10%；单列Natural Gas约58–66%。用机组月均值相对燃料组季度均值拆分后，煤约25–30%、管道气约51–68%已是机组间差异。固定燃料标签的统一EF压力盒因而不能代表未见地区的机组构成不确定性。

科学上它把“月度全燃料种类发电量仍不够”具体拆成活动量/组合层与机组燃料转换效率层；它不是平均电网CIF，也没有预测/留区结果。用户所说中国月报全发电类别能支持份额/活动量信息，但不自动提供燃料消耗、热耗、机组效率或同口径排放。V245因此是当前最有实证基础的研究转向，而不是最终算法突破。后续应在真实CEMS的多BA留区中比较：统一燃料EF、源区EF先验、少量目标粗效率信息/实测校准，小时尾数覆盖与区间宽度；还需先配平CEMS化石机组毛负荷与BA全电网净发电分母。完整报告见[V245](V245_CEMS_EMPIRICAL_EF_UNCERTAINTY_RESULTS.md)。

## V246/V247：月度热输入增益通过跨年供体确认，但煤电异质性仍是硬反例

V246同年空间BA留出里，加入目标BA月度燃料热输入后，跨区排放迁移相较只用月度燃料发电量大幅提升。V247进一步冻结2024供体→2025目标、目标BA从供体剔除：49个目标BA、586 BA-month、99.765%筛选后发电量有供体支持。BA-month中位APE 5.42%→0.316%，全队列WAPE 4.78%→1.18%；Pipeline Gas WAPE 6.43%→0.16%，煤4.41%→2.31%，自然气12.00%→0.89%。所以热输入价值不是同年泄漏才产生。

但DUK/WALC煤电是重要失败点：热输入方案煤因子误差约32%/34%，目标实测煤强度远低于供体；油类整体也会变差。因而“源组合×热输入转换层”只是强机制候选，仍需目标本地煤质/热值或极少量排放测量识别剩余燃料因子。且目标端热输入是当月聚合字段，任务更接近新区域月度排放估计/小时调度 nowcast，不是月初即知的未来预测；CEMS样本仍不是含非化石发电的全网CIF。

**最新判断：已经找到清晰且跨年的信息增量，尚未找到足够新颖、覆盖可靠的小时算法突破。**下一轮重点做直接小时风险集合：用月度本地热输入锚定总量/转换层，测试能否降低小时区间而维持90%整日覆盖；煤电差异大的DUK/WALC作为冻结压力区，PACE/LGEE问题继续保留作跨信息机制压力测试。完整记录见[V246](V246_CEMS_HEAT_INPUT_TRANSFER_RESULTS.md)、[V247协议](V247_TEMPORAL_CEMS_TRANSFER_PROTOCOL.md)、[V247结果](V247_TEMPORAL_CEMS_TRANSFER_RESULTS.md)。

## V248：一日本地实测因子校准显著降低目标月排放误差

在2024同月外区供体CO₂/热输入因子→2025目标BA的跨年设计上，再用目标燃料的随机本地实测日CO₂/热输入比校准因子，目标月本地总热输入为规模锚。50次随机重复，586配对BA-month。0日先验WAPE 1.178%；1个本地日降至0.239%（SD0.027），3日0.129%，7日0.079%，12日0.055%。煤电WAPE 2.31%→0.39%（1日），目标压力区DUK平均月APE 17.41%→4.51%，WALC 13.54%→1.84%。油类也改善，但其抽样曲线不单调。

这是目标燃料转换率局地修正的强证据：问题不是小时形状模型再复杂一点，而是外区因子和目标机组群存在局地差异，一个本地日的直接 CO₂/heat-input比例就能消除大部分月度水平偏差。不过这是一项**月度排放估计/回顾性锚定**实验，使用了目标整月heat-input总量，抽样日直接用于估计同月平均因子；它既没有评小时标签覆盖，也没有提供未来月初预测。它不能称小时CIF突破。

V249仅做下一项关键验证：在对齐时间口径的单时区BA，冻结12个小时观测日，使用直接CEMS小时CIF标签，测量heat-input/local-factor机制是否在维持90%整日覆盖的同时缩窄区间，并保留DUK/WALC煤电压力案例。若小时覆盖/效率无增量，则本方向应收敛为月度强度预算/数据价值论文，不强行包装小时预测算法。详情见[V248协议](V248_SPARSE_LOCAL_FACTOR_CALIBRATION_PROTOCOL.md)、[结果](V248_SPARSE_LOCAL_FACTOR_CALIBRATION_RESULTS.md)。

## V246：月度热输入直接改进目标区燃料转换率迁移

在2025 CEMS标记实测CO₂、实测热输入、正运行/毛负荷及非混合主燃料类别下，按EIA-860映射49个BA。做目标BA完全留出的空间迁移，按同日历月、同燃料从其他BA估系数；成对比较“目标月燃料发电量×供体CO₂/MWh”与“目标月燃料热输入×供体CO₂/mmBtu”。共586个BA-month，供体覆盖目标样本99.75%发电量。增加目标本地月热输入后，BA-month等权中位绝对百分比误差从5.57%降至0.38%，平均从7.35%降至1.22%；全队列排放WAPE从4.91%降至1.11%。煤和天然气类都有同方向改善，油类较差。

这是现阶段最有价值的方向候选：将源组合与燃料转换效率分开，利用一个按燃料聚合的本地月热输入总量吸收目标区机组效率异质性，再迁移跨区CO₂/热输入因子。方法的乘法核算身份本身是标准物理关系，V246是同年同月空间留区、CEMS毛发电口径而非全网CIF，因此还不是原创定理或完成的论文突破。现在进入V247：2024同月其他BA供体→2025目标BA的时间外推；只有未来时间仍有明显增益后，才进入逐小时排放覆盖与区间效率实验。详情见[V246](V246_CEMS_HEAT_INPUT_TRANSFER_RESULTS.md)。

## V249–V251 CEMS小时区间：全供体复算后的主结果

执行审计发现，V249 runner r1把供体BA误限在目标BA名单内，导致V250/V251部分燃料/月份因无名单内供体而跳过。r1结果仅保留作诊断，修正经过见[供体池更正记录](V249_V251_DONOR_POOL_CORRECTION.md)。r2对2024 EIA-860映射的68个BA聚合供体，并逐目标排除目标BA；目标名单、年份、中心、随机种子和区间校准预算均不变。因为r1标签已查看，r2是口径修正后的复算，不称盲测。

V249开发/压力队列（BANC、CPLE、DUK、ERCO、NYIS、SCEG、SRP）84 BA-month中，按BA等权，发电量锚→热输入锚→一日本地因子宽度为230.25→196.35→176.92 g/kWh，整日24小时覆盖92.59%→92.52%→92.52%。热输入相对发电量窄13.34%（BA bootstrap 95% CI −19.54%至−5.23%）；一日本地因子相对热输入再窄6.53%（CI −16.83%至−0.13%），但此队列含既有压力区DUK。局地因子改善主要在DUK和ERCO，不能据此估计未见地区的普遍收益。[V249修正结果](V249_CEMS_HOURLY_INTERVAL_TRANSFER_RESULTS.md)。

V250、V251是两批未进入V249的目标地区。严格标签筛选后，V250有4个可评分BA，V251有6个，合计10个BA、120 BA-month；未能评分的预登记区保留为缺测，不补换。两批队列合并按BA等权后，发电量锚、热输入锚、一日本地因子的区间宽度分别327.99、251.75、250.56 g/kWh，24小时整日覆盖92.50%、92.54%、92.55%。热输入锚比发电量锚窄20.26%（BA bootstrap 95% CI −29.69%至−10.81%），Winkler低20.02%，9/10区间宽变窄，覆盖差仅+0.04个百分点（CI −0.17至+0.25）。这是当前最强的地区外实测结果：月热输入信息在这批新BA上提高小时区间效率，未显示整日覆盖下降。

一日本地因子相较热输入锚只窄0.61%（CI −1.86%至+0.37%），Winkler低0.60%（CI −1.83%至+0.36%），仅6/10区间宽变窄。它没有在两批新地区中形成可确认的普遍增益。V249局地收益集中在DUK等开发/压力点，不能称为已迁移的原创算法。

## 论文立意更新

当前最可信的方向是CIF预测的**信息预算与可识别性基准**：比较逐类月发电量、目标月热输入、小时总量、极少量目标排放观测对月水平与小时尾风险覆盖/区间宽度的贡献，明确每类字段何时有用、何时不足。热输入字段在中国用户确认的月报中尚未证实存在，故美国CEMS增益是信息价值研究，不是中国可部署性证明；这些预测用目标完整月边际和月内校准日，属于月后重建，不是月初预测。标签是筛选化石机组毛负荷强度，也不是全网CIF。

方法新意门槛很高。少样本 conformal、迁移校准与分布异质性已有成熟研究；2026年ICML已有依可观测偏差诊断修正跨域conformal校准的工作；CEMS研究也已揭示负荷率导致机组效率和排放强度变化。故“加门控”“加分层收缩”“第一次把一天用于本地因子”都不足以单独支撑算法创新。[Few-shot conformal](https://proceedings.mlr.press/v139/fisch21a.html)、[covariate-shift calibration](https://proceedings.mlr.press/v108/park20b.html)、[2026 conformal calibration transfer](https://proceedings.mlr.press/v306/doula26a.html)、[CEMS机组运行强度研究](https://www.nature.com/articles/s41467-025-59800-4)。中国江苏也已有基于碳流和GNN-LSTM的高时空分辨率CEF预测研究，因此稿件需清楚区分“富信息条件下的区域预测”和“无高频标签的新区预算受限迁移”。[江苏省短期CEF预测研究](https://www.sciencedirect.com/science/article/pii/S0959652625023753)。

下一步不再围绕单日本地因子堆模型。先在同一全供体、直接实测标签口径下测试信息预算梯度：月发电量-only、月热输入、目标月少量本地日因子、目标小时/日观测；保持校准日预算和覆盖门槛一致。再检验热输入信息增益是否可被更廉价的公开字段代理。只有发现不依赖目标CEMS、在真正新地区仍保持覆盖且显著缩区间的机制，才重新打开算法稿路线。EPA小时率需乘运行时长，且数据可能处于监测位置级并经分摊，相关口径限制不变。[EPA排放数据指南](https://www.epa.gov/system/files/documents/2022-07/CAMD%27s%20Power%20Sector%20Emissions%20Data%20Guide%20-%2007182022.pdf)。

## V252/V253：年报式热耗信息能否代替目标月热输入

V252用2024目标BA逐燃料年度热耗率（全年热输入/毛发电量）×2025目标月发电量×2024其他BA同月燃料CO₂/mmBtu，重分析V247的49区、586 BA-month。所有目标燃料都有年度热耗率支持，覆盖目标毛发电量100%。总排放WAPE从月发电量供体基线4.78%降至2.01%，BA-month均值/中位APE从7.29%/5.42%降至3.60%/1.75%；BA成簇bootstrap显示WAPE降低幅度95%区间36.8%–72.1%。这说明年度热耗率可能是在中国“逐类月发电量”之外的最小有价值字段候选。

但分燃料负迁移不能忽略：柴油WAPE从9.87%升到20.95%，煤、管道气、自然气则改善。目标年度直接CO₂/MWh更准（总WAPE1.45%），但新CIF区域没有这一目标历史因子时不属于可用信息；目标月热输入参照WAPE1.18%，则是月后信息上界。[V252结果](V252_ANNUAL_LOCAL_HEAT_RATE_TRANSFER_RESULTS.md)。

V253把年度热耗率中心带到相同10个目标BA的小时标签，四种中心共用12个本地完整日做风险校准。区间宽度按BA等权由发电量基线324.33降至年度热耗率259.87 g/kWh，下降16.20%（BA bootstrap 95% CI −26.88%至−5.67%），10区中7区变窄，整日覆盖92.27%→92.10%。目标月热输入参照进一步窄约5.74%，直接年度CO₂/MWh比年热耗率仅再窄约1.01%。因此年热耗率是相较一日本地CO₂样本更现实的“粗统计锚”，但V253仍给所有方法同样12个**目标本地小时风险校准日**，尚未解决真正无目标高频历史的冷启动。[V253协议](V253_ANNUAL_HEATRATE_HOURLY_INTERVAL_PROTOCOL.md)、[结果](V253_ANNUAL_HEATRATE_HOURLY_INTERVAL_RESULTS.md)。

**当前最值得验证的破局点：年度本地燃料热耗率＋全源区逐燃料因子供体能否在零目标小时校准下维持可靠覆盖。**下一轮用2024非目标BA日风险分布校准2025目标区的年度锚区间，不能使用目标2025小时标签或目标2025热输入来构造区间；目标小时标签只在最后评分。若覆盖明显低于90%，则年热耗率只改善月中心，不能支撑可信小时区间；若覆盖达标但区间显著更宽，要量化为实现冷启动可信度的真实信息成本。方法新颖性仍未解决，2026 ICML的跨域conformal校准工作已覆盖偏差诊断和校准迁移；投稿贡献应优先聚焦**有直接CEMS标签的信息预算基准与最低字段合同**，而非泛化的迁移/conformal新算法。[Conformal Calibration Transfer](https://proceedings.mlr.press/v306/doula26a.html)。

## V254/V255：零本地小时的外区风险迁移与相似度选源均未过关

年度热耗率中心下，V254用2024其他BA来源风险q90给2025已打开的10个目标BA区间，整日覆盖87.77%、宽233.51 g/kWh、Winkler 291.05；目标本地12日参考覆盖92.10%、宽259.87、Winkler268.07。来源区间更窄但欠覆盖，统一抬q会显著增宽且不能修复GVL等局部低覆盖。

V255按燃料份额与年度热耗率距离选择K=5/10/20/全部供体，在2024来源BA留一中没有任何K通过预设覆盖门槛；按预先规则回退至全部供体，故目标结果等同V254。它否定了当前简单相似度门控的有效性，不支持“加一个相似度权重”作为创新点。[V254](V254_ZERO_LOCAL_HOURLY_CALIBRATION_RESULTS.md)、[V255](V255_SIMILARITY_CONDITIONED_SOURCE_RISK_RESULTS.md)。

## V256：本地风险标签预算存在随机—顺序裂缝

固定年度热耗率中心，在10个已打开BA、120 BA-month，随机同月抽取校准日时，9日方案整日覆盖89.98%（4/10 BA低于90%）；12日为92.30%（10/10 BA经验覆盖高于90%）；18日为94.78%；24日为91.94%，但平均仅剩6.4个测试日，区间宽度的非单调来自有限样本次序统计量与测试集变化。12日结果重现V253本地校准基线。

但按时间顺序用月初连续日校准后续日，9/12/18/24日整日覆盖仅80.24/84.00/88.08/84.99%，都未达到90%。因此随机拆分里“约12天足够”的说法不能外推为真实在线预测预算；同月风险有顺序漂移。现阶段最清楚的结论是：公开粗统计可以改善区间中心，本地小时标签随机抽样可以校准同月风险，但既没有解决零目标小时冷启动，也未验证顺序上线场景。下一轮优先测试滚动校准，以区分“风险会被近期观测追踪”与“必须拥有同月随机标签”两种情形。[V256协议](V256_LOCAL_CALIBRATION_DAY_BUDGET_PROTOCOL.md)、[结果](V256_LOCAL_CALIBRATION_DAY_BUDGET_RESULTS.md)。

## V257：滚动风险更新跟踪了部分月内变化，但未解决地区条件欠覆盖

在V256相同预测日期配对下，最近n个已完成日滚动重算q优于只用月初固定窗口。n=18时整日覆盖从88.08%升至90.61%，Winkler从296.90降至269.24；平均区间宽度增加5.81 g/kWh，差异区间跨0。均值覆盖的BA bootstrap区间88.08%–92.92%，仍跨90%目标；10个目标区里4个低于90%，包括SEC 83.2%、GVL 85.2%。滚动9/12日仍整体欠覆盖，n=24滚动覆盖86.18%。因此滚动更新是有用强基线，但不是可靠新地区机制：至少在已打开队列里，它能追踪一部分时间变化，不能消除地区尾风险异质性，也没有展示可靠缩窄区间。[V257协议](V257_ROLLING_RISK_CALIBRATION_PROTOCOL.md)、[结果](V257_ROLLING_RISK_CALIBRATION_RESULTS.md)。

## V258：更强排放中心也未解除GVL的尾风险欠覆盖

固定18日滚动风险窗口和相同预测日期，替换年度热耗率中心为年度目标直接CO₂/MWh、再替换为目标月热输入oracle。总体覆盖与区间宽差别不大；目标月热输入oracle把等BA平均区间从260.48窄到250.85 g/kWh，Winkler改善10.89，但GVL覆盖仍只有84.5%（年度热耗/年度直接中心均85.2%），区间仍很宽；SEC从83.2%到87.0%，仍未达标。说明GVL欠覆盖主要不是中心偏差，关键剩余风险信息在粗统计中心之外。该oracle结果使用目标月实测热输入，不能作部署方法。[V258协议](V258_CENTER_VS_RISK_ABLATION_PROTOCOL.md)、[结果](V258_CENTER_VS_RISK_ABLATION_RESULTS.md)。

## V259：分开上、下尾后Winkler改善，覆盖门槛仍未通过

在最近24个本地日分数的月末滚动预测上，分别校准每日上尾和下尾，将整日覆盖从对称区间的86.18%点估计升至87.33%，平均宽度从226.89降至214.89 g/kWh，Winkler从250.20降至230.50；宽度差CI跨0，覆盖差CI亦跨0，只有Winkler改善区间不跨0。仍有6/10 BA欠90%覆盖，SEC只有76.4%。非对称区间更有效率，但未达可信覆盖要求；Bonferroni/单尾split分位数是标准办法，不构成新算法。[V259协议](V259_ASYMMETRIC_HOURLY_INTERVAL_PROTOCOL.md)、[结果](V259_ASYMMETRIC_HOURLY_INTERVAL_RESULTS.md)。

## V260：月利用率预测来源尾风险尺度，跨年区间校准仍失败


但已打开2025目标上的容量回归原始区间只有84.07%整日覆盖；按来源BA-month残差90分位修正则97.44%覆盖、区间宽420.21、Winkler427.54，效率较差。事后用70分位看似覆盖91.47%、宽260.8，但这是目标标签打开后扫描，只能提示校准迁移不匹配，不能当成确认性结果。此轮得到的是**月容量利用率与风险尺度之间的来源域信号**，不是已解决新区可信区间的突破；下一步只用来源域嵌套留BA CV选择残差修正，再对目标作已打开的跨年诊断。[V260协议](V260_CAPACITY_FACTOR_TAIL_RISK_PROTOCOL.md)、[结果](V260_CAPACITY_FACTOR_TAIL_RISK_RESULTS.md)。

## V261：来源域嵌套选80分位残差后覆盖达标，效率仍输简单基线

用2024来源BA的外层5折留区、外折训练BA内部交叉拟合风险残差，事前选择校准分位数。p=0.80是唯一通过来源CV双门槛且区间最窄的候选（来源整日覆盖90.33%，80.07% BA-month覆盖≥85%，平均宽578.2）。冻结后应用已打开2025目标：整日覆盖94.80%，宽314.13，Winkler336.80。它比V254简单全源q90参照更宽且Winkler更高（233.51/292.72），也弱于12日本地风险日参考（259.87/268.07）。所以容量利用率特征虽能改善来源q90尺度预测，却没有转化为跨年有效的覆盖—效率突破。[V261协议](V261_SOURCE_NESTED_RISK_CALIBRATION_PROTOCOL.md)、[结果](V261_SOURCE_NESTED_RISK_CALIBRATION_RESULTS.md)。

## V262：小时总量有5%–10%误差时仍收窄全月尾数锐界

围绕中国可得的天气重建小时负荷不等于实测发电总量这一限制，在V239已打开EIA代理样本上做有界测量误差压力实验。固定每日总量和月燃料边际，PACE三个季节月以及CPLE/ERCO/AZPS各一个对照月，给小时曲线增加±5%/±10%观测盒；ERCO两个压力端点未在20秒限时内证明，排除汇总。另5个已证明BA-month上，鲁棒尾数集合相对每日-only平均收窄83.8小时（5%盒）和78.6小时（10%盒）；相对同样本精确小时曲线只损失5.0/10.2小时。真实固定EF尾数全部被包含，已证明鲁棒集合都嵌套于每日-only集合。

这提示小时总量通道不要求完美测量：在该EIA固定EF可行集上，±10%边际误差尚未抹掉它对全月超阈频率的识别价值。但它只验证集合几何，不是小时CIF预测准确率或覆盖率；样本5/6、误差模式人为构造，不能据此将中国天气重建负荷视作实测发电。潜在贡献进一步具体化为**粗月燃料边际＋有误差小时总量的尾风险识别前沿**，新颖性仍须与鲁棒部分识别文献区分，并扩充样本/误差机制及真实小时CIF外测。[V262协议](V262_HOURLY_TOTAL_MEASUREMENT_ERROR_PROTOCOL.md)、[结果](V262_HOURLY_TOTAL_MEASUREMENT_ERROR_RESULTS.md)。

## V263：一小时相位包络保留小时总量风险识别增益

沿用V262五个两端可证明BA-month，对每日曲线施加一小时循环相位错位，并允许每小时真实总量在观测曲线前后相邻点的范围内变化。20/20个端点均证明；五例平均尾数集合宽度从每日-only的729.2小时降至642.8小时，精确小时总量是640.4小时。即使允许一小时局部时序不确定，信息收益平均只损失2.4小时；PACE三月仍仅收窄21–43小时，CPLE与AZPS的增益较大。

V263与V262形成“幅度误差＋局部时标误差”的初步鲁棒性证据，但均为EIA活动量与固定EF代理的已开标签可行集，不是预测覆盖验证。方法学候选可聚焦**可获得小时总量的误差容忍界与风险值**；在拓展完整16 BA-month、多小时偏移/平滑系统误差以及独立CIF响应之前，仍不称算法突破或中国可部署创新。[V263协议](V263_ONE_HOUR_PHASE_AMBIGUITY_PROTOCOL.md)、[结果](V263_ONE_HOUR_PHASE_AMBIGUITY_RESULTS.md)。

## V264：全cohort相位包络复制通过，完整锐界样本12/16

将V263的一小时错位＋相邻三点包络冻结规则扩展到V239全部16个BA-month。64个端点中60个达到最优证明；12个案例的相位与精确小时条件双端闭合，CPLE 4/10月和ERCO 4/10月未证明、排除汇总。12例中相位包络下尾数锐界平均宽616.3小时，对比每日-only 729.7小时，收窄113.3小时（15.5%）；精确小时曲线是613.3小时，相位不确定仅增加3.0小时。12/12真实固定EF尾数见证包含且风险集合嵌套成立。PACE三季仅缩窄21–43小时，体现风险信息价值明显异质。

这把候选增量推进为有跨BA/月份复制的**尾风险信息价值与有界时间误差前沿**，但没有证明一个新的预测算法，也没有测CIF真实概率覆盖。数据仍是EIA生成侧活动量＋固定EF代理，CIF非独立观测；仍需评估更现实的误差结构、因子异质和标签外部效用。若能拿到的只有中国各类月发电量，小时尾风险仍不能由这些边际识别；本轮提示新增小时总量曲线可能是最低有效时间通道，但中国负荷重建不能直接代替发电测量。[V264协议](V264_FULL_COHORT_PHASE_AMBIGUITY_PROTOCOL.md)、[结果](V264_FULL_COHORT_PHASE_AMBIGUITY_RESULTS.md)。

## V265：水平EF校准精度是小时尾风险信息的前置门槛

对V240的16个阈值侧解析证书做精度敏感性前沿。统一正EF相对误差盒达到5%、7.5%、10%、20%、40%、60%时，分别有1/16、3/16、4/16、5/16、9/16、11/16 BA-month跨越500阈值。最小临界误差为PACE 10月3.63%；另有AZPS 10月6.62%、PACE 7月7.44%、PACE 4月8.76%。因子盒跨阈值时，比例燃料分配构造证明小时总量曲线仍不能收窄尾数外包络的全宽范围。

由此形成比“多收小时点”更具体的最小本地信息顺序：先把燃料排放/转换因子不确定度压到阈值一侧，再用小时总量时间形状收紧联合尾风险。该门控在数学上清晰，也与V245观察到的机组/燃料EF异质性相关；但通用鲁棒部分识别已存在，不能声称公式或MILP新颖。下一步必须评估该门控是否能在独立系统减少无效测量、提升尾风险决策；现有证据仍是已打开EIA固定EF代理，不是预测覆盖或中国验证。[V265协议](V265_EF_PRECISION_FRONTIER_PROTOCOL.md)、[结果](V265_EF_PRECISION_FRONTIER_RESULTS.md)。

## V266：相位误差与幅度误差联合后仍保留大部分小时总量信息收益

把V264一小时相位包络和V262乘性测量误差联立，固定同一批12个相位参考案例。±5%与±10%误差下，分别有10/12案例双端证明，合计44/48端点最优；PSCO 4月和SOCO 10月有未证明端点，均排除于锐界均值。完整配对中，日度-only平均界宽730.9小时，相位-only为627.3小时，联合误差为633.1/637.8小时。相对日度-only仍缩窄97.8/93.1小时，即保留同一子集相位-only收窄幅度的94.4%/89.9%。已证明案例均包含其EIA固定因子代理尾数见证，并嵌套于日度-only集合。

联合误差并未消掉V264的部分识别信号，但只是既有结果的稳健性补充。仍然假设每日总量精确可用，小时代理曲线通过EIA燃料发电量构造；不是中国用户月报所能提供的输入，也不是预测覆盖。PACE的收窄仍只有20–41小时，显示价值对地区/月份有明显异质性。当前更合理的门槛是：证明因子支持和小时总量信息合同在目标地区可得后，再进行预注册的独立验证；本轮不构成新算法突破。[V266协议](V266_JOINT_PHASE_AND_MEASUREMENT_ERROR_PROTOCOL.md)、[结果](V266_JOINT_PHASE_MEASUREMENT_ERROR_RESULTS.md)。

## V267/V268/V269：EF阈值门控需要燃料组水平和区间形状

用2025 EPA CEMS实测状态标记样本，按州—季度聚合煤、气、油组因子；用覆盖至少90%组内州—季度毛发电量的区间/半径，代入V240相同16个EIA BA-month做跨系统压力测试。燃料组异质性不同：煤半径9.8–13.0%、气6.8–9.2%、油/石油焦14.8–27.6%，油类只有10州且发电量远小于煤/气。目标均值门控按**全源发电量**作分母，校正后的项目EF中心＋CEMS离散半径有4/16跨500（AZPS Oct、PACE Apr/Jul/Oct），CEMS因子中心＋同半径有2/16（PACE Jul/Oct），非对称最窄90%支持也是2/16（PACE Jul/Oct）；项目中心统一±10%则是另一套简化风险设定。燃料组水平、区间中心和区间宽度会改变门控结果，不能用一个通用“EF±r%”代替。

首次运行的V267–V269门控分母曾漏计非化石发电，旧r1门控结果作废；CEMS因子支持区间未受影响，正式结论使用校正的r2结果。

## V271：非对称EF支持集合首次与小时信息边际价值联立验证

在同一16个已开EIA BA-month上，把V269校正后的燃料特异、季度非对称EF支持区间直接送入尾计数MILP；各燃料EF按全月共享，稳健下界/上界分别在所有燃料EF区间的低端/高端求解。64个主端点全部证明；3个额外单调性诊断端点未证明，不纳入结果。月边际集合平均宽734.1小时；加入精确日总量为727.8小时；再加小时总量为589.3小时。相对月度信息小时总量平均收窄144.9小时，相对日总量收窄138.5小时（19.0%），14/16案例变窄。

关键匹配是：V269非对称EF月均集合恰在PACE 7、10月跨500；V271这两例小时尾数集合仍为全宽`[0,744]`，精确小时曲线没有任何收窄。其余14个不跨阈值的案例收窄8–362小时。这符合阈值侧非识别定理，并把此前“因子校准后再看时间信息”的候选路径推进为**目标因子支持门控 → 时间信息价值曲线**。所有下界仍为0、界宽仍普遍很大；这是固定因子响应代理上的部分识别，不是可信预测覆盖。

这目前是最有实证支撑的TransCIF创新候选：燃料组因子水平/非对称支持决定小时信息是否值得采集，再按本地数据预算评价可识别尾风险。它仍非已确认突破：EF区间由美国CEMS州—季度支持跨系统迁给EIA BA，非中国目标校准；没有未见地区确认、概率预测校准或独立小时CIF覆盖。下一步应冻结一套**目标侧EF数据合同**后，在外区对比简单月锚、标准动态重组和条件区间强基线；未达标前保留未见BA。[V271协议](V271_ASYMMETRIC_EF_TEMPORAL_VALUE_PROTOCOL.md)、[结果](V271_ASYMMETRIC_EF_TEMPORAL_VALUE_RESULTS.md)。

## V270：最新文献核对后的创新边界

2026年中国电网动态因子研究已用潮流/碳流平衡做节点—时间分辨核算，但依赖拓扑、机组因子、机组出力、负荷和支路潮流；AAAI-26 CIF预测工作已用多频率局部时序及跨变量结构；通用的含噪协变量部分识别也有既有鲁棒优化方法。因此本项目不能把动态因子、wavelet、多频分解、迁移学习或鲁棒MILP单独称为创新。

当前唯一值得继续检验的主假设是：**目标燃料EF水平＋非对称支持的阈值门控，与小时尾风险可识别集及同预算信息采集曲线的耦合**。这套组合对“只有完整月度燃料发电量、没有本地小时标签”的场景有清晰应用问题，也能指导先校准因子还是先采时间信息；但目前仅有美国CEMS→EIA的压力传递，离稳定、可迁移的论文方法仍差直接目标因子来源、与强基线的同预算外区比较和独立小时标签覆盖验证。[V270文献与创新核对](V270_LITERATURE_AND_NOVELTY_CHECK.md)。

## V272：验证标签渠道分层，避免“估算”与“实测”混称

核实到 EIA-930 除已有燃料发电 CSV 外，还可下载 BA 级小时 CO₂ 排放工作簿：含生产侧、燃料分项、进出口及 MRIO 消费侧排放。它比固定因子手工proxy更接近系统级估算标签，但燃煤/气/油的排放率由BA年度机组群估计，且以小时燃料发电为输入；因此它是**官方估算型标签**，不是独立小时CEMS分子。仓库当前并未保存这类 CO₂ 工作簿。

此发现不替代已完成的 CEMS 实测机组级迁移，也没有新增性能结果。后续必须把 EPA CEMS直接机组标签、EIA系统级估算标签和中国模型小时投影分轨：前者限定被监测化石机组，第二类区分生产/消费并披露年度因子与燃料输入共享，第三类只能用于同模型模拟迁移。找到可公开的 EIA 标签并做同合同对照，有机会检验 V271 是否只在固定EF代理下成立；中国真实小时计量仍需授权数据。[V272 证据层级审计](V272_HOURLY_TARGET_EVIDENCE_HIERARCHY.md)。

## V273/V274：官方EIA标签上，简单供体小时形状未形成可信增量

直接下载7个BA的EIA发布工作簿，并冻结2019–2023训练、2024风险校准、2025评分。分别做生成侧和消费侧CIF回测；按当地日期切年，并使每个目标BA的历史标签都不进入自身的源形状或区间半宽校准。初版UTC年切分及全目标共享校准半宽均作废，正式结果使用按当地日期及逐目标排除后的重跑。

目标月标签均值作为oracle中心，测试只回答条件在真实目标月均值已知后，跨BA时钟曲线能不能迁移小时形状。生成侧供体星期小时形状把等BA MAE从32.99略升到33.22 g/kWh、区间宽度从222.43增至223.16，日同时覆盖从84.67%升至85.34%，仍远低于90%；ERCO只有28.8%整日覆盖。消费侧形状也没有让总体覆盖达90%，月×小时供体区间要宽约27%才略增覆盖，而Winkler明显变差；ERCO月锚整日覆盖仅18.1%。因此跨BA典型日内曲线在EIA系统级估算标签上没有通过可信区间门槛，不能作为V271的有效时间信息补救。

这轮主要负结果进一步收窄了突破空间：需要找到能够识别目标局地风险/燃料效率/消费侧流入来源的新增低频字段或极小目标测量预算，不能只增强供体日历分辨率。评估仍是最终修订版EIA估算标签、目标月oracle中心和已开BA cohort；不是严格发布时间回测、直接CEMS整网标签或中国验证。[V273生成侧结果](V273_EIA_GENERATED_CIF_SHAPE_TRANSFER_RESULTS.md)、[V274消费侧结果](V274_EIA_CONSUMED_CIF_SHAPE_TRANSFER_RESULTS.md)。

## V275/V276：月燃料构成有风险尺度信号，燃料EF和净进口并未补足尾部

用EIA工作簿中每类月发电量构成与月发电总量预测日最大小时偏差q90；按目标BA逐区排除历史、2019–2023训练、2024跨区风险校准、2025评分。目标月均值仍为oracle中心。消费侧燃料构成模型相比同口径pooled-source全量区间，宏日覆盖从84.79%升到85.78%，宽度219.00降至209.70 g/kWh，Winkler从322.52降到278.81；6/7 BA达到90%，ERCO仍只有35%。生成侧更差（覆盖84.47%、宽271.23、Winkler343.81；BANC仅16.3%）。

加入净进口比率使消费侧宏覆盖降至85.30%、区间宽212.90、Winkler292.00；再加入本地煤/气/油CO₂因子也变差，因子＋进口为覆盖83.82%、宽232.11、Winkler308.28。现有月度燃料边际可以识别部分平均风险尺度，却不能稳定识别尾部风险；增加已尝试的EF或进口字段没有实证价值。所有结果依赖EIA估算CIF与oracle月中心。[V275协议](V275_COARSE_MIX_CONDITIONAL_RISK_PROTOCOL.md)、[V276协议](V276_LOCAL_FACTOR_LEVEL_RISK_ABLATION_PROTOCOL.md)。

## V277/V278：固定弃答门控可筛出明显迁移失败，但不能保证选择集覆盖

V277冻结无需调参规则：只有燃料构成风险半径比跨区pooled-source q90更窄时才发布。原汇总曾只统计有接受日的BA，宏接受比例因此虚高；脚本已修正，保留零接受BA并按全部有效日计算接受率。2025开发期接受约65.5%有效日，接受集覆盖92.52%，但数据已打开，只能作为候选。

V278按冻结规则在未参与2019–2024拟合/校准的2026时段（七BA、1–10月）做时间外检验，只用八类月燃料占比和月总发电量。门控接受65.56%日，接受集日覆盖仅87.88%，低于90%；BANC为81.14%，CISO为63.53%，FPL为88.96%；ERCO被全部拒答，但MISO也全部被拒答。全量风险模型相对pooled-source只将宏覆盖从80.42%提高至82.41%，宽度219.00降到205.11、Winkler344.17降到316.21，均未达可信覆盖。

2026仅10个月/BA的月风险排序相关弱且异质（BANC/CISO/NYIS Spearman为-0.05/-0.18/-0.18）；燃料构成特征无法识别部分局地尾风险。事后所需q90乘数从BPAT的0.964到BANC的5.080，差异极大，排除“统一加宽即可修复”的解释。该选择机制作为拒答诊断基线有用，但不形成预测突破；月度发电构成、进口和年燃料因子都不足以约束小时尾部。下一步的科学问题应从“更复杂地回归风险”转向明确尾风险可识别的条件/下界，或量化增加何种最小时间观测（小时总发电/燃料分项）才有价值。EIA工作簿标签仍是估算值，月中心是oracle，且没有未见BA或中国验证。[V277协议](V277_SELECTIVE_RISK_ABSTENTION_PROTOCOL.md)、[V278冻结协议](V278_SELECTIVE_2026_CONFIRMATION_PROTOCOL.md)、[V278结果](V278_SELECTIVE_2026_CONFIRMATION_RESULTS.md)。

## V279：EF支持区间改变最少小时观测合同（候选机制扩展）

在V226已打开的26 BA、74 BA-month上，将点EF行空间选出的自适应字段、固定四字段、因子支持感知字段和完整八类小时数据放入同一CEMS燃料特异季度EF支持LP。2026年局地随机12日校准后，支持感知字段平均需3.46个小时数值字段（含总发电量），57/74个月不超过四字段；13个月需五字段。它在每BA-month上的稳健小时可行范围宽度与八类全测仅差浮点舍入量级。整日名义覆盖98.57%，稳健集合整日包含率93.52%；支持感知方案比固定四字段平均窄11.81 g/kWh、Winkler低11.76，区域bootstrap区间均不跨0，而稳健包含率差区间跨0。与源区朴素半径比，Winkler差的区间跨0且本地方案更宽，故没有证实效率支配。

这是比V228更贴近因子不确定性的候选创新组件：根据月燃料活跃支持与燃料特异EF区间，预先选择能识别小时排放范围的聚合通道；名义EF下的最少字段可能漏掉独立变化的油/气分项。它仍不是突破确认：同一V226 cohort已打开；CEMS州—季度支持不是目标BA因子置信集，`other`固定400；小时可行集是逐小时边际LP，区间包含诊断读取了目标完整小时结构，且EIA名义CIF为燃料生成量×固定EF。需要新来源的目标EF合同和未查看系统确认。[V279协议](V279_ASYMMETRIC_EF_SENSOR_FRONTIER_PROTOCOL.md)、[结果](V279_ASYMMETRIC_EF_SENSOR_FRONTIER_RESULTS.md)。

## V280：EIA估算标签外测否定最少通道增量，暴露PACE风险尺度异质

在此前困难区PACE/LGEE下载2025官方BA小时CO₂工作簿，以新标签源压力测试V279支持感知最少通道。两区各12个月，每月随机抽12个完整日、重复100次，并用目标月标签均值作为oracle中心。support_minimum（3–4个小时数值字段）、fixed_four、full8三方案逐次指标完全重合：等BA小时覆盖99.67%、日同时覆盖97.95%，平均区间宽501.50 g/kWh、Winkler 503.71。相较之下，source_loo宽239.08但小时覆盖75.52%、日覆盖51.37%。区域拆分显示PACE source_loo日覆盖仅4.05%，而LGEE为98.70%；本地校准方案在PACE为95.95%、LGEE为99.95%，代价是PACE宽682.60、LGEE宽320.41。

因此V279最少通道机制没有在新标签源上优于固定四字段或full8；区间全局偏宽且LGEE明显过覆盖，PACE要以极宽区间才接近标称日覆盖。问题已从“再减少哪些小时字段”转到“为什么同一source风险尺度在PACE严重失配、LGEE基本适用，以及如何把小时标签/目标EF合同错配从小时形状风险中分开”。这提示下一轮比较目标局部校准、可迁移风险尺度和因子支持失配诊断，并要求在预测时不读取测试日结构的约束下做区间效率评估。

边界：EIA CO₂是BA燃料活动与年度因子产生的估算标签，非直接CEMS；LP支持为美国CEMS季度EF合同，不是目标EF置信集；中心为目标月oracle；PACE/LGEE曾在项目其他代理分析中出现，故是新标签源压力检验而非未看地区确认。事后集合包含率用完整测试小时燃料结构，仅供诊断。[V280协议](V280_EIA_OFFICIAL_LABEL_PACE_LGEE_PROTOCOL.md)、[结果](V280_EIA_OFFICIAL_LABEL_PACE_LGEE_RESULTS.md)。

## V281/V282：本地上一年因子支持修复LGEE过宽，但PACE尾风险仍需本地化

V281在已打开的EIA目标工作簿中读取公开的逐燃料CO₂ Factor列，事后与CEMS季度支持核对：PACE活跃煤小时的因子100%高于CEMS支持上界（目标1,063–1,086，对比季度支持上界1,014–1,033 g/kWh）；LGEE煤/气主力因子完全在支持内，只有极少量油电小时（全年72小时）的因子高于上界。2024本地因子范围覆盖2025 PACE煤因子全范围，而跨区CEMS范围覆盖不到。该差异与V280供体基线在PACE日覆盖仅4.1%、LGEE达98.7%的失效模式一致。

据此形成V282事后候选：用上一年本地活跃燃料因子范围替代跨区CEMS支持；当年仍用相同12个完整日校准、相同随机拆分和同一oracle月中心。在PACE/LGEE两BA的24个月上，平均区间宽从501.50降至426.51 g/kWh（−14.95%），Winkler从503.71降至430.00（−14.63%），日同时覆盖从97.95%降至95.54%，仍超过90%。LGEE宽度降36.9%、日覆盖96.3%；PACE宽度只降4.6%、日覆盖94.7%，仍极宽。CEMS分支逐重复与V280完全重现；source_loo简单基线宽239.08但宏日覆盖仅51.37%，PACE严重失效。

这形成首个更像机制而非“加宽”的可检验候选：**上一年本地燃料EF支持 + 少量本地小时燃料观测校准尾风险**。证据仍是两区事后探索，目标月中心oracle，EIA官方标签估算，且中国当前已知信息只有各类月发电量，未确认有可构造上一年本地EF的燃料消耗/热耗率信息。不能宣称突破或可部署。由于相同的12日抽样预算下效率改善且两个BA日覆盖仍超过90%，下一步可以冻结这一规则，选一个从未打开官方BA小时CO₂工作簿的区域作外部检验；若不能保持覆盖或PACE型目标依旧极宽，就应把结论收敛为目标因子信息是必要数据，而非算法优势。[V281结果](V281_TARGET_EF_SUPPORT_AUDIT_RESULTS.md)、[V282方案](V282_PRIOR_YEAR_LOCAL_EF_SUPPORT_PROTOCOL.md)、[V282结果](V282_PRIOR_YEAR_LOCAL_EF_SUPPORT_RESULTS.md)。

## V282事后尾部尺度诊断与V283封存预检

直接对已打开的2025标签做事后描述：PACE 363个完整日的日最大偏差经验q90为310.0 g/kWh，LGEE 365日为80.0；V282本地上一年EF区间的平均半宽分别325.46/101.05，CEMS支持为341.30/160.20，简单source_loo为119.54。PACE目标小时形状尾部本身约是LGEE的3.9倍，所以统一供体尺度低估PACE、而本地EF支持主要消除LGEE过宽；PACE风险不能靠更准的EF水平单独解决。

尝试冻结未开CO₂标签的SIKE作V283外区，但只用EIA-930燃料活动量预检即发现2025仅覆盖6–12月7个月，达不到冻结的10个月门槛；没有下载/打开其官方标签，也未换目标。另核对后发现SIKE此前进入V279固定EF代理队列，所以即使覆盖够也只能算新官方标签源压力测试，不是完全未查看BA确认。V283结果与边界见[预检报告](V283_SIKE_PREFLIGHT_RESULTS.md)和[冻结协议](V283_SIKE_LABEL_PREFLIGHT_PROTOCOL.md)。当前真正未分析且具备全年多燃料发电的区域需从全BA历史/标签访问表重新审计；水风单燃料BA不适合验证燃料因子迁移机制。

## V284：9日是本地上一年EF先验的最小有效小时校准预算

在V280已开PACE/LGEE队列上，以同一随机日序列嵌套比较9、12、15个校准日。上一年本地EF支持+9日校准达到宏日覆盖93.45%、宽413.41、Winkler419.02；同预算CEMS支持为97.51%、490.77、493.29；source_loo为51.38%、239.08、621.12。9日先验版比CEMS支持窄15.76%、Winkler低15.06%，所有24个BA-month平均日覆盖均≥90%（最差月90.62%）。增加到12/15日提高覆盖但增宽，表明在当前开发标签下可把9日作为最小有效预算候选。

该增量仍由仅两区的事后实验支持，PACE区间仍宽637 g/kWh；不能称独立突破。下一关应是预先冻结规则后的严格未查看BA确认。V283 SIKE因2025只有7个月而未开CO₂标签，且SIKE曾进入V279代理队列，不算盲测。仓库初步盘点显示全年多燃料且此前未参与任何CIF风险开发的BA候选可能已耗尽；必须严查标签与代理队列，不能把“新EIA标签文件”冒充“新地区”。[V284方案](V284_LOCAL_DAY_BUDGET_PRIOR_EF_PROTOCOL.md)、[结果](V284_LOCAL_DAY_BUDGET_PRIOR_EF_RESULTS.md)。

## V285：严格未查看多燃料BA队列暂缺

对仓库2025 EIA-930的62个BA活动量与历史实验路径盘点后，全年且≥4燃料类候选主要为ISNE、NEVP、WACM、PACW、AECI；都已进入之前风险/CIF或外部数据实验，不能当真正未看BA。余下未见BA主要是水电/风电单燃料区，缺少燃料EF组成变化，不能确认V282机制。SIKE仅7个月且V279代理队列已用；HST两个月、WWA九个月。故目前没有合格“未查看+全年+多燃料”区域，不应把新CO₂标签源误称新区域。V282/V284仍是两BA开发候选；下一步真确认需新区域/独立标签源或用户提供未参与开发的本地数据。[外区库存审计](V285_UNSEEN_MULTIFUEL_BA_INVENTORY.md)。

## V286：上一年本地EF支持在时间外显著增效，但严格覆盖门槛未通过

冻结9日方法后，对PACE/LGEE 2020–2023共48个BA-month（此前未作目标评分）做时间外检验。上一年本地EF方法宏日覆盖92.73%、宽344.54、Winkler351.46；同预算CEMS支持为98.25%、448.81、450.24；source_loo为54.55%、228.31、404.71。相对CEMS宽度降23.2%、Winkler降21.9%，但严格冻结门槛要求各BA-year≥90%，PACE 2023只有89.30%，12个月中7个月低于90%（最低84.67%），故判为未通过复制门槛，不调同一标签。

事后V287发现PACE活跃煤因子在2020、2021、2023全小时100%超出上一年本地支持；2022只有10.3%落在上一年区间内。PACE 2023目标因子范围1,064.6–1,086.7 g/kWh，2022支持仅1,021.3–1,035.5；LGEE煤因子跨年包含明显更稳定。该漂移与PACE 2023欠覆盖同向，说明“上一年EF锚定”在局地年度结构变化时会漏风险。下一步做当前年EF oracle敏感性，确认局部因子信息是否能修复，区分EF漂移与小时形状风险；之后重新设计需要当前年哪种最小低频信息，不能宣称V282已跨年稳健。[V286协议](V286_PRIOR_YEAR_EF_TEMPORAL_HOLDOUT_PROTOCOL.md)、[结果](V286_PRIOR_YEAR_EF_TEMPORAL_HOLDOUT_RESULTS.md)、[V287事后漂移审计脚本](../scripts/v287_temporal_ef_support_drift_audit.py)。

## V288/V289：本年因子漂移需更新，但现有标签没有as-of发布时间证据

V288把目标年的完整本地因子支持作为不可部署oracle，复用V286样本拆分。PACE 2023日覆盖从上一年EF的89.30%升至91.15%，但宽度539.40升至546.53、Winkler548.05升至553.35；因此当前年EF能修复部分水平错配，但不能压缩PACE尾部风险。

V289尝试以“上一年支持 ∪ 目标年此前月份EF支持”的滞后并集更新。2020–2023的8个BA-year日覆盖都超过90%，宏覆盖94.30%、宽360.86、Winkler366.24；比CEMS支持窄19.6%、Winkler低18.7%，source_loo仍只有54.55%日覆盖。相较仅上一年EF，月度覆盖低于90%的BA-month由13/96降至3/96；PACE 2023日覆盖由89.30%升至92.80%。但宽度增加4.7%，仍有PACE 2022-02和2023-09等3个月欠覆盖；CEMS支持反而96个月全过90%，代价是更宽。

这把候选机制推进为**滞后本地EF漂移更新 + 小时尾风险校准**，但还不能宣称突破：V289方案在目标数据打开后提出，EIA历史工作簿没有发布vintage，无法证明早期月份EF当时可用；同时区域不是地理盲测，且中国目前已确认只有各类月发电量，尚无当期EF更新字段。真正的创新需求现更清晰：当只知来源月发电量而不知道目标年EF是否漂移时，怎样给出对漂移稳健且不过宽的小时风险界限。需要研究可观察的燃料质量/热耗代理，或明确不可识别时的最小附加信息及拒答条件。[V288结果](V288_CURRENT_YEAR_EF_ORACLE_SENSITIVITY_RESULTS.md)、[V289方案](V289_LAGGED_CURRENT_EF_UPDATE_PROTOCOL.md)、[V289结果](V289_LAGGED_CURRENT_EF_UPDATE_RESULTS.md)。

## V290：增加本地小时校准日能消掉经验欠覆盖，但这不是算法增量

在相同PACE/LGEE 2020–2023队列内嵌套比较9/12/15/18个本地完整小时校准日。滞后本年EF支持的日同时覆盖由9日94.30%升到12日95.67%、15日96.51%、18日97.05%；重复均值低于90%的月由3/96降至0/96，最低月均覆盖90.56%。12日区间宽373.49、Winkler377.25；相较同预算CEMS支持宽19.0%、Winkler低18.4%，但覆盖也低3.06个百分点。相对自身9日，12日宽度增加3.50%、Winkler增加3.01%，增量来自新增本地小时标签预算。18日上一年EF方法亦在所有月重复均值过90%，而区间略窄于18日滞后更新。source_loo仍只有54.6%宏日覆盖。

9日分支在r2逐次复现V289，指标最大差均≤2.3e−13。首版r1只在派生宏表把滞后方法月计数复制到各基线；保留r1且不用于结果，修正后完整重跑r2。该实验仍是两BA、既有已查看标签、月中心oracle及目标小时校准，不是中国信息预算或外区确认。它只量化了本队列的本地标签价值，未产生创新方法。近期ACI/ECI在线共形已处理漂移反馈，generation-based CIF也已有区间walk-forward预测，因此仅把EF范围滞后更新或增加滚动/在线校准不能作为原创算法。[V290结果](V290_NESTED_LOCAL_RISK_BUDGET_RESULTS.md)、[创新缺口与路线](V290_NOVELTY_GAP_AND_ROUTE.md)。

## V291：GB 14区历史API不是独立实际标签

为寻找独立于EIA/固定因子代理的区域外测，核查NESO官方GB区域碳强度API。对North Scotland的2018与2024历史单日端点做schema预检，均返回14区体系的半小时区域`intensity.forecast`和`generationmix`，没有独立`actual`字段；官方方法用30分钟区域生成/需求预测和简化GB交流潮流模型估计消费侧强度。当前CSV仍是持续更新版本，没有历史发布vintage。因此不能把过去时间戳返回的预测模型输出冒充新区域真实标签；它未来可作为基线，但需要先保存实时发布版并与后续真实值配对。V227已有GB全国燃料结构的生成侧固定EF代理外部时间检查，但没有通过Winkler效率门槛。V291因此否定了一个看似现成的盲外测渠道，未打开可用新小时真值或产生预测性能增量。[V291通道预检](V291_GB_REGIONAL_LABEL_PREFLIGHT.md)、[NESO区域方法](https://carbonintensity.org.uk/)。

## V292：弱监督共形的月度弱覆盖对小时真值没有足够约束

评估Cauchois等（JMLR 2024）的弱监督预测推断能否补上月度燃料边际下的小时覆盖缺口。沿用已打开EIA 2026年7–9月非开发队列，令每个小时的弱集合为活跃燃料EF极值区间，月锚为预测中心。锚点单点、全宽0的区间已对弱集合达到100%相交覆盖，但对EIA生成侧小时代理的覆盖只有12.16%；固定全宽400时代理覆盖98.25%，仍有至少一个BA-month仅80.56%。强覆盖全体BA-month达到100%需要全宽800。该结果吻合弱覆盖定义本身：`C(X)∩W≠∅`，不是标签落入预测集。

因此普通弱共形迁移不能构成方法突破，也不能把形式弱覆盖写成小时风险可信。下一条研究主线应是“预算化新增字段→目标联合风险函数的锐界收缩→支持不足时拒答”，并比较每类字段的边际识别价值。该EIA样本已看过，数字是适配反例，不是新区域确认。[V292协议](V292_WEAK_COVERAGE_COUNTEREXAMPLE_PROTOCOL.md)、[结果](V292_WEAK_COVERAGE_COUNTEREXAMPLE_RESULTS.md)。

## V293：月度全燃料边际对阈值小时数仍留下近全月的锐界

在V231的已打开EIA 2026年7–9月队列上，对固定燃料EF、逐燃料月量和任意正小时总量，按五个阈值复核月度-only精确尾数界。注意该月度-only解析界已在V87/V237实现，不是V293的新定理。500 g/kWh下，52/74 BA-month的活跃类别EF跨阈值，平均包络宽715.8小时；另22个月份的类别EF位于阈值同侧，尾数固定为0或T。370行EIA代理尾数全部落在解析包络内。

这是对已存在识别结果的多阈值、较大队列复核，只增强证据广度，没有原创预测方法或盲测增量。一般部分识别LP与线性规划的信息收集文献也已存在，故信息预算前沿需依靠CIF专属的新识别对象、合法数据合同和独立标签确认来产生贡献，不能把“按字段求LP”直接当方法创新。[Finkelstein等，UAI 2021](https://proceedings.mlr.press/v161/finkelstein21b.html)、[SIAM J. Optimization信息收集论文](https://doi.org/10.1137/12086279X)。下一步按风险函数与新增字段预算计算CIF锐界收缩，并检验是否有CIF独有机制。[V237结果](V237_NESTED_DAILY_HOURLY_RISK_INFORMATION_RESULTS.md)、[V293协议](V293_MONTHLY_ONLY_TAIL_COUNT_PROTOCOL.md)、[结果](V293_MONTHLY_ONLY_TAIL_COUNT_RESULTS.md)。

## V294：中国公开月度多燃料表是估算产品，不是独立逐类观测

为落实用户所说的各类月度发电量，直接审计Ember当前省级月表和其2026年8月更新方法。CSV有2016-01至2026-09的31省×8源月度发电，同时列出排放与强度；但Ember方法明确NBS仅发布规上火电合计，煤/气/其他化石/生物质的月度分拆依靠Carbon Monitor份额、年度校准及装机结构IPF，风光还用NEA和ERA5/物理模拟修正。排放强度在CSV内可由排放/电量列算术复原，不能作为独立CIF标签。Ember因此是有价值的公开估算输入和同源一致性目标，不是直接实测验证数据。

这修正了“类别齐全即可当作已观测燃料边际”的默认假设：如果用户原表就是Ember，则算法输入有估算/来源依赖；如果用户使用CEC或监管原表，需另行审核口径和修订时点。候选研究应把数据估算预算和目标标签证据等级拆开；这项审计尚非算法突破。[Ember页面](https://ember-energy.org/data/china-electricity-data/)、[China Methodology 2026-08](https://storage.googleapis.com/emb-prod-bkt-publicdata/public-downloads/ember_china_generation_methodology.pdf)、[V294审计结果](V294_CHINA_MONTHLY_SOURCE_LINEAGE_RESULTS.md)。

## V295：停止重复阈值门控，寻找目标EF支持的独立依据

按“检查现有实现后再决定”的原则，复查上轮准备继续的阈值支持门控。仓库已有V240比例分配全宽定理、V265误差精度临界前沿、V268因子中心/非对称支持比较、V271日/小时风险集合的信息价值实验，以及V279支持感知小时字段选择。V271还在16个BA-month实测了跨阈值两例小时总量无收窄、其余14例收窄。重复同一EIA队列/指标无法带来新贡献，故不重跑。

下一步应把问题转到V240/V271缺失的前提：**怎样用目标可得的低频信息构造有覆盖依据的目标燃料EF支持集？** V294显示Ember月燃料分项经过估算；MEE年度因子是消费侧且口径不同；跨区CEMS因子盒只是经验压力支持。新方法若不能给出目标EF支持的合法来源/校准置信度，那么阈值门控和后续hourly-LP只能是条件分析。优先审计用户实际源或CEC原表是否另含煤耗/燃料消耗/按燃料排放，并区分全国、区域、省级以及发布时点。现阶段没有证据支持宣称已有算法突破。

## V296：中国月度“所有发电类型”仍须按火电内部粒度和小时信息拆解

复核了CEC两个原始月报PDF与国家统计局规上工业能源生产发布。CEC月报是全国表，发电量列为总量、水电、火电、核电，另有全国平均供电煤耗；并不等于省级煤/气/油逐月量。国家统计局月度能源发布列火电、水电、核电、风电、太阳能等来源组，但“火电”仍是聚合项，且限规上企业。国家能源局另定期发布全国年初至当月累计供电煤耗率（2025年1–6月300.7、1–8月304.6 g/kWh），对全国热耗趋势可能有用，但它不是目标省级每月EF。Ember的省级细分则是估算产品，不能视为逐燃料实测或独立小时标签。

信息边界更清楚：若省级月度表确实细分到各燃料且有同口径本地EF，它能算/约束**月均**CIF；它仍不能从月边际识别小时CIF形状或跨阈值尾数。CEC全国供电煤耗目前也不足以填补目标省EF支持，因为其地域、供电/发电边界与煤气内部混合都错位。该轮不是新性能实验或突破；报告[V296](V296_CHINA_MONTHLY_INFORMATION_BOUNDARY.md)冻结字段合同和下一关。要推进可落地性，用户实际表头/来源/粒度是关键输入；继续研究时避免把五类能源统计直接称作逐燃料月量。

## V297–V299：平均锚点的信息价值高，但测量精度脆弱；解析集合经独立LP复核

V297在同一358个EIA BA-month、滞后粗容量预算下，比较粗技术月量、假想目标月均CIF、阈值侧MWh和详细燃料月量。零误差的目标月均CIF锚可把平均可行K跨度压到接近详细燃料边际（τ=600: 223.9 vs 223.8小时；τ=900: 187.8 vs 140.0小时）。但这属于oracle信息价值，不是可实施预测。

V298测得精度脆弱：月均锚只允许±0.25 g/kWh误差时，平均K跨度已升为356.4/319.9小时（τ=600/900），负证率从精确锚的约53.6%降至约20.7%，大部分月份回到弃答。此误差带是敏感性设定，不是中国计量误差实测值；它表明直接采月均CIF要发挥作用，得有非常精确且同边界的本地CIF/EF数据。

V299用独立两块燃料分配LP复核解析法，在10个按冻结规则选出的BA首月、两阈值和两误差带共40组中，所有完整K可行集完全一致。该审计修复了之前HiGHS尺度容差的假阳性与K集合排序比较问题；同时也确认误差带±0.25时，不同BA可行跨度从0–28小时到1–744小时，不能声称普适窄区间。线性规划匹配只验证识别集数学实现，未验证标签、统计覆盖或中国迁移。

方向判断因此更明确：**不要把“预测月均CIF再桥接小时尾部”作为主创新**，精度要求与中国公开数据口径不符。高EF侧月MWh在因子支持稳定且逐燃料月量可得时，信息结构更贴近阈值风险；然而V99表明阈值附近燃料EF区间会使该通道失效，而V294/V296表明国内公开省级燃料拆分本身有估算依赖或只到火电聚合。当前更可信的论文贡献候选是“目标统计信息预算下的CIF小时风险锐界/弃答基准 + 因子支持不确定性”，算法新意尚未达成。下一轮应对中国公开可见类别合同做同预算、同因子支持的区间前沿，并报告何种新增时间信息能使预测从不可识别转为可识别，不能继续把EIA固定因子代理的零误差oracle锚当作算法增量。

复现与完整限制见[V297–V299结果汇总](V297_MONTHLY_CIF_ANCHOR_VALUE_RESULTS.md)。

## V300/V301：官方省级年度因子给出一条新的水平锚路线

继续核对中国公开资料时，发现MEE/NBS在2025-12-31发布了2023年省级电力平均CO₂因子，覆盖30省；此前V296没有纳入。官方方法明确因子用于消费侧电力核算，省级计算包括本地发电排放及省际/区域净输入电量，2021–2023因子计算方法和数据来源一致。[2023公告](https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/t20251231_1139517.html)、[方法说明](https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202404/W020240412827267566470.pdf)。这补上了“完全没有省级年度水平标签”的空白，同时也证明它不能直接替代本地发电侧小时CIF。

V300对2023目标做30省留一地区测试：仅用其他省2021/2022官方因子和前一年Ember发电结构拟合，再从留出省2022五组结构预测2023官方因子。五组非负混合模型MAE/RMSE为0.0493/0.0770 kgCO₂/kWh，相比训练省份均值0.1121/0.1510，MAE降56.1%；八类燃料模型MAE降55.2%，没有胜过五组模型。25/30省优于均值，但北京、青海、上海、吉林等残差仍大，北京绝对误差达0.3553 kgCO₂/kWh。V301将目标改为2022、仅用2021因子训练，MAE从0.1207降到0.0596/0.0598，复制出约50%的降低。

这构成截至目前最具体的**新地区水平锚候选**：公开粗发电结构似乎已足以迁移年度平均水平，不必依赖省级小时CIF历史或煤气拆分。但V300/V301仍是年均层面结果，MEE标签是消费侧并纳入跨省电量；Ember特征历史版本不完整，目标与输入来源也不等于用户自己的报表。结果没有覆盖、小时形状或超阈风险指标，不能据此宣称小时预测突破。已公布年因子约滞后两年，也限制实时部署。

下一步转向残差机制：用可公开年度跨省送受电量检验是否能解释大误差省份，确定本地发电构成之外的最小年度水平输入；若该字段无法获得，则把省际流量作为消费侧目标的不确定项。完成后再把“年度水平锚 + 小时形状/风险模块”放回EIA生成/消费侧同边界标签比较覆盖和效率。不能将年度平均点误差下降替代小时预测证据。完整设计、原始逐省残差和限制见[V300/V301报告](V300_OFFICIAL_PROVINCIAL_EF_MIX_TRANSFER_RESULTS.md)。
### V302：年度省级送受电粗代理未解释年度EF迁移残差（2026-10-30）

在 V301 已冻结的 2022 省级留一预测上追加事后机制诊断，转录《中国电力统计年鉴2024》同年分省用电、发电、跨省送出量。供需差额与有符号残差 Spearman ρ=0.097（p=0.611）；送出占比 ρ=−0.218（p=0.248）。各用一个变量做跨省留一线性残差修正，MAE/RMSE 均由 V301 的 0.0598/0.0814 恶化至约 0.0867/0.108。见[V302结果](V302_OFFICIAL_FLOW_RESIDUAL_DIAGNOSTIC_RESULTS.md)。

这轮负结果缩小了有效信息候选：省级年度发电/用电差额或单向送出总量不足以解释消费侧官方EF中由电力交换带来的偏差。下一步应先查能否取得目的地受入电量、来源电网/路径碳强度的公开数据；若没有，年均来源混合只能留在水平锚，不能据此宣称小时CIF迁移。V302仍为同年事后诊断、年鉴数值转录与30省小样本，不能证明无潮流效应。当前尚无足够证据称为算法突破。

### V303：新增同年分省受入电量仍不能校正V300残差（2026-10-30）

检索到2025年《中国电力》论文表3列出2023年各省跨省送出与受入总量。将受入量除以全社会用电量，事后与V300 2023五组模型留省残差对齐：受入占比对有符号残差Spearman ρ=0.193（p=0.307），绝对残差ρ=0.231（p=0.219）；单变量跨省留一残差修正使MAE从0.0493增至0.0730、RMSE从0.0770增至0.1002。详见[V303结果](V303_DIRECT_RECEIPT_RESIDUAL_DIAGNOSTIC_RESULTS.md)。

该发现排除了“受入电量体量可通过统一单变量关系可靠修正”这条简单路线，但不排除受入来源碳强度的作用。相应文献比较了来源地、受端、煤电净调入与绿证口径，报告不同核算方法会显著影响部分区域的间接排放核算；因此真正缺失的信息应是受入量的来源/电源构成与目标口径，而非只增加总量。此处是信息预算/标签边界证据，不是新算法或小时预测结论。
### V304–V305：把缺失OD矩阵转为锐界和最小观测预算（2026-10-30）

根据MEE省级因子计算说明，来源省/区域因子需乘以净受入电量；2025年《中国电力》论文又提供2023年区域送出/受入边际总量与路线集合。两侧边际精确闭合于59,433,862万kWh。将未知13条路线电量作为运输多面体，官方区域因子加权后，论文所列路径支持下华北、南方、西南进口强度被点识别；华东界[0.4456,0.5689]、华中界[0.2685,0.6036]仍宽。完全不使用路径结构时，华北、南方、西南界宽分别扩大到0.248、0.389、0.208。见[V304/V305结果](V304_V305_REGIONAL_FLOW_IDENTIFICATION_RESULTS.md)。

更强的结构结果：13条边的等式秩11、只有2个自由度；对华东/华中进口因子以及所有受端因子的联合点识别，均最少需要补测2条独立双边流量。西北→华东与西南→华东是一组足够线路。所有值仍是2023区域回顾性分析，零宽依赖路径表穷尽、边际和EF精确。它是最小信息/可识别性创新候选，不是小时CIF预测结果；运输LP本身不新，下一步要验证它能否推广到中国省级公开数据并改善真实标签下预测区间，才可能支持论文方法贡献。

### V306：公开UHVDC线流目前不能直接补成V305的区域OD流量

核对国家能源局2023可靠性报告中的年度点对点直流输电量与线路端点后，复奉、锦苏、建苏三条明确由四川送至华东的线合计801.02亿kWh，占论文西南区域全部方向送出边际1019.70亿kWh的78.6%。该部分量不超总边际，可作部分物理通道观测，但缺少区域内交流交换及其余通道，不能等同于完整的西南→华东OD值。宾金送端跨四川和云南，缺少分源拆分；不能把整条线路量强行计入西南。吉泉+灵绍为1106.73亿kWh，落在V305西北→华东可行区间内，但这只说明数值不冲突，不证明相同计量口径。不同官方/行业公开报告对同一线路年量也存在差异。

继续回读原文PDF后确认，区域表1/2的电量统计来源为《中国电力统计年鉴》，表中只给区域可达关系和发送/接收边际；表3给各省发送/接收总量，但没有省际OD配对。NEA线路可靠性表是另一张统计表。当前没有证据把某条物理线路量与年鉴某条区域边定义为完全相同计量对象，故V307只能保留条件敏感性。原文没有提供可以直接填补OD矩阵的线路到区域分解。

因此V305仍只是条件性的数学最小观测设计，公开线级电量尚未验证其落地性。下一关是获取论文区域划分与原始统计边界，并判断月度逐类发电数据能否搭出独立的年度消费侧/发电侧约束；不能将物理线路量和净区域交换直接混算。详见[V306审计](V306_HVDC_MARGIN_COMPATIBILITY_AUDIT.md)及复算脚本 `scripts/v306_hvdc_accounting_compatibility.py`。

### V307：部分通道下界可能收紧锐界，但尚未跨过口径验证

在V304运输多面体上，对西南→华东边加入三条四川起点线路合计801.02亿kWh的条件下界。若保留100%该线路量，华东进口因子界由[0.4456,0.5689]缩至[0.4456,0.4863]，华中界由[0.2685,0.6036]缩至[0.4930,0.6036]；两区宽度均下降约67%。保留80%时宽度仍下降约54%。这是有意思的部分流量观测价值：一个边的下界通过总边际耦合同时约束两个接收区。

当前只能称条件LP敏感性，因输电线路统计与论文净交换量的计量点、损耗、区域映射尚未一致；未观测小时标签，不是小时CIF预测增量。新的候选方法方向是“线级部分观测→区域OD可行集→来源因子锐界”的信息预算框架，但LP不新，必须先证实多年度省级同口径数据和独立标签预测价值。详见[V307协议](V307_PARTIAL_CORRIDOR_BOUND_PROTOCOL.md)、[结果](V307_PARTIAL_CORRIDOR_BOUND_RESULTS.md)。

### V308：省级总送受量聚合后对区域OD约束冗余

将Quan等论文表3的31省发送/受入总量按七区域汇总，逐区与表1/2的区域跨区交换边际比较。所有区域边际都低于对应省级跨省总量；例如华东跨区送出853万kWh，而区内省级跨省发送合计16,350,856万kWh，华东跨区受入29,783,268万kWh，而省级受入合计51,376,666万kWh。总省际量163,293,824万kWh约为区域跨区总量59,433,862万kWh的2.75倍。因此把各区域省级总量仅作为容量上界，不会收紧V304运输多面体。

这与V302/V303的负结果一致：单个省级发送/受入体量没有足够来源—目的配对信息。下一步所需数据应是已配对的OD/结算流，或经过口径验证的物理线路约束；重复获取更细的未配对总量不是最小信息突破。详见[V308协议](V308_PROVINCIAL_MARGIN_INFORMATION_AUDIT_PROTOCOL.md)、[结果](V308_PROVINCIAL_MARGIN_CAPACITY_AUDIT_RESULTS.md)。

### V309：增加滞后月度燃料组成季节项反而恶化跨省年因子预测

在同样30省整省留出的MEE/NBS年度消费因子目标上，以2020/2021滞后月Ember五组份额的一至二阶傅里叶项增加年度NNLS信息。2022目标年度五组NNLS MAE为0.0598，季节一阶/二阶为0.0718/0.0920；2023目标NNLS为0.0493，季节一阶/二阶恶化到0.0708/0.0835 kgCO₂/kWh。一阶季节项2023相较NNLS的MAE增量0.0216，省份bootstrap区间[0.0100,0.0334]，且只有8/30省改善。故否决简单月内季节特征迁移路线；月度细节没有自动补上消费侧跨区来源信息。

这项负结果说明当前最强证据仍是滞后年度五组结构作为年水平锚；复杂月度分量没有带来跨省泛化，也不能替代小时标签。候选突破须转回“最小可观察OD/小时聚合信息→锐界或风险区间”并获取独立标签；不能继续堆叠月季节特征。详见[V309协议](V309_MONTHLY_SEASONAL_MIX_TRANSFER_PROTOCOL.md)、[结果](V309_MONTHLY_SEASONAL_MIX_TRANSFER_RESULTS.md)。

### V310：新公开年度省际 OD 矩阵提供信息结构实例，但没有小时预测突破

新审计到 Xiong 等 2026 年 Zenodo v1 数据集，提供中国 31 省 2023 年技术发电量、技术 CF、省级 CF 分布、输配电生命周期因子和省级送端—受端年度矩阵。矩阵含 116 条非零边、总量 1,834.23 TWh，列/行总量闭合。用比例共享节点流追踪重建源数据省级 CF 均值后，31 省 Pearson 相关 0.998、MAE 0.0603；相较本地技术构成加权的 MAE 0.0652，流向信息有排序价值，但点误差收益仅约 7.5%。再加数据集自身 T&D 因子，MAE 为 0.0339。

这是 V305 省际 OD 最小信息假设的年度可操作例子，但输入和比较目标全部来自同一生命周期研究记录，不是独立验证；该记录未明确省级 CF 的核算侧，且生命周期 CO₂e 与小时发电侧 CIF 的边界不同。已有中国时空 CIF 论文采用 node-flow 方法，故单独移植比例共享/flow tracing 不新。当前该数据增强“成对省际流量能补充省级边际”的证据，但未证明它能提高 hourly prediction coverage 或区间效率。下一步需优先获得并审核用户实际月度逐技术发电表及版本/口径；再构造月平均生成侧 factor 与跨省年度 OD 的明确边界情景，并只在独立小时标签上检验。详见[V310 协议](V310_ZENODO_ANNUAL_OD_AUDIT_PROTOCOL.md)、[结果](V310_ZENODO_ANNUAL_OD_AUDIT_RESULTS.md)。

### V311：即使年度 116 条 OD 路线已知，月度受电量仍不可识别

使用 V310 年度 OD 矩阵与 Ember 2023 省级月总发电代理，建立非负月流量 LP：各路线全年合计固定为 OD 年量，送端每月总外送不超过当月发电。31省×12月共372个省份月份全部可行，但各自受电量锐下界都是0 TWh；毛受电占“本地发电+毛受电”比例的平均界宽0.546（中位0.598）。均匀拆分年度路线量是可行情景，但均匀份额均值0.168，仅是额外假设，不能当作观测。Ember代理比Zenodo同省技术发电合计低1.59%，故该约束是跨来源压力测试，不是完全同口径的官方实测。

因此，月度各类发电结构与年度省际 OD 即使都有，也不能锁定该月的来源构成；月度受电边际、带月份的双边流量或受电来源碳强度是更直接的新增信息候选。由于缺少月度需求、线路容量、流损和调度数据，本 LP 的范围有意宽；它只是“在当前信息预算下可行集仍然大”的证明，不是小时 CIF 预测区间或新算法突破。下一轮应等预算受控地比较月受电总量、少数月OD边和来源碳强度三种新增信息，找对小时风险最有识别价值的最小字段。详见[V311协议](V311_ANNUAL_OD_MONTHLY_RECEIPT_BOUNDS_PROTOCOL.md)、[结果](V311_ANNUAL_OD_MONTHLY_RECEIPT_BOUNDS_RESULTS.md)。

### V312：等预算信息价值的最坏情形比较指向省级月受电边际

用双副本LP对所有可能实测字段值取最坏条件区间直径，比较进口源碳暴露代理。对372个省份月份目标，在341个独立月度数字预算下，知道各省月受电总量平均保证缩窄85.3%区间；知道31条年度电量最大路线的月流量缩窄42.2%；按年度估计碳量排序的31条路线缩窄46.1%；仅有11个全国月受电总量没有保证缩窄。这个两副本定义不依赖均匀调度拆分作为伪真值，是当前最有力的最小字段候选。

该结果仍是源因子固定的年度生命周期代理。OD 与源 CF 同属Zenodo数据集，月总发电上界来自Ember估算；目标不是小时CIF。路线子集只测试两种冻结启发式，没有穷举最优路线组合。现阶段可主张的信息设计贡献是“省级月受电统计比少量大路线量更稳健识别直接进口碳暴露”，不能主张覆盖可信小时风险。下一步要把各月源EF/目标因子支持不确定性纳入，再检验数据预算排序是否仍成立，然后才开展小时盲测。节点流追踪本身已有成熟工作，不能作为创新主张。详见[V312协议](V312_SAME_BUDGET_IMPORT_INFORMATION_PROTOCOL.md)、[结果](V312_SAME_BUDGET_IMPORT_INFORMATION_RESULTS.md)。

### V313：中国区域市场规则开始提供时间戳买卖方流，但这只是物理 OD 子集

国家能源局全国月报披露跨省跨区**市场交易**总量，没有省对省或全口径省级月受电表；但 2026 西北省间互济细则规定日/月市场信息包含买方、卖方、交易时段和电力电量，日前后支持15分钟级日内出清。华中（东四省）市场也已开始连续结算试运行。由此，V312高价值统计“省级月受电量”目前不是已验证的全国公开全口径字段；更可操作的下一步是取区域互济市场的时间戳交易流作为**已观测子集**，与年度OD边际共同约束未观测流。

这可能把最小信息设计从静态年量推进到前瞻可用的交易记录，但市场成交/出清量不等于结算落地电量，更不等于总物理潮流。没有清算时间戳和物理口径核对之前只能估计“市场交易来源碳风险”，不可称为用电侧小时CIF。下一步先审计公开交易平台是否可下载真实带时段的买卖方—成交电量表，再决定做观测子集锐界，避免用新闻总量代替OD输入。详见[V313字段可得性审计](V313_CHINA_MONTHLY_FLOW_FIELD_AVAILABILITY_AUDIT.md)。

### V315/V316更正与V319：月报式五类别仍有区间识别价值，但尚非CIF突破（2026-10-30）

复核发现V315/V316两副本LP的进口来源排放强度误用受端省份，而正确归属应为送端省份。这使两轮旧区间宽度与信息收益数字失效；旧结果文档已加更正标识，保留原文件以便追溯。V319已按送端省×技术×月份因子重跑完整31省×12月，并在相同8源潜变量上对比月总发电上限、火/水/核/风/光五类上限、八类细分上限，以及341个省月受电字段的组合。

更正后，五类相较总量上限在372省月格中平均缩窄条件锐界21.5%（省份聚类bootstrap 95%区间15.3%–27.9%）；五类再细成八类、且不加受电字段时仅多缩窄1.07%。341受电数字单独加在总量上限上只有0.06%平均收益；与五类上限联合时为5.25%。这提示在当前代理可行集里，五类能源结构比单独的受电边际提供更基础的信息，二者联合有互补作用。

这是有效的信息预算/可识别性候选，不是小时预测或真实消费侧CIF结果。V319的因子和逐月发电均来自Ember同一估算产品；V310年度OD统计与其月供给边界尚未共同闭合；分母仍是本地发电量，目标是进口排放暴露代理。LP与字段预算不是新算法，且五类映射尚未对用户原表逐字段核验。下一步需用用户原始月表建立字段合同，取得同边界需求/流量或独立小时标签；若无法获得小时标签，应将论文主张明确限制为信息预算下的月度锐界基准。详见[V319结果](V319_CHINA_MONTHLY_CATEGORY_INFORMATION_RESULTS.md)。

### V320：NBS月表存在，但规上工业口径不能直接当全系统外送上限（2026-10-30）

找到OREO汇编的NBS省级月发电表：31省、总量及火/水/核/风/光，许可CC BY 4.0；公开年发电及年用电表可作同源对照。方法明确月发电只覆盖规上工业、2月值为1—2月累计，年发电则覆盖全部电厂。2023按2月累计值加3—12月后，月表合计8854.86 TWh，仅为同源年全电厂9456.44 TWh的93.64%；少约601.58 TWh，省均覆盖率93.1%（84.3%–99.0%）。因此月度分类不能当作所有来源电量的硬上限；V319区间缩窄数值不能原样外推到这个真实字段合同。

同源年全口径发电与年用电分别为9456.44、9223.80 TWh。再拼入V310年OD后，`G+OD净受电-D` 在8省为负（合计−97.85 TWh），正残差合计330.49 TWh。数据边界仍未闭合，说明公开月类别可用于发电结构，但月OD与消费侧平衡不可直接拼接。OREO公开月消费页当前需申请访问，年消费表不能替代月需求分母。

正向进展是取得了可复现的真实公开五类别月表；V321已把年全口径减月规上发电作为“未观测供给池”，并把天津缺失水电设为总量内未知。详见[V320结果](V320_NBS_MONTHLY_GENERATION_SCOPE_AUDIT_RESULTS.md)及[V321结果](V321_UNOBSERVED_GENERATION_REMAINDER_RESULTS.md)。

### V321：补入未观测供给后，月分类收益明显收缩，收受信息呈互补（2026-10-30）

将NBS月度规上发电缺口601.58 TWh作为不按月分配的年度剩余池，设未知运营EF∈[0,1] kgCO₂e/kWh，并把天津缺水电设为总量内未知。五类月约束在无受端字段时，341省份×期中平均锐界只缩窄3.62%（中位0；13/31省平均有缩窄；省份bootstrap 95%区间1.81%–5.76%），明显低于V319未处理范围差时的21.5%。加入310个省×期受电数字后，五类信息相较月总量字段的平均配对缩窄为11.20%（中位0.30%；17/31省平均有缩窄；95%区间6.47%–16.25%）；受电字段单独加入月总量模型的收益近零。这提示类别与受端字段之间存在互补，但只在部分省期明显。

这是公开字段范围内的结构性部分识别证据，非小时覆盖验证或标准消费侧CIF。受端需求、损耗与V310路线口径仍未闭合，剩余EF支持也是显式情景。V319原21.5%不可挪用到NBS表。V322已将残余EF上限扩至0.5、1.0、1.5并通过逐省期单调性审计；论文需要报告因子支持—区间宽度前沿，而不单报字段收益。详见[V321结果](V321_UNOBSERVED_GENERATION_REMAINDER_RESULTS.md)与[V322结果](V322_RESIDUAL_FACTOR_SUPPORT_SENSITIVITY_RESULTS.md)。

### V322：因子支持敏感性保留正向平均信息价值，但常见省期的收益不稳定（2026-10-30）

把未观测年残余EF上限从0.5扩大至1.0、1.5 kgCO₂e/kWh后，五类月数据相对月总量的平均配对缩窄依次从10.75%降至3.62%、2.79%；省期中位缩窄在后两档为零。加入310个假设受端月字段后，平均增益分别为15.99%、11.20%、8.78%，中位为17.28%、0.30%、0.21%。三档省份聚类bootstrap区间都高于零，但区间宽，且场景端点未验证。逐省期锐界随因子支持扩大单调不缩窄，计算审计通过。

因此类别信息的平均作用不是只由 `[0,1]` 单一点设定造出，但效益大小与覆盖的省期高度依赖残余风险支持。创新机会收敛为“统计覆盖缺口＋因子支持共同驱动的最小观测预算/锐界前沿”，不是通用点预测器。下一关要用真实用户表核 schema，再寻求独立受端小时标签；在这之前尚不能称作小时CIF突破。详见[V322结果](V322_RESIDUAL_FACTOR_SUPPORT_SENSITIVITY_RESULTS.md)。

### V323：等预算最坏情形比较排除“月度残余供给剖面”作为首要补测字段（2026-10-30）

把年度全口径减月度规上发电的已知省级剩余量作为年度总量约束，新增假想每省10个独立月度残余发电量（第11期由年度总量推出），与310个省×期受电总量做同预算比较。双副本LP对所有可能的共享观测值剖面取最坏条件区间直径，避免用Ember拟合形状假装实测。五类供给约束下，残余发电剖面字段对341个省期目标平均及中位收窄均为0；月受电量平均缩窄8.03%，中位仅0.30%，187/341省期为正、17/31省平均为正，省聚类bootstrap 95%区间4.05%–12.71%。两类字段同时给出不优于受电量单独给出。只有总量供给约束时，二者均几乎没有价值，显示受电字段与五类发电结构在此可行集内互补。

由此，当前最小信息候选应优先转向接收端月度受电统计，或同预算的少数成对来源—目的流量；继续补月度残余供给形状无法在最坏情形下识别更多进口排放暴露。受电收益高度异质，不能写成普遍区间缩窄；该结论仍受`[0,1]`残余EF代理支持、年度OD跨源口径、缺少消费平衡和假想字段分布等限制。尚未验证小时CIF预测或覆盖率，也没有算法突破。详见[V323结果](V323_MONTHLY_RESIDUAL_FIELD_VALUE_RESULTS.md)。

### V324：年流量最大的31条线路与全省月受电统计价值接近（2026-10-30）

在同一五类别供给约束和310字段预算下，比较逐省10期月受电合计与年度OD流量最大的31条路线各10期月流量。后者覆盖年度OD总量77.56%。受电字段平均缩窄8.03%、中位0.30%；稀疏OD字段平均缩窄7.24%、中位0；逐省期正收益分别为187/341和154/341，正省平均分别17/31和14/31。线路减受电的平均收益差为−0.79个百分点，省聚类bootstrap 95%区间[−3.03, +1.19]个百分点，未支持任一单字段设计明确胜出。620字段合并后平均缩窄9.11%，只比受电字段多1.08个百分点，且中位仍只有0.30%。总量供给臂下所有新增字段仍近乎无效。

因此，少数大线路的成对流量可作为接近受端合计统计的信息替代，但不能称为更高效或新的算法突破；当前实现仍是LP锐界基准，选择规则仅按年度流量排序，字段效果在省份间分布稀疏。下一轮应冻结年度OD、因子差异和供给信息驱动的候选选边规则，在跨年或留区条件下比较其是否能稳定超过受端合计与大流量基线，并扩展残余EF支持。若无显著、广泛且跨年稳健的增量，应将贡献收敛为最小观测基准/负结果，而非夸大为CIF预测创新。详见[V324结果](V324_SPARSE_MONTHLY_OD_VALUE_RESULTS.md)。

### V325：滞后碳强度差异评分没有改善稀疏线路观测（2026-10-30）

将线路按“2023年度OD电量×2022 Ember来源省与受端省年均发电加权碳强度差的绝对值”排序，选31条线路、每条测10期月流量，与V324年流量前31条和310个受端月总量比较。五类供给约束下，碳差异选线平均缩窄5.91%、中位0；年流量选线为7.24%、中位0；受端总量为8.03%、中位0.30%。碳差异线路的正收益为143/341省期、13/31省平均；其流量覆盖仅65.04%，而年流量前31条覆盖77.56%。碳差异减年流量的平均收益差−1.33个百分点（95%区间−4.62至+1.22），减受端统计−2.12个百分点（−5.71至+0.42）；未显示显著劣势，但点估计均更差。将其与受端字段合并后也没有超过年流量线路加受端字段。

因此否决简单年度碳强度差异评分，不能把直觉上的“高碳差线路”包装成创新。下一轮将测试受端覆盖均衡的选线设计：每个有入流的受端先覆盖一条可行线路，再用年流量补齐固定线路预算。若覆盖均衡也不能扩展正收益省份并改善中位区间，停止继续堆选线启发式，回到受端统计/信息预算基准或明确报告负结果。V325仍为单年直接进口暴露代理，不是消费侧CIF或小时预测突破。详见[V325结果](V325_CONTRAST_RANKED_ROUTE_DESIGN_RESULTS.md)。

### V326：受端均衡选线扩大正收益覆盖，但不胜过受端月统计（2026-10-30）

按每个受端选一条年度最大入线路，得到31线、310个独立月流量数字，覆盖年度OD总量55.15%。相比全球年流量前31线路（77.56%覆盖），受端均衡线平均缩窄由7.24%略降至6.83%，但正收益省期从154/341增至187/341，正省份从14/31增至17/31，中位从0增至0.30%。这表明覆盖约束可把收益分散到更多省期。受端均衡线与全球大流量线的平均收益差−0.41个百分点（省聚类95%区间−3.09至+2.31）；与受端月总量差−1.20个百分点（−2.94至+0.03），均不能证明明确优劣。其正收益格数/省数与受端月总量相同，但总体平均更低。加受端字段后620字段平均收益8.57%，也低于全球大流量线加受端字段的9.11%。

这第三个选线启发式没有形成突破，且显示宽度均值与受益覆盖之间有可解释的权衡。停止继续调年度线路排序。下一步回到真实小时标签的未见地区预测：测试本地月度燃料类别与滞后年度水平锚能否改善小时分布/阈值风险预测，对比无锚、持久性、季节、标准动态模型与合规保形基线；只有独立小时标签上覆盖维持且区间效率改善才算算法增量。若无小时标签可用，则收敛为信息可识别性基准，不能写成小时CIF预测突破。详见[V326结果](V326_RECEIVER_BALANCED_ROUTE_DESIGN_RESULTS.md)。

### V327：公开NBS月度分项在两个留出年份都保留年度水平迁移信号（2026-10-30）

用V320 OREO/NBS-derived实际月度发电类别替换V300的Ember mix，目标仍为官方MEE 2021–2023省级年度消费侧因子，按省LOPO，2022目标训练2021、2023目标训练2021–2022，输入滞后一年。五组NNLS pooled 2022/23 MAE为0.0517、RMSE 0.0706、P90绝对误差0.1197 kgCO₂/kWh；同拆分Ember为0.0545/0.0792/0.1288；训练省均值为0.1164/0.1582/0.2081。NBS与Ember相比MAE点估计改善5.1%、RMSE改善10.9%，且两个目标年均有改善；但省份配对绝对误差差为−0.00280，聚类bootstrap 95%区间[−0.00965,+0.00194]，仅14/30省NBS较好，增量未通过预设不确定性门槛。

这是比估算mix更直接的中国公开字段路径，但仍只有两个时间目标；MEE标签是消费核算，NBS月表是规上范围，历史发布时间和修订vintage不明，不能外推小时CIF或覆盖率。下一步按同LOPO折把已知“年全电厂−月规上”缺口作为显式未知组并与NBS五组结构联合拟合，测试统计覆盖差异本身是否影响年度水平迁移。详见[V327协议](V327_NBS_REPORTED_MIX_TRANSFER_PROTOCOL.md)、[结果](V327_NBS_REPORTED_MIX_TRANSFER_RESULTS.md)。

### V328：把规上覆盖缺口设为第六个来源因子没有增益（2026-10-30）

在V327同样的30省、2022/2023目标和省留一折上，将月表五类份额乘以月表/同源年全口径发电覆盖比，并增加“未观测剩余份额”第六类，用NNLS学习共同因子。scope-aware模型MAE在2022为0.05794、2023为0.04658、合并为0.05226 kgCO₂/kWh，均略差于V327五类归一模型的0.05700/0.04646/0.05173；合并RMSE从0.07060升至0.07210，P90绝对误差从0.11972升至0.13113。配对绝对误差差为+0.00053（95%区间−0.00432至+0.00520），14/30省更好，未达到预注册晋级标准。

这说明统计覆盖缺口在V321物理硬约束中必须显式建模，但把它变成共享第六因子并不能改善跨省年度水平预测。别把数据口径问题直接包装成可预测信号。下一轮升级空间划分：用七大统计地理区整区留出，测试公开NBS月结构能否迁移到真正未见区域；如果整区外推失败，省级LOPO增益不能支持冷启动主张。目标仍是年消费因子，训练/特征历史vintage和发布时效也须单列，不能声称小时CIF突破。详见[V328协议](V328_SCOPE_AWARE_NBS_TRANSFER_PROTOCOL.md)、[结果](V328_SCOPE_AWARE_NBS_TRANSFER_RESULTS.md)。

### V329：整区域留出保留粗组成信号，但未证明NBS优于Ember（2026-10-30）

按冻结的七大统计区逐区留出30省，2022/2023两个前向目标年上，NBS滞后构成NNLS MAE为0.0746 kgCO₂/kWh、Ember为0.0695，训练均值为0.1227；等区域MAE分别0.0757、0.0715、0.1205。两种构成模型都明显优于区域外训练均值，但NBS没有稳定胜过Ember：只有2/7区域NBS更好，NBS−Ember区域聚类差为+0.00419，95% bootstrap区间[−0.00260,+0.01466]。NBS在西北明显失效（2022/2023 MAE 0.1953/0.1608），东北2023也比训练均值差（0.1157 vs 0.0354）；西南则两种构成都远优于均值，说明错误不能仅按可再生占比概括。拟合的东北风电因子大于1.14 kgCO₂/kWh，提示组分共线和系数弱识别。

当前结论是“整区冷启动下有年度粗构成信号”，而非“可普遍迁移的CIF预测器”。消费侧MEE目标与本地发电结构的偏差可能与跨省交换和口径相关，但本实验不能证因。下一步应先测训练集可计算的特征支持/外推诊断是否能提前标记失效，再测试年发电—用电差额是否在构成之外提供增量；必须继续整区域外层留出，且标签仍限于年度消费侧因子。详见[V329结果](V329_CHINA_REGION_BLOCK_COLDSTART_RESULTS.md)。

### V330：嵌套空间调参的弱岭惩罚大幅修复西北/东北误差，但区域收益不够广（2026-10-30）

在外层整区留出内，仅用外层训练区域和可用历史标签做内层空间验证，从冻结的六档λ中为非负NNLS选择岭惩罚。相对V329原始NNLS，NBS岭NNLS的合并省加权MAE由0.07461降至0.05144 kgCO₂/kWh（−31.1%），等区域MAE从0.07567降至0.05450；两个目标年均改善。收益集中在先前严重偏高的西北（2022/23 MAE 0.1953/0.1608降到0.0295/0.0208）和东北（0.1126/0.1157降到0.0901/0.0834）。

但只有3/7区域相对原NNLS变好，北方和西南两年都变差；区域聚类bootstrap差值95%区间为[−0.06765,+0.00825]，跨零。当前预注册的广泛增益门槛未达，普通岭回归也非独立创新点。它揭示的机会是风险自适应：只在训练可识别性/特征支持弱时增强正则，否则保留原NNLS。下一轮将冻结自适应规则并与固定岭、原NNLS同外层比较，禁止按已见西北结果做地区专用开关。结果仍为年度MEE消费因子迁移，不是小时CIF。详见[V330协议](V330_SPATIAL_NESTED_RIDGE_PROTOCOL.md)、[结果](V330_SPATIAL_NESTED_RIDGE_RESULTS.md)。

### V331：简单组成距离不能判断岭惩罚何时有益（2026-10-30）

对60个省份—年份外层预测，比较训练组成最近Hellinger距离、NNLS到岭预测的调整幅度、NBS与Ember预测分歧三个训练可算分数。前两项与岭模型逐例收益的Spearman相关分别为0.10、0.28，区域聚类95%区间均很宽且含零；组成距离AUC为0.50。NBS—Ember分歧相关0.35、AUC 0.64，可能含有提示，但七区域聚类区间[−0.14,+0.66]仍不能排除无效。西南平均组成距离最高却被岭惩罚伤害，西北则有较大收益，简单距离阈值不能分辨。

因此不继续拟合逐省门控规则，以免在七个区上过拟合。当前结果更像信息不足的证据：本地五类发电结构既未描述跨省来源，也未稳定标记消费因子迁移风险。要继续逼近突破，需要有配对来源/受电信息或独立小时生成侧标签；同一小样本上反复调回归参数无法证明覆盖可靠。详见[V331协议](V331_COLDSTART_SUPPORT_DIAGNOSTIC_PROTOCOL.md)、[结果](V331_COLDSTART_SUPPORT_DIAGNOSTIC_RESULTS.md)。

### V332：把热类拆为煤/气等八类并未改善整区迁移（2026-10-30）

在V329同一七个区域外层留出与2022/2023前向标签上，使用同一Ember月度构成输入分别拟合8类（煤、气、其他化石、生物质、水、核、风、光）和聚合5类（热、水、核、风、光）NNLS。省加权MAE分别0.0801和0.0695，等区域MAE分别0.0792和0.0715；细分仅3/7区域改善，细分减粗分的区域平均MAE差+0.00768，7区bootstrap区间[−0.01676,+0.03993]。细分改善西北/东北，却使北方恶化0.0949 kgCO₂/kWh。

这说明增加源类别会同时增加可解释信息和待估系数，不能仅因标签更细就认为迁移更准。下一轮对5类和8类分别做训练区内嵌套岭惩罚，判断该负结果是否由弱识别导致；若仍无广泛增量，停止类别拆分路线。数据仍是回溯Ember估算构成和年消费侧MEE标签，不代表用户月表或小时CIF。详见[V332协议](V332_FUEL_GRANULARITY_REGION_COLDSTART_PROTOCOL.md)、[结果](V332_FUEL_GRANULARITY_REGION_COLDSTART_RESULTS.md)。

### V333：正则化缓和8类过拟合，但正则后的5类模型仍更稳（2026-10-30）

在相同外层整区折内，分别用训练区域嵌套调λ的岭NNLS拟合Ember 8类与5类组成。岭惩罚使8类省加权MAE从0.0801降到0.0571，5类从0.0695降到0.0490；但同预算公平对比的8类岭仍劣于5类岭（区域等权MAE 0.0552 vs 0.0511），仅3/7区更好，差值95%区域bootstrap [−0.01794,+0.03591]。它对未正则5类有4/7区的增益，却并未给细燃料统计带来稳健独立价值；北方仍明显变差。

因此终止“靠拆细燃料种类就得到更好跨区年度预测”的当前路线。年度最强项是标准的五组岭组成回归，不能称为算法创新。下一步从同一小标签集移开：利用V245–V246已经暴露的机组燃料转换效率差异，测试本地热耗率/效率或成对受电来源信息在小时实测标签上的增量；中国应用仍需证明相应字段真实可得并核对历史发布时间。详见[V333协议](V333_NESTED_REGULARIZED_FUEL_GRANULARITY_PROTOCOL.md)、[结果](V333_NESTED_REGULARIZED_FUEL_GRANULARITY_RESULTS.md)。

### V334：中国公开年鉴热耗率在整区留出中带来跨区年度增量（2026-10-30）

审计《中国电力统计年鉴2024》省级6000 kW及以上电厂表2-41发电煤耗与表2-43供电煤耗，取滞后年度值缩放本地热电份额。30省整省级LOPO中，NBS五类NNLS的MAE由0.05173降至0.04381/0.04393 kgCO₂/kWh；七个统计区整区留出中，NBS等区域MAE由0.07567降到0.06180/0.06178（发电煤耗/供电煤耗），两个前向目标年均改善。Ember五类基线也有约11%区域等权MAE改善。西北与东北的偏高明显缓解，说明“组成×本地转换效率”比继续拆细热组更有希望。

但仍不是已确认的论文算法突破：NBS增益仅在4/7或3/7区出现，区域bootstrap区间都跨零；东部/中部等地略恶化。年鉴历史值的发布时点晚于目标，不能称实时合法冷启动；标签仍是消费侧年因子，不是小时CIF。另，r1 runner漏建2023目标行，已保留并标为无效；正式r2包含2022和2023完整评分，结论只据r2。下一轮用训练区内嵌套选择混合强度，测试能否保留西北/东北修复而减少其它区的负迁移，再决定是否值得上小时标签。详见[V334协议](V334_LOCAL_THERMAL_HEATRATE_TRANSFER_PROTOCOL.md)、[结果](V334_LOCAL_THERMAL_HEATRATE_TRANSFER_RESULTS.md)。

### V335：嵌套正则后，本地热耗率在Ember整区迁移仍有窄幅增量（2026-10-30）

在七统计区完全留出、2022/2023前向目标下，对五组构成基线和“热份额×滞后本地煤耗率”采用相同嵌套岭NNLS。Ember源的发电煤耗率调整使区域等权MAE从0.05112降至0.04692，两个年份的MAE与P90均下降；4/7区域改善，七区域配对聚类bootstrap差值95%区间[−0.00797,−0.00043]。供电煤耗率版本的点估计也下降，但区间跨零。NBS源的两种费率均在4/7区改善，bootstrap均跨零。

这是目前中国年度冷启动线上最具体的本地效率信息增量，但算法只是标准岭模型加上简单、可解释的热份额尺度，区间下界仅略低于零，只有七个聚类；更关键的是年鉴历史发布日期尚未对齐目标时点，不能据此宣称可实时预测。结果也不是小时CIF或覆盖验证。下一关先核对用户实际月表类别、计量单位、累计口径、统计范围和发布时间，再按真正合法信息集做外推。若效率字段的历史发布时间不支持目标时点，则它只能作为回顾性解释量；论文主线应转为泄漏安全的跨区信息预算/部分识别基准。详见[V335协议](V335_NESTED_HEATRATE_FEATURE_TRANSFER_PROTOCOL.md)、[结果](V335_NESTED_HEATRATE_FEATURE_TRANSFER_RESULTS.md)。

### V336：训练区内选择效率迁移强度没有超越固定全量校正（2026-10-30）

在整区留出内层同时选择煤耗校正强度γ与岭惩罚，纯构成模型对应γ=0。训练折常选γ=0.5–0.75，但外层增益不稳定：NBS供电煤耗版本等区MAE为0.04956，对纯构成0.05450，5/7区改善，聚类区间[−0.01132,+0.00015]仍跨零；Ember发电煤耗版本0.04947对0.05112，4/7区改善，区间[−0.00431,+0.00091]。所有分支中固定γ=1的整体点误差反而更低。训练样本不足以学出可靠的区域自适应门控，因此不保留自适应γ作为创新方法。V336-r1对照曾错误复用同一个λ，已留档但不采用；结论只看各固定臂独立调λ的r2。详见[V336结果](V336_NESTED_HEATRATE_SHRINKAGE_RESULTS.md)。

### V337：省内月度季节形状没有稳定补充年度构成信息（2026-10-30）

在同一整区折中，将五类年度平均占比扩为“年度均值＋各类月度首谐波正余弦系数”，对Ember和NBS分别嵌套调岭惩罚。NBS pooled等区MAE从0.05450到0.05294，但2022变差、2023改善，仅3/7区变好，聚类区间[−0.00836,+0.00302]跨零。Ember由0.05112略升至0.05161，亦只有3/7区变好。NBS季节特征排除了Jan–Feb累计的2月观测。结果否决在这组年度标签上继续扩充季节特征；形状/水平拆分若要成为CIF贡献，应直接在独立小时风险标签上验证，而不能由年度平均因子代替。详见[V337协议](V337_MONTHLY_SHAPE_LEVEL_TRANSFER_PROTOCOL.md)、[结果](V337_MONTHLY_SHAPE_LEVEL_TRANSFER_RESULTS.md)。

### V338–V341：中国月度技术组与装机能否约束小时尾部风险（2026-10-30）

用户确认国家统计局月度口径包含火电、水电、核电、风电和太阳能，火电不拆煤/气/油。官方月度产量属于规上工业口径；热类仍聚合煤、油、气、余热余压余气和生物质，不能从五类月量直接得燃料加权CIF。V338检查2023年341个省月/时段里，滞后装机没有月均发电量超装机上限；这只是必要条件，不检验小时峰值。详见[V338协议](V338_CHINA_LAGGED_CAPACITY_FEASIBILITY_PROTOCOL.md)。

V339首次小时K外界把2022/2023装机臂错标且重复累计行，作废；V340修复装机映射后，仍把天津缺失水电项隐式当零，不能用于完整五技术池结论。两份原始结果均保留并在各自修正协议标记。V341仅保留五类齐全的330个省—月（30省×11期），并把目标定义为五类已报发电量之和；总发电量仅用于口径差额审计，不强行配平。以thermal EF `[400,1000]` g/kWh情景、其它技术直接运行排放因子为0，输出的是K小时数的外包络[min K,max K]，不是完整整数可行集，也不是实测小时CIF。

V341的330省月完整案例中，月度五类电量单独给出平均外包络宽约795小时。加入2022滞后省级装机后，τ=100降至469.9小时（缩窄40.9%），τ=300降至682.0小时（缩窄14.3%）；用2023年末装机则为483.4/686.2小时。τ=300仍有55–64%弃权。宽度改善是依赖固定排放因子和装机小时上界的条件部分识别信息价值，缺中国小时标签，不能说覆盖率、预测准确率或突破已成立。参见[V341协议](V341_CHINA_COMPLETE_CATEGORY_HULL_PROTOCOL.md)、[结果](V341_CHINA_COMPLETE_CATEGORY_HULL_RESULTS.md)。

下一步仍需真实逐小时发电标签或跨区域外部数据验证识别覆盖与合法发布时间。若该信息不可得，论文新意应落在信息预算基准、统计口径/版本账本和透明弃权，不应声称新预测算法。真正的中国未见区小时确认暂时无法执行，因为现有NBS月量并无小时真值。

### V342：五技术组K外包络在完整中国省月样本上达到锐界（2026-10-30）

把已知的V92/V102锐组容量求解器逻辑独立移植到中国五类技术组，使用完全相同的330省月、月度边际和两套CEC装机表，在thermal EF=400和1000 g/kWh两端分别枚举完整整数K集合。独立锐界端点与V341外包络在660个省月×阈值组合、所有信息臂上完全一致；两端点K集合的并集每一行都填满[minK,maxK]区间中的全部整数（660/660无缺口）。由于固定逐小时分配下提高thermal EF不会降低K，连续EF支持的范围也不能超出此端点区间。V341因此不是松的外界，而是在本次固定组别模型条件下的sharp区间，且整数区间全可达。

宽度信息量与V341一致：τ=100下滞后装机把平均区间从795.42小时缩到469.86小时；τ=300则仅从795.77缩到682.05小时，仍有大量弃权。这是目前“公开粗信息能识别什么风险”的最强中国证据，但并非新的锐界算法（已有V92/V102），也不是预测覆盖验证。最可信论文位置是严格信息预算与合法统计口径下的CIF部分识别/基准框架；创新还需通过最小附加信息实验（特别是thermal有效因子/内部燃料组成对τ=300的价值）及独立逐小时标签验证。详见[V342协议](V342_CHINA_SHARP_GROUP_HULL_PROTOCOL.md)、[结果](V342_CHINA_SHARP_GROUP_HULL_RESULTS.md)。

### V343：热电因子信息与装机约束存在明显互补，时序仍是主缺口（2026-10-30）

对τ=300，在thermal EF支持[400,1000]、[500,900]、[600,800]和oracle点700下，使用V342完整锐界求解器重复330省月。仅月度五类电量时，支持从[400,1000]缩到精确700，平均K区间只缩0.40小时（795.77→795.36）；加2022滞后装机后则缩182.17小时（682.05→499.88），加2023年末装机缩174.54小时（686.20→511.66）。EF信息与容量约束强互补；即使给oracle精确EF=700和滞后装机，仍留下约500小时可行区间，因此高阈值尾部的主要未知是小时调度/组内时序，而不只是热电因子水平。

这给了一个较聚焦的最小信息价值假说：若能在合法公开渠道获得省级热电内部燃料/热耗信息，它只有在与装机硬上界共同使用时才有显著价值；但它本身不足以形成窄小时风险预测，仍需至少一种合法的小时/日内形状或独立小时标签。700 g/kWh及嵌套因子带只是oracle敏感性，不是中国省级校准参数；无小时标签、无历史发布时点校核，不能宣称经验覆盖。详见[V343协议](V343_THERMAL_EF_SUPPORT_VALUE_PROTOCOL.md)、[结果](V343_THERMAL_EF_SUPPORT_VALUE_RESULTS.md)。

### V344：新文献收紧了创新空间，主线应改成可审计的信息预算基准（2026-10-30）

最新已发表近邻Li et al. (Scientific Data, 2026)已经公开31省、2025–2060、五规划情景的8760小时省级CEF模型投影；它用电源—电网—负荷—储能优化和碳流计算，并以2021–2023官方年因子验证（MAPE≤2.525%）及小时碳流平衡验证。其GTSL/PSOS核心优化代码不开放，作者也指出无小时未来CEF数据可做预测对比。另一个更直接的月到日近邻Wang et al. (SSRN, 2025-12)用月度燃料发电量、气象降尺度、CarbonMonitor-China日排放标签，在内蒙古/江苏/云南三省训练RF/GB/XGBoost/LightGBM并做时间交叉验证；可访问摘要没有证明零标签整省留出，因此不能等同于目标省冷启动，但它明显压缩了“月数据降尺度到高频CEF”的创新空间。Lyu et al. 2026 SSRN预印本另用小时负荷模型、PyPSA-China与碳流分析消费侧因子；波形/小波的新CIF预印本则用澳洲的小时CIF、负荷、风光/火电和气温输入。详见[V344文献审计](V344_RESEARCH_POSITIONING_LITERATURE_AUDIT.md)。

这使“我们用五类月量再训练一个小时CIF模型”很难成为新意，且五类月量不能验证消费侧进口因子。现在可守的窄贡献是假设透明、按真实发布日期冻结的信息预算，对点预测、部分识别边界和系统模型投影分轨评测；V343显示装机与热电因子互补，但即使oracle因子+滞后装机仍留下约500h的τ=300范围。下一关要在Wang等的三省日标签之外，做真正的整省留出与无目标日排放监督迁移，进一步对比零本地CIF标签的信息预算；再用独立小时标签数据做外部机制验证。当前仍是候选框架，未达到算法或实证突破标准。

### V345：年度全口径差额和装机不能定位月度规下电量（2026-10-30）

把同源年度全电厂总发电量减去NBS月度规上总发电量，得到31省共601.58 TWh未覆盖差额。再给每个省使用2022或2023总装机作为月发电上限，按年度差额守恒计算每个报告期可分配范围。2023装机下，341个省月每一个都允许放入该省全年全部差额；2022装机下335/341个也允许，剩下6个仍能放入99.8%以上。月度分配下界全为0，单期分配范围的中位宽度19.245 TWh。

因此V341–V343基于规上五类月度边际的K区间不能直接解释为全省全口径CIF小时风险边界。年度总量和装机并没有把未覆盖电量的时序压窄；其技术组成与因子也未知。这是一项数据合同/识别边界结果，没有新增小时风险算法，也没有实测覆盖验证。下一步要么寻找有发布时间记录的省级全口径月度锚，要么将残差作为独立未知组纳入小时K可行集，量化其对V341边界的扩宽。详见[V345结果](V345_CHINA_RESIDUAL_MONTHLY_SCOPE_RESULTS.md)和[协议](V345_CHINA_RESIDUAL_MONTHLY_SCOPE_PROTOCOL.md)。

集合关系补充：在V345容量约束下，每个省月的未覆盖差额下界为0，全年差额可放在其他报告期。故全口径扩展模型若允许该0残差配置，旧的V341–V343 K集合仍包含于新集合，旧宽度是新宽度下界。该推论不依赖残差EF，但不提供实际扩宽小时数；要算扩宽量仍须给残差的合法月度边界、技术组成/EF支持与小时出力上限。

### V346：把年度差额纳入小时K集合后，装机带来的区间收益大幅回落（2026-10-30）

在V341的330个完整省月上加入0到年度全口径差额的月残差，残差EF支持取0–1000 g/kWh，热电EF仍为400–1000；并用省级总装机作为残差组小时上限。此上限与已报技术容量独立，允许总出力超过系统装机，所以结果是保守外包络。加入残差后，τ=100的2022滞后装机平均K宽度从469.9增至604.0小时（+134.1）；τ=300从682.0增至795.5小时（+113.4），330省月中328个外包络覆盖整期。2023回溯装机得到相近扩宽。仅月度信息臂原本已近整期，平均变化分别不足1小时。

这表明装机收益依赖全口径月度残差约束；粗月报统计口径一旦留缺，小时风险可识别性几乎退回整月宽度。它是明确的负结果和论文框架线索，尚非实证覆盖或算法创新。残差EF支持未经本地标定，官方NBS查询数值尚未逐省下载核对。V346详见[协议](V346_SCOPE_RESIDUAL_K_HULLS_PROTOCOL.md)和[结果](V346_SCOPE_RESIDUAL_K_HULLS_RESULTS.md)。

### V347：估算月度燃料组成与装机上限互补，但燃料估算不是真值（2026-10-30）

将Ember 2023省级月度8类燃料估算折算成热电有效EF区间，与NBS规上类别＋全口径残差的V346外包络配对。若将Ember分项当作准确月度边际，τ=300的平均宽度在滞后装机下从795.9缩到498.3小时（缩窄297.5小时），在2023回溯装机下从795.5缩到510.2小时（缩窄285.3小时）；仅月度臂几乎不变。τ=100下容量臂也缩窄约170–177小时。此信息价值与V343一致指向“组成×容量”互补。

证据边界很强：Ember燃料份额经过模型估算和校准，不是官方直接燃料观测；其总电量均值比NBS规上总量高约5.7%–5.8%。滞后装机臂有2个西藏省月物理不可行，已作为不可比单元报告。当前收益是在把估算月mix当作正确约束时的条件敏感性，不能称实测覆盖或同预算预测增益。下一步应给燃料估算加入误差集合，检验约285小时的价值是否稳健；收益若对来源误差敏感，不能把Ember估算引作破局输入。详见[V347协议](V347_EMBER_COMPLETE_MIX_VALUE_PROTOCOL.md)和[结果](V347_EMBER_COMPLETE_MIX_VALUE_RESULTS.md)。

### V348–V349：匹配因子支持后，月度估算mix比补总量带来更大条件识别增量（2026-10-30）

V348把各省月煤、气、其他化石、生物质份额分别扰动±100%，再与燃料因子支持集合求交。τ=300、滞后装机下，328个可行省月平均K宽度从NBS残差臂795.85小时降到519.07小时，缩窄276.78小时；月量单独臂几乎没有宽度改善。扰动比例是压力情景，不是Ember误差校准。

V349-r1因五技术对照EF[400,1000]窄于燃料分项EF外包络[0,1100]，不适合归因，已明确作废。修正后的r2对三臂同一省月和容量逐一配对：NBS月量＋年残差宽795.85小时；Ember估算总量聚合为五技术、统一热电支持[0,1100]后宽773.55小时；在同一Ember总量/容量和因子外包络下加入±100%份额误差的燃料mix后宽519.07小时。总量/口径臂贡献22.30小时，燃料mix再贡献254.48小时，后者在276/328格变窄、52格相同、没有格变宽。燃料mix相对总量控制的增量约占两段总缩窄的92%。

因此在这个K部分识别模型中，最有价值的附加粗信息是热电内部燃料组成，而不是只补齐省级月度总发电量；但当前收益仍完全条件于Ember估算mix，没有真值、小时标签或经验覆盖检验。官方NBS `fsMonthData` 页面可按省查询月度发电量，其可见类别是合并火电/水电/核电/风电/太阳能，并未展示煤/气/油火电拆分。网页抽取器只能识别该页需要JavaScript；本次运行环境连接官方站点时也返回连接关闭，因此官方逐省数值尚未直接拉取复核。下一步应优先建立官方源逐省当期值数据账本，确认累计/当期与发布时间，然后寻找独立小时标签；在此之前不能将此结果写成已验证的小时预测创新。详见[V349协议](V349_EMBER_TOTAL_VS_MIX_ABLATION_PROTOCOL.md)、[结果](V349_EMBER_TOTAL_VS_MIX_ABLATION_RESULTS.md)。

### V350：燃料mix与装机约束存在强交互，单独mix几乎不识别小时尾部（2026-10-30）

在V349同一328省月、同一Ember总量和[0,1100]火电EF外包络下，对“聚合热电EF vs ±100%误差燃料mix”与“无有效容量上限 vs 滞后2022 CEC容量”做2×2锐K集合消融。五技术组宽度从无容量的796.63小时降至有容量的773.55小时（容量增益23.08）；燃料mix臂从795.82降至519.07小时（容量增益276.74）。mix在无容量时仅缩窄0.81小时；加容量后缩窄254.48小时。差中之差互补量253.67小时，276/328个省月为正，0个为负，中位数266.5小时。

这是模型内可解释的信息互补：月度组成边际需与可用小时出力上限配合，才明显排除τ=300尾部可行分配。它不是经验预测准确率/覆盖验证，且Ember月mix与±100%份额误差均非观测真值，装机也是历史代理。故当前中国论文最稳的定位继续是“合法信息预算下的小时风险部分识别与互补机制基准”，而不是宣称已有新预测器。要称算法/实证突破，必须取得可审计的省月燃料因子/组成以及独立小时标签。详见[V350协议](V350_FUEL_MIX_CAPACITY_FACTORIAL_PROTOCOL.md)、[结果](V350_FUEL_MIX_CAPACITY_FACTORIAL_RESULTS.md)。

### V351：容量互补在份额误差±100%下仍保留（2026-10-30）

将V350的同一2×2因子设计扩展到V348的全部燃料份额误差带。互补缩窄量从误差0%时274.35小时、327/328省月为正，逐步到误差±100%时253.67小时、276/328为正；五档均无负互补。组成信息在无有效容量约束时只缩窄0.81–0.87小时，有容量时仍缩窄254.48–275.22小时。

这让“热电mix × 出力上限互补”的模型内结论不依赖单一误差档，但份额误差集合仍人为设定，数据也仍为Ember估算；误差带未经观测校准，且没有小时CIF标签，因此不构成经验预测增益。下一步破局不是再调误差参数，而是取得官方省级月度边际原始序列、合法容量发布时间，并找到外部小时标签，才能检验这种互补是否对应真实覆盖。详见[V351协议](V351_MIX_CAPACITY_COMPLEMENTARITY_FRONTIER_PROTOCOL.md)、[结果](V351_MIX_CAPACITY_COMPLEMENTARITY_FRONTIER_RESULTS.md)。

### V352：互补效应随风险阈值升高而衰减，不能泛化成统一规律（2026-10-30）

固定±100%燃料份额误差、328省月和同一[0,1100]EF外包络，将V350因子设计扩到τ=100/300/500。燃料mix×装机差中之差分别为357.34、253.67、115.80小时；τ=500还有9/328省月的交互项为负。τ=100/300下mix无容量边际仅约1小时，但τ=500已为49.49小时。因此均值上互补在三档均为正，强度却明显阈值依赖，不能说所有高尾部风险都遵循同一机制。

τ=300四臂宽度与V350相同，提供了复算一致性。下一步按省聚类估计互补量的不确定性/地区主导度，再决定是否值得形成机制主张。即使聚类结果稳定，Ember估算mix、回顾性容量和无小时真值仍意味着它只是信息几何，不是实测预测增益。详见[V352协议](V352_THRESHOLD_FACTORIAL_SWEEP_PROTOCOL.md)、[结果](V352_THRESHOLD_FACTORIAL_SWEEP_RESULTS.md)。

### V353：省聚类后阈值衰减仍稳定，但西藏τ500出现饱和型负互补（2026-10-30）

对V352按30省聚类，20,000次bootstrap后等省平均互补τ100/300/500分别为355.68/252.30/112.68小时，95%区间依次为[318.59,389.83]、[200.88,301.09]、[54.14,168.42]。τ100−300差103.38小时、τ300−500差139.62小时，区间均不跨零；留一省均值范围亦保持正。

但τ500的9个负省月全部来自西藏，西藏省均互补−399.33小时：±100% fuel mix本身已将区间缩到0小时，装机再无法收窄，因此与“需要容量才识别”的主模式不同。这显示负交互是识别集饱和导致，不能忽略地区/阈值异质性。当前机制可表述为“组分与容量的识别互补依赖基线是否饱和”，不能包装成统一线性增益。V353是全体省份有限样本bootstrap敏感性，不是普遍总体保证；仍无小时真值。下一步解析西藏EF支持与容量边界，寻找是否为物理/口径差异或模型支持导致的饱和。详见[V353协议](V353_PROVINCE_CLUSTER_INTERACTION_BOOTSTRAP_PROTOCOL.md)、[结果](V353_PROVINCE_CLUSTER_INTERACTION_BOOTSTRAP_RESULTS.md)。

### V354：EF支持位于阈值哪一侧，完整解释τ500全部负互补（2026-10-30）

把±100%误差下的热电有效EF边界与τ直接比较。τ500的328个配对省月中，20个省月EF支持全在阈值以下；这20格的燃料mix K区间都为0。全部9格负交互都在这类中且均为西藏；其它11格虽然mix也已将K宽度压为0，但聚合对照本身无容量增益，所以交互为0。西藏上界约440 g/kWh < τ500，模型假定非热电直接EF=0，因此mix信息本身已饱和识别K=0，容量无从再缩窄。

其余支持全在阈值以上的省月，容量互补均值为τ100 357.34h、τ300 260.05h、τ500 139.99h；跨越/触及阈值的τ300 9格互补均值仅27.56h，τ500 11格为0且mix宽度仍等于整期。由此得到一条更精确的可检验机制：先判断热电EF支持相对风险阈值的位置，再判断容量是否有识别空间；全在阈值下可能已饱和为K=0，跨阈值通常仍宽，支持全高于阈值时容量与组成的互补更明显。该规则目前只是Ember估算与非热电EF=0假设下的几何诊断，绝非真实中国小时CIF安全判别。详见[V354协议](V354_THRESHOLD_SIDE_SUPPORT_DIAGNOSIS_PROTOCOL.md)、[结果](V354_THRESHOLD_SIDE_SUPPORT_DIAGNOSIS_RESULTS.md)。

### V355：河南官方省级月报可大幅缩小候选年度残差，但口径尚未对齐（2026-10-30）

用户给出的国家统计局页面可按省查看月度发电统计；补查国家能源局河南监管办月报发现，2023年公开了总量及水/火/风/光四类的月值和累计值。4月、8月直报页本轮未检索到，但分别可用5月减3月累计及9月减7月累计还原；单月类别闭合误差最大0.01亿千瓦时。监管办12月累计总发电343.273 TWh，对比本地NBS衍生年度全厂353.464 TWh低10.191 TWh（2.88%）；而NBS衍生2–12月月序列年度化合计317.10 TWh。它显示有机会把V345的36.364 TWh年度差额压小约72%，但2023监管办只写“全省”，未明确写“全口径”；2026才出现明确标签。跨源残差不能冒充同口径误差真值，而且监管办把火电合并，无法提供煤/气/油mix。

这是一个值得拓宽到多省的官方数据来源线索，不是小时预测或覆盖增益。必须先完成公开发布版本、统计范围和跨月差分审计，再以保留的统计范围残差重做V346；即使识别区间变窄，仍需独立小时标签验证。详见[V355协议](V355_HENAN_REGULATOR_MONTHLY_SOURCE_AUDIT_PROTOCOL.md)、[结果](V355_HENAN_REGULATOR_MONTHLY_SOURCE_AUDIT_RESULTS.md)。

### V356–V357：国家统计局省级入口可扩成统一账本；云南累计锚只带来小幅、阈值依赖的识别增量（2026-10-30）

用户提供国家统计局[分省月度数据入口](https://data.stats.gov.cn/dg/website/page.html#/pc/national/fsMonthData)，确认可以直接选择省份查询。它使全国统一月度技术边际账本具有清晰的官方入口；但火电仍聚合、统计范围为规上工业，而且还需保存官方原始响应、单位、当期/累计含义、查询时点和修订版本。不能把现有OREO派生表当作本项目已经完成的逐省原始官方抓取。

跨省来源审计发现，云南能源局并列公布全省总量与规上分类电量。用官方全省累计总量减本地NBS五类边际形成复合残差质量锚后，V357在云南2024年11期sharp K集合中，τ=300等期平均宽度从796.36降至755.73小时（40.64小时、5.1%），仅3/11期缩窄；τ=100完全无改善。τ=300按目标小时加权缩窄37.88小时。它说明稀疏累计节点能排除少量时期残差配置，但收益小、依赖阈值，没有小时标签与经验覆盖验证，不构成预测算法突破。云南“1—10月”页面日期早于统计期结束，已从主结果剔除，仅作敏感性分析。详见[V356](V356_CROSS_PROVINCE_PUBLIC_SCOPE_SOURCE_AUDIT_RESULTS.md)、[V357协议](V357_YUNNAN_CUMULATIVE_SCOPE_TO_K_PROTOCOL.md)及[V357结果](V357_YUNNAN_CUMULATIVE_SCOPE_TO_K_RESULTS.md)。

下一步应利用国家统计局省级入口建立有来源快照的31省月度账本，核实跨省缺项、期间语义和口径差异后重跑部分识别；再找独立小时标签。论文主线仍是有限信息预算下的跨区CIF风险基准、统计范围/版本账本、sharp风险集合与弃权评测；没有新算法或标签验证前不声称算法创新或覆盖达标。

### V358：河南逐月独立总量锚比年度残差锚更有识别价值，但τ300收益仍弱（2026-10-30）

河南能源监管办2024年12份月报均有当月总发电量及水/火/风/光类别。逐月报告值彼此闭合至0.01亿千瓦时，并与12月累计值闭合至同样的舍入量；报告只说“全省发电量”，没有明确写“全口径”。与本地NBS/OREO五类月量拼接后，得到11期合计35.774 TWh的跨源复合差额，再用V342锐K solver做同EF、容量的年度锚/逐月锚配对。

年度残差改为逐月已知后，等期平均K宽度τ100从639.27降到492.82小时（缩窄146.45小时、22.9%，11/11期改善）；τ300从798.55降到764.09小时（缩窄34.45小时、4.3%，只有4/11期改善）。小时加权改善分别146.41和48.27小时。比V357稀疏节点更有信息，但高风险τ300仍多期无增益、区间仍宽；最重要的是来源差不是非负真实规下发电真值，没有任何小时标签能检验经验覆盖。故此结果支持把“统计频率匹配/来源差异预算”列入基准，不是CIF预测算法突破。原始12月摘录、逐月协议和结果见[数据CSV](../data/china/henan_nea_monthly_generation_2024.csv)、[V358协议](V358_HENAN_MONTHLY_TOTAL_ANCHOR_PROTOCOL.md)、[结果](V358_HENAN_MONTHLY_TOTAL_ANCHOR_RESULTS.md)。

源间技术差分显示来源误差有符号：NEA减NBS年总分类量的水/火/风/光分别为+0.797/−8.614/+7.809/+35.782 TWh。NEA分类边际直接替换NBS后τ300平均K宽度反而比NBS臂大约112小时；这说明只给净总量差建立一个非负unresolved组会掩盖技术来源错位。下一步更值得探索的是总量守恒下的多源技术边际区间协调，并与V358单一残差组和单源类别臂对照；须把多源来源不确定性作为方法对象，而非假装分类一致。

V359已形成待验证算法设计：潜在技术边际落在两个官方来源逐组包络内，并满足独立月总量守恒，随后把整个来源可行集传播到小时K锐界。核心难点是现有V342 solver只对固定技术边际锐，不能仅用各来源端点并集代表联合集合；新solver需要按组容量折点分片后做精确LP。若该方法最终只是通用区间数据协调与旧sharp solver拼接，创新性可能不足；要过关必须完成精确性证明、误差带敏感性、跨省复算和独立小时标签验证。设计见[V359](V359_SIGNED_MULTISOURCE_RECONCILIATION_DESIGN.md)。

### V359首轮：联合来源LP未优于较宽的单源风险集合，25%误差扩展使τ300全期弃权（2026-10-30）

实现了联合`x`（未知真实分类边际）和`e`（事件小时内分类电量）的LP，而非各来源区间端点简单取并。每个候选K直接检验是否存在满足`q·e>0`、`q·(x−e)≤0`、来源边际区间、总量守恒和容量约束的联合解；零超额边界另检查非风险期仍有正电量。12个小样本塌缩区间场景与V342固定边际锐solver完全一致。

河南2024两来源包络δ=0的τ100/τ300平均宽度为506.91/778.82小时，分别等于监管来源单臂；比NBS单臂478.09/666.91及V358净残差臂492.82/764.09均没有变窄。逐组支持扩大±25%后τ300平均宽度为798.55小时，即11/11期全期宽；扩大±50%保持全期宽。τ100宽度也随来源误差扩大而上升到548.64/590.45小时。

因此V359“联合多源协调可以缩区间”的假说否决。它只可能作为诚实的来源不确定性评测臂，未来需小时真值验证其是否修复单源窄区间的漏覆盖；当前没有覆盖数据，也没有算法预测增益。实现、单元宽度、源码与研究判定见[V359结果](V359_SIGNED_MULTISOURCE_RECONCILIATION_RESULTS.md)、[LP脚本](../scripts/v359_signed_multisource_margin_hull.py)及[结果目录](../results/v359-signed-multisource-margin-hull-r2/)。

下一轮应找第二个至少覆盖12个月的省级独立总量来源，按相同合同重复实验，并优先获得小时CIF验证源。国家统计局分省月度入口能统一目标省技术边际，但本环境直连仍不可用；现有官方NBS接口原始快照尚待补齐。

### V360：官方分省月表的范围核对新增第二省证据，不能与全省监管总量直接拼接（2026-10-30）

用户再次明确国家统计局`fsMonthData`可直接按省查看。既有V320–V322已使用NBS派生的31省月表做过识别实验；此次尝试本环境直接打开官方JS入口仍遇到连接关闭，因此新增一个谨慎的跨来源点核对：2024年12月河南NBS规上发电量305.90亿kWh vs NEA全省327.45，江苏547.10 vs NEA全省577.01，监管办数值分别高6.58%与5.18%。NBS值来自公开转录表，河南/江苏监管值来自各自官方通报，正式数据集还需官方portal原始导出复验。单月两省不足以估计统一修正率，但说明河南源差不是唯一观测到的案例。

判断：官方NBS省级月量是可用的统一公开输入预算；它的规上工业统计范围不能当成全省全口径月供给上限。2月是1–2月累计，且NBS答复说明各月当期值不必然加总到累计值，累计数据也会复核修订；因此基准应保存回测时点可得快照，不能仅抓当前最终历史序列充当实时可得输入。火/水/核/风/光月量能锚定粗统计组成，仍未给煤/气/油火电内部构成及小时调度/风险形状。新核对及官方来源见[V360](V360_NBS_PORTAL_AND_CROSS_SCOPE_CHECK.md)。

下一轮省际独立源搜索继续，但V358/V359不能作为统计边界已闭合的预测证据；正式新地区小时覆盖评测仍需独立小时标签和来源版本快照。

### V361：江苏11个匹配月期均复现全省监管值高于NBS规上值（2026-10-30）

继续完成江苏2024年12份监管办电力运行通报的月发电量摘录，并配对公开转录的NBS地区月表。按NBS报告窗口将1–2月合并、3–12月逐月匹配，11/11期NEA全省值高于NBS规上值；逐期相对差的等期均值6.25%。NEA 12月年累计6703.75亿kWh，比NBS年度累计6330.7亿高373.05亿（5.56%）。但NBS当期各报告值简单和6295.0亿比年累计少35.7亿，官方答复说明企业范围和累计复核使月值不必然可加。因此逐月差额总和不能当作规下供给；V361只增强跨源范围不一致的证据。

原值与每期两侧URL在[江苏配对CSV](../data/china/jiangsu_nea_nbs_monthly_scope_check_2024.csv)，方法边界见[V361审计](V361_JIANGSU_MONTHLY_SOURCE_SCOPE_REPLICATION.md)。NBS值仍是公开转录，需官方入口原始导出复验。研究决策不变：NBS月度粗类是合法输入候选；监管办全省量独立作为来源敏感性；完成小时标签同口径匹配前，不宣称源差校正提升了小时覆盖或区间效率。

### V362：上海官方多目标因子确立标签分轨需求，未形成小时预测突破（2026-10-30）

上海生态环境局/统计局新公开2020–2024年度公共电网平均因子及剩余组合因子。官方方法明确为自然年消费侧指标：分子含本地公用电厂燃烧排放与市外净调入化石电力隐含排放；剩余组合因子还排除已有绿色权益声明的非化石电量。上海碳市场分别用公共电网平均因子确定配额、用剩余组合因子核算外购电，绿电核算为0。

上海公共电网平均值与MEE/NBS省级平均值在2021–2023分别相差−0.1098/−0.0999/−0.0954 kgCO₂/kWh（相对MEE −18.82%/−17.08%/−16.63%）。这是真实的官方来源/口径差异诊断，不能叫预测误差；两类年度消费侧因子不可直接拼成同一训练标签。公共平均与剩余组合自身差距到2024年为0.0139 kgCO₂/kWh，说明基准至少应分别登记标签目标、边界、年份、发布日期、版本和可用时点。

V362实质结论：上海序列可作为单城的官方口径/版本验证轨，不能作为小时CIF真值，也不足以估计通用跨区预测效果。国家统计局动态省级入口本轮仍无法从当前环境直接连接（`ERR_CONNECTION_CLOSED`）；没有新抓取省份数值或真实小时标签。该发现支持论文采用“目标estimand分轨＋合法发布时间账本”的基准设计，但不构成方法/算法突破。完整数字、比较表和限制见[V362](V362_SHANGHAI_OFFICIAL_TARGETS_AND_IDENTIFICATION.md)。

### V363：官方全生命周期技术因子可支持邻近任务，不能替代运行期小时CIF（2026-10-30）

生态环境部等部门公开了2023、2024年全国电力碳足迹因子，按煤、气、水、核、风、光、生物质等技术拆分，并另列输配电。官方方法为技术生命周期排放（上游、建设、运营维护、退役等）加输配电排放后除以全国用电量，目标为kgCO₂e/kWh，而非TransCIF的运行期小时CO₂/kWh。它适合独立的年度生命周期足迹任务，不能拿来充当小时边际CIF或与MEE省级平均CO₂因子混训。

省级安徽2024电力足迹搜索结果提供了可能的第二个地区级官方标签入口，但本轮未取到省厅原始PDF、未核对边界与版本，故不纳入数据集或实验。国家足迹只有两个全国年份，同样不足以评估跨区迁移。完整路线判断见[V363](V363_OFFICIAL_LIFECYCLE_FACTOR_ROUTE_ASSESSMENT.md)。

### V364：NBS派生粗类别省月占比可提前发出阈值风险“不可识别”证书（2026-10-30）

把V240阈值侧构造用于2022–2024年31省、规上口径NBS派生`thermal/total`月占比，共1023省期；显式按Jan–Feb累计处理。若合并火电有效因子支持`[L,U]`满足`L×火电占比<τ<U×火电占比`，则对任意同一小时总发电曲线，都能构造两个完全相同月边际的世界分别实现`K=0`和`K=H`。这是一张完整风险区间证书，不是通过算法预测实际K。

主支持档`[400,1000] g/kWh`下，τ100/300/500分别有64/409/833个省期（6.3/40.0/81.4%）可给出该证书；支持放宽到`[350,1100]`后，τ300上升到679/1023（66.4%）；窄至`[700,900]`时只有18/1023（1.8%）。所以结果揭示的有效控制量是当地火电有效因子的**支持宽度及其相对阈值位置**：更多小时总量曲线本身无法解除跨阈值的月边际不可识别；需要可信本地因子支持或小时组成/调度限制。

这是框架层候选增量：将可证明的阈值可识别性预检作为预测前置门控，可对结构性不可识别的省期拒绝输出虚假窄区间。它是V240定理在用户中国粗类别数据上的直接量化应用，不是新定理或经验预测器；因子支持未经本地标定，NBS数值来自OREO转录而未从国家统计局原始接口重抓，无中国小时真值或覆盖验证。因此只能称**最贴近真实数据预算的可识别性证据**，还未达到算法突破。逐省逐期结果和局限见[V364](V364_CHINA_NBS_THRESHOLD_SUPPORT_CERTIFICATE.md)及[`summary.json`](../results/v364-china-nbs-threshold-crossing-certificate-r1/summary.json)。下一轮要以同一门控对接独立小时标签，检查相对标准条件区间基线能否在不损失覆盖下缩窄区间或合理弃权。

### V365：把全宽识别障碍转成阈值条件化的本地测量目标（2026-10-30）

对每个触发V364证书的省月，求比例混合见证跨阈值的有效热电因子`e*=τ/s`。当前支持`[L,U]`要退出完整`[0,H]`区间证书，最小单侧端点收紧量是`min(e*−L,U−e*)`；其意义是让某一边界达到临界值，不等于测量标准差或实际置信区间宽度。在主档`[400,1000]`、τ=300的409/1023省期中，`e*`中位461.3 g/kWh，最小单侧支持收紧中位60.5、P90 208.8 g/kWh；若保持支持中心对称收缩，对应每端中位30.2、P90 104.4。放宽EF支持至`[350,1100]`后，证书增至679期、24省，单侧收紧中位66.1。

这把“本地EF有用”具体化为按省月和阈值的采集条件：测得的有效EF支持须整体落在`τ/s`一侧，才能排除这条比例混合全宽见证。它支持“阈值条件化采集/弃答”作为自创算法接口，但还没有观测成本、误差模型或小时标签验证，因此目前只是测量目标排序，不能宣称自适应策略提升覆盖或效率。逐期临界值和支持变化见[V365](V365_THRESHOLD_CONDITIONED_MEASUREMENT_FRONTIER.md)。


### V366：省级月占比显著优于全国汇总的结构性预筛（2026-10-30）

在同一NBS派生规上月表、相同`[400,1000] g/kWh`假设热电因子支持和`CIF>300`目标下，用省级本地thermal/total替代全国汇总占比。全国占比为992/1023省期给出比例混合全宽K证书；省级占比下为409/1023，593期（全国全宽证书案例的59.8%）不再满足这条全宽构造。此结果确认省级页面的关键价值是保留区域结构、筛出仍需补充本地因子或小时燃料构成信息的省期。它不是小时预测收益：因子支持仍是未校准情景，原始官方页面接口尚未抓取，小时CIF标签与覆盖/区间分数评估缺失，且官方火电类别仍未细分为煤/气/油。复现见[V366](V366_PROVINCIAL_INFORMATION_GAIN.md)。

### V367：异地小时形状迁移中，本地月结构优于留一地区汇总锚（2026-10-30）

用7个EIA平衡区、2022–2024年245有效区月和175,560小时做leave-one-region-out回测：留出目标区小时代理标签，只让其它6区供给小时形状。若目标区仍可提供完整8燃料月边际，本地月均强度加同一迁移形状比留一地区池化月均强度的小时MAE低110.0 g/kWh，7/7区更优；τ300月超阈小时绝对误差少460.5小时（7/7），τ500仅少9.2小时（1/7），τ700无事件、无区分力。该结果证明全源别月结构在代理域有空间价值，但输入比中国官方5类统计更丰富，故V368再收紧为仅本地火电占比。

### V368：仅本地热电占比的增量在τ300显现，因子情景改变收益（2026-10-30）

保持同一个留一区小时形状，唯一对比是用目标区本地`(煤+气+油)/总发电`月占比或其它6区池化占比形成小时强度水平。在热电EF=700 g/kWh、τ300下，本地占比让每区月平均尾数误差从541.2降到383.4小时，区域等权配对收益153.7小时，七区均优；EF=500时收益188.6小时、6/7区优。可是EF=900时收益119.6小时但只有2/7区优、置信区间触及0；EF700、τ500本地信号略负，τ700无事件。迁移小时形状自身增量很小，主要收益来自本地月水平。

因此已找到一个**有经验支持但尚未达突破门槛的候选**：按当地月热电占比修正迁移水平，并按`τ/s`与有效EF支持的相对位置决定是否输出尾风险估计或保留宽区间。它符合中国更粗的统计信息合同，也解释了阈值依赖；但EF范围未经中国校准、七个美国地区已被查看、标签为固定EF派生代理且尚无区间覆盖测试。下一步须比较局部EF区间传播与强条件分位数/保形基线，在同一未参与校准的地区上报告覆盖、宽度与弃答率。详见[V367](V367_LEAVE_REGION_OUT_COARSE_MIX_RESULTS.md)与[V368](V368_THERMAL_SHARE_ONLY_TRANSFER_RESULTS.md)。

### V369：本地热电占比改善区间效率，但地区尾部仍欠覆盖（2026-10-30）

跨拟合七个EIA区的小时CIF代理区间中，本地非对称残差分位数的90%/95%经验覆盖为87.4%/92.6%，均宽213/247 g/kWh；池化月水平对应覆盖74.6%/78.2%、均宽479/516。局部水平明显省宽，但最差区覆盖仅68.2%/76.3%，不满足稳健覆盖要求。τ300尾数区间均宽仍为月长约79%/81%。当EF支持`[400,1000]`跨阈值时，多数区月区间接近全宽；未跨阈值的少数期较窄。这支持阈值边界作难例预筛，未构成覆盖达标。

### V370：把跨阈值门控直接变成弃答未通过覆盖—宽度门槛（2026-10-30）

对τ300，83%的区月触发门控。跨界返回全宽、其余条件校准后，总区间宽83.3%月长、覆盖99.2%，还不如直接本地区月计数残差区间（宽79.3%、本折经验覆盖100%）。跨界直接弃答只保留16.7%目标，保留子集条件校准覆盖仅75%。τ500保留子集覆盖100%但区间零宽，主要因为这些子集代理标签无超阈小时，具有事件稀少的平凡性。当前 gate 是可信预警器，不是已验证的高效选择性算法。下一步应寻找能提高非跨界校准样本的信息或找到更合适的独立标签/地区，而不能把此次结果包装成成功创新。

### V371：五类组成相似度加权的小时形状迁移未产生实用增量（2026-10-30）

使用目标区本地热电占比做水平、以水/核/风/光/其它占比计算留一区小时形状donor权重，和等权donor比较。τ300时最近3个donor将平均绝对尾数误差从383.8降到380.0小时（95%区域聚类区间改善0.5–7.7小时），仅占基线误差约1%；小时CIF MAE没有改善。kernel带宽0.15/0.30也只有2–3小时收益。τ500收益不足1小时且区间含0。依预设门槛关闭“更复杂地按月组成挑形状”的路线，不再做带宽网格寻优。该轮支持了一个更清晰的研究结论：省级月结构主要带来本地**水平**信息，复杂迁移形状不是当前破局点。详见[V371](V371_COMPOSITION_WEIGHTED_SHAPE_RESULTS.md)。

### V372：省级占比的信息增量存在，但不构成单调收窄或预测突破（2026-10-30）

把V366单点比较扩展为4种火电EF支持×3个阈值。12个组合中，8个本地占比的V364全宽证书数低于全国占比；但证书也会新增，例如基准EF支持τ100时全国证书0、本地64，窄支持τ500时本地又新增56个证书。这些都是**充分证书是否触发**的差异，不是sharp K集合宽度；因子端点与阈值依赖明确。省级面板适合做基准的地域难度分层，现有结果不能称省级输入“消除了模糊”或降低了预测风险。详细表和判定见[V372结果](V372_PROVINCIAL_INFORMATION_ROBUSTNESS_RESULTS.md)。

用户再次指出国家统计局`fsMonthData`页面可直接按省查询。该页面确为应采集的官方入口；当前环境对`data.stats.gov.cn`页面和新版数据API的请求均连接重置，所以V320/V364/V372仍用NBS来源的OREO转录表，而非官方端点的原始快照。下一步数据任务不是增加算法复杂度，而是取得该入口的响应后，完成指标/地区/单位/期别合同和逐省期差异审计，再用V342 sharp solver复算实际K集合宽度。没有原始快照及小时真值，不能将这一支线称为算法突破。

### V373：本地月组成相较全国组成缩窄τ300锐K集合（2026-10-30）

在V341同一2023年30省完整队列上，使用V342固定因子sharp K solver与两种CEC容量版本。只在两臂都满足当地容量的262/330省期配对样本上，τ300本地组成的平均K宽度比全国平均组成窄80–83小时；129期更窄、133期不变、没有配对期变宽。另有68期全国组成反事实不可行，不能并入配对宽度。τ100只有约19小时均值收窄且81期变宽，明确存在阈值异质性。此结果把省级占比的价值推进到实际可行集合宽度，但仍只是假定EF、年度装机上界与规上统计剩余量下的识别，不是点预测、覆盖或算法创新。详见[V373结果](V373_NATIONAL_VS_LOCAL_MIX_SHARP_K_RESULTS.md)。

### V374：全口径范围残差抵消了τ300省级组成收益（2026-10-30）

将V345月度规上范围残差（0至年度差额）与V346宽残差EF/容量松弛加入V373同一组成对照。262个两臂均可行期在τ300下，无论本地还是全国构成都回到796.9小时近整月外包络，V373的80–83小时缩窄完全归零。τ100仍缩窄约23小时，但平均宽度仍581–613小时且81期变宽。此结果表明当前省级组成增量只在规上边界/已固定范围残差的条件模型内成立；想预测全省全口径风险，必须先观测或约束统计范围之外的电量。现阶段破局点更像“最小新增信息与弃权基准”，不是已证实的迁移算法。详情见[V374](V374_LOCAL_MIX_WITH_SCOPE_RESIDUAL_RESULTS.md)。

### V375：省级组成优势需要把月度范围残差约束到很窄（2026-10-30）

固定V373共同可行的262个省期，只改变每月未覆盖电量最大值相对V345年度差额的比例。τ300下残差上界为10%时，本地混合组成比全国平均窄35.7小时；25%时仅8.7小时；达到50%后两臂都为796.9小时整月宽，省级组成增量归零。它把“最小新增字段”具体化为目标月全口径总发电量或等价逐月残差约束。10%只是当前面板、固定因子支持与装机松弛下的敏感性位置，并非已验证的普适精度。没有实际小时标签，仍不能声称覆盖或预测增益。详见[V375](V375_SCOPE_RESIDUAL_INFORMATION_FRONTIER_RESULTS.md)。

数据来源线索：云南省能源局公开的[2024年1—8月简报](https://nyj.yn.gov.cn/xwzx/nyxw/202409/t20240925_3196950.html)同时列“全部发电量”、规上发电企业发电量与水/火/风/光累计值，可作为完整范围累计质量锚。V357对云南累计节点的试验只将τ300等期平均锐K宽度缩窄5.1%、小时加权缩窄4.48%，说明单靠稀疏累计锚还不够；下一步需核实更多省份是否以稳定频率公开同口径全量月值。

### V376：物理共享装机约束未改变τ300近整月风险集合（2026-10-30）

把全省总装机施加为每小时跨全部已报技术与范围外残差的共享上限，使用精确有理数事件超额优化；冗余共享约束逐项复现原V342锐K集合。V374的262个配对省期里，两臂仍全部为796.85小时全月宽，本地/全国组成没有任何区别。由此看，允许月度范围残差取较宽质量与EF支持才是τ300集合重新变宽的主因；只把各技术容量耦合起来不足以解决它。详见[V376](V376_SHARED_SYSTEM_CAPACITY_RESULTS.md)。

### V377：固定零残差后共享装机不带来额外收窄（2026-10-30）

把V376范围外残差固定为零后，本地组成平均锐K宽713.5小时，全国组成796.9小时，差83.4小时，与V373完全一致；共享系统装机相较原独立容量模型平均收窄0小时。和V376比较可见，造成风险集合回到全月宽的关键因素是宽范围残差支持，而非技术间总容量耦合。零残差只是规上边界内的乐观对照，不能代表全口径中国目标。论文增量仍指向数据边界与最小本地信息识别，不是预测算法突破。详见[V377](V377_SHARED_CAPACITY_NO_SCOPE_RESIDUAL_RESULTS.md)。

### V378：上海直接NBS导出核验强化了本地组成价值，也暴露累计值不可拆月（2026-10-31）

用户提供NBS分省月度页的上海火/风/光原始CSV，并确认该省水电、核电为零。11个非空单月期（2025年9–12月、2026年3–9月）的33个火/风/光数值与V320 OREO派生面板全部一致；三类值与五类合计最大差10 GWh。上海热电份额为94.13%–96.86%，均值95.47%；同期全国池化份额为60.17%–67.69%。在V366同一假设EF支持[400,1000]、τ300混合阈值诊断下，全国11期全都触发跨阈值全宽证书，上海11期都不触发，且本地混合均值支持下界全都高于300 g/kWh。全时期阈值扫描显示`τ∈[270.76,376.53)`内全国份额11期都触发、上海份额11期都不触发，说明τ300结果并非单点选择造成。它提供上海本地月混合信息的原始来源核验，并说明全国份额先验会显著误判上海均值阈值位置；该证书仍不是小时K宽度或覆盖结果。

数据合同也得到实际检验：2026年2月当期字段空白，累计值是1–2月；2025年12月火电累计增量与当期值相差1.9亿kWh，2025年11月风电相差0.7亿kWh。国家统计局解释企业统计范围会变、累计值会复核修订，因此累计量与各月当期量不要求相加闭合；不能从稀疏累计值静默反推单月值。详见[V378](V378_SHANGHAI_NBS_EXPORT_AUDIT_RESULTS.md)。这让中国基准的实际采集路径更可信，但只有一个高热电省份且没有小时标签，依然不足以证明算法突破。

### V379：省级份额的信息作用随阈值反转，适合做识别状态图而非“本地优越”指标（2026-10-31）

将同一`[400,1000] g/kWh`假设EF支持和V364充分证书放到2025年3–12月、2026年3–9月的31省单月报告期，共527省期。τ300下全国池化份额对527期全给出全宽证书；省级份额仅278期触发，249期（47.2%）不再触发，且无新增证书。可本地解除的包括高热电份额地区（如上海、安徽、北京）和水电占比高地区（四川、云南、西藏），对应同一阈值下均值支持跨界的不同方向。

本地信息的作用不是单调的：τ100/200时全国证书数为0，本地新出现30/58期；τ700/800时全国也为0，本地新出现248/120期。因而该结果更支持“省级组成×阈值”的双向识别状态地图，而非声称本地信息一律减少不确定性。候选创新是把风险问题的阈值、当地组成、EF支持和合法信息源一起纳入可迁移benchmark，输出哪些省期可识别/仍需补信息的审计结果。V379仍依赖V320 OREO转录、省级范围和未校准EF场景，没有小时CIF标签、预测覆盖或真实sharp K增益；目前是基准设计候选，不是算法突破。下一批原始核验选四川、吉林、福建、青海，覆盖水、风、核、光极端。详见[V379](V379_RECENT_PROVINCIAL_MIX_REGIME_RESULTS.md)。
### V380：更强月度组成相似度校准有小幅分数增量，仍未解决地区欠覆盖（2026-10-31）

基于V369 leave-one-EIA-region-out交叉拟合残差，测试目标月热电份额核与完整8燃料mix总变差核加权分位数，并把强核估计与全局本地分位数插值。τ300、标称95%下，50%插值把区域均区间分数由327.86降至321.04（7/7地区分数更好，7地区cluster bootstrap 95%差`[-11.63,-3.24]`），平均宽度基本持平246.75→246.98 g/kWh；实际覆盖却只有93.13%，最差地区76.58%，仍未达到标称95%。标称90%时分数改善bootstrap区间跨零，且最差地区覆盖下降。

窄核的欠覆盖更严重：8燃料TV带宽0.10使标称95% τ300区间宽度收窄至168.8 g/kWh，但最差地区覆盖仅4.7%；全局/核混合放宽到0.40后，覆盖近似回到基线，效率增量很小。故停止继续调核带宽。V380说明本地组成距离不能可靠预测小时残差可迁移性；平均分数增量虽存在但不能包装成覆盖保持的突破。可留下“全局与相似组成校准保守插值”作为低级基线，主要破局缺口仍是小时调度/来源信息或独立高频标签。详见[V380结果](V380_MIX_SIMILARITY_CALIBRATION_RESULTS.md)。

### V381：上海附件原件复核通过，逐小时负荷候选仍缺独立数据合同（2026-10-31）

用户再次提供上海火/风/光三份NBS导出，并确认水电、核电为零。三份附件哈希均与V378归档原件一致，没有新增月份、变量或独立样本；V378已将五类分母和零值处理纳入，故不重复运行并保留原报告。另审查CEpower公开README：项目称省级8760小时负荷来自电力公司实测，但没有给出数据公司、时段、版本、数据许可或独立误差核验；历史回测还被列为未来工作。它可保留为待核数据候选，当前不能充当实测发电分母或小时CIF验证标签。证据和下一步见[V381](V381_SHANGHAI_EXPORT_RECONFIRMATION_AND_LOAD_SOURCE_AUDIT.md)。

因此下一推进点是多省原始月表的跨结构复现，优先选四川、吉林、福建、青海与上海形成水/风/核/光/火结构对比；这能检验省级粗组成识别状态图是否稳健，但仍不等于小时预测突破。算法晋级仍需要同预算小时标签、强基线和覆盖—效率联合收益。

### V382：公开年度煤/气/油派生表可做额外预算，但与NBS火电总量不闭合（2026-10-31）

发现2020年省级煤/气/油“供电量”技术报告，并核查了其年鉴燃料消费、标准煤折热比例分配和相同厂用电率等假设。上海派生总供电量82162 GWh，高于NBS转录火电毛发电量80350 GWh 2.26%；表内煤/气/油合计76276 GWh，又低于NBS火电量5.07%，并有5886 GWh表内未分到煤/气/油的差额。故年度细分是**边界需建模的额外信息情景**，不能作为完美硬锚。值得进一步测试的候选方法是“年度燃料总量/比例区间＋公开月火电边际＋未解释残差的部分识别传递”；这还只是数据结构与方法假说，没有小时标签预测或覆盖/效率结果。复现与条件见[V382](V382_CHINA_ANNUAL_FUEL_BRIDGE_AUDIT.md)、脚本`v382_china_annual_fuel_bridge_audit.py`。

### V383：年度燃料总量对月度煤/气份额几乎没有边际识别力（2026-10-31）

在接受V382派生年燃料量、并将其与NBS火电年量差显式作未分类残差的条件下，上海2020的11个NBS报告桶×4组月度分配仍有30个自由度。每个桶的煤份额和气份额边际锐界仍分别是完整`[0,1]`；油因全年量小，上界可限制到3.29%，未分类残差上界则仍高达29.1%–88.2%。这证明“把年度结构复制到每月”是模型假设，不是年度边际的统计含义。通用自由度公式不构成理论创新；它把需要新增的最低信息具体到月度内部组成约束，并支持后续对“可行集合＋集合内迁移概率”的实验。仍无小时CIF验证和效率收益，算法突破未成立。见[V383](V383_ANNUAL_TO_MONTHLY_FUEL_IDENTIFICATION_RESULTS.md)及脚本`v383_annual_monthly_fuel_identification.py`。

### V385：上海近期原始月表不等于小时尾风险识别增量（2026-10-31）

将用户提供的2025年9–12月与2026年3–9月上海火/风/光11期直接NBS当期值（并按用户确认水、核为零）与V320对应记录逐项核验后，运行V342精确sharp K solver。火电因子支持400/1000 g/kWh、其余组直接运行因子置零，分别与全国当月池化构成比较。2023装机约束下上海本地11/11期可行，全国构成11/11期均因风电边际超过该省2023风电容量代理而不可行，不能作配对宽度比较；这是本地mix对容量边际一致性检查有用，但也暴露装机代理过旧。去掉容量后，τ300平均锐包络宽732.1小时（约整月），本地构成相对全国构成只窄1小时；τ100无缩窄。虽然上海火电份额为94.1%–96.9%，在宽火电因子支持且无小时调度约束时，风险K仍近乎整月不识别。完整结果见[V385](V385_SHANGHAI_RECENT_SHARP_K_RESULTS.md)。

这强化而非推翻了路线判断：NBS月度粗类别能做省级结构分层与容量一致性筛查，却不能单独产生可信窄小时风险区间。当前没有算法突破；更有价值的下一个技术方向是把信息预算按“粗技术量、同口径当期装机/可用率、小时燃料或直接排放/进口碳流约束”分层，并量化何种最小新增测量才能把近整月sharp set收紧。该方向仍须经独立中国小时标签验证。国家统计局JS入口仍无法从运行环境直连，直接上海CSV可以核验但没有补齐其他省份原始快照。

### V386：上海实际受电量证明消费侧目标需独立输入合同（2026-10-31）

上海市经信委2024年电力平衡：全社会用电1984亿kWh、本市发电1085亿、市外来电899亿，闭合到0；本地发电占54.69%，市外输入占45.31%。同报告称本地新能源发电103.5亿kWh（用电量5.22%），可再生能源消纳622亿kWh（用电量31.35%）。这些官方数字说明，用户NBS技术月表只描述本地生成的一部分，不能单独作为消费侧公共电网因子的充分输入。若固定本地强度、只允许进口平均EF在敏感性档[0,1000] g/kWh内变，消费侧年度均值的不确定宽度已达453.1 g/kWh；这只是场景算术，不是实测EF区间。2025年上海市政府称市外来电仍“接近五成”，未把模糊文字换成精确值。

这个量化推进了路线判断：论文应把V108已定义的generation-based小时风险与消费/网络流因子分成两个estimand轨。前者可用用户月技术边际做低频输入，但当前仍需缩窄小时K界并验证覆盖；后者必须额外取得同口径逐时受电量、送端区域强度、网络损耗与绿电属性。南方电网2026年宣布广西15分钟市地级可溯源数据库，说明真实测量验证渠道存在，但公开报道没有给出可下载样本，不能算已取得标签。V386计算、数据来源与限制见[V386](V386_SHANGHAI_CONSUMPTION_BOUNDARY_RESULTS.md)。当前仍未得到算法/覆盖突破；更好的论文原创主张是双estimand、双信息账本的冷启动识别基准，并在实际小时数据许可到位后验证。

### V387：跨来源年度拆分揭示 Ember 相对误差带过窄，且全局校准会伤害上海（2026-10-31）

将2020年1mi1表中30省煤、气供电量的`coal/(coal+gas)`与Ember估计的同年省级煤气比例对照。两侧均为方法估算，并非直接燃料真值；差异MAE 5.20个百分点、P90 9.86点、最大21.00点，整体偏差为−4.41点。模拟V348相对份额扰动后，±25%只覆盖7/30个1mi1派生比例；因此不能把±25%结果当作跨来源稳健性。省份留一全局偏差修正虽把MAE降至4.19点，却把上海留出误差由2.18放大至6.82点；单一全局偏差存在区域负迁移。用省份留一跨来源半径迁移到上海，2025–26月均碳足迹区间宽度约0.096 kgCO₂e/kWh（校正中心），未校正中心约0.114；煤气完全未知的条件界约0.441，±25%相对扰动约0.072。故有价值的候选方向是由多个来源/口径的分歧生成区域可迁移燃料支持集，并对负迁移扩大区间；但目前只有一个历史年派生对照，没有真实燃料误差或中国小时CIF覆盖验证，尚非论文级方法突破。细节见[V387](V387_CROSS_SOURCE_FUEL_SHARE_AUDIT_RESULTS.md)。


### V388：最小本地新增信息可具体落到“燃料品种×火力发电投入”（2026-10-31）

用户本轮重新提交的上海火/风/光CSV哈希与归档原件一致，没有新增信息；2026年2月当期字段为空，不把累计值拆月。新查到的上海官方能源报表制度205-2表按能源品种记录加工转换投入，并把火力发电与供热等用途分开，且按月度周期填报。这比笼统索取“更详细本地数据”更具体：年度或月度燃料品种×火力发电投入是可定义的信息预算层。但该省级聚合当前未取得，报表字段的存在不等于数据公开；年度公开“用于加工转换”合计还混有供热、炼焦等活动，不能冒充发电燃料份额。新增燃料锚仍无法给出小时调度/排放标签。结果仅收敛了最小数据合同，没有算法或小时预测增量；建议后续按公开五类月量、年燃料投入、月燃料投入、小时燃料/排放四层进行同预算部分识别与覆盖—效率比较。详见[V388](V388_MINIMUM_LOCAL_FUEL_INFORMATION_AUDIT.md)。


### V389：上海官方目标口径差约为近期年度漂移的6.8倍（2026-10-31）

上海公共电网官方年度因子2021—2024自身跨度为14.9 gCO₂/kWh；与MEE/NBS上海省级年度因子在2021—2023同年对照，差值为95.4—109.8 gCO₂/kWh，平均绝对差101.7 gCO₂/kWh，约为前述跨度6.8倍。该差异不是预测误差，因两个官方标签的系统边界、绿色权益扣除和数据版本尚未证实一致。官方公式还说明上海公共电网消费因子由本地火力发电燃料投入、市外净调入化石排放和供电量共同构成；本地燃料投入来自能源平衡表“火力发电”子项，跨区电量及分母来自电网企业统计。说明用户NBS月技术量并不能单独定义消费侧标签。对论文设计的推进是：双estimand与分层信息预算不只是概念上的谨慎，而是由官方量级差异支持的必要设计；目前仍无算法增量、小时真值或跨省外部验证。详见[V389](V389_SHANGHAI_ESTIMAND_BOUNDARY_SCALE_AUDIT.md)。

### V390：目标知情oracle表明“选哪三天”不是主要破局点（2026-10-31）

在V202相同25个EIA平衡区、449地区月、相同3天×4个六小时块×8类燃料预算下，构造直接读取目标月真实分时燃料边际的三日oracle上限。τ300、5%风险门槛下，oracle归一化锐区间宽度为0.74144，均匀选日为0.74362，改善只有0.218个百分点（约1.6小时/月）；25个地区中7个oracle更窄、18个并列。高风险正证书率从6.00%升至9.78%、弃权率相应下降，但该分类增量也依赖不可能部署的目标信息。oracle仍有真实K包含性，但这只是确定性集合覆盖核验，不是概率覆盖率。

因此停止“仅优化选日”的主创新线：在极乐观上限下区间仍几乎不动，实际选日模型不能把它作为方法收益。优先比较同预算新增燃料细分/时间分辨率，以及直接小时风险信息；要求真实中国小时标签才能检验预测覆盖和区间效率。V390仍是EIA估算代理的信息上限实验，不是中国小时CIF预测结果。详见[V390协议](V390_ORACLE_ACTIVE_DAY_UPPER_BOUND_PROTOCOL.md)与[V390结果](V390_ORACLE_ACTIVE_DAY_UPPER_BOUND_RESULTS.md)。

### V392：官方NBS五省长面板确认粗组成分层，但阈值鲁棒性随回溯期显著变窄（2026-10-31）

本轮打通国家统计局分省月度JSON接口，从官方入口拉取上海、四川、吉林、福建、青海2018年1月至2026年9月的总发电量及火、水、核、风、光当期值。共3,150条地区×技术×月份记录；2,580条有值、570条空值。435省月有总量与火电值，411省月六类均完整。1—2月当期字段为空，未用累计值差分补齐。上海附件33个非空火/风/光当期值与官方API逐项一致。官方2,580条非空值和V320 OREO转录值在10 GWh官方显示精度上2,580/2,580匹配；380个原始非精确差全部是太阳能、最大5 GWh，且四舍五入到官方精度后全部匹配。

用官方热电/总发电量本地份额及`[400,1000] g/kWh`因子支持，对τ300月均支持区间阈值证书复算435省月：本地份额148期跨阈，全国池化份额405期跨阈，271期（62.3%）状态不同。上海/四川/青海多从全国跨阈状态移出，吉林多期仍跨阈，福建全部跨阈；本地信息的作用随地区资源结构而异。五类分母与总量分母的状态在411个完整期中只差1期。它支持“阈值条件化、省级识别难度分层”成为基准设计候选。

长回溯也收紧了结论边界：最近55个完整地区月的状态共同稳定区间约`[294.69,376.53)`，而2018—2026的435期共同稳定区间仅`[299.68,300.40)` g/kWh。阈值地图有实际省级信息价值，但固定τ结果对历史结构变化敏感，必须报告连续阈值扫描。V392仍不是预测或小时K结果：官方历史数据没有as-published vintage，五省按既有结构反差选取，NBS与V320不是独立测量源，且没有中国小时CIF标签、概率覆盖或区间效率增益。它是新的**数据可审计性与识别基准增量**，还不能称算法突破。详见[V392审计](V392_NBS_OFFICIAL_LONG_PANEL_AUDIT.md)及[V392快照](../results/v392-nbs-official-provincial-long-panel-r1/)。

### V393：遵守统计发布时滞后，本地粗组成仍识别阈值状态，但不是小时预测突破（2026-10-31）

核对国家统计局实际发布日期后，月初预测不能使用上月构成：2026年1—2月报告于3月16日发布、3月报告于4月16日发布、4月报告于5月18日发布。因此V393以`t-2`作为目标月月初可用的最后月度特征，`t-1`只作为目标月中旬更新敏感性。严格沿用V392跨阈证书定义，状态为`400×热电份额 ≤ 300 < 1000×热电份额`，即假设热电EF支持[400,1000]、其他技术运行EF为零时月均支持区间是否跨越300 g/kWh。缺失值严格保留。月初`t-2`样本345省月，本地滞后证书状态一致率95.4%（灵敏度94.8%、特异度95.7%）；仅五省池化状态一致率33.6%。在2023—2026共同可比的145省月前向时间段，本地`t-2`一致率99.3%，2018—2022拟合的省份静态状态为80.0%，仅五省池化静态状态为60.0%。

这不是小时预测模型效果：上海可判定月份始终为不跨阈，福建始终跨阈，四川和青海多为完全低于阈值，变化几乎集中于吉林；前向样本中本地动态相对省份静态的差异几乎由吉林单次状态转变贡献，五省又是刻意选出的结构对照。池化五省不能代表全国。指标只是月均支持识别，不是小时CIF误差、K区间覆盖或sharpness。当前价值是为“公开月度本地组成作为最小合法信息预算”的时效合规识别层提供初步时间前向证据；必须在真正未见省份、真实小时CIF标签上与全国锚及标准动态基线比较后，才可能支撑方法贡献。上海水电、核电在81个非缺失月份官方值均为零，24个缺失期仍未推定为零；用户上海火/风/光CSV的33个值与API相符。详见[V393结果](V393_RELEASE_LAG_IDENTIFIABILITY_RESULTS.md)及脚本`analyze_v393_release_lag_identifiability.py`。

### V394：31省前向状态识别显示本地滞后组成胜过全国池化，但仍未触及小时CIF（2026-10-31）

把NBS官方当前月度面板从五省扩展到31个省级地区，2018-01至2026-09，19,530条地区×六指标记录，3,563个缺失值原样保留。V392五省3,150行与新抓取逐项完全一致；全省面板与V320转录的15,451个非空重叠值全部在官方10 GWh显示精度半格内，13,182个原始值精确一致，2,269个非精确值全为太阳能且最大5 GWh，按HALF_UP舍入均匹配。该比较是同源转录精度审计而非独立源验证。

沿用V392同一跨阈证书`400×s≤300<1000×s`，月初严格用`t-2`本地份额预测目标状态。在2,139个共同有效省月，准确率89.1%、balanced accuracy 88.8%；排除目标省的其余30省池化锚分别为38.8%、50.2%，整体池化锚39.0%、50.4%。在2023—2026的899个前向省月，本地滞后状态balanced accuracy 87.8%，2018—2022拟合的省级静态状态为80.0%，排除目标省池化锚50.0%。但省级准确率差异很大：浙江55.1%、重庆58.0%，多省因阈值状态长期稳定而达到100%；前向错误集中在重庆、浙江、河北、内蒙古、黑龙江等地区。

这是目前最强的**公开信息预算→均值阈值状态识别**实证增量：本地月组成可刻画国家聚合抹掉的省级结构，并在月初时效规则下优于总量池化和静态区域状态。它仍不能声称可信小时CIF预测：标签由同一发电构成导出，目标不是小时强度；没有中国小时CIF、K覆盖率、区间宽度或proper score。省份留出在此处是逐个排除目标省的池化基线，不等于训练一个算法后在真正未见省份验证。当前论文破局点应收敛为“有限公开本地信息下的阈值条件识别基准”作为可验证中间贡献；要立小时CIF主张，下一门槛仍是独立小时碳标签和冻结算法的未见省验证。详见[V394结果](V394_ALL_PROVINCE_RELEASE_LAG_STATE_RESULTS.md)。

### V395–V399：滞后组成的源区收益未能迁移为可靠小时风险（2026-10-31）

V395在七个EIA代理区用2022训练、2023校准、2024开发测试，月初可用`t-2`热电份额使95%小时区间评分和宽度显著改善，但300 g/kWh尾部计数最差区覆盖只有41.7%。V396按组成条件化风险半径、V397直接预测尾频均未解决计数覆盖。V398按V395冻结规则预测六个外区后再评分，小时覆盖仅77.8%、最差12.2%；LGEE热电份额0.980–0.990，超出所有源区2022份额最大值0.761，属于强外推；PACE份额在源支持内但小时覆盖仍仅72.1%，说明单一月组成不能给出小时风险。V395内部标量范围诊断同样不能简单转成拒绝规则：低组成BPAT虽在源范围之外仍覆盖良好，范围内MISO仍有风险不足。V399将八燃料细分加入小时/月份交互后，虽比无组成模型窄，但对更简单的热电份额模型覆盖及尾部计数更差；也不符合中国月度统计的信息预算。V400对合法滞后热电份额月变化量做源区直接尾频预测，300阈值误差与评分均变差，500阈值窄度略改善但评分与覆盖变差。上海附件与现存归档完全相同，V385已作33值核验和11月精确集合分析；用户本次提供确认而未扩展时间跨度。整体证据不构成算法突破，记录与下一步见[V395–V400状态](V395_V399_TRANSFER_BREAKTHROUGH_STATUS.md)。

### V401：上海本地发电数据与消费端CIF存在近半电量的边界缺口（2026-10-31）

上海市经信委2025年度报告称，全社会用电量2089亿千瓦时，本地发电1091亿（52.2%），市外来电998.4亿（47.8%）；本地可再生发电占本地发电11.9%，全市可再生能源消纳占用电量34.1%。因此用户NBS表对生产端组成有用，但不能单独表示消费端供电结构；用户CSV注释限定规上工业法人。2025年末NBS累计风电+光伏仅42.4亿千瓦时，而全口径年报本地可再生发电129.7亿；差87.3亿，NBS火电958.3亿与报告推算的全口径非可再生发电961.3亿接近，显示统计范围缺口主要落在可再生量（技术归因仍需核实）。在热电EF 400–1000且其他技术运行EF设零的简化支持下，NBS子集热电占比95.76%，全口径年报隐含非可再生占比88.11%，年均生产侧proxy差30.6–76.5 g/kWh（EF=700时53.6）；不是实测因子。上海水电、核电本地统计为零并不意味着市外输入没有水电或核电。即便假设本地年均因子已知，进口因子在0–1000 g/kWh支持下仍使消费端年均因子的进口不确定宽度最多达478 g/kWh；这是边界条件下的代数部分识别，不是预测置信区间。该方向有实际价值，但生产/消费因子差异和跨省隐含碳已有研究，不能把“上海依赖外来电”当作论文创新。进一步将87.3亿千瓦时年可再生缺口仅按年末8.42 GW容量上界分到月、Jan–Feb作为合并桶，在EF=700 g/kWh时，3–12月每月仍有233.9–318.8 g/kWh的生产侧强度可行宽度（Jan–Feb桶222.0）；年度锚并未确定月度水平。候选贡献应是按目标边界和统计覆盖、发布时间建立可迁移的信息预算/识别基准；小时预测要补上逐时进口来源/因子或独立交付端标签。见[V401边界审计](V401_SHANGHAI_ESTIMAND_BOUNDARY.md)。

### V402：小时级中国因子文献不等于独立小时真值，上海附件没有扩展样本（2026-10-31）

用户新提交的上海火、风、光CSV与已归档原件三份SHA-256完全一致，仍是2025Sep–2026Sep的上海规上工业月度当期/累计导出；没有新增月份、省份或字段。上海水电、核电零值来自此前NBS官方快照，而不是这三份附件中的列。V394已从NBS官方接口覆盖31省2018–2026月度面板，因此这次附件核验加强了输入可追溯性，但没有增加样本。

前沿数据审查发现两类高分辨率参考，但都不能当作小时实测真值：2025年《中国工程科学》江苏示例以电网拓扑、分时电量与省间交换计算2023年四个典型日、13个地级市的小时因子；2026年Scientific Data公开31省、五种情景、2025–2060的8760小时预测因子，来自规划/调度仿真，技术验证与年度官方因子比较，不是独立小时标签。清华Zenodo配套数据含省级因子、技术模型与省间交换矩阵，但页面未标明历史小时实测真值。可把这类数据用于明示的“机制参考一致性”压力测试，不能报作真实CIF预测覆盖或泛化精度。

结论进一步收敛：目前成立的论文增量是**发布日期与统计覆盖约束下，公开月度组成对省级因子支持/阈值可识别性的跨省基准**；现有证据仍不足以证明可信小时CIF预测、小时概率覆盖或算法突破。任何高频声称需单独的独立小时交付端或排放标签。按目标类型拆成月/年官方生产侧、小时机制推演参考、独立小时交付侧三层；不得混合评分。完整来源边界和下一关见[V402](V402_HOURLY_LABEL_PROVENANCE_AND_BENCHMARK_GATE.md)。

### V403：固定本地热电份额成为有竞争力的年度因子冷启动基线（2026-10-31）

将上一年度本地热电份额乘以预设热电因子中点700 g/kWh，与2021–2023 MEE/NBS消费侧省级年度因子比较。90个省年 MAE 为54.37 g/kWh，优于同年全国份额 proxy 的124.14；τ300分类89/90，但正类比例达90%，不能只报准确率。2022–23与五组NNLS的配对MAE差为−1.25 g/kWh，省份cluster bootstrap 95%区间[−20.17,16.49]，无法确认点预测优势。以`[400s,1000s]`作支持时，59/90个省年可安全作阈值判定，均无错判；87/90目标落入该支持。目标是年度消费侧因子，不是小时标签，且特征存在NBS历史修订风险。详见[V403](V403_LOCAL_MIX_OFFICIAL_ANNUAL_FACTOR_RESULTS.md)。

### V404：零标签本地物理锚的年度区间在嵌套留区比较中较窄（2026-10-31）

本轮补齐脚本后重建 V404，发现原始未脚本化 NNLS 指标与嵌套协议无法重建，故保留旧文件但不再当作可信主结果。可重建结果中，90%名义覆盖时本地固定锚为29/30覆盖、平均宽234.3 g/kWh；五组NNLS为27/30、287.2，区间分数分别252.2和477.4，覆盖不同，且七区bootstrap差区间跨零。95%时后验测试β=.5本地锚30/30、宽270.2，NNLS仅27/30、宽623.8；本地分数优势明显，但并非同覆盖对比。β=.5仍是在查看目标结果后挑选，且目标为年度消费侧标签，不能称算法确认。当前可复现候选是“本地物理锚 + 区域分离校准”；它仍需未看年份和冻结超参数确认，且不支持小时CIF或正式覆盖保证。见[V404](V404_NESTED_REGION_CONFORMAL_MIX_INTERVALS.md)。

### V405：内层留区选择残差尺度没有超过固定锚（2026-10-31）

把2022标签限定为内层调参、2023统计区作为完整外层测试，重新选择热电份额残差尺度β。7个外层区中6个选择β=0，只有Central选择β=1，没有区域选择β=.5。所选规则相对固定β=0在90%名义下的宏区间分数只改善0.017 g/kWh，95%下反而差0.57；覆盖率相同。事后固定β=.5的95%区间仍最窄且30/30覆盖，但合法内层选择无法重现它，故判为调参选择噪声，停止把幂尺度作为创新方向。结论收敛到本地物理锚/分离校准是强基线，算法创新尚无增量。详见[V405](V405_NESTED_REGION_SCALE_SELECTION_RESULTS.md)。

### V406：嵌套地区交叉验证选择物理锚—NNLS收缩权重，出现覆盖相近的区间增益（2026-10-31）

将预测中心改为`λ×(700×本地热电份额)+(1−λ)×五组NBS NNLS`，每个测试区仅以其余地区2022标签的留区MAE选择λ，再评2023整区留出。6/7外层区选择λ=.75，1区选择λ=1。90%覆盖名义下预测覆盖29/30，与纯本地锚相同；区间宽234.3→229.7 g/kWh，proper interval score 252.2→244.5；95%下覆盖仍29/30，宽285.2→265.8、分数307.5→288.4。选权重减纯锚的七区簇bootstrap区间分数差分别−8.42 `[−12.64,−4.29]`、−20.54 `[−28.53,−11.45]`，6/7区改善。点MAE仅46.85→44.65，配对区域bootstrap跨零；故目前是区间效率候选增量，不是点预测突破或正式覆盖保证。虽然外层标签未用于单区调参，方法族是在此前浏览2023结果后提出，不能将bootstrap当作确认性推断。必须冻结后用新年度/独立地区确认。详见[V406](V406_NESTED_PHYSICAL_ANCHOR_SHRINKAGE_RESULTS.md)。

### V407：前向诊断显示内层选权重不稳，固定收缩仍是待确认候选（2026-10-31）

只用2021标签和2020结构做跨区留出，预测2022年度因子。内层权重选择在90%名义下区间分数271.1，差于固定λ=.75的240.1，也差于本地锚261.2；覆盖同为28/30。固定λ=.75较纯本地锚平均区间窄16.0 g/kWh、分数低21.1；95%下也更窄、分数低31.1。此结果中90%内层选择相对固定λ=.75的七区宏分数差为+42.0，簇bootstrap 95%区间`[2.3,86.9]`，六区中只有两区更好，故撤回“学习权重有稳定增益”的主张。固定λ=.75也不是确认性结果：它是在查看V406的2023表现后选作候选，V407目标标签曾参与此前权重开发，且目标仍为年度消费侧因子。附件上海三份NBS导出哈希与V378/V381归档一致，无新增时段；V378已核明当期值、累计值和统计范围。详见[V407](V407_TEMPORAL_FORWARD_SHRINKAGE_DIAGNOSTIC.md)与[V378](V378_SHANGHAI_NBS_EXPORT_AUDIT_RESULTS.md)。

### V408：阈值侧燃料月量在开发区外近似保留详细燃料边际的小时计数集合（2026-10-31）

在预先冻结的V90/V97开发 roster 之外取41个EIA平衡区，按固定质量门槛得到29区、247个2025区月。前一年粗装机约束仅140/247区月可用；缺失装机约束始终计入全分母并对容量决策记abstain。τ=600时仅输入煤+油月电量，τ=900时仅输入煤月电量，均复用V97固定EF集合构造。无容量时，risk-side与详细燃料边际的区均K集合宽分别为480.36/480.34小时与453.58/453.55小时；600下239/247集合宽逐条完全相同、900下236/247相同，所有其余差异不超过1小时。加同一可行容量约束后，条件区均宽为305.10/299.68小时与212.20/209.10小时，容量可用行中123/140与127/140精确同宽，最大差56/65小时。观测代理K均包含，详细边际集合逐行包含于risk-side集合。

这支持一个更清楚的研究方向：**阈值特异的最低月度燃料信息预算，可否保留可审计的小时尾数识别集合**。开发区外的同源proxy复制让“压缩燃料报告仍能近似保持集合”成为有力机制候选，但不是CIF小时预测优势。容量只覆盖56.7%区月；无容量K集合区均仍约453–480小时宽，阈值1%下容量方案正判定仅0–0.29%、弃权约77.6–79.3%；标签是固定EF生成代理，side-field假设精确且系统并非整体盲测。故不能把“近充分”写成高实用准确率或可信概率覆盖。

新上传的上海火电、风电、太阳能NBS导出在33个非空当期值上逐项精确匹配V392官方接口快照，未增加月份；11个重叠月的官方水电、核电当前值均为数值0。上海火电占统计总发电量94.13%–96.86%，在V392明示`[400,1000]`热电EF假设下，月加权均值支持下界约376.5–387.4 g/kWh，故可判为高于300的粗月均支持状态。但火电仍未拆煤/气/油，也没有小时调度或逐时排放；这批数据不能直接提供V408所需煤+油/煤月字段，更不能识别小时K。完整复核见[V409](V409_SHANGHAI_UPLOAD_RECONCILIATION.md)。

**判断与下一步**：目前最值得继续的不是再堆预测网络，而是做“最小信息字段—识别集合—拒绝决策”基准：同一输入预算比较粗五类NBS月表、阈值侧燃料字段、年度/容量约束及详细燃料上界；对缺失/错报/修订设非零预算，并在独立小时排放真值上评价K集合覆盖、宽度和有效决策。对中国，现有上海上传只足以作低频组成锚。若无法取得独立省级小时真值，论文需把主贡献收敛为识别/信息预算基准，不能声称小时预测突破。详见[V408协议](V408_THRESHOLD_SIDE_REGION_HOLDOUT_PROTOCOL.md)与[V408结果](V408_THRESHOLD_SIDE_REGION_HOLDOUT_RESULTS.md)。

### V410–V411：中国逐小时情景参考下，低频热电锚有水平信号但小时尾部仍不可用（2026-10-31）

新增使用CC BY 4.0 Figshare v3 的中国31省、2025年逐小时规划情景因子作**模拟压力参考**，明确不当作实测小时标签。S1仅用于训练/校准，S2–S5依次作为负荷增长/碳预算情景外推。目标月只给` t−2 `官方NBS火、水、核、风、光月结构；五类和同月总量均需有效，所以每省仅保留May–Dec 8个月。按留一省外训练与逐省留出90%同时区间校准，跨4情景共评31省×8月。这里仍有两项范围不匹配：NBS月发电是规上生产侧，情景因子经省际潮流做碳流计算，且NBS历史接口没有 as-published 版本。

S2中source-only小时曲线的区均MAE为202.3 g/kWh；700×本地滞后热电份额锚降至138.5；五类组成+日历Ridge为141.5，未胜过简单锚。其区均90%月内同时覆盖为90.7–91.9%，平均区间仍宽938–962 g/kWh，且北京锚/Ridge在8/8月同时覆盖为0。τ300/500/700诱导K集合全含情景目标，但宽度约607–735小时；τ300全弃权，τ700下Ridge仅12.5%否定判定、87.5%弃权，没有正判定。故不能把region-mean cover包装成可信可迁移覆盖。

事后再试“本地热电锚定水平＋源区小时月均去中心化形状”：S2–S5区均MAE降至133.5–134.5、RMSE降至147.6–148.6，但同时区间仍宽941.5、最差省覆盖0–12.5%、τ700 K宽632小时、90.3%弃权。该小的点误差信号来自已打开的情景标签，只属探索；没有联合覆盖—效率增量，不能晋级方法。

这轮提供的真正进展是归因更清楚：本地粗份额确有低频水平信息，常规形状迁移即使小幅降点误差也无法解释网络受电/小时调度风险。优先创新方向应为**估计对象分层的最小信息预算基准**：发电侧月均支持、受电/消费侧碳流暴露、小时尾部依次区分；对每级输入报告可识别集合、同预算强基线、最差地区覆盖与弃权。要求信息字段确实公开可复现；没有目标匹配的小时真值时，只能定位识别基准，不能宣称小时预测突破。详见[V410协议](V410_CHINA_SIMULATED_CIF_SCENARIO_STRESS_PROTOCOL.md)、[V410结果](V410_CHINA_SIMULATED_CIF_SCENARIO_STRESS_RESULTS.md)与[V411探索输出](../results/v411-level-shape-scenario-exploratory-r1/)。


### 2026-10-31 仓库整理与来源更正

四日成果、当前证据优先级、停止分支及投稿缺口集中到[四日成果报告](reports/2026-10-28_31.md)。根README与文档/脚本入口已整理，保留历史协议和实验路径。小摘要原样复制到Git内快照，大产物不移动。整理旧V194–V200时发现V410五情景标签早已打开，因此V410的S2–S5只能称本轮未用于拟合/校准的开发复测，不能称项目首次未知情景确认；详见[V410–V411更正](V410_V411_PROVENANCE_CORRECTION.md)。原协议和机器结果保留，不倒改历史。
