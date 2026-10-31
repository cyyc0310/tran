# V27：同月合并类别预算下，普通监督模型能否解除供体比例支持限制

2026-10-26，Sol。运行前合同。用户最新澄清所有可再生与不可再生电源类别均有月发电量，本轮按完整电源类别能推同口径总量处理；不默认有煤／气拆分，不要求年度细分。美国受控MG仍为POS＋核水风光，POS=煤气油other，是代理映射，不认证中国统计中的具体类别／全口径／发布vintage。V26条件年度草稿未运行并继续PAUSED，不执行部分分母缺失分支。

V25合并类别失效不能直接推为信息不足：重加权供体ratio与插件均值受源比例凸包限制，普通监督映射及折外残差仍未检验。本轮只是固定ordinary强控制，**不宣称原创方法、覆盖保证或完整强概率模型比较**。七已查看源2025H1、原1145有效日；未知地区保持关闭。

## 不变数据与唯一新增拟合

父清单 `docs/evidence/monthly-transfer-v25-verification.json` SHA `740e8b1f18c93ec4bb63136da59cbb026ff24233587c17ca1569b2e6d90f5ad3` 及其闭包逐项绑定。直接复用V25 MG law draws、conditional IDs/fallback IDs与R1六源proper库，不重拟合月预报，不重新选择kernel/seed，不新读proper小时文件。源训练是已有bank归档中的月组成监督；外层评估源不在其训练库。年度数值不用作任何模型输入，父证据闭包中后验年度文件只是hash继承，不作特征。

每outer，从允许六源proper bank按(region, year, month)去重，同月donor_mix须exact一致。X仅五个可见组成POS＋核水风光，y为POS内部coal/gas/oil/other比例。POS=0月不存在r标签，排除该fit标签并记录，但不删除bank或小时评价。源实际月组成作为训练X、隐燃料比例作为训练监督响应；不是当天已发布的本地粗预报特征。查询X是已封存MG预测粒子，粗预报不确定性沿粒子传递，不使用查询隐藏真值或M8 forecast。未引入本地月总量作为独立新特征，故不能称完整利用了全部公开绝对量。

所有final和各fold权重为 `N/(R*n_region)`，源地区等质量、区内每月等质量、均值1；R和N都依该fit队列重算。两固定模型：Ridge(alpha=1,fit_intercept=True)，ExtraTreesRegressor(n_estimators=128,min_samples_leaf=2,max_features=1.0,random_state=17023,n_jobs=1)。不标准化、不调参、不按2025分数选特征。ET点模型是训练比例的凸平均，属于非线性内插参考；不能把它当成点外推证明。Ridge在simplex投影后可能超出训练r凸包，但是否发生必须实际诊断，不能预先断言。

每个源的比例残差由严格leave-one-source-out模型产生：当前outer已排除，在其六源中再排除该源的所有月，五源拟合后预测该源月。delta=真实源r−投影前普通回归预测r（与query使用同一投影前定义）。不得用final六源模型的训练内残差作校准。来源是现有源监督，并非目标fine或未查看区。LOSO只是防止直接训练内乐观，不给分布外覆盖保证。

## 七臂与匹配关系

1. `MG-recompose_conditional`：原V25 raw/fine/IDs精确复现。
2. `Ridge-residual_anchor`。
3. `Ridge-residual_recompose`。
4. `ET-residual_anchor`。
5. `ET-residual_recompose`。
6. `Ridge-plugin_anchor`。
7. `Ridge-plugin_recompose`。

plugin的比例为 `Proj_simplex4(fit6(X_querydraw))`；residual为 `Proj_simplex4(fit6(X_querydraw)+delta_effective_donor_month)`。若有效donor月POS0、且queryPOS0无需r，残差取0并flag；正queryPOS按原MG fallback必须有正POS有效供体。目标无额外年度context，不能从V26引入年度比例。

每粒子输入的MG POS总份额draw和其它四个份额不变；按归一比例分配后的POS总和允许正常浮点求和误差，改变的是POS内部比例。有效POS供体有fallback时为j，否则i；LOSO残差所属地区/月、POS q、支持mask都随同一j，非POS q和support仍i。原donor IDs、fallback IDs不改变。训练月键、样本数、权重、fold排除、point预测、折外预测／残差和投影前后值全部留档。缺源fuel时q=1沿用显式默认，不伪称有该fuel的真实形状支持；逐粒子报告不支持的POS能量比例和组内EF超出原供体EF范围的比例。

每对anchor/recompose共享fine_draw和q。加性锚公式 `rawCIF_i + EF·(fine_draw−donor_mix_i)`；ratio为原物理重组。先重建原MG，raw、fine与IDs/fallback均exact，否则中止、不seal；没有tol降级或按结果换算术。原MG背景新增列只作诊断，不改变旧分数。

## 门槛、评分与可证伪解释

全181起点×128粒子×24小时，7臂×原5倍率0.75/1/1.25/1.5/2=35曲线；裁边／spread、全部指标、有效日与V25相同。全七区预测seal后才读取七已查看源2025H1响应评分。预定40,075日行、245地区行、35宏行；6个匹配对比×7源×5倍率=210对比行。

六个对比为每个新recompose对其anchor（Ridge residual、ET residual、Ridge plugin），Ridge residual对Ridge plugin，Ridge residual对MG、ET residual对MG。每区日均再七区均，宏90–95%、worst≥85%；不合格fallback仍false，不扩倍率。固定factor配对与各臂按源规则选factor都报告，CRPS/width/IS/level/shape/上下漏齐全。

若只缩窄失覆盖、只加宽cover或某源改善另源崩溃，不晋级。若普通回归／LOSO残差解除MISO缺口，应归因模型支持与信息利用，不称月表信息量自动增加，也不称新算法。若仍失败，只否定这些固定普通控制，不证明所有MG方法不可能。若仅plugin与residual差异，可能涉及中心／分布／q关联／投影，不能只叫方差收益。

未来只有在完整同预算强概率参考上满足既有≥5%CRPS下降、width不增、IS下降且覆盖合格，才另立原创机制与冻结确认合同；本轮不会自动打开未知区。完整异方差／条件分位数模型、年度only、真实中国字段映射/vintage均未完成。科学价值是先检验有限源比例支持是否被误当信息论障碍。

prepare/predict/score三阶段独占写，不覆盖失败；三份预审先完成再prepare。独立实际审计不得导入V27 producer，应重fit全部final/LOSO模型、检查月份去重／fold／权重／特征预算、独立projection/effective q/路径/评分与首尾hash；所有失败保留，原receipt不回填。
