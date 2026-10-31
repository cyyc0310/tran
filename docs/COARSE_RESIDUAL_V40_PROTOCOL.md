# V40：粗残差与能源结构的同预算校准对照

2026-10-27，Sol，实际校准前固定。普通强基线试验，不预告原创性。父V39清单SHA `d6dd3c74e2c3c094008dee1be78934f9ff2e6afc1cde6a5b764cbe83fe5ccb26`。复用42个五源Month profile，不重跑V39，不新增fine监督键或小时/未知地区读取。

## 同规模profile ensemble

每outer T使用V39中六个inner profile，分别排T和s。源s的校准标签与特征都对应排T/s的同一profile；预测T时也对这六个profile逐component应用相同校准器，再等权平均EF。未校准、均值、D、S、DS五臂全用这个ensemble，不能把五源残差校准器直接贴到六源V38 head上。

每outer六个源级样本、等权。标签是V39固定late12月平均`EF_true−EF_pred`，不读early真实EF，也不把154个月当154个地区。源曲线含部署2025-01-01截止合法取得的其他源late监督，所以校准是回溯构造，不是2023时点前推。将late12月均值修正用于单月，仅称持续平均偏差修正，不声称horizon完全匹配。

## 合法过去12条月报特征

源pseudo-origin固定2023-07-01；目标仍42个2025-01至06月初。每个来源先按source、period_end/available_at不晚于origin选择最近12条2022年起粗记录，再转换总量和核/水/风/光份额。POS用补集，绝不转换当期coal/gas/oil/other细份额。不得取late实现量或真实POS分母作特征。

每历史月m独立使用m.year−2的V33 OP库存及其发布检查，不用query/pseudo-origin的年份回填整段历史。令q=C/sum C、z=POS_MW/sum C、xhat=q*u_profile(month)，d=z−sum xhat。C总量及sum xhat须有限正；零容量不是物理缺席证明，不加epsilon。特征固定为：D=mean12(d)，S=mean12((EF_pred−400)/600)，其中EF_pred=(xhat·[1000,400,800,400])/sum xhat。没有交互项、天气或额外状态输入。

源窗口按元数据预查SHA `99c09880b9705cfa75da8ab0418b43c541172919f575fca93ff4b2338f4904cb`，七源均2022-06至2023-05；需实际重建后核对。目标12条可能不连续且陈旧，全部保存keys、end/release、月龄、跨度和月份组成，不补月，不称源/目标窗口同分布。

## 固定五臂

名称为EnsembleNone、ResidualMean、ResidualD、ResidualS、ResidualDS。后四臂每outer分别拟合一次，共28个新小校准器；None不拟合。Mean仅截距，D/S选相应一列，DS选两列。

响应为源late EF偏差/1000。所用列以该outer六样本计算均值和总体标准差；严格常数列记录mask、scale=1、训练和查询标准化列均置0，不补epsilon。设计矩阵Z含截距，固定求解`(Z.T Z + diag(0,1,...)) theta = Z.T y`，即SUM平方损失、alpha=1、截距未罚。记录scaler、常数mask、设计、系数、normal-equation残差；不得按结果改正则、增加输入或删源。

每个query/component基线EF由该月C和profile给出。raw_EF=base_EF+1000*预测偏差；None raw_EF就是base_EF。所有臂先逐component固定clip至[400,1000]，再六component等权平均，保存raw/clipped/clip flag及完整ensemble。CIF粒子仍等于该ensemble EF乘原128个MG_POS粒子，不改变MG，不把component散布当完整风险分布。均值/中位数都从原粒子计算。

## 完整比较与执行

原七源/42query全部预测、特征和参数封存后，再读取原35个合格2025评分标签。原V38的1330 case参考数值精确保留，新增350行，共1680 case、336 region、48 macro、336去单源敏感性、42 support。ISNE仅一个评分月，旧参考的原信息预算和源响应差别继续披露。内部EF与发电侧固定因子CIF代理都评价，不能只靠相反偏差抵消的CIF。

只有DS优于S，才支持D在结构信息之外提供增量；只胜Mean不够。需同时看EF/CIF、逐源及去单源敏感性。所有五臂完整报告，不调阈值、挑月或授权新地区。即便出现点增量，也不自动构成方法创新或小时覆盖证据。

输出exclusive，prepare/predict/score唯一实际运行；固定五个新代码/测试/合同、已封profile/标签、直接粗表/容量/MG/参考输入hash和环境。两份独立预审通过后运行；预测started、完整seal和score输出建立hash关联，完整预测验证必须先于评分标签。独立临时公式重算特征、28个校准器、component/clip/ensemble和全部评分，预锁atol1e−9/rtol0，原1330参考numeric exact。不另建大harness，不重跑已通过的旧阶段。原小时、旧目标、未知地区和方法冻结资格保持关闭，长期目标active。
