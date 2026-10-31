# V35：零训练条件积分诊断

2026-10-26，Sol。正式计算前固定，仅读已封V34的七个源区、42查询与原35评分月；不训练模型，不改预测，不打开小时、旧目标或真正未知地区。V34仍按其原合同解释，不能由本轮事后诊断改写已冻结评分。

## 数据和复现

固定父清单为`docs/evidence/fleet-information-v34-verification.json`，SHA `c9e6a6f4f91465407268d5dbe7a0ba56eedfaf8fecc1a5bad5c643ff4259437e`。核其完整闭包、当前环境、五个代码/测试/合同、独立代码与数学预审后，以exclusive写新目录和开始收据；失败保留，不覆盖。全部计算成功后才写表及completion，输入首尾哈希不变。只消费已绑定的V34 NPZ/JSON、旧Size-Ridge系数及原月标签路径，不解析原容量档，不调用fit。

查询只读origins、keys、arms、MG_draws、EF_draws/CIF_draws及其原point、query_feature_Size/SizeCap/IntensiveCap/DirectEF。原标签仍用同截止日、coverage和口径规则，42行支持须与V34逐字节相同，35个EF均已定义。地区内先平均、再七源等权；ISNE只有1月，不据单月判断全年稳定性。35月有时间/地域依赖，全部诊断为描述性证据。

## 粗组成分布

每个可评分月计算POS均值、中位数、总体标准差、均值与中位数误差；分位点固定为0.05/0.10/0.50/0.90/0.95，使用numpy线性分位数。报告80%/90%中心区间及粒子min/max包含率，端点包含；不使用epsilon扩宽。less/equal计数按精确比较，经验midCDF=(less+0.5×equal)/128，只是有ties的描述量，不声称有限样本均匀性或做独立性检验。

POS经验CRPS为mean|X−y|−0.5mean|X_i−X_j|。G5经验energy score用相同形式及五维Euclidean距离，均使用128²有放回经验对；不把它们解释为CIF区间评分。另以最近合法月的G5作无需拟合的点对照：POS绝对误差及G5 Euclidean误差。零标准差保留，不作除法或补epsilon。

两项分数均采用越小越好的loss约定，是已有proper scoring rules，不是新指标；参考[Gneiting与Raftery，2007](https://sites.stat.washington.edu/people/raftery/Research/PDF/Gneiting2007jasa.pdf)及[IPP原文Table 1的CRPS表达](https://arxiv.org/html/2309.10083v2)。经验分布分数的数学性质不替代本任务中的联合或条件校准验证。

输出35行case、7行region、1行macro。均值或边际包含率接近名义值不证明联合/条件校准；较低proper score也不认证跨源泛化。

## 点预测目标

对11个已封臂都计算原均值点和现有粒子的中位数点，保留CIF/EF绝对、平方、有符号误差，以及地区RMSE的七源均值。原mean points与粒子均值核对后仍使用原封存point，不能改旧基线。输出770 case、154 region、22 macro。

条件均值适合平方损失，MAE的Bayes点是完整条件分布的中位数；但本研究粒子只传播粗组成不确定性，未包含全部内部EF风险。因此这些粒子的中位数只是普通摘要对照，不认证为真实CIF的条件中位数，也不构成算法创新。

## 固定Ridge条件head诊断

只选已有系数即可精确重建的四臂：Size-Ridge、SizeCap-Ridge、IntensiveCap-Ridge、DirectEF-Ridge。不重新fit树或任何回归。核原feature names、scaler、coef/intercept、outer身份与封存查询，先复现全部42查询的原EF粒子，绝对容差1e−9、rtol=0，超差整轮失败。比例头沿原simplex投影，DirectEF沿原[0.4,1]截断及1000缩放。

之后仅在35个有标签月，把head前五个currentG5特征换为实际粗组成，过去上下文及参数不变，检查EF残差。输出140 case、28 region、4 macro。这个oracle使用额外未来信息，不是部署方法；不构造真实G输入却仍乘预报G的混合CIF，也不列入公平CIF排名。oracle增益不能单独诊断测量错配，真G下偏差也不能单独归因跨源迁移。

## 判读

本轮没有新的方法筛选、显著性阈值或未知地区授权。先看MG组成是否有明显描述性失配、真实G条件下的EF偏差是否仍在，以及mean/median损失差异能否解释排名。三者不能用同一组35月完全识别。若需要新机制，另在真实拟合前固定小规模对照；不根据本轮结果调宽区间来制造覆盖。

最终独立审阅重算核心公式和数据支持即可，不再建立另一套训练或大型审计流水线。研究目标仍是同预算可信转移及覆盖—效率增量，本轮本身不证明方法突破。
