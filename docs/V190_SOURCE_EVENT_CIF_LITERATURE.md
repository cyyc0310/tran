# V190文献与已有实现边界

2026-10-29，Sol，准备阶段检索；以下使用原论文/作者机构网页，不以搜索摘要证明不存在先例。本次是可证伪候选，不声称已证明原创。

|原工作|已存在内容|V190不得重复包装|
|---|---|---|
|[Meinshausen 2006 QRF](https://jmlr.org/papers/v7/meinshausen06a.html)|森林邻域权重估计完整条件分布|经验叶分布、quantile森林不是新算法|
|[Tibshirani等2019 weighted CP](https://arxiv.org/abs/1904.06019)|在适用covariate shift和密度比条件下加权校准|ESS或重加权不能单独叫原创，也不能无条件移植保证|
|[Cauchois等 Robust Validation](https://arxiv.org/abs/2008.04267)|f-divergence不确定集下的稳健预测集|未识别目标shift半径不能冒充稳健保证|
|[Barber等 Beyond Exchangeability](https://arxiv.org/abs/2202.13415)|加权和非交换条件下的稳健推断/相应假设|不能把源块删留称jackknife+覆盖保证|
|[Wang等2021 class-conditioned Wasserstein DRO](https://arxiv.org/abs/2109.03676)|类别条件源分布与Wasserstein稳健优化|类别内均衡或Wasserstein本身已有先例；此处不声称首创类别条件稳健方法|
|[Sagawa等 group DRO](https://arxiv.org/abs/1911.08731)|组最坏风险及正则对泛化的重要性|源最坏、group reweighting不能自动当贡献|
|[Fine-grained robust conformal inference](https://arxiv.org/abs/2402.13042)|分开已识别协变量shift与有界条件shift|源计数不能识别未知目标条件shift|
|[ECTR 2026](https://arxiv.org/abs/2601.22944)|环境内tail重权与共享不变性风险、KL浓度控制|不能声称环境条件tail重权是新方向；V190不实现其训练目标/TV不变假设|
|[LADDER 2026](https://arxiv.org/abs/2607.26458)|固定源模型、无目标标签的推断源重权|无标签source weighting已有原作；V190不假定能以目标style识别应信任的源|

原作至少摘要/方法定义已核；不是全部相关文献穷尽，也未独立复现其论文代码。若本轮出现正结果，仍须针对最接近方法完整理论与实现复审。当前不赋予method_breakthrough=true。

库内V119/V140已有支持缺口fallback；V159/V160已有标量/活动/响应/CDF监督的经验叶重组；V164/V165已有月锚world重权/粗分布；V172/V173已有目标零概率赔率和矩保护；V174/V175已有proper源OOF PIT和CRPS/组max再校准，均不得抹去。V190只检验新限定组合：在相同合法物理ratio atoms和同一固定kernel上，保存一个源事件层总质量、仅改变层内源贡献，按**本次query的CIF分布transport影响**选层；它是否提供普通全局源均衡、ESS和source-spread以外的价值，需要同预算实证反驳。恒等式本身是基本混合CDF性质，不称新定理。

源事件分布不识别目标未来事件概率，源支持单一/缺失时也不能从计数给出未知类别质量。V190保存缺失/单源标记，不制造伪事件或以零质量称风险稳定。所有有限样本覆盖均仅开发评价，无distribution-free新域保证。
