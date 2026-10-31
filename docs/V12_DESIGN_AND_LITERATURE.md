# V12 设计与原始文献审查

2026-10-24，Sol。运行前审查；不构成实验结果。只读既有 V11 结果、方法实现与公开原始文献；未读取新地区小时标签。审查起点见 [V11 结果](WEATHER_RECOMPOSITION_V11_RESULTS.md)：W donor 对经过 B 回归替换的 donor 明显改善，但对 native 的 CRPS 仅改善 0.62%，差值描述区间跨零，PACE/LGEE 小时覆盖仍失败。因此 V12 首先检验严格 native 基点上的天气增量，再比较标准条件概率模型；当前组件本身不支持原创算法命名。

## 1. 严格 native 基点与差分天气

设 donor 轨迹为 `q = a * s`，每个燃料的 `a = mean_hour(q)`、`mean_hour(s)=1`。定义 `delta(x)=shape_W(x)-shape_B(x)`。天气仅更新：

`s_new = simplex24(s + tau * [delta(query) - delta_OOF(donor)])`；`q_new = a * s_new`。

`tau` 表示天气强度，建议与 spread 倍率 `kappa`、分位数概率符号分开。R1 的天气强度固定为 `.25,.5,1`。无 W/`tau=0` 必须直接返回原 q，不能先投影、标准化或重新拟合 B。`delta(query)==delta_OOF(donor)` 同样应保持 q。主 B 是共同支持的精确 native；B global/hetero/QRF 是另外的概率参考，不能声称其情景等于 native。

donor 的 B、W 两预测器都须对其地区折外；外层留区源地区从所有内层形状模型、donor 库、组成模型、缺失填补、尺度、残差库和 QRF 排除。固定同一 donor 身份、组成情景、随机数及 spread 设置，才能将主 B/W 差解释为天气干预。日幅度为零保持零；unsupported 燃料保持既有 flat prior，不由天气创造支持。

逐情景核查燃料日幅度和共同 donor 身份。simplex 可改变天气增量及未解释形状误差，也可能在原 donor 的零发电小时创造正值；必须报告投影率、增量衰减、原零值上的新增质量。日燃料总量守恒不意味着 CIF 等权日均值或目标月组成精确守恒。CIF 有逐小时随机分母，不应把合法月组成 EF 点积强加为日等权 CIF 均值。

## 2. 两轮是同一开发过程

R1：比较三档差分天气 donor，并完成 B/W global、双通道 hetero、QRF 参考；固定 native 与简单锚。R1 只用源 2025H1 留区分数选择 tau 和每个预定分支的 spread。源 gate 为宏覆盖 `[90%,95%]`、最差地区 `>=85%`；合格者按 CRPS 选。若无人合格，仍标记失败，按事先写清的覆盖违约函数及固定 tie-break 选择诊断候选，不得把诊断候选改称过 gate。

R2：只继承 R1 源选 tau，不能用 R1 目标分数改变它。条件 QRF 的全局混合固定 `p in {0,.25,.5}`，B/W 预算共享相同 p 候选集、支持、树数和 spread 网格；global/hetero 不套用此 p。源选 p 后冻结两轮全部目标预测，再打开六个已查看目标 2026H1 分数，是最清楚的执行顺序。

复用源 2025H1 作 R1 选择及 R2 选择，使 R2 是条件于 R1 的开发扩展，不能称两个独立确认。七源、六个已反复查看目标及六个月块 bootstrap 不会因增加情景/配置/轮次数而增加独立样本数。两轮网格、回退、停止规则必须在首轮源分数前一并冻结；如后续根据新结果增加网格，明确记为下一次探索。模型选择准则自身也会过拟合，选型偏差可与算法差异同量级；固定配置后的 bootstrap 不包含这项开发不确定性。[Cawley 与 Talbot，JMLR 2010](https://www.jmlr.org/papers/v11/cawley10a.html)

## 3. QRF 参考的准确名称和概率合同

QRF 原文以森林邻域权重估计完整经验条件分布，再反演其 CDF 得到分位数；取树均值的分位数不能替代这一过程。叶内保留完整响应观测、每树归一化再平均是关键。[Meinshausen，JMLR 2006](https://www.jmlr.org/papers/volume7/meinshausen06a/meinshausen06a.pdf)

V12 可采用固定 `RandomForestRegressor(n_estimators=128,min_samples_leaf=20,max_features=.7,bootstrap=True)`，响应为源 2024H2 留区端到端 24 小时误差。24 输出共享均值损失分裂，再以同一校准日索引抽整条残差，形成合法的联合经验条件分布；边际分位数从各小时加权 CDF 取得。应称“多输出均值分裂、叶加权的 QRF-style 整日残差参考”。这保留抽中残差日的小时配对，不等于已证明目标真实 copula 被识别，也不是逐小时独立拟合 24 个原始标量 QRF。

专门面向多变量完整分布的森林已有原始研究，并使用不同于普通均值 CART 的分布分裂。V12 这个可执行强参考不能据此代表所有条件分布方法已经穷尽。[Ćevid 等，Distributional Random Forests，JMLR 2022](https://www.jmlr.org/papers/v23/21-0585.html)

约定 `b_i = 1 / n_region(i)`；所有训练校准日穿过每棵已拟合树。第 t 棵树对查询 x 的叶权重为：

`w_ti(x) = b_i * I[leaf_t(X_i)==leaf_t(x)] / sum_j(b_j*I[same_leaf])`；`w_cond = mean_t(w_t)`。

这是全训练校准日叶分布，含该树 bootstrap 未抽中的日；不是仅 in-bag 叶分布。全局分布 `w_global=b/sum(b)`。R2 精确混合 `w_p=(1-p)*w_cond+p*w_global`，再抽整日，属于分布混合。不能把两条残差路径按 p 算术平均，后者会改变并经常收缩方差。conditional leaf 内的地区占比不一定相等；它表示使用地区平衡先验后的条件质量，不能宣称每个叶都严格等地区。

本机只读核查：执行环境 `../tran-stable/.venv/bin/python` 的 scikit-learn 为 **1.9.0**。该版本 bootstrap 按传入 sample_weight 的归一化概率抽行，树拟合使用抽中次数；不会再将原 sample_weight 乘入一次。后处理上述全训练叶 CDF 再使用 b_i 是公开声明的分布估计合同，不能再暗加 bootstrap multiplicity。其加权均值一般不等于 `forest.predict`。应封存 sklearn 版本和拟合/叶质量策略，不可套用其他版本的内部实现假设。官方 API 支持多输出、返回每树 leaf ID，并提供抽样索引；`predict` 仍是树预测均值，不直接输出条件分位数。[scikit-learn 官方 RandomForestRegressor 文档](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)

这一 residual 合同使用确定性合法组成与重组中心，误差已经包含组成预测误差；不可再叠加完整粗情景/donor 风险。若拟合尺度或条件森林时再次使用校准误差，其预测评估必须保持外层地区排除；源内诊断校准覆盖若涉及同一误差的重用，要进一步折外或显式标为训练诊断。

条件分位数与“90%覆盖保证”是不同命题。CQR 用独立 proper/calibration 划分对分位数区间作校准，其经典结论是交换性下的边际覆盖；不是任意条件覆盖，也不会自动跨季节、跨地区成立。只有来源区间的上下界并不定义完整 CRPS 分布。V12 直接叶 CDF 有完整分布，但若未实施相应 conformal 步骤就不要称 CQR 或继承该保证。[Romano、Patterson 与 Candès，NeurIPS 2019](https://papers.neurips.cc/paper_files/paper/2019/file/5103c3584b063c431bd1268e9b5e76fb-Paper.pdf)

## 4. 必须保留的结论边界与诊断

源、目标均保留宏平均及逐区 CRPS、90%覆盖/宽度/IS、最差地区，另报日水平 L 和中心化 D 的 CRPS/energy score。PACE/LGEE 不得被宏覆盖掩盖：V11 PACE 日水平覆盖约90.56%而小时覆盖约80.88%，不能预设只放大日水平就足够；LGEE 全局误差库曾达到约99%覆盖却显著变宽且 CRPS 恶化。

实用性能 gate 与原创机制结论分开：源 gate 合格；已查看目标宏覆盖至少90%、最差地区至少85%；相对 native、简单锚及同预算 hetero/QRF 强参考要求至少5% CRPS 改善、配对差描述区间上界<0、宽度不增加、IS改善。未满足则明确失败；即使满足，当前差分控制、尺度回归、QRF 和分布混合仍是标准组件，只能支持进一步确认，不能自行变成原创机制证据。

不得按 PACE/LGEE 目标残差调整参数；可以报告合法特征支持、源地区残差质量、尾部、投影率、日水平/形状分解等预定诊断。源与六目标通过上述标准之后，仍需另行冻结天气预报代理的地点、时效、版本、缺失及独立确认合同，才考虑新地区。此次审查不授权打开新地区。
