# 五类粗月报条件下的整国任务式训练协议

日期：2026-11-01，运行前冻结。目标是填补“训练时模拟中国信息条件”中的具体空缺：在每个外层整国留出内，用其余国家构造伪目标 episode，比较 episodic domain-generalization 训练和同结构 ERM；模型推理只读取五类滞后月份额、其历史摘要及日历。该实验是可证伪的算法基线，不预先称为新方法。

## 目标和边界

- 数据：已查看的 Energy-Charts NL、ES、BE、AT、SE；2022小时标签训练，2023上半年来源校准，2024外层整国评分。
- 标签：同一固定排放因子生成侧小时 CIF 代理。不是实测/官方CIF，也不构成中国精度验证。
- 目标信息：五类月发电量（火、水、核、风、光）聚合为份额，精确使用`t−2`到`t−7`六个月；日历特征及其与当小时的交互。包括预测年内已经滞后发布的月统计，但每个预测月只读到`t−2`及更早月份，绝不读取当月或未来月份。火电输入将所有不可在中国五类口径中单独观察的燃料及其他可再生一并计入“火电”，避免给模型第六类目标统计。
- 月份额由当前修订小时序列回算，只模拟两个月时滞；不是历史as-issued发布回放。目标2024小时发电和标签只在预测文件封存之后用于评分。

## 对照

1. 五类等预算的比例强重组：训练源小时燃料组成与小时形状，目标只提供滞后五类月统计。
2. 日历/粗月 Ridge、粗月 HGB：普通强统计模型，同样使用五类特征。
3. 小型同结构多层感知器的 ERM 与 episodic domain-generalization（MLDG式）训练。episode 的meta-train由外层源国构成，meta-test为轮换留出的一整个伪目标国家；内层一步仅用meta-train标签更新虚拟参数，outer loss计算在伪目标的粗月输入和小时标签上。测试时不更新参数，不访问目标小时标签。

两种神经训练共享网络、优化器、训练步数、种子、国家均衡抽样和输出范围，仅改变训练目标。三个固定种子。目标区间用同一外层源国2023H1残差绝对值90%分位数校准；不保证跨国覆盖。

## 报告与判定

按国、按月报告MAE、日内形状MAE、覆盖率、区间宽度和区间分数；同时列国宏平均和最差国。主要比较MLDG对ERM的配对变化，并与五类强重组及Ridge/HGB比较。单一平均点数、相对神经弱基线改善或靠加宽区间提升覆盖，均不算成功。晋级需要跨国方向一致、宏MAE至少改善5%、覆盖不下降且区间分数不恶化；五国只属开发集，不据此打开未查看确认国家。

## 文献边界

MLDG以模拟训练域/测试域偏移的episodic目标训练，是既有领域泛化方法；本实验若有增益，只能说明该训练范式在粗月CIF迁移中的适用性，不能单独主张算法原创。整国留出要优于逐小时随机划分；这是防止同一区域和时间序列泄漏的关键设计。

- Li et al., *Learning to Generalize: Meta-Learning for Domain Generalization*, NeurIPS 2018: https://homepages.inf.ed.ac.uk/thospeda/papers/li2017metaDG.pdf
- Li et al., *Episodic Training for Domain Generalization*, ICCV 2019: https://openaccess.thecvf.com/content_ICCV_2019/papers/Li_Episodic_Training_for_Domain_Generalization_ICCV_2019_paper.pdf
- Janssens et al., *TMAML: Temporal Model-Agnostic Meta-Learning for Cold-Start Time Series Forecasting*, arXiv 2026: https://arxiv.org/abs/2610.04547
