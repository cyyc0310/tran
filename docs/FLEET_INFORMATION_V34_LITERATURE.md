# V34：本轮比较与相关工作的边界

2026-10-26，Sol。此记录写于V34真实拟合前，用于限定创新主张，不输入模型。

[Electricity Maps的Tier C / GPZD官方方法](https://www.electricitymaps.com/data/methodology)已面向没有小时记录、只有月度或年度量的地区，结合详细地区训练数据、天气、地理与容量信息进行时间分解，并用优化对齐各mode总量。因此，月报加装机、迁移学习、非负和守恒这些部件的组合已经有直接先例。当前任务的不同约束是POS组内细分没有本地标签；这是一项需要检验的任务差异，尚不是新方法。

[Schivley等2018年的原论文](https://www.cmu.edu/ceic/assets/docs/publications/published-papers/2017-and-2018/schivley-2018-environ-res-lett-13-064018.pdf)在Methods中用分燃料发电量除以容量和小时数计算容量因子。它使用的信息比B0粗月报丰富。容量归一化和学习利用率应作为普通基线，不能由命名产生新颖性。

[Singh等，UAI 2025，Learning from Label Proportions and Covariate-shifted Instances](https://proceedings.mlr.press/v286/singh25b.html)研究完整标注源实例与目标bag标签并存的hybrid LLP，给出泛化界并比较LLP和域适应基线。本次实际核读论文页摘要，PDF链接读取失败，未审核证明。它提示“源细标签＋目标聚合标签”已有成熟问题表述；我们的粗燃料量既不是小时CIF均值，也不完整给出四类比例，不能直接套用其保证。

[Jin、Egami、Rothenhäusler，PNAS 2025，Beyond reweighting](https://www.pnas.org/doi/pdf/10.1073/pnas.2427181122)研究可观测协变量漂移对未知条件漂移的预测作用。已核正文第1、3、4节及公式(4)、(9)：标准化漂移可用于刻画未知条件变化的尺度，其区间需要对漂移比率作分布假设或校准。论文对象是跨站点效应估计，不能据此认证小时CIF覆盖。但它直接排除了把“用可观测漂移预测不可观测误差”这一泛化想法本身当作我们的创新。

下一步应由结果决定：若库存只修正月度平均水平，就把它作为强基线保留；若压力或库存失配能在六源训练、外源留出中预测剩余误差，还须与普通异方差或条件分位数模型比较。只有证据表明一种针对粗统计未识别方向的机制，在同B1预算下稳定改善覆盖与区间效率，才值得进一步论证方法贡献。七源反复探索不等于独立确认，未知地区继续关闭。

## 首次模型运行后的口径核查

[EIA在2025-05-27的官方说明](https://www.eia.gov/todayinenergy/detail.php?id=65344)明确指出，电厂月年报与BA小时报告包含的机组不同：后者限于BA管理或可见的机组，可能漏掉配电侧未计量资源，也可能包括境外计量机组。文中建议通过Plant ID与Generator ID连接新公布的930机组名单。原文同时说明新调查在2024年启动，因此不能把当前名单直接回填为2022等早期起点已知的成员映射。

已核[930A当前填报说明](https://www.eia.gov/survey/form/eia_930a/instructions.pdf)第1、3、4页，其目的正是明确860中的哪些机组进入930报告；这与860的Plant BA字段并不等价。[930当前说明](https://www.eia.gov/survey/form/eia_930/instructions.pdf)第6页还区分实际燃料与未知时采用主燃料的填报方式，允许部分满足保密条件的类别报OTH。年度主燃料容量与小时燃料标签可能错配。因此本轮容量只作库存代理；没有根据这些一般机制认定BPAT或任一源的具体失效原因。当前说明条文也没有全部认证为2025各起点的历史版本。
