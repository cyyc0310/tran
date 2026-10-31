# V42 后续判断：先补已知强控制

2026-10-27，Sol。只读科学备忘录；尚无足够独特、可归因的新机制。本文件不是V43合同，以下blend与诊断均为候选草案，未运行、没有实际结果。

V42显示容量的作用有明显地区差异：扩池SizeCap-ET在MISO显著改善，在BPAT/FPL发生负迁移；扩池B0 Size-ET的宏观结果反而退化。这支持研究“容量增量何时能迁移”，但尚不能区分调度差异、登记边界错配与模型错设，也不能将整包扩池收益归因于单一源覆盖因素。

## 最小候选顺序

1. 可先用封存EF粒子考察固定全局blend，例如Expanded B0/B1、Legacy/Expanded B1之间的少量预锁权重，保留原端点。逐粒子混合后计算CIF，均值、中位数全报；权重和完整比较表须在新合同中固定，不能按地区评分挑选。
2. 比较必须保留原最强B0 Size-Ridge及CoalOilUpper，不能只把退化的Expanded Size-ET当强参考。现有CoalOilUpper是旧训练范围定义的封存规则；如果后续重算expanded范围的容量锚或gate，必须另行预锁，不能说V42已经做过。
3. 若仍值得继续，可检验“源外B1−B0收益的方向，能否由合法粗历史和库存特征预测”。候选是同412月内、outer与inner双排除的月级比较误差及邻域条件异质性诊断；源均衡、按地区检验，不能把412个月或共享数据的配对称为独立域，也不回到六个平均偏差上堆复杂修正器。

若源外收益方向缺乏稳定性，或改善仍集中少数源、EF与CIF不能同时匹敌已有强参考，应停止当前B1纠偏分支。新合同需要明确可证伪判据；本备忘录不先决定更多架构或精确性能阈值。失败只能否定当前表示与迁移假设，不能证明所有B1方法不可能。

若转而探索年度实际煤/油发电信息，应明确列为B2预算的信息价值问题，不把额外响应信息当算法创新。本轮合同不包含这些新字段；原小时、保留地区和新增供体2025细分继续关闭。

## 已核原始论文与实际读取范围

以下均访问官方全文PDF，并阅读所列相关主文；未通读全部附录，未将其理论保证直接套用到当前数据。

- Hoffman、Mohri、Zhang，NeurIPS 2018，[Algorithms and Theory for Multiple-Source Adaptation](https://proceedings.neurips.cc/paper_files/paper/2018/file/2e2079d63348233d91cad1fa9b1361e9-Paper.pdf)。读§2–4，尤其论文第3–5页的分布加权组合、条件分布差异项及估计分布下的界。多源密度加权已有先例；输入覆盖不能单独认证预测可靠。
- Amortila、Cao、Krishnamurthy，COLT 2024，[Mitigating Covariate Shift in Misspecified Regression with Applications to Reinforcement Learning](https://proceedings.mlr.press/v247/amortila24a/amortila24a.pdf)。读§2.1–2.2、Proposition 2.2及DBR公式/定理。即使条件均值稳定，模型错设也可能放大协变量偏移误差；Ridge退化本身不能证明条件机制改变，DBR也已有先例。
- Tsai等，AISTATS 2024，[Proxy Methods for Domain Adaptation](https://proceedings.mlr.press/v238/tsai24b/tsai24b.pdf)。读§3–4及§6.2，尤其PDF第3–4、9页的结构假设、completeness、positivity与真实数据多域适配失败。登记容量有解释性不等于满足因果proxy识别条件；不能仅凭proxy命名主张识别。

本次仅形成候选科学判断，未修改producer、正式运行、主结果或封存输入，未重新拟合，也未写入任何新实验成绩。
