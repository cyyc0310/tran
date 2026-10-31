# V30文献核查：上下文、地区特有因素与连续域迁移已有强近邻

2026-10-26，Sol。针对性原始文献核查，不是系统综述，也未复现或比较其分数。以下定位不改变已经锁定的V30特征、同队列学习器或信号规则。

[Brenner等，Learning Interpretable Hierarchical Dynamical Systems Models from Time Series Data，ICLR2025正式论文](https://proceedings.iclr.cc/paper_files/paper/2025/file/d4c961804d08e55d898cce944206d455-Paper-Conference.pdf)：实际核摘要、§1及§3.1–3.2。它以共享群体参数与低维个体特征生成不同系统的动态模型，训练共享参数和个体特征，并讨论少量新数据下的迁移。因此“共享＋地区特有因子”“低维环境状态”本身不是我们的新颖性。本文是动力系统重构，不是公开燃料月量反推小时CIF；个体观测／目标序列与我们隐藏燃料、无本地小时标签的预算不同。未核全部实验、附录或复现结论。

[Cai等，Continuous Domain Generalization，NeurIPS2025主会官方摘要](https://proceedings.neurips.cc/paper_files/paper/2025/hash/35c1d69d23bb5dd6b9abcd68be005d5c-Abstract-Conference.html)：本轮只核正式摘要、原始arXiv版本信息，未读完方法／证明。该工作处理时间、地理及社会经济等连续域因素，用结构化参数迁移与descriptor gate应对不完整描述。因此“连续统计年龄＋地区状态＋门控／transport”不能仅靠组合名称声称原创。我们尚未证明合法月度context足以描述隐藏燃料条件偏移；现V30只检验普通模型能否利用这些信息，不能援引其理论作为CIF覆盖保证。

[Hong等，Unify and Anchor: A Context-Aware Transformer for Cross-Domain Time Series Forecasting，2503.01157v1](https://arxiv.org/abs/2503.01157v1)：本轮只核原始摘要与版本页，页面仍标conference under review，不当作已录用主会。其以频率分解与外部上下文域锚支持跨域预测。因此“用context作为迁移锚”已有直接近邻；把Ridge／ET多加历史和总量也只是普通特征控制，不能直接算原创方法。未获取其完整训练预算或复现性能，不声称同预算击败该模型。

## 对论文立意的实际含义

更有待研究的部分是：在严格发布时间和粗统计预算下，哪些隐藏燃料状态可由可见上下文预测，哪些仍依赖不可检验的迁移假设；这些差别如何进入小时风险，而不是把所有误差统一当条件噪声。V30即使有oracle-current-group条件改善，也只说明有可利用的统计信号；需要去除oracle、接合法未来MG预报和小时端强参考，才可能支撑任务与机制主张。

本轮保留简单模型、同队列与对照来定位信息来源；未因看到这些论文就添加VAE、Transformer或域门控。是否值得设计自己的机制，由正式结果、区域代价和前瞻预算验证决定。
