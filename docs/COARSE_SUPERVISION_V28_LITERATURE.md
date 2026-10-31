# V28：多尺度概率预测的近期原始文献核查

2026-10-26，Sol。针对性核查，不是系统综述或已复现实验；不会因换成扩散模型就宣称创新。

[Huk等，JMLR2026，Probabilistic Rainfall Downscaling](https://jmlr.org/papers/v27/23-1381.html)：本轮只核原始出版摘要。其两阶段分别拟合条件边际及空间依赖，并用censored Gaussian copula和评分规则估计依赖。因此“把水平分布与依赖分开建模”本身已有成熟近邻，不足以作为我们的原创点。空间降尺度与燃料能量比值不是同一映射，不能直接移植其覆盖解释。

[Huang等，2026-10-14预印本v1，Generative modelling for multivariate spatio-temporal downscaling of monthly climate fields](https://arxiv.org/html/2610.19885v1)：核全文§1及§2.1–2.2，不宣称读完附录或复现分数。其训练使用ERA5日场，另有月平均场，并通过月条件生成日尺度场；训练/验证与2015–2023评价分开，采用日场autoregressive上下文。**不能仅以“月到细尺度、概率生成、时间一致性”为新颖性**。其月条件与目标日对应；本任务只取得滞后已发布月统计，未来粗状态还需预报，且CIF为潜在燃料能量比值。该区别是任务边界，不是原创机制证明。

[Molinaro等，Universal Diffusion-Based Probabilistic Downscaling，v3 2026-04-20](https://arxiv.org/html/2602.11893v3)：核§1–3、摘要和原始版本页；版本页标ICLR2026 workshop，不能称主会论文。它在配对粗/细再分析上训练，再用于不同上游预报，明确把非唯一降尺度建模为条件分布。解耦上游预报与细尺度生成也不是我们的独有想法；未据此比较或复制其性能。

V28真正先检验的是：源区火电内部比例可利用较长期粗标签，而小时调度仅有较短fine覆盖，二者训练队列是否必须同步。保持普通Ridge/ET、q、残差池与本地MG预算的时间窗消融只是必要强控制；即使有效，更多标签与较近期样本的收益也不能算新算法。潜在方法贡献必须进一步解决异步监督后条件关系和风险迁移失配，并胜过完整同预算普通概率参考，而不只是重命名two-stage/coarse-to-fine。
