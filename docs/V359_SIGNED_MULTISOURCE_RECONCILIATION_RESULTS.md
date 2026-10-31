# V359：联合来源LP不缩窄河南高阈值风险区间

完成日期：2026-10-30。协议与方法见[V359设计](V359_SIGNED_MULTISOURCE_RECONCILIATION_DESIGN.md)，实现为[联合LP脚本](../scripts/v359_signed_multisource_margin_hull.py)，逐阈值/期/因子K集合在[结果目录](../results/v359-signed-multisource-margin-hull-r2/)。

## 算法实现与核对

把潜在技术边际`x`和风险事件期内的技术发电质量`e`一起作为LP变量。对每个K，约束：

- 每类`x`位于NBS与监管源区间之内，且总和等于监管月总量；
- `0≤e_g≤x_g`、`e_g≤Kc_g`、`x_g−e_g≤(H−K)c_g`，确保组内出力上限可实现；
- 风险期排放超额`q·e>0`，非风险期超额`q·(x−e)≤0`。

这直接检查是否存在一个来源协调后的技术边际和小时分配支持K。对阈值边界，还另查非风险期超额等于零但仍有正发电的情形，避免零出力时段被当作有效的低CIF小时。thermal EF端点400/1000、τ=100/300均运行；小时总数8784。12个小型塌缩来源区间场景、共12组因子/阈值比较，联合LP给出的K集合与V342既有固定边际锐解完全一致。LP严格超额检查容差为1e-7，数值算法的普适精度尚无形式证明。

## 消融结果

来源基础盒按逐技术两来源值的最小/最大形成；放宽情景在此基础上将下界乘`1−δ`、上界乘`1+δ`，并维持官方总量守恒。所有基线使用同一2023滞后CEC容量、阈值和EF端点。

| 技术边际支持范围 | τ=100协调集合平均宽度 | τ=300协调集合平均宽度 | τ=300期级观察 |
|---|---:|---:|---|
| NBS单源基线 | 478.09 h | 666.91 h | — |
| 监管月报单源基线 | 506.91 h | 778.82 h | — |
| 两源之间包络（δ=0） | 506.91 h | 778.82 h | 与监管单源端点宽度逐期相同 |
| 包络扩大±25% | 548.64 h | 798.55 h | 11/11期均为全期宽 |
| 包络扩大±50% | 590.45 h | 798.55 h | 11/11期均为全期宽 |

对照V358的“总量逐月锚＋NBS类别＋净残差组”，τ=100/300宽度为492.82/764.09小时。V359两源包络在基础范围下分别比它宽14.09/14.73小时，25%扩展后τ300全期弃权。两源协调臂在基础及扩展支持下均没有任何统计期比NBS或监管单源集合更窄。

## 决策

V359否决了“用两个来源间包络提高区间效率”的当前主张：来源矛盾需要诚实传播，但一旦这样做，区间至少与较宽来源相同，轻度扩误差就让τ300完全不可识别。这是数据质量/方法边界的负结果，不是新预测器的增量。

它仍给出一条可能有论文价值的评测问题：在真实小时标签上，单源较窄区间是否因错把类别来源当真值而失去覆盖；多源协调的额外宽度是否换来覆盖修复？无小时标签就无法回答，也不能称协调集合“更可信”。下一步先找第二省复现有符号类别差，再检查是否存在同时可得NBS、监管月报与小时CIF的独立数据。若没有标签，该方法只能留作基准数据口径臂。

## 文献定位更新

联合LP本身不能作为论文创新点。异源数据部分识别/融合已经在NeurIPS 2025论文中形成sharp bounds和敏感性前沿框架；更早的统计匹配文献也分析来源误分类下的部分识别区域。因此V359的“LP把源间不确定性传播到K”是电力CIF问题上的应用扩展，是否有方法新颖性需要逐式查重，当前证据更支持否定独立算法主张。

相邻应用也在逼近：2025年中国近实时CO₂研究比较月度统计、年度能源公告和组合更新策略，发现单一数据源的及时估算有明显误差，组合公告与月度数据改善了年/月误差；2026年Scientific Data已发布覆盖中国31省的2025–2060逐小时CEF规划投影。因此“用粗统计推出中国省级小时因子”已不是宽泛空白。更可守的价值可能是：针对从未有小时标签地区的统一信息预算基准，明确区分统计量预测、物理系统情景投影和部分识别，并用逐源版本/统计范围账本解释覆盖失效。前提仍是有至少一种独立小时标签验证模型区间覆盖。

- Lanners et al., [Data Fusion for Partial Identification of Causal Effects, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/dd4e79c2228c43954cc074abf2f85a07-Abstract-Conference.html)。领域不同，但说明“融合异源信息并构造识别界”不是本研究独有算法概念。
- Di Zio & Vantaggi, [Partial identification in statistical matching with misclassification](https://doi.org/10.1016/j.ijar.2016.12.015), *International Journal of Approximate Reasoning* 82 (2017), 227–241。
- Hu et al., [Notable uncertainties in near real-time CO₂ emission estimates in China](https://www.nature.com/articles/s41612-025-00991-4), *npj Climate and Atmospheric Science* (2025)。作者比较月度统计、年度公告和组合动态更新，直接提示来源频率/发布时点/口径差会改变估算误差。
- Li et al., [High temporal and spatial resolution projected electricity carbon emission factors of China from 2025–2060](https://doi.org/10.1038/s41597-026-07272-6), *Scientific Data* (2026)。这是31省逐小时CEF规划投影，不等同于无标签新地区的经验预测验证，但压缩了泛化的“中国小时CEF模型”新颖空间。
- Wang et al., [Inherent spatiotemporal uncertainty of renewable power in China](https://doi.org/10.1038/s41467-023-40670-7), *Nature Communications* 14 (2023), 5379，提供中国省级风光小时预测误差分布；是小时形状/误差机制的邻近数据，不是CIF小时真值。
