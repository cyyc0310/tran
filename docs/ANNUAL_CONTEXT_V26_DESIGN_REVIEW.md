# V26：过去年度细分的最小条件信息价值试验

2026-10-26，Sol。只读科学设计建议；没有运行真实新预测，没有解析新的小时训练标签或目标／未知区数值。保持旧封存文件不变。实际V26 protocol和代码尚待稳定后三审，本文不提前批准生产执行。

**建议先固定五臂：原MG背景＋年度固定比例的additive／ratio＋年度锚加普通源比例残差的additive／ratio。** 暂不加年度趋势或年龄匹配算法；先判断新年度信息本身、标准传播映射与源变化先验分别能否解释增量。不是新机制或已证中国数据渠道。

## 1. 信息条件与已知失败线索

用户仅有公开月／年统计，但具体字段未确认。V26假设在五类合并月表之外，还能取得过去年度八燃料发电细分；年度fine-fuel统计是新增的**粗频率本地信息**，不能说MG不加信息就恢复水平，也不能将耗用量／产量当发电份额。源proxy按V24已审核完整region-year派生；完整12条月记录仍只覆盖原eligible小时，不是官方全年普查。

[V25结果](MONTHLY_TRANSFER_V25_RESULTS.md)显示MG在MISO明显低估水平，而2023年度POS组内EF649.565超出outer源比例支持max542.949。该观察是事后诊断；凸包只限制latent比例及其EF线性函数，不是完整CIF路径上限，不证明2025等于2023。它支持一个可反驳的信息试验，不能当成已成功方法。

未来查询只能使用满足period_end、available_at<=origin的年度记录。2023完整七源proxy年在模拟2024-01-31发布；2024仅五源完整，模拟2025-01-31后才可用，ERCO／ISNE因缺月继续使用更老完整年。FPL2022不完整，不填缺月、不偷用未来2023。不把年度复制成12个观测，也不认证这些模拟时间是中国实际年报vintage。

年度读取先按地区和字符串时间筛合法记录，再转数值；每个source-donor context独立用其自己的origin与合法年度，而不是用query当时可知的更新年表回填历史。

## 2. 所有新臂共同保持什么

- 仅七个已查看源2025H1；七outer库排除对应评估源的fine。无新目标或未知区授权。
- V25已审核的MG五维law、128粗draw、供体IDs、POS fallback IDs、有效q、原MG raw及父hash全部重用；不重拟合MG，不改变路由／日历核／seed／倍率和评分支持。
- 训练／预测只读已封R1 proper和V25归档，不新parse小时fine；全七区预测先seal后，评分才读取明确允许的七源2025H1响应。R1 bank不含2025评价真值，不能以旧CRPS表代替新路径评分。
- 每个ratio／additive配对用完全相同fine_draw；不叠加另一套完整fine残差、年度误差law或额外粗粒子。原MG是较少信息的背景，AY-fixed才是同年度预算下的关键普通参考。

V25归档未直接保存sampled_q时，必须从同一父库、i/j及置换规则重建已声明有效q；不能将NPZ保存布局误差当算法变化。先复现原MG raw／draw／IDs，失败则保留诊断并关闭后续seal。

## 3. 年度比例与五臂定义

P={coal,gas,oil,other}，年度完整mix为m^Y；POS质量>0时，年度比例

`a_f=m_f^Y / Σ_{k∈P}m_k^Y`，`f∈P`。

它是过去年度POS内部燃料比例，不是整区CIF，也不是未来月份额。query固定锚a_t取最近已合法发布的完整年度。对于粒子，沿用MG的POS粗draw g_P与四个零EF组draw；AY的fine_mix在P内为g_P r，P外不变。

| 臂 | r的定义 | 配对解释 |
| --- | --- | --- |
| 原MG conditional | 原V25同donor的组内比例 | 必须精确回到较少信息背景 |
| AY-fixed additive | r=a_t | 用rawCIF_i+EF·(m_AY−m_i)整日平移 |
| AY-fixed ratio | 同一r、同一fine_draw及有效q | 与上行比较普通线性／比值映射，不独享年度信息 |
| AY-sourceResidual additive | `r=ProjSimplex4(a_t+δ_i)` | 普通源组成偏移，不命名新算法 |
| AY-sourceResidual ratio | 同一r、同一fine_draw及有效q | 与上行匹配传播；与AY-fixed比较变化先验 |

有效POS donor为j（若V25对i触发借用）或i；令其proper月比例为r_d、其真实donor日期为t_d、地区为s_d。必须使用

`δ_i = r_d − a_(s_d, latest released at t_d)`。

不可用j的比例配原i的日期／年度，也不能拿source2024完整年在H1 donor时点尚未知的值算δ。proper r_d作为源监督响应可来自部署前已知的整proper月归一化；这不是H1当天已发布逐日预测误差，需保留回顾性source-teacher语义。

该加法和simplex投影是标准组成残差传播。δ未作零均值约束，所以它同时改变中心、散布和δ–q关联，投影后均值还可进一步改变。AY-sourceResidual相对AY-fixed不只是“风险方差增益”；即使得分提高，也不能仅据此宣称新的依赖机制。记录未投影向量、投影L1幅度和活跃边界等诊断。

### 最终回退规则

1. target没有合法完整年度、或其年度POS质量为0：两类AY都回到原MG比例并记录target_missing／target_zeroPOS，不把过去零POS当未来物理零。
2. target年度比例有效，但有效source donor没有合法年度、或其年度POS为0：该粒子取δ=0，sourceResidual退到AY-fixed；**不退原MG**。分别记录source_missing／source_zeroPOS，δ=0不表示观察到了零漂移。
3. 有效q沿V25。若POS借j，支持mask的POS列也来自j，其余列来自i。年度新比例可能把质量放到source未支持燃料，该q仍为旧默认1；必须计`Σ_f m_f 1{effective_support_f=0}`及相应粒子／小时比例，不能伪称新质量有可迁移源形状支持。

固定年度比例可能显著改善latent水平，却把煤／油放到平坦先验q上。保留原q不意味着CIF noise保持不变；改变mix后R的中心／形状／尾部都会变化。两种传播都须报告clip前后及unsupported质量，避免把数值裁边或默认q当作新风险认证。

## 4. 为什么暂不实现age-matched或trend

H1 proper从2024-01-25开始。2024-01-31后多数donor使用刚发布2023年度；此前仅有少数日期可能使用老2022年，FPL该年还缺失。2025Jan查询使用2023年，其发布年龄约336–365天；Jan31后有2024年者变年轻，ERCO／ISNE则继续变陈旧。故“全源有年度残差”不等于“拥有查询age的足够独立支持”。

应报告query/source context的年度、period_end age与release age，按预先固定年龄段给context数量、独特source-year／source-month、ESS／回退率及匹配差。重复相同年度的日记录不能算数百个独立年度漂移样本。年龄匹配不能仅靠放宽kernel后宣称完成；older query age超出source支持应显式标记。

若此轮显示真正增量，后续可另冻结：在同一proper donor上构造“故意不取最新、但当时已发布的上一年”上下文，覆盖更老age。此举是合法减信息，不是读未来；但同一fine响应的多context版本高度相关，不能把它们当独立新增样本或让某源因版本多而增加权重。缺年必须保留明确处理。那时再和标准age-aware状态空间／趋势模型比较，不在本轮看结果后补一个有利版本。

年度趋势至少需两个合法完整年；query和H1proper的可用年数并不一致，FPL2022缺失会混入不同回退。因此本轮先不把trend当已完成强控制。固定年度锚是最低限度的强ordinary参考，若它已解释增量，继续开发的目标应是剩余漂移和概率效率，而非给标准锚重新命名。

## 5. 预定可反驳解释与停止规则

建议五臂×原五factor，共25条曲线；source队列、统计约定及完整七区seal规则不变。主比较分两层：

- **信息层**：AY-fixed相对原MG，新增过去年度细分是否在coverage合格时改善CRPS／IS而不增宽。该对比跨信息预算，只能称条件信息价值。
- **方法层**：AY-sourceResidual相对同映射AY-fixed是否有独占增量；各自ratio−additive保持fine_draw一致。该对比仍同时可能包含mean／spread／关联变化，不自动成为原创机制证据。

保留V25完整M8强参考并标更丰富信息；不要只和失败MG比。按臂选factor时仍宏90–95%、worst≥85%，无合格则failed fallback；同时给固定factor匹配结果和逐区上下尾／level／shape。不同粒子、重复日和多轮已查看源选择不构成独立确认，月份CI仅作描述。

停止／降级解释提前写明：

- 年度固定锚已得到同等收益：普通年度信息解释了改善；没有额外残差机制增量。
- sourceResidual只通过大幅增宽提高覆盖，或IS／CRPS恶化：不晋级，不继续用倍率挽救。
- latent EF更接近过去年度但2025评分无改善：不能说年度锚不可能有效；只能说此固定先验／漂移假设没有兑现预测价值。
- MISO改善而其它区变差、老年度query／unsupported质量集中过差：保留完整异质性，不按地区开专门规则或用结果选择年度。
- 只有覆盖合格，且同预算残差法相对固定年度锚达到至少5%CRPS下降、width不增、IS下降，才值得补完整强概率模型、age／trend以及下一确认合同。即使达到，也不是创新或自动target授权。

本五臂没有穷尽年度信息下的直接回归、异方差、QRF、EMOS或条件状态空间强控制，不得据此说超过全部概率方法。年度信息本身可能足以缓解V25的latent支持缺口，但其是否适合小时预测、保留效率以及能否实际获得中国字段，仍须分别验证。

## 6. 设计结论

按上述有效donor联动、as-of年度与回退、原MG重放和评分门禁，五臂小实验在科学上可运行，并比同时加趋势／年龄模型更易归因。当前应先验证“新增年度细分＋普通锚”能否产生合法预测价值，而非继续堆叠区间层。组成偏移、比值重组、源残差及粗到细预测已有近邻工作；潜在贡献仍是严格信息合同及不可迁移风险的可证伪处理，不能预先宣称原创。

本设计已纳入根代理最终回退约定：缺source年度时δ=0退fixed，缺target年度时两类AY退MG。实际protocol／代码稳定后需再按字节审查；本文不替代正式preflight和独立实际数值审计。
