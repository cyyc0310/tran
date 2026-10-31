# V190：源事件层的CIF风险影响与概率law稳定化

2026-10-29，Sol，在V190任何拟合/性能评分前冻结。基于V189已查看油电例外，不是独立确认。直接检验完整CIF law、90%覆盖/宽度/CRPS/Winkler，不重做零分类；文献边界见[V190文献](V190_SOURCE_EVENT_CIF_LITERATURE.md)。普通重权不自动原创。

## 时间/地区/样本合同

沿用V188 cutoff=2024-07-01与原六折proper源和158 H2 test查询，仍27地区（Q3 78月26区/Q4 80月27区）。源小时只原asof4096预算中截点前且当前源M已发布的Apr–May子集，17,262/16,417/17,630/17,649/18,542/19,368小时，不补样；sourcefine缺SPA照实记录，proper粗/小时集合不混。

校准取原calibration roster中2024-04/05的合法canonical查询，共55月，六折10/10/10/10/8/7月、5/5/5/5/4/4可用地区。proper/cal/test地区严格不交叉；少数cal地区无早期月不补造，名单及实际UTC范围另存。构造模型时逐cal profile断言整月合成release≤July cutoff，所有该月有效小时UTC均早于cutoff，不仅凭月份字符串筛选。July固定模型可使用全部截点前proper标签，回算Apr/May cal时可能使用晚于该cal origin的其他区源标签；这是July时点跨区回顾性校准，**不是cal逐月实时前向**。模型与半径在July前数据上冻结，H2只更新合法已发布本地月协变量，不用H2标签选方法/阈值。没有历史真实vintage或新域交换性覆盖保证。

## 共同learner与16个固定方法

两个ET32/leaf64/max_features1/seed20261028、地区平衡的scalar CIF残差kernel（20/long84），训练标签为源小时CIF减该源当时合法月能量锚；proper源u activity按V159定义，以过去月总量尺度和.99S+.01proper prior载体归一，非q=1补全。query以相同载体B重组每个原source atom，CIF=EF·(B*u)/sum(B*u)，G>0严格验证。先构造预测后读取query标签。

6个普通基线：anchor_calendar（同源48-state残差law）、direct20/84纯经验叶锚残差分布、hetero84（V156 HGB mean+proper五地区OOF Poisson方差，原100iter/leaf64/15nodes/+25规则，1025正态点离散及[0,1000]clip）、physical84_leaf、physical84_bg=.95leaf+.05地区平衡背景。hetero7次fit，加2ET=9模型/折、54/六折；没有新小时或目标监督。monthly_context内部仍执行既有harmonic Ridge辅助计算，其预测不用于本实验，9/54只计风险模型。V156脚本有top-level argparse，**不import**，仅按明确参数重建同预算/本cutoff基线，不将原全年归档冒充July前向参照。

physical84_bg共同atoms/weights上9个固定law修改：source_uniform、source_delete_mix（固定kernel删一源CDF平均，不重训、不称jackknife+）、event_mass_full、event_risk_half/full、oil_nonzero_full、uniform_mix10、ess_mix_anchor、source_spread_mix_anchor。再加uniform_range，合计16法。uniform_range以1025个中点的Uniform[0,1000]离散近似计CRPS、[0,1000]全支持区间计覆盖/宽度，明确不是其90%等尾区间。

## 事件层与候选机制的精确定义

16个proper源层A=(fuel f,zero/nonzero)，只在源当前M_f>0上定义；q中未知当前月正负不参与。它是source atom的元数据层，不能称目标真实事件概率。背景5%使每个已有源atom质量为正，不能补出未观测类别。

对一个clock，原law为w、atom CIF x（共用合法ratio值）。层总质量p_A；r_s为层内各支持源的归一贡献；k为支持源数；H=sum r_s²；gamma=clip((H−1/k)/(1−1/k),0,1)，数值零≤64eps清为0。k<2时不更新且标不可均衡；缺类/单源都不能称已解决。

层内等源候选w_eq,i=w_i/(k*r_s)（i∈A），其他不变；它严格保存p_A，不声称其他重叠层质量或能量矩也不变。full更新w_new=(1−gamma)w+gamma w_eq（仅A内）；half取gamma/2。等源目标是候选稳定化假设，不是目标源真实概率估计。

W1影响D_A=∫|CDF_new−CDF_old|dc，在同atom有序网格上精确累积质量差。因只改层内且保p_A，D_A=p_A*gamma*W1(F_A,F_A,eq)，这是基本混合CDF恒等式，不是覆盖定理。event_risk选择D_A最大的一个层（固定fuel/class顺序打破tie）；event_mass只按p_A*gamma选层，检验CIF风险影响是否超出普通集中度。零影响全部不改。oil_nonzero固定oil/M>0/raw>0层，是V189明确动机对照，不按本轮held效果挑层。16层完整支持/质量/集中度/风险影响都保存。

source_uniform在整个law使各支持源质量相等；source_delete_mix为每个支持源删除后剩余w重归一的等CDF平均。ess_mix_anchor以整个law源H的同gamma与anchor_calendar混合；source_spread_mix_anchor以各源条件CIF均值的等源std/(原law std+该std)混合anchor。uniform_mix10固定.9物理+.1全范围1025点Uniform近似。它们均为普通控制，不声明理论保证。

目标未观测类别/新ratio atom仍不可识别。重权候选不制造support，full物理support miss应相同；cal半径可能扩出经验support，但不代表创造了训练事件。选层可能改变均值/其他燃料层概率，诊断保存，不冒称月量精确守恒或V173矩保护。

## 校准、分母和gate

全55cal+158test共213query先构造该折所有law并封存hash/诊断，再读该折cal真值算半径及test评分。cal仅按各法基础5/95分位误差的地区平衡q90（pooled）与max-region q90；法/强度固定，不用cal优化超参。9个修改法同时报告自身recalibrated和共同physical84_bg fixed半径；6基线+uniform_range仅自身半径。14+36=50记录/test，共7900。uniform_range两个cal标签相同仅为表结构重复，不当独立结果。

主要all_available27地区，按小时后地区等权；Q3/Q4各自分母明确。保留原全部gate：宏覆盖≥90%、最差≥85%，CRPS相对同合同强direct/简单锚/异方差/普通源均衡/ESS/spread/保守控制至少降5%、Winkler下降且宽不增加，地区paired描述范围支持，不能只过弱参照。只有开发同时满足才讨论进一步确认合同；无CIF gate不得开真正未查看地区，目标active。大量相关比较仅开发描述，不宣称distribution-free或原创已经成立。

## 预检、存储与审计

六折无held评分预检：54模型指纹、12首尾query全law未来M扰动/raw=None exact；100随机数值case验证权重/层质量/共同support/W1混合恒等式，单源/缺类/恒定CIF反例。source输入仅已发布prefix；HGB proper OOF按地区非小时随机。正式指纹须与预检一致；完整input hash begin/end和atomic fold/resume。

完整law无需重复保存数十GB原子权重：保存全部source矩阵/角色/前缀/OOF/树指纹，以及213×16×48=163584个law x/w字节hash、基础分位/均值/全部层诊断，再由审计完整重拟合/重建law核每条hash。共享构造与树生长的范围明确；独立权重公式/W1-CDF积分、每fold首尾query×两clock×所有物理重权法的合成y CRPS独立积分，全部真实分数/半径/root/宏/比较重算，缺类/单源绝不填成已知无风险。
