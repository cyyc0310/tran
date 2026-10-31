# V48：重组路径内的完整水平与风险控制

2026-10-27，Sol。父V47清单4e5a3cfd29b6c6db424f734e89bbe05f7e7ba253b7dea652b4a1b994075df87c。V47已证明普通水平纠正能改善开发日期上的平均覆盖，但小时形状几乎不变。本轮只改变小时底座，沿同预算检验较好的重组形状是否能带来区间效率；不是新方法、独立确认或方法冻结。

## 固定输入与控制

使用原七个开发地区2025H1的181个UTC日、128粒子。原V29 Scale-recompose raw记为R，对应fine_draws记为F；核F与Scale-anchor fine_draws逐值相同。原query_mean、MG和V47的15个level S均按同日同粒子逐值配对。origins、keys、MG、15level registry必须与V47 literal exact，不把月初seed延用到其它日。不读训练审计数字、不重建头、MG、条件比例或shape bank，零fit。

U=F dot EF8，P=MG_POS*(query_mean4 dot EF_POS)。全部15 S输出四模式：Recompose-Replace=R−U+S；Recompose-Keep=R−P+S；Recompose-CenterOnly=R+mean_particle(S−P)；Recompose-FlatKeep=对应Keep每粒子24小时平均重复。60个重组底座控制全部固定，不只挑Ridge或最佳scale。S仍不识别完整r4；加性桥接不是新的燃料物理重组、月能量闭合或新center风险校准，原R的燃料重组性质不自动传给加常数的新路径。

另加15个Recompose-AlignedKeep-S控制：C=R−mean_hour(R)+mean_hour(A)，A为原Scale-anchor raw，路径为C−P+S。它保留旧Keep-A的逐粒子raw日均及其与S/MG的配对，同时替换为R的居中形状；不是仅由普通A/R底座差异归因shape。增加两invariant：AlignedKeep日均逐粒子等于旧Keep-A，居中shape等于R；前者另直接对父V47实际Keep数组核，不只自身公式。clip后日均/形状可改变，评分包括clip交互，不识别独立调度/尾部机制。

旧V47的63臂完整保留，新增75臂共138：B0 38/B1 99/M8 1。新增B0 20/B1 55，原3参考只在旧63中出现一次。纯wrapper复用V47 build_paths的相同公式，以R填充其三个内部参考槽后立即丢弃这三个槽；只输出75新路径（60个原模式加15个AlignedKeep），不将临时槽冒充原参考或读取M8作为B0输入。原63预测不重生成，旧360675日指标所有数值按round_trip直接复制，重新从完整day表汇总。

每模式的代数边界沿V47：Keep−Replace=U−P可有非零均值；CenterOnly与Keep的ensemble mean整条曲线仅在raw/clip前相同；FlatKeep只保留raw/clip前日均粒子分布。非Flat单粒子居中形状保留的是R的形状，不是原anchor的形状，也不保证跨粒子分位数或clip后保持。零POS和raw越界只诊断，不临时修补。

## 评分与比较

1267query全预测，评分原V47冻结的1145日，支持BPAT176/CISO178/ERCO180/FPL179/ISNE89/MISO166/NYIS177，122缺日原样标记。全部新预测seal后才能读取旧七源2025-01-01至2025-07-01的有效24h generation，source_days先筛时间与有效24h再转数值；支持不符整次失败，不随新分数改日期。PACE/LGEE、真正未知及新增donor2025fine继续关闭。

沿原先clip→point→factor→clip，五factor(.75,1,1.25,1.5,2)，primary1，第7/122顺序统计量；八原指标、path simultaneous coverage、spread clipping全保留。七区内平均后等权，保留逐区和去地区敏感性；小时/日期不当独立域。全部旧与新五点曲线同列，包括V47普通Replace固定Mix factor1.5等强参照；不根据新分数拟合尺度或只比较原锚factor1。macro flags仍相对原Scale-anchor factor1的width/CRPS/IS，条件coverage≥.90且最差区≥.85；不等于coverage≥anchor。

新75×5×1145=429375日，加旧360675共790050；4830地区、690宏、4830去地区、1267支持，95025新raw诊断。报告全部15水平的架构匹配与Keep/CenterOnly匹配；指出预算差异、平均与最差地区、负迁移以及clip影响，不由某一漂亮curve点宣称创新。

## 执行与验证

exclusive results/recompose-bridge-v48，prepare/predict/score各一次，失败保留。每源NPZ只保存新raw75×181×128×24、levels15×181×128、MG、日期/键/registry/U/P；JSON记录来源与原六加新增两invariants，原参考沿父路径绑定。七archive全部grid/hash验证后score_started才真值，逐源加载限内存。五producer/test/protocol、两实际落盘预审、父manifest和当前直接输入先预锁；只解析父1089 metadata并核本轮依赖，不重遍历旧数据。cached数值在freeze/prediction_started后打开。

独立临时公式重建75路径/95025diagnostics/原六与新增两invariants、MG/日期/keys/levels literal exact、429375新日评分与全790050汇总、旧360675全部numeric exact；tol1e−9/rtol0。无需新fit或大审计框架。若收益仍可由普通平移/缩放解释，不能以架构拼接取代创新目标；下一步才形成严格outer排除的源小时风险数据集，与全局校准/异方差/条件分位数比较，检验可识别粗风险和未解释level/shape分开处理的价值。长期目标active，未知地区不打开。

首版两预审及7测试报告已完成，但未执行formal或读取新数值。上述AlignedKeep在首次正式启动前加入，保留首版记录，最终五文件重新核pin，两份-r1预审实际落盘后才启动；不覆盖首版报告。
