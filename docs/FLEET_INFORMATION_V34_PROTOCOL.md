# V34：装机信息是否改善粗月报条件下的跨区预测

2026-10-26，Sol。此合同在任何V34真实模型拟合和结果查看前固定。本轮回答新增信息是否有用，所有方法均按普通基线解释；不以B1胜B0作为方法创新，不打开旧目标或真正未查看地区。

## 问题与数据

B0是完整粗类别月度发电量，POS内煤、气、油、other细分未知。B1另加V33整理的公开登记装机代理。装机与发电量、登记BA与发电统计辖区可能不一致，装机零值也不认证物理缺席，不能用作未来小时出力硬界。B1在中国场景能否取得及字段是否同口径尚未验证。

仅BPAT、CISO、ERCO、FPL、ISNE、MISO、NYIS七个已查看源区。复用V31的2025年1—6月42个起点、每起点128个合法MG预测粒子和既定35个可评分月。ISNE只有一个评分月，不能据此判断其全年稳定性。训练仍是V30封存的2022年1月至2024年6月六源L24 cohort，标签发布截止2025-01-01；每个outer排除该源，地区均衡权重不变。复用封存训练矩阵和查询特征，不新增细燃料、天气或小时数据。

装机只读V33的OP和Operable两张28行汇总表。每个训练起点和查询起点按年份减2选择唯一库存年度，并验证已登记发布日期不晚于该起点；缺行、重复、负值、非有限值或POS总容量非正均失败，不回退其他年度。2020—2023库存对应2022—2025起点。当前下载档不是已认证首发字节，沿用V33的历史版本局限。

## 固定比较

臂顺序为：Size-Ridge、Size-ET、Capacity-OP、Capacity-Operable、Utilization-OP、SizeCap-Ridge、SizeCap-ET、IntensiveCap-Ridge、IntensiveCap-ET、DirectEF-Ridge、DirectEF-ET。

前两臂直接复制封存V31预测，逐值保持一致，不重复拟合。Capacity两臂用各口径POS内四种燃料装机比例作为发电比例。Utilization-OP在六源训练集学习全局相对利用率：比例正比于容量乘以利用率，气电log利用率固定0，另三参数以0初始化，范围[-6,6]。用scipy least_squares拟合地区均衡的四比例平方误差，max_nfev=10000，ftol/xtol/gtol均1e-10；优化失败或非有限值则整轮失败，不按测试结果调参。缺库存类别保持0，因此该基线无法纠正登记库存支持缺失。

SizeCap用原Size的31特征，加OP内四项POS容量比例和一项压力。压力严格为exp(log_MW0)×lag0_POS/cap_POS，log_MW0恢复的是最近合法月总平均MW；不能乘当前MG_POS粒子，压力在同一起点的128粒子间相同，不截断。IntensiveCap用原Trend的30特征加上述5项，移除绝对规模。压力和容量比例在发电量与容量同时倍增时不变，这是特征性质，不是未经验证的性能主张。

上述两特征族各拟合Ridge和ExtraTrees，目标为四燃料比例。Ridge alpha=1并仅用训练权重计算标准化；ET为128树、min_samples_leaf=2、max_features=1.0（浮点数，使用全部特征）、random_state=17023、n_jobs=1。比例预测沿用V30 simplex投影。DirectEF用SizeCap的36特征和同两模型，直接回归由现有训练比例计算的EF/1000，预测截到[0.4,1]再恢复EF；此截断来自既定排放因子范围，不由装机认证。共7个新fit/outer、49个fit，没有超参数搜索。

Utilization的最小二乘残差乘sqrt(sample_weight)，记录训练Jacobian秩、active_mask和最优性，不依据测试结果补设筛选。固定气电参考也不保证参数可识别，拟合系数不解释为物理利用率。既定因子下EF=400+600×coal比例+400×oil比例，gas与other在EF中不可区分；标量预测改善不能证明四比例被识别。

每个粒子的CIF等于MG_POS乘EF，点预测取粒子CIF的平均，EF点预测另取EF粒子平均。不能以平均MG乘平均EF替代CIF粒子平均。两类点误差均保留，各源先平均月误差再做七源宏平均。目标是本研究既定固定燃料因子的发电侧代理CIF，不是实测排放或消费侧含进口CIF。

## 指标、判定与下一步

主要指标是七源macro CIF MAE，另报RMSE、signed error和EF误差，保留逐月、逐源结果。macro RMSE为七源各自RMSE的均值，与V31一致。35月的逐key支持及EF定义状态保持不变，不按库存支持删月。九个新臂各与两旧臂比较；另比较同模型IntensiveCap与SizeCap、DirectEF与SizeCap，以及Utilization-OP与Capacity-OP、Capacity-Operable与Capacity-OP，共24组、168条逐源contrast。

新增信息的实用筛选要求：相对Size-Ridge和Size-ET均至少降低5%宏CIF MAE，各至少改善4源，每源不超过参考10%退化，MISO与ERCO均不得退化。小参考误差仍使用原绝对容差1e-12，不用重新设分母来放宽约束。每个臂完整报告，不按成绩删臂。该筛选只是探索性信息价值证据；七个源已多次查看，不作确认性显著性主张。

即使通过，仍不授权未知地区测试，也不称为机制突破。先检查收益来自库存组成、规模归一化还是普通回归；若强基线已解释收益，专门机制必须在同B1预算内再比较。若所有臂不通过，则分析库存边界、利用率与粗统计错配，不能依赖统一加宽区间掩盖失败。小时风险、覆盖率和区间效率须另立实验，本轮不由粒子离散程度推导覆盖保证。

## 执行与复核

固定V31、V33证据闭包、五个V34代码/测试/合同文件、独立代码与Python/科学预审、环境版本后prepare。输出使用exclusive写，失败保留。predict开始前写独占收据；全七源预测与hash seal完成后才允许score读取已查看源区标签。预测只读封存V30训练审计、V31白名单NPZ成员和V33汇总CSV，不解析raw容量档。score必须与V31的42行label_support保持同一支持。原句柄退出和日志保留。

NPZ白名单精确为origins、arms、query_feature_Size、MG_draws、CIF_point、EF_point、CIF_draws、EF_draws、projection_l1。Trend取Size前30列，不额外读取别的成员；预测不接收2025评分标签或评分输出。训练只提取同outer的Size两头六源metadata，核对keys/y/weights/时点一致。任何容量异常都整轮失败，不能删样本改变队列。

代码、父清单、绑定输入与预审须为普通非symlink文件。prediction seal与completion都绑定prediction_started的SHA，评分前核marker所指freeze一致；开始收据不作为进程存活证据。

结果先由独立审阅核对数据时点、外区排除、强基线和判定，再决定是否值得开展同预算风险机制。不能把审计通过与科学突破混写。
