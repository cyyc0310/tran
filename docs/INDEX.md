# 研究文档入口

**2026-11-02 GB→US同公式CN5固定因子代理开发对照：** 英国小时燃料结构训练、源区校准后迁移到美国BA；支持审计只保留五个全年小时覆盖≥80%的区，七区全量队列因CISO/ISNE小时标签缺失未通过门槛。五区粗重组MAE较700锚低8.7%，但90%覆盖仅79.7%、Winkler更差27.6%，MISO覆盖23.7%；因此点误差增益未形成可信区间收益，也不是盲地理/as-issued确认。[结果与边界](GB_TO_US_2025_CN5_PROXY_RESULTS_20261102.md)、[全区协议](GB_TO_US_2025_CN5_PROXY_PROTOCOL_20261102.md)、[支持子集协议](GB_TO_US_2025_CN5_PROXY_SUPPORTED_PROTOCOL_20261102.md)、[运行器](../scripts/run_gb_to_us_2025_cn5_proxy.py)、[封存预测与评分](../results/gb-to-us-2025-cn5-proxy-r4/)

**2026-11-02 台电公开Q2小时外部复核已冻结预测协议并提取滞后月输入：**使用美英源2022训练/2023H1校准、官方当前修订月表`t−2…t−7`五类份额；封存后才解封台电小时响应。该目标结构预检时曾打开少量记录，且输入非历史首发快照，因此不称盲确认/as-issued回测。[预检](TAIPOWER_PUBLIC_HOURLY_TARGET_PREFLIGHT_20261102.md)、[冻结协议](TAIPOWER_CN5_EXTERNAL_TRANSFER_PROTOCOL_20261102.md)、[运行器](../scripts/run_taipower_cn5_external_transfer.py)、[公开月输入及哈希](../data/taipower-public-hourly-target-r1/)

**台电小时目标范围审计未通过：**正向映射覆盖仅92.62%，逐月有效小时不足99%，且机组小时总量与全国月报差47%–56%；无精度分数可解释。小时数据更接近台电自营月总量，但仍差17%–26%，不能声称范围匹配。[结果与决策](TAIPOWER_CN5_EXTERNAL_TRANSFER_RESULTS_20261102.md)、[状态/评分封存](../results/taipower-cn5-external-transfer-r1/scored/)

**2026-11-02 AEMO燃料组成响应头顺序跨域复核：**US+GB源燃料份额头在固定EF代理上降低宏MAE，但只在合并来源时有效；区间分数较最佳强基线差17.9%、NSW/QLD严重欠覆盖。此为已看目标上的方法复核，不是盲测或原创方法确认。[结果](AEMO_FUELSHARE_RESPONSE_TRANSFER_RESULTS_20261102.md)、[冻结协议](AEMO_FUELSHARE_RESPONSE_TRANSFER_PROTOCOL_20261102.md)、[运行脚本](../scripts/run_aemo_fuelshare_response_transfer.py)、[封存证据](../results/aemo-fuelshare-response-transfer-r1/README.md)

**2026-11-02意大利燃料份额响应头外部复核：**严格CN5燃料组成头胜比例重组和历史因子头，但低于同预算direct_latest；覆盖86.4%、Winkler也未胜最强基线，停止在该小时代理调参。该国此前已见日尺度标签，小时目标为修订数据固定因子代理，不是盲确认。[结果](ITALY_FUELSHARE_RESPONSE_TRANSFER_RESULTS_20261102.md)、[冻结协议](ITALY_FUELSHARE_RESPONSE_TRANSFER_PROTOCOL_20261102.md)、[运行脚本](../scripts/run_italy_fuelshare_response_transfer.py)、[封存证据](../results/italy-fuelshare-response-transfer-r1/README.md)

**2026-11-02日本九区冻结方法外部复核：**US/GB源模型迁移到日本同公式固定因子代理时，粗重组日内形状误差下降35%，但宏MAE比700锚高2.7%、覆盖只有37.3%，仅3/9区胜锚；保留为新国家外部负结果，不是官方CIF或as-issued确认。[结果](JAPAN_CN5_EXTERNAL_TRANSFER_RESULTS_20261102.md)、[协议](JAPAN_CN5_EXTERNAL_TRANSFER_PROTOCOL_20261102.md)、[预测与分区证据](evidence/japan-cn5-external-transfer-r1/README.md)

**2026-11-02月均锚定相对形状伪目标复核：**新乘法归一将区间宽度从旧加法混合607.8收回到336.1，但MAE71.88仍比强燃料份额头54.80高31.2%，Winkler也高9.1%；相对700锚仅1.8% MAE增益、未到晋级门槛。已停止当前配方，不在已看九国上调参。[结果](CN5_MONTHLY_ANCHORED_RELATIVE_SHAPE_RESULTS_20261102.md)、[冻结协议](CN5_MONTHLY_ANCHORED_RELATIVE_SHAPE_PROTOCOL_20261102.md)、[脚本](../scripts/run_cn5_monthly_anchored_relative_shape.py)、[机器证据](evidence/cn5-monthly-anchored-relative-shape-r1/README.md)

**2026-11-02美英双向预测逐月复核：**US→GB的粗形状重组在已看GB目标12个月都小幅胜源火电因子锚；GB→US增量受MISO严重欠覆盖限制，整体分数掩盖了逐BA风险。[逐月分析](CROSS_COUNTRY_PILOT_MONTHLY_BREAKDOWN_20261102.md)、[机器分数及哈希](evidence/cross-country-pilot-r1/monthly_posthoc_all_methods_summary.json)

**2026-11-02官方发电强度伪目标专家未跨域迁移：**七个EIA BA留区伪目标选择的专家回退在源内MAE降低4.3%，迁至GB后反而较同预算HGB高4.9% MAE；700热电份额锚与源燃料重组基线都更好。关闭当前距离门控回退算法分支，保留作负结果基线。[结果与路线判断](CROSS_COUNTRY_GENERATION_PSEUDOTARGET_EXPERTS_RESULTS_20261102.md)、[冻结协议](CROSS_COUNTRY_GENERATION_PSEUDOTARGET_EXPERTS_PROTOCOL_20261102.md)、[脚本](../scripts/run_generation_pseudotarget_experts.py)、[封存证据](evidence/generation-pseudotarget-experts-r3/README.md)

**2026-11-02跨国双轨小时任务目录：**以任务而非统一排行榜登记严格CN5伪目标、US↔GB官方发电/消费估算、固定因子代理、前瞻未评分预测及中国无标签推理；各行标出输入vintage、标签口径、支持量、确认状态和证据路径，避免将发电代理与官方消费强度合并排名。[任务目录与比较边界](CROSS_COUNTRY_BENCHMARK_CATALOG_20261102.md)、[机器可读CSV](CROSS_COUNTRY_BENCHMARK_CATALOG_20261102.csv)

**2026-11-02 CN5小时燃料组成月闭合实验：**源区小时燃料响应加月均组成Ridge闭合明显劣于原小时燃料份额头和比例重组，确认严格五类粗月报无法稳定识别火电内部月组成；停止该路线。[结果与判断](CN5_MONTHLY_COMPOSITION_CLOSURE_RESULTS_20261102.md)、[冻结协议](CN5_MONTHLY_COMPOSITION_CLOSURE_PROTOCOL_20261102.md)、[脚本](../scripts/run_cn5_monthly_composition_closure.py)、[封存证据](evidence/cn5-monthly-composition-closure-r1/README.md)

**2026-11-02 CN5滞后月边际IPF协调：**把源小时形状投影回预测时可见的`t−2`五类月边际后，九国LOCO预测与原比例重组几乎数值相同，点误差和区间均无增益；强燃料份额头仍更优。审计证明原比例重组已隐含满足该月边际，故停止重复协调分支。[结果与判断](CN5_MONTHLY_MARGIN_RECONCILIATION_RESULTS_20261102.md)、[冻结协议](CN5_MONTHLY_MARGIN_RECONCILIATION_PROTOCOL_20261102.md)、[脚本](../scripts/run_cn5_monthly_margin_reconciliation.py)、[审计与机器输出](../results/cn5-monthly-margin-reconciliation-r3/)

**2026-11-02 US→GB发电侧固定因子代理2025时间留出：**在相同美国2022源训练/2023校准模型下，粗月报小时轮廓重组相对源火电水平锚的MAE改善4.1%，覆盖高0.98个百分点且区间略窄；与2024开发结果排序一致，但两个月明显欠覆盖，且目标数据已在其他形状研究中查看。[结果及跨年复核](US_TO_GB_2025_GENERATION_PROXY_RESULTS_20261102.md)、[协议](US_TO_GB_2025_GENERATION_PROXY_PROTOCOL_20261102.md)、[脚本](../scripts/run_us_to_gb_2025_generation_proxy.py)、[封存证据](evidence/us-to-gb-2025-generation-proxy-r1/README.md)

**2026-11-02英国ET 5.4历史版本追溯：**2024发布页快照保留了原工作簿链接，但附件自身未见公开网页归档，原URL已重定向当前修订版；严格as-issued英国回测需申请旧表或从未来发行开始留档。[专项审计](UK_ET54_ASISSUED_ARCHIVE_AUDIT_20261102.md)、[历史输入审计更新](ACTUAL_MONTHLY_VINTAGE_AUDIT_20261101.md)、[网页及机器证据](evidence/uk-et54-vintage-archive-audit-r1/README.md)

**2026-11-02创新点复核：**跨区零样本迁移和消费侧潮流预测已有直接相关工作；建议把贡献落在滞后可见信息合同、双估计目标、整域嵌套评测与可识别性。详见[文献复核与下一轮决策](CROSS_COUNTRY_CIF_NOVELTY_REASSESSMENT_20261102.md)。

**2026-11-02下一轮信息预检：**严格CN5九国开发数据的小时负荷字段覆盖99.78%–100%，可以运行独立的滞后负荷信息价值实验；它暂不属于已确认的中国主输入预算。[覆盖结果与冻结建议](CN5_LAGGED_LOAD_PROFILE_INFORMATION_PREFLIGHT_20261102.md)

**2026-11-02负荷Oracle已筛查并关闭当前因子头路线：**目标小时实际负荷使严格CN5因子头宏MAE略升0.25%，未通过5%预设增益门槛。[结果](CN5_HOURLY_LOAD_ORACLE_INFORMATION_RESULTS_20261102.md)、[冻结协议](CN5_HOURLY_LOAD_ORACLE_INFORMATION_PROTOCOL_20261102.md)、[预测与哈希证据](evidence/cn5-hourly-load-oracle-r1/README.md)

**2026-11-02双轨创新方向与小时可识别性首轮检查：**近期多频CIF与组件式迁移已有相邻论文；新方向先量化M/M+C信息预算下小时形状的等价集合，再决定是否值得开发约束预测器。[文献与路线决策](DUAL_TRACK_INNOVATION_DECISION_20261102.md)、[CN5预算等价反例](CN5_MONTHLY_BUDGET_HOURLY_IDENTIFIABILITY_PILOT_20261102.md)

**2026-11-02仅公众数据下的下一阶段决策：**用户确认无市场成员账号；冻结中国M/M+C与消费侧公开信息边界，提出“月能量闭合的源形状集合迁移”作为下一项待证伪实验。[研究决策与门槛](PUBLIC_ONLY_NEXT_RESEARCH_DECISION_20261102.md)

**2026-11-02严格CN5源子集分歧区间实验：**在同中心下覆盖提高但区间宽70.5%、Winkler恶化33.2%，按预设效率门槛停止该风险尺度分支。[协议](CN5_SOURCE_SUBSET_DISPERSION_PROTOCOL_20261102.md)、[九国结果与封存](CN5_SOURCE_SUBSET_DISPERSION_RESULTS_20261102.md)

**2026-11-02用户再次确认发电/消费双轨；官方规则显示中国小时预测字段值得审计：**新能源分类型、水电、发电总量及省间联络线预测列入规则候选，但逐省历史签发版本与实际公开覆盖尚未验证。下一轮先审来源、发布时间、归属与完整度，再决定M+F或C1+Fflow能否进入模型。[路线与门槛](DUAL_TRACK_INNOVATION_DECISION_20261102.md)、[国家能源局规则](https://zfxxgk.nea.gov.cn/2024-01/31/c_1310763726.htm)、[浙江省披露附表](https://zjb.nea.gov.cn/dtyw/tzgg/202512/P020251220333999314238.pdf)

**2026-11-02浙江小时预测入口预检：规则将省级曲线列为市场成员可见，未证实社会公众可取；历史曲线仍未取到。** 2026年规则要求披露信息留存/可查询至少两年，但不取消访问范围；当前环境访问失败，未取得可封存曲线，不能启动M+F/C1+Fflow历史评分。[逐层审计和下一步](CN_HOURLY_FORECAST_VINTAGE_PREFLIGHT_20261102.md)

**2026-11-02月度预算小时信息阶梯已完成：**九国目标小时oracle显示，风光单独只将LP界宽从1000降至808，水核风光一起降至234、全部非火电降至176 g/kWh；目标出力真值是oracle，不是预测特征或概率覆盖。2022装机条件臂仍因2024出力超库存613次而不可信。[协议](CN5_MONTHLY_BUDGET_LP_INFORMATION_PROTOCOL_20261102.md)、[结果与双轨解释](CN5_MONTHLY_BUDGET_LP_INFORMATION_RESULTS_20261102.md)、[runner](../scripts/run_cn5_monthly_budget_lp_information.py)、[r5机器结果/哈希](../results/cn5-monthly-budget-lp-information-r5/summary.json)

**2026-11-02小时总量/日热电统计信息价值检验：**七个EIA BA中，月报＋小时总量后尾部可行集仍宽70%；再加每日热电量只额外缩窄约5%，MISO 500 g/kWh仍宽86%。两个额外信息均为oracle，不是预测覆盖或消费侧结论。[协议](V401_MONTHLY_MARGIN_TAIL_IDENTIFICATION_PROTOCOL.md)、[结果](V401_MONTHLY_MARGIN_TAIL_IDENTIFICATION_RESULTS.md)、[MILP runner](../scripts/v401_monthly_margin_tail_identification.py)、[r7机器摘要](evidence/v401-monthly-margin-tail-identification-r2/summary.json)

**2026-11-02优先级4区间压力测试：**九国源域整段最大残差把FR已看目标区间扩大3.7–4.4倍，但FR原小时q90区间本就100%覆盖；关闭裸最坏域扩宽路线，不能视作新域覆盖保证。[协议](CN5_DOMAIN_MAX_INTERVAL_DIAGNOSTIC_PROTOCOL_20261102.md)、[结果](CN5_DOMAIN_MAX_INTERVAL_DIAGNOSTIC_RESULTS_20261102.md)、[机器证据](evidence/cn5-domain-max-interval-r2/README.md)

**重要口径更正：**部分旧Energy-Charts实验实际使用六类粗输入（含`other_renewable`），不能作为中国火、水、核、风、光五类预算结果。引用前先读[输入预算更正与影响范围](ENERGY_CHARTS_INPUT_BUDGET_CORRIGENDUM_20261101.md)；严格五类重跑见该文档链接。

先读[跨域迁移研究计划](CROSS_DOMAIN_TRANSFER_RESEARCH_PLAN_20261031.md)了解用户纠偏后的目标与研究顺序，再读[四日成果报告](reports/2026-10-28_31.md)和具体实验记录。新计划覆盖旧文档建议转向部分识别的路线判断，不改变历史数值及失败结论。完整逐步日志 [BREAKTHROUGH_SEARCH](BREAKTHROUGH_SEARCH.md) 是过程记录，不宜单独作为最终论文结论。

| 问题 | 优先阅读 |
|---|---|
| 最新：CN5伪目标源筛选按目标月粗组成条件化后只改变2/108次选择，未改善强基线区间 | [结果与路线判断](ENERGY_CHARTS_TARGET_CONDITIONED_SOURCE_DROP_CN5_RESULTS_20261101.md)、[冻结协议](ENERGY_CHARTS_TARGET_CONDITIONED_SOURCE_DROP_CN5_PROTOCOL_20261101.md)、[证据包](evidence/energy-charts-target-conditioned-source-drop-cn5-r1/README.md)、[脚本](../scripts/run_energy_charts_target_conditioned_source_drop_cn5.py) |
| 最新：水平锚＋重组形状的严格五类伪目标LOCO未超过强重组，区间过宽 | [九国结果与路线判断](CN5_PSEUDOTARGET_LEVEL_SHAPE_RESULTS_20261101.md)、[冻结协议](CN5_PSEUDOTARGET_LEVEL_SHAPE_PROTOCOL_20261101.md)、[逐国封存证据](../results/cn5-pseudotarget-level-shape-r2/)、[脚本](../scripts/run_cn5_pseudotarget_level_shape.py) |
| 最新：GB→US 2025官方发电强度开发时间留出未超过简单锚，源校准区间明显欠覆盖 | [结果报告](CROSS_COUNTRY_2025_OFFICIAL_GENERATION_TRANSFER_RESULTS_20261102.md)、[冻结协议](CROSS_COUNTRY_2025_OFFICIAL_GENERATION_TRANSFER_PROTOCOL_20261102.md)、[评分证据](evidence/gb-to-us-2025-official-generation-r2/README.md) |
| 最新：US→GB 2025 NESO官方发电强度反向留出：简单热电锚胜过燃料重组，重组靠过宽区间达到全覆盖 | [结果报告](CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_GENERATION_RESULTS_20261102.md)、[冻结协议](CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_GENERATION_PROTOCOL_20261102.md)、[预测/评分哈希](evidence/us-to-gb-2025-official-generation-r1/README.md)、[NESO官方目标快照](evidence/neso-national-intensity-2025-r2/manifest.json)、[最终运行器](../scripts/run_us_to_gb_2025_official_generation.py) |
| 最新：US→GB 2025官方消费强度开发留出，强重组代理改善点误差但区间仍欠覆盖 | [结果报告](CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_CONSUMPTION_RESULTS_20261102.md)、[冻结协议](CROSS_COUNTRY_2025_US_TO_GB_OFFICIAL_CONSUMPTION_PROTOCOL_20261102.md)、[评分证据](evidence/us-to-gb-2025-official-consumption-r4/README.md) |
| 最新：GB→US 2025官方消费强度开发留出不复制US→GB的重组收益，MISO严重欠覆盖 | [结果报告](CROSS_COUNTRY_2025_GB_TO_US_OFFICIAL_CONSUMPTION_RESULTS_20261102.md)、[冻结协议](CROSS_COUNTRY_2025_GB_TO_US_OFFICIAL_CONSUMPTION_PROTOCOL_20261102.md)、[评分证据](evidence/gb-to-us-2025-official-consumption-r3/README.md) |
| 2025美英官方小时强度双轨综合判断与四项研究任务状态 | [方向不对称、端点缺口与后续算法门槛](CROSS_COUNTRY_2025_BIDIRECTIONAL_SYNTHESIS_20261102.md) |
| EIA-930双轨标签时间戳与覆盖审计 | [标签通道审计](CROSS_COUNTRY_2025_OFFICIAL_EIA_LABEL_AUDIT_20261102.md)、[EIA证据](evidence/eia930-official-co2-2025-r1/README.md)、[NESO源域历史](evidence/neso-national-intensity-2022-2024-r1/manifest.json) |
| 最新：EIA消费侧标签拆分显示MRIO分配的来源排放进入小时CIF，净交换量不足以描述消费侧差异 | [标签分量核查](EIA_CONSUMPTION_LABEL_FLOW_DECOMPOSITION_RESULTS_20261101.md)、[审计脚本](../scripts/audit_consumption_label_flow_decomposition.py)、[逐区结果](../results/consumption-label-flow-decomposition-r1/region_decomposition.csv) |
| 最新：同小时净流量Oracle上限仍不双向改善消费侧迁移 | [结果与路线判断](HOURLY_FLOW_ORACLE_INCREMENT_RESULTS_20261101.md)、[冻结协议](HOURLY_FLOW_ORACLE_INCREMENT_PROTOCOL_20261101.md)、[配对脚本](../scripts/run_hourly_flow_oracle_increment.py)、[机器汇总](../results/hourly-flow-oracle-increment-r2/endpoint_metrics.csv) |
| 最新：消费侧滞后净受电信息未带来可信跨域增量；月统计不足以修复小时强度与覆盖 | [结果与路线判断](NET_INTERCHANGE_INCREMENT_RESULTS_20261101.md)、[冻结协议](NET_INTERCHANGE_INCREMENT_PROTOCOL_20261101.md)、[配对脚本](../scripts/run_net_interchange_increment.py)、[配对机器结果](../results/net-interchange-increment-r3/endpoint_metrics.csv)、[原始NESO流量清单](../data/neso-interconnector-flows-r1/source_manifest.json) |
| 最新：消费侧潮流输入审计（丹麦CBPF与CBET差异及能量平衡残差，未评分） | [覆盖、字段边界与下一步](ENERGY_CHARTS_PHYSICAL_FLOW_AUDIT_20261101.md)、[原始响应与哈希清单](../data/energy-charts-flow-audit-r1/audit_manifest.json)、[小时诊断](../data/energy-charts-flow-audit-r1/hourly_balance_and_flow_comparison_2024.json) |
| 最新：M+F细燃料水平/小时形状双轨配对，发电与消费增量不对称 | [结果与判断](FINE_FUEL_DUAL_TRACK_RESULTS_20261101.md)、[冻结协议](FINE_FUEL_DUAL_TRACK_PROTOCOL_20261101.md)、[预测/评分证据](evidence/fine-fuel-dual-track-r3/README.md)、[运行脚本](../scripts/run_fine_fuel_dual_track.py) |
| 最新：伪目标报告覆盖校正未优于identity，关闭该方法分支 | [结果与研究决策](PSEUDOTARGET_REPORTED_MIX_CORRECTION_RESULTS_20261101.md)、[冻结协议](PSEUDOTARGET_REPORTED_MIX_CORRECTION_PROTOCOL_20261101.md)、[源区/英国机器证据](evidence/pseudotarget-reported-mix-correction-r1/README.md)、[脚本](../scripts/run_pseudotarget_report_coverage_correction.py) |
| 最新：回到“外国小时监督、仅中国公开粗月报”的主线及下一步晋级门槛 | [研究重点与论文主张边界](CHINA_TRANSFER_RESEARCH_FOCUS_20261101.md) |
| 最新：EIA真实发布月报滚动至小时迁移，州—BA边界错配使效用结论无效 | [协议](ACTUAL_ISSUE_MONTHLY_TRANSFER_PROTOCOL_20261101.md)、[结果与口径审计](ACTUAL_ISSUE_MONTHLY_TRANSFER_RESULTS_20261101.md)、[逐月预测与归档证据](../results/actual-issue-gb-to-nyis-r5/) |
| 最新：UK→US Lower 48真实月报发布下分开评分官方发电/消费强度；简单锚优于模型重组 | [双轨结果与水平/形状诊断](LOWER48_ACTUAL_ISSUE_DUAL_TRACK_RESULTS_20261101.md)、[预测前协议](LOWER48_ACTUAL_ISSUE_DUAL_TRACK_PROTOCOL_20261101.md)、[封存预测、逐月结果与哈希](../results/lower48-actual-issue-dual-track-r4/)、[预测/评分脚本](../scripts/run_lower48_actual_issue_dual_track.py)、[事后水平/形状诊断脚本](../scripts/analyze_lower48_level_shape_transfer.py) |
| 最新：英国官方ET 5.4月报替换NESO回算输入的同队列跨国预测敏感性 | [结果与路线判断](GB_OFFICIAL_MONTHLY_INPUT_TRANSFER_RESULTS_20261101.md)、[输入对齐协议](GB_OFFICIAL_MONTHLY_INPUT_ALIGNMENT_PROTOCOL_20261101.md)、[迁移冻结协议](GB_OFFICIAL_MONTHLY_INPUT_TRANSFER_PROTOCOL_20261101.md)、[脚本](../scripts/run_gb_et54_official_input_transfer.py)、[证据包](evidence/gb-et54-official-monthly-input-r1/README.md) |
| 最新：US↔GB发电侧2026年11月剩余小时的双向真实前瞻预测已封存，待评分 | [结果状态](PROSPECTIVE_GENERATION_US_GB_NOV2026_REMAINDER_RESULTS.md)、[冻结协议](PROSPECTIVE_GENERATION_US_GB_NOV2026_REMAINDER_PROTOCOL.md)、[预测脚本](../scripts/run_prospective_generation_us_gb_nov2026_remainder.py)、[封存预测与哈希](../results/prospective-generation-us-gb-nov2026-remainder-r1/README.md) |
| 最新：AEMO五区官方日CDEII核验与43,800行回顾性小时代理构建 | [结果与边界](AEMO_CDEII_DAILY_PROXY_VALIDATION_20261101.md)、[验证/构建脚本](../scripts/validate_aemo_daily_proxy.py)、[官方CSV、小时代理、哈希与逐日证据](../results/aemo-cdeii-validation-r3/) |
| 最新：法国冻结预测改用RTE官方终版发电强度标签复核，点误差方向保留但区间严重过宽 | [结果与研究边界](RTE_FR_FROZEN_FORECAST_LABEL_AUDIT_20261101.md)、[评分脚本](../scripts/validate_rte_frozen_fr_confirmation.py)、[汇总指标与哈希](../results/rte-eco2mix-2024-frozen-fr-audit-r1/) |
| 最新：US↔GB 2026年12月双向发电侧前瞻预测已封存，使用t−3/t−4月量检验额外信息滞后 | [预测状态与边界](PROSPECTIVE_GENERATION_US_GB_DEC2026_RESULTS.md)、[冻结协议](PROSPECTIVE_GENERATION_US_GB_DEC2026_PROTOCOL.md)、[封存脚本](../scripts/run_prospective_generation_us_gb_dec2026.py)、[哈希和预测](../results/prospective-generation-us-gb-dec2026-r1/) |
| 最新：US↔GB 2026年12月消费侧前瞻预测已封存，七个美国BA与GB分开待评分 | [结果与边界](PROSPECTIVE_CONSUMPTION_US_GB_DEC2026_RESULTS.md)、[冻结协议](PROSPECTIVE_CONSUMPTION_US_GB_DEC2026_PROTOCOL.md)、[封存脚本](../scripts/run_prospective_consumption_us_gb_dec2026.py)、[哈希及机器审计](../results/prospective-consumption-us-gb-dec2026-r1/) |
| 最新：12月双轨预测预评分审计发现消费侧负校准中心，另封存非负支持集敏感性 | [边界审计与敏感性](PROSPECTIVE_DEC2026_PRE_SCORE_QA_20261101.md)、[消费敏感性预测与哈希](evidence/prospective-dec2026-nonnegative-support-r1/consumption/)、[发电敏感性预测与哈希](evidence/prospective-dec2026-nonnegative-support-r1/generation/) |
| 最新：非负支持集投影在严格五类九国已看LOCO中保持覆盖并缩窄区间，但不提升点预测 | [配对结果与研究边界](NONNEGATIVE_SUPPORT_CN5_LOCO_RESULTS_20261101.md)、[逐月指标及预测证据](evidence/cn5-nonnegative-support-loco-r2/README.md) |
| 最新：非负支持集投影在US↔GB双标签轨保持覆盖；负中心只部分解释消费侧误差 | [双轨配对诊断及边界](NONNEGATIVE_SUPPORT_DUAL_TRACK_RESULTS_20261101.md)、[逐月/逐BA指标与哈希](evidence/paired-nonnegative-support-dualtrack-r2/README.md) |
| 最新：同BA EIA-923实际月统计下的跨国小时轮廓重组信息价值 | [当前20个BA月配对结果、旧审计复现漂移与限制](EIA923_ACTUAL_BA_INFORMATION_VALUE_RESULTS_20261101.md)、[冻结诊断协议](EIA923_ACTUAL_BA_INFORMATION_VALUE_PROTOCOL_20261101.md)、[复现脚本](../scripts/run_eia923_actual_ba_information_value.py)、[机器结果证据包](evidence/eia923-actual-ba-information-value-r1/README.md) |
| 最新：额外细分月度燃料构成及小时轮廓的增量/归因 | [20个BA月配对结果与路线判断](EIA923_FINE_FUEL_COMPOSITION_INCREMENT_RESULTS_20261101.md)、[增量冻结协议](EIA923_FINE_FUEL_COMPOSITION_INCREMENT_PROTOCOL_20261101.md)、[事后归因协议](EIA923_FINE_FUEL_PROFILE_DECOMPOSITION_PROTOCOL_20261101.md)、[增量脚本](../scripts/run_eia923_fine_fuel_composition_increment.py)、[归因脚本](../scripts/analyze_eia923_fine_fuel_profile_decomposition.py)、[增量证据](evidence/eia923-fine-fuel-composition-increment-r1/README.md)、[归因证据](evidence/eia923-fine-fuel-profile-decomposition-r1/README.md) |
| 最新：美英双向细分燃料小时响应转移与源区风险半径诊断 | [双向结果与晋级判断](CROSS_COUNTRY_FINE_FUEL_BIDIRECTIONAL_RESULTS_20261101.md)、[冻结协议](CROSS_COUNTRY_FINE_FUEL_BIDIRECTIONAL_PROTOCOL_20261101.md)、[源区留一风险诊断](CROSS_COUNTRY_FINE_FUEL_RISK_CALIBRATION_PROTOCOL_20261101.md)、[双向证据](evidence/cross-country-fine-fuel-bidirectional-r1/README.md)、[风险证据](evidence/cross-country-fine-fuel-risk-r1/README.md) |
| 最新：严格`t−2`中国五类月报的发电/消费小时尾风险双轨复核 | [结果与下一步](LAGGED_COARSE_RISK_TRANSFER_RESULTS_20261101.md)、[冻结协议](LAGGED_COARSE_RISK_TRANSFER_PROTOCOL_20261101.md)、[发电侧证据](evidence/lagged-coarse-risk-generation-r1/README.md)、[消费侧证据](evidence/lagged-coarse-risk-consumption-r1/README.md)、[复现脚本](../scripts/run_lagged_coarse_risk_transfer.py) |
| 最新：月组成预测误差能否识别小时点误差与尾部覆盖 | [事后风险可识别性结果](LOCAL_MONTHLY_RISK_IDENTIFIABILITY_RESULTS_20261101.md)、[逐BA相关与哈希](../results/local-monthly-risk-identifiability-r1/README.md)、[复现脚本](../scripts/analyze_month_mix_vs_hourly_risk.py) |
| 最新：严格中国五类统计下的国别火电因子专家外层留出 | [九国结果与路线判断](CN5_FACTOR_EXPERT_TRANSFER_RESULTS_20261101.md)、[冻结合同](CN5_FACTOR_EXPERT_TRANSFER_PROTOCOL_20261101.md)、[封存预测与复现证据](../results/cn5-factor-expert-transfer-r1/README.md) |
| 最新：CN5源训练燃料组成响应头宏点误差改善，机制核验支持动态份额作用，但CZ仍欠覆盖 | [结果与论文边界](CN5_FUEL_RESOLVED_RESPONSE_RESULTS_20261101.md)、[r1协议](CN5_FUEL_RESOLVED_RESPONSE_PROTOCOL_20261101.md)、[H2校准协议](CN5_FUEL_RESOLVED_H2_CALIBRATION_PROTOCOL_20261101.md)、[复现证据及机制诊断](evidence/cn5-fuel-resolved-response-r1/README.md)、[诊断脚本](../scripts/analyze_cn5_fuel_resolved_mechanism.py) |
| 最新：CN5燃料组成头的法国正向确认未在波兰冻结外部确认中复制，暂不支持稳定算法增量 | [波兰结果与边界](CN5_FUEL_RESOLVED_PL_CONFIRMATION_RESULTS_20261101.md)、[预测前字段审计](CN5_FUEL_RESOLVED_PL_PREFLIGHT_20261101.md)、[冻结协议](CN5_FUEL_RESOLVED_PL_CONFIRMATION_PROTOCOL_20261101.md)、[封存预测/评分证据](../results/cn5-fuel-resolved-pl-confirmation-r3/)、[运行脚本](../scripts/run_cn5_fuel_resolved_pl_confirmation.py) |
| 历史：CN5燃料组成响应头在未评分法国获得一次冻结外部正向信号，尚需独立国家复现 | [法国结果与边界](CN5_FUEL_RESOLVED_FR_CONFIRMATION_RESULTS_20261101.md)、[预测前协议](CN5_FUEL_RESOLVED_FR_CONFIRMATION_PROTOCOL_20261101.md)、[预测/哈希证据](evidence/cn5-fuel-resolved-fr-confirmation-r1/README.md)、[运行脚本](../scripts/run_cn5_fuel_resolved_fr_confirmation.py) |
| 最新：燃料组成模型第二外部域候选爱尔兰在预测前因泥炭/others标签口径不兼容而拒收 | [预测前审计与决策](CN5_FUEL_RESOLVED_IE_PREFLIGHT_20261101.md)、[原始数据和元数据哈希](evidence/cn5-fuel-resolved-ie-preflight-r1/README.md) |
| 最新：CN5组成加权区间没有增量；同月等权校准更窄但最差国仍失效 | [九国结果](CN5_COMPOSITION_WEIGHTED_CONFORMAL_RESULTS_20261101.md)、[冻结协议](CN5_COMPOSITION_WEIGHTED_CONFORMAL_PROTOCOL_20261101.md)、[证据包](evidence/cn5-composition-weighted-conformal-r1/README.md)、[脚本](../scripts/run_cn5_composition_weighted_conformal.py) |
| 最新：CN5欠覆盖主要受目标热电因子失配；本地年度/月度细燃料信息改善均值但不保最差域覆盖 | [两轮信息价值结果](CN5_LOCAL_THERMAL_FUEL_INFO_VALUE_RESULTS_20261101.md)、[证据包](evidence/cn5-local-thermal-fuel-info-r1/README.md)、[年度协议/脚本](CN5_LAGGED_LOCAL_FUEL_MIX_INFORMATION_PROTOCOL_20261101.md) / [runner](../scripts/run_cn5_lagged_local_fuel_mix_information.py)、[月度协议/脚本](CN5_LAGGED_MONTHLY_FUEL_MIX_INFORMATION_PROTOCOL_20261101.md) / [runner](../scripts/run_cn5_lagged_monthly_fuel_mix_information.py) |
| 最新：目标区最小历史小时标签预算与逐日滚动校准 | [双轨结果](LOCAL_HOURLY_LABEL_BUDGET_RESULTS_20261101.md)、[冻结分析合同](LOCAL_HOURLY_LABEL_BUDGET_PROTOCOL_20261101.md)、[逐日/逐BA证据](../results/local-hourly-label-budget-r3/README.md)、[运行脚本](../scripts/run_local_hourly_label_budget.py) |
| 最新：US↔GB端到端小时预测按发电/消费目标双轨配对，M、M+C与强重组同队列比较 | [结果与中国路线判断](PAIRED_ENDPOINT_TRACK_TRANSFER_RESULTS_20261101.md)、[冻结协议](PAIRED_ENDPOINT_TRACK_TRANSFER_PROTOCOL_20261101.md)、[机器证据包](evidence/paired-endpoint-track-transfer-r2/README.md)、[复现脚本](../scripts/run_paired_endpoint_track_transfer.py) |
| 最新：US→GB伪目标残差在发电/消费双标签同队列、同H2校准下复评分 | [结果与晋级判断](PAIRED_PSEUDOTARGET_ENDPOINT_TRACK_RESULTS_20261101.md)、[冻结协议](PAIRED_PSEUDOTARGET_ENDPOINT_TRACK_PROTOCOL_20261101.md)、[封存证据](evidence/paired-pseudotarget-endpoint-tracks-r3/README.md)、[审计脚本](../scripts/audit_paired_pseudotarget_endpoint_tracks.py) |
| 最新：BANC逐BA风险欠覆盖与月报源支持诊断 | [失效归因](LAGGED_COARSE_RISK_FAILURE_DIAGNOSTIC_20261101.md)、[源支持证据](evidence/lagged-coarse-risk-support-r1/README.md)、[诊断脚本](../scripts/analyze_lagged_coarse_risk_support.py) |
| 最新：多月滞后五类统计轨迹对小时风险的配对消融 | [结果与决策](LAGGED_TRAJECTORY_RISK_RESULTS_20261101.md)、[冻结协议](LAGGED_TRAJECTORY_RISK_PROTOCOL_20261101.md)、[生成侧证据](evidence/lagged-trajectory-risk-generation-r1/README.md)、[消费侧证据](evidence/lagged-trajectory-risk-consumption-r1/README.md)、[运行脚本](../scripts/run_lagged_trajectory_risk_transfer.py) |
| 最新：五类中国口径下M与M+C年度装机配对消融 | [冻结协议](FIVE_GROUP_CAPACITY_INCREMENT_PROTOCOL_20261101.md)、[结果报告](FIVE_GROUP_CAPACITY_INCREMENT_RESULTS_20261101.md)、[封存证据](../results/five-group-capacity-increment-r2/README.md) |
| 最新：严格CN5容量信息预算的2025时间留出 | [结果与路线判断](CN5_CAPACITY_2025_TEMPORAL_RESULTS_20261102.md)、[冻结协议](CN5_CAPACITY_2025_TEMPORAL_PROTOCOL_20261102.md)、[逐国预测、评分和哈希](../results/cn5-capacity-temporal-2025-r1/)、[运行脚本](../scripts/run_cn5_capacity_temporal_holdout_2025.py) |
| 最新：官方月报M与年度煤/气装机结构M+C的双向增量 | [US↔GB结果](EIA923_CAPACITY_INCREMENT_RESULTS_20261101.md)、[冻结协议](EIA923_CAPACITY_INCREMENT_PROTOCOL_20261101.md)、[运行证据](evidence/eia923-capacity-increment-r1/README.md) |
| 最新：火电因子头平均增量对国家与月份的敏感性 | [配对块重抽样诊断及结论](FIVE_GROUP_FACTOR_INCREMENT_UNCERTAINTY_20261101.md)、[复现脚本与机器结果](../results/five-group-factor-increment-uncertainty-r1/) |
| 最新：留BA伪目标校准与区间半径因子诊断 | [结果与研究判断](CROSSFIT_DOMAIN_RISK_CALIBRATION_RESULTS_20261101.md)、[冻结协议](CROSSFIT_DOMAIN_RISK_CALIBRATION_PROTOCOL_20261101.md)、[预测和评分证据](evidence/crossfit-domain-risk-calibration-r2/README.md) |
| 最新：新国家90%区间的有限样本域数门槛 | [国家级共形秩审计及下一轮设计](CROSS_COUNTRY_DOMAIN_LEVEL_COVERAGE_LIMIT_20261101.md) |
| 最新：五类火电因子头跨时间窗复核未通过最差域覆盖门槛 | [协议](FIVE_GROUP_FACTOR_TEMPORAL_CHECK_PROTOCOL_20261101.md)、[结果与冻结判断](FIVE_GROUP_FACTOR_TEMPORAL_CHECK_RESULTS_20261101.md)、[封存证据](../results/five-group-factor-temporal-check-r1/README.md) |
| 最新：严格五类预算伪目标因子响应收缩未通过晋级门槛 | [结果与路线判断](CN5_PSEUDOTARGET_FACTOR_RESPONSE_RESULTS_20261101.md)、[冻结协议](CN5_PSEUDOTARGET_FACTOR_RESPONSE_PROTOCOL_20261101.md)、[证据包](evidence/cn5-pseudotarget-factor-response-r1/README.md)、[复现脚本](../scripts/run_cn5_pseudotarget_factor_response.py) |
| 最新：严格五类GroupDRO嵌套伪目标未改善最差域或区间 | [结果与研究边界](CN5_GROUPDRO_PSEUDOTARGET_RESULTS_20261101.md)、[冻结协议](CN5_GROUPDRO_PSEUDOTARGET_PROTOCOL_20261101.md)、[封存证据](evidence/cn5-groupdro-pseudotarget-r1/README.md)、[运行脚本](../scripts/run_cn5_groupdro_pseudotarget.py) |
| 最新：严格CN5正则softmax组成回归未超过燃料份额HGB（负结果） | [九国结果与停止判断](CN5_SOFTMAX_COMPOSITION_TRANSFER_RESULTS_20261101.md)、[冻结协议](CN5_SOFTMAX_COMPOSITION_TRANSFER_PROTOCOL_20261101.md)、[封存证据](evidence/cn5-softmax-composition-r1/README.md)、[运行脚本](../scripts/run_cn5_softmax_composition_transfer.py) |
| 最新：严格CN5非线性log-ratio HGB仍未超过份额HGB（负结果） | [九国结果与路线停止判断](CN5_LOGRATIO_HGB_TRANSFER_RESULTS_20261101.md)、[冻结协议](CN5_LOGRATIO_HGB_TRANSFER_PROTOCOL_20261101.md)、[封存证据](evidence/cn5-logratio-hgb-transfer-r1/README.md)、[运行脚本](../scripts/run_cn5_logratio_hgb_transfer.py) |
| 最新：CN5燃料份额机制消融显示增益主要由煤动态头驱动，跨域仍不稳定 | [九国消融结果与论文边界](CN5_FUELSHARE_COMPONENT_ABLATION_RESULTS_20261102.md)、[冻结协议](CN5_FUELSHARE_COMPONENT_ABLATION_PROTOCOL_20261102.md)、[预测、区间、成分与哈希](../results/cn5-fuelshare-component-ablation-r1/)、[运行脚本](../scripts/run_cn5_fuelshare_component_ablation.py) |
| 最新：滞后非热电组成轨迹对发电侧小时响应仅有局部增量，未解决90%覆盖 | [九国结果与路线判断](CN5_LAGGED_TRAJECTORY_RESPONSE_RESULTS_20261101.md)、[冻结协议](CN5_LAGGED_TRAJECTORY_RESPONSE_PROTOCOL_20261101.md)、[预测封存/评分证据](../results/cn5-lagged-trajectory-response-r1/README.md)、[运行脚本](../scripts/run_cn5_lagged_trajectory_response.py) |
| 最新：总发电量趋势候选在新年份复现小幅增量；90%覆盖及未见地理确认仍未完成 | [2024开发结果](CN5_TOTAL_VOLUME_TRAJECTORY_RESULTS_20261101.md)、[2025时间外协议和结果](CN5_TOTAL_VOLUME_TRAJECTORY_2025_TEMPORAL_PROTOCOL_20261101.md) / [评分](CN5_TOTAL_VOLUME_TRAJECTORY_2025_RESULTS_20261101.md)、[目标数据预检](CN5_TOTAL_VOLUME_TRAJECTORY_UA_PREFLIGHT_20261101.md)、[2025逐国封存证据](../results/cn5-total-volume-trajectory-2025-r1/README.md)、[开发/时间复核脚本](../scripts/run_cn5_total_volume_trajectory.py) / [时间复核运行器](../scripts/run_cn5_total_volume_trajectory_2025_holdout.py) |
| 最新：CN5总量趋势未迁移至US→GB官方发电强度；点误差、覆盖与Winkler均未改善 | [冻结协议](CROSS_COUNTRY_US_TO_GB_VOLUME_TREND_PROTOCOL_20261102.md)、[结果与路线判断](CROSS_COUNTRY_US_TO_GB_VOLUME_TREND_RESULTS_20261102.md)、[正式runner](../scripts/run_us_to_gb_volume_trend_experiment.py)、[封存预测/评分/哈希](evidence/us-to-gb-volume-trend-r1/README.md) |
| 最新：预测器分歧对小时误差风险只有局部提示，CZ/HU/SI低分歧仍高风险 | [事后诊断与下一步门槛](CN5_PREDICTION_DISAGREEMENT_RISK_DIAGNOSTIC_20261101.md)、[逐日数据与哈希](../results/cn5-prediction-disagreement-risk-r1/)、[诊断脚本](../scripts/analyze_cn5_prediction_disagreement_risk.py) |
| 最新：严格双重整国留出分歧校准仍未保持覆盖并缩窄区间，关闭当前风险映射分支 | [结果与路线决策](CN5_DISAGREEMENT_RISK_PSEUDOTARGET_RESULTS_20261101.md)、[双重留出协议](CN5_DISAGREEMENT_RISK_NESTED_PROTOCOL_20261101.md)、[r2伪目标分数/哈希证据](../results/cn5-disagreement-risk-pseudotarget-r2/)、[运行脚本](../scripts/run_cn5_disagreement_risk_nested_pseudotarget.py) |
| 最新：源月报支持度能否提示跨国欠覆盖 | [五国留国诊断、误报边界及中国含义](LOCO_PUBLIC_INPUT_SUPPORT_RISK_20261101.md)、[分析脚本与逐月结果](../results/loco-public-input-support-risk-r2/summary.json) |
| 最新：冻结支持度告警在CZ/RS留国扩源折复核 | [域级失效、CZ完整字段反例和RS缺失压力测试](EXPANSION_PUBLIC_SUPPORT_RISK_20261101.md)、[机器结果和哈希](../results/expansion-public-support-risk-r2/summary.json) |
| 最新：源伪目标嵌套学习目标欠覆盖风险 | [六国LOCO、proper score增量及逐国失效](SOURCE_PSEUDOTARGET_RISK_META_RESULTS_20261101.md)、[冻结协议](SOURCE_PSEUDOTARGET_RISK_META_PROTOCOL_20261101.md)、[封存机器结果](../results/source-pseudotarget-risk-meta-r2/summary.json) |
| 最新：中国31省输入上的源风险模型无标签应用 | [模型分数的外推边界与后续标签验证要求](CHINA_SOURCE_RISK_META_APPLICATION_20261101.md)、[逐省分数与哈希](../results/china-source-risk-meta-r2/summary.json) |
| 当前主线与下一轮优先级 | [跨国训练、零目标小时标签部署与实验计划](CROSS_DOMAIN_TRANSFER_RESEARCH_PLAN_20261031.md) |
| 最新：四项研究目标的统一证据台账和论文边界 | [跨国小时预测、预算训练、燃料响应与区间进度](CROSS_COUNTRY_EVIDENCE_LEDGER_20261101.md) |
| 最新：US/GB→AEMO五类发电侧跨域开发检验：US+GB粗Ridge改善三地平均点误差，但90%覆盖仅46% | [结果与失效边界](AEMO_CN5_CROSS_DOMAIN_TRANSFER_RESULTS_20261102.md)、[冻结协议](AEMO_CN5_CROSS_DOMAIN_TRANSFER_PROTOCOL_20261102.md)、[封存预测/评分/哈希](../results/aemo-cn5-cross-domain-transfer-r2/)、[预检脚本](../scripts/preflight_aemo_cn5_fuel_monthly_support.py)、[运行器](../scripts/run_aemo_cn5_cross_domain_transfer.py) |
| 最新：AEMO跨域结果统一源/目标燃料EF后的标签敏感性：粗Ridge MAE改善但NSW/QLD严重欠覆盖 | [配对结果与分轨判断](AEMO_CN5_COMMON_EF_CROSS_DOMAIN_RESULTS_20261102.md)、[冻结协议](AEMO_CN5_COMMON_EF_CROSS_DOMAIN_PROTOCOL_20261102.md)、[封存证据](evidence/aemo-cn5-common-ef-cross-domain-r5/README.md) |
| 最新：五国整国留出及年度装机增量 | [粗月输入基线、荷兰失效和装机负结果](ENERGY_CHARTS_LOCO_RESULTS_20261101.md)、[LOCO协议](ENERGY_CHARTS_LOCO_PROTOCOL_20261101.md)、[装机预算协议](ENERGY_CHARTS_CAPACITY_PROTOCOL_20261101.md)、[证据包](evidence/energy-charts-loco-capacity-r3/README.md) |
| 历史：Energy-Charts六组信息预算九国扩展（含其他可再生） | [六组敏感性结果](ENERGY_CHARTS_NINE_COUNTRY_LOCO_RESULTS_20261101.md)、[冻结协议](ENERGY_CHARTS_NINE_COUNTRY_LOCO_PROTOCOL_20261101.md)、[封存证据](evidence/energy-charts-nine-country-loco-r1/README.md) |
| 历史：Energy-Charts六组预算年度装机消融 | [六组敏感性结果](ENERGY_CHARTS_NINE_COUNTRY_CAPACITY_RESULTS_20261101.md)、[冻结协议](ENERGY_CHARTS_NINE_COUNTRY_CAPACITY_PROTOCOL_20261101.md)、[封存证据](evidence/energy-charts-nine-country-capacity-r1/README.md) |
| 历史：Energy-Charts六组预算伪目标源筛除 | [六组敏感性结果](ENERGY_CHARTS_PSEUDOTARGET_SOURCE_DROP_RESULTS_20261101.md)、[冻结协议](ENERGY_CHARTS_PSEUDOTARGET_SOURCE_DROP_PROTOCOL_20261101.md)、[封存证据](evidence/energy-charts-pseudotarget-source-drop-r1/README.md) |
| 最新：严格中国五类输入下九国留出、装机和源筛除 | [五类主基线](ENERGY_CHARTS_NINE_COUNTRY_CN5_RESULTS_20261101.md)、[装机增量](ENERGY_CHARTS_NINE_COUNTRY_CN5_CAPACITY_RESULTS_20261101.md)、[伪目标源筛除](ENERGY_CHARTS_PSEUDOTARGET_SOURCE_DROP_CN5_RESULTS_20261101.md)、[输入预算更正](ENERGY_CHARTS_INPUT_BUDGET_CORRIGENDUM_20261101.md) |
| 最新：五个Energy-Charts源到SMARD德国跨平台留出 | [德国分月失败与强基线比较](ENERGY_CHARTS_TO_SMARD_GERMANY_HOLDOUT_RESULTS_20261101.md)、[冻结协议](ENERGY_CHARTS_TO_SMARD_GERMANY_HOLDOUT_PROTOCOL_20261101.md)、[预测与审计证据](evidence/energy-charts-smard-germany-holdout-r1/README.md) |
| 最新：月度发电量/年度容量利用率未带来跨国迁移增量 | [五国结果与荷兰覆盖诊断](MONTHLY_CAPACITY_UTILIZATION_TRANSFER_RESULTS_20261101.md)、[冻结协议](MONTHLY_CAPACITY_UTILIZATION_TRANSFER_PROTOCOL_20261101.md)、[预测与输入证据](evidence/monthly-capacity-utilization-r1/README.md) |
| 最新：伪目标训练选择、源专家门控和区间失败 | [逐轮结果与论文创新边界](PSEUDOTARGET_TRANSFER_RESULTS_20261101.md)、[伪目标LOCO协议](PSEUDOTARGET_LOCO_SELECTION_PROTOCOL_20261101.md)、[组成条件专家协议](COARSE_CONDITIONAL_EXPERT_TRANSFER_PROTOCOL_20261101.md)、[预测/校准证据](evidence/coarse-conditional-experts-r5/README.md) |
| 最新：官方消费侧US BA伪目标专家门控未迁移至GB | [GB结果及失败归因](OFFICIAL_PSEUDOTARGET_BA_ENSEMBLE_RESULTS_20261101.md)、[冻结协议](OFFICIAL_PSEUDOTARGET_BA_ENSEMBLE_PROTOCOL_20261101.md)、[运行证据](evidence/official-pseudotarget-ba-ensemble-r1/README.md) |
| 最新：五类中国预算下整国episodic训练未超过强重组 | [MLDG式训练结果和下一步](EPISODIC_COLDSTART_RESULTS_20261101.md)、[冻结协议](EPISODIC_COLDSTART_TRANSFER_PROTOCOL_20261101.md)、[封存证据](evidence/episodic-coldstart-r1/README.md) |
| 最新：门控专家的组成支持距离仅部分解释收益 | [事后诊断与路线判断](COARSE_EXPERT_SUPPORT_DIAGNOSTIC_20261101.md)、[分层数据证据](evidence/coarse-expert-support-diagnostic-r1/README.md) |
| 最新：整国伪目标残差训练仍未超过强迁移基线 | [源端收缩结果与研究判断](CROSS_FITTED_PSEUDOTARGET_RESIDUAL_RESULTS_20261101.md)、[冻结协议](CROSS_FITTED_PSEUDOTARGET_BACKOFF_PROTOCOL_20261101.md)、[封存证据](evidence/crossfit-pseudotarget-residual-r3/README.md) |
| 最新：伪目标组成条件训练重加权未达门槛 | [实验结果与后续研究重点](PSEUDOTARGET_INSTANCE_REWEIGHTING_RESULTS_20261101.md)、[冻结协议](PSEUDOTARGET_INSTANCE_REWEIGHTING_PROTOCOL_20261101.md)、[预测与运行证据](evidence/pseudotarget-instance-reweighting-r1/README.md) |
| 最新：跨国官方小时强度标签接入审计 | [EIA-930与NESO标签定义、可比轨道和执行顺序](OFFICIAL_HOURLY_CIF_LABEL_AUDIT_20261101.md) |
| 最新：本地发电量对消费侧CIF的识别边界 | [EIA发电/消费强度差分解及中国信息需求](OFFICIAL_HOURLY_CIF_LABEL_AUDIT_20261101.md)、[逐BA证据](evidence/generation-consumption-intensity-gap-r1/README.md) |
| 最新：真实EIA-923月报下的官方消费强度跨国预测 | [US↔GB结果、覆盖与输入预算判断](EIA923_OFFICIAL_CONSUMPTION_INPUT_BUDGET_RESULTS_20261101.md)、[冻结协议](EIA923_OFFICIAL_CONSUMPTION_INPUT_BUDGET_PROTOCOL_20261101.md)、[封存证据](evidence/eia923-official-consumption-input-r1/README.md) |
| 最新：官方消费侧标签下的年度装机增量 | [双向结果与失效原因](OFFICIAL_CONSUMPTION_CAPACITY_TRANSFER_RESULTS_20261101.md)、[实验协议](OFFICIAL_CONSUMPTION_INTENSITY_TRANSFER_PROTOCOL_20261101.md)、[封存证据](evidence/official-intensity-transfer-capacity-r1/README.md) |
| 年度煤气容量年份更新是否改善US↔GB迁移 | [2020对2022配对回测与版本限制](CAPACITY_2022_OFFICIAL_PAIRED_DEVELOPMENT_RESULTS.md)、[EIA/DUKES容量值审计](evidence/public-capacity-2022-asof-2024-r1/audit_summary.json)、[预测和指标证据](evidence/capacity-2022-paired-official-development-r4/README.md) |
| 最新：2026年11月GB→US消费侧真实发行版前瞻预测（待评分） | [封存状态与协议档案限制](PROSPECTIVE_GB_TO_US_NOV2026_FORECAST_RESULTS.md)、[重建合同说明](PROSPECTIVE_GB_TO_US_EIA923_NOV2026_PROTOCOL.md)、[七区预测文件与哈希](evidence/prospective-gb-to-us-eia923-nov2026-r1/README.md) |
| 最新：2026年11月GB→US同BA五类输入严格完整行前瞻预测（发电侧代理，待评分） | [冻结协议](PROSPECTIVE_GB_TO_US_EIA923_STRICT_NOV2026_PROTOCOL.md)、[两区封存状态与限制](PROSPECTIVE_GB_TO_US_EIA923_STRICT_NOV2026_RESULTS.md)、[预测入口](../scripts/run_prospective_gb_to_us_eia923_strict_nov2026.py)、[延迟评分](../scripts/score_prospective_gb_to_us_eia923_strict_nov2026.py)、[预测哈希证据包](evidence/prospective-gb-to-us-eia923-strict-nov2026-r1/README.md) |
| 最新：2026年11月US→GB配对前瞻预测（待评分） | [封存状态与解释边界](PROSPECTIVE_US_TO_GB_NOV2026_FORECAST_RESULTS.md)、[冻结协议](PROSPECTIVE_US_TO_GB_NOV2026_PROTOCOL.md)、[本地预测与哈希](../results/prospective-us-to-gb-neso-nov2026-r1/) |
| 同任务的最新2025年度煤气装机候选（待评分） | [装机增量候选与可识别性结果](PROSPECTIVE_CAPACITY_2025_CANDIDATE_RESULTS.md)、[运行脚本](../scripts/run_prospective_capacity_2025_candidate.py)、[预测/哈希证据](evidence/prospective-capacity-2025-candidate-nov2026-r1/README.md) |
| 最新：官方小时标签下源伪目标专家增量未跨国稳定复现 | [US→GB回退结果和论文边界](OFFICIAL_PSEUDOTARGET_BA_BACKOFF_RESULTS_20261101.md)、[冻结协议](OFFICIAL_PSEUDOTARGET_BA_BACKOFF_PROTOCOL_20261101.md)、[封存证据](evidence/official-pseudotarget-ba-backoff-r1/README.md) |
| 最新：伪目标残差修正能否迁移到官方消费强度 | [源端留BA选择与GB负迁移结果](OFFICIAL_PSEUDOTARGET_RECOMPOSITION_RESIDUAL_RESULTS_20261101.md)、[冻结协议](OFFICIAL_PSEUDOTARGET_RECOMPOSITION_RESIDUAL_PROTOCOL_20261101.md)、[封存证据](evidence/official-pseudotarget-recomposition-residual-r1/README.md) |
| 最新：生成侧一致代理下的伪目标残差配对复验 | [与官方消费标签的轨道对照](PSEUDOTARGET_RECOMPOSITION_RESIDUAL_GENERATION_PROXY_RESULTS_20261101.md)、[冻结协议](PSEUDOTARGET_RECOMPOSITION_RESIDUAL_GENERATION_PROXY_PROTOCOL_20261101.md)、[封存证据](evidence/pseudotarget-recomposition-residual-generation-proxy-r2/README.md) |
| 最新：2020年度煤／气装机同预算增量 | [三国结果与创新边界](OFFICIAL_ANNUAL_COAL_GAS_CAPACITY_RESULTS_20261101.md)、[冻结协议](OFFICIAL_ANNUAL_COAL_GAS_CAPACITY_PROTOCOL_20261101.md)、[封存证据](evidence/annual-coal-gas-capacity-r1/README.md) |
| 最新：扩大到2400小时后天气是否有增量 | [R6b天气预算对照与失效边界](EXPANDED_ANCHOR_ISSUED_RESULTS.md)、[固定协议](EXPANDED_ANCHOR_TRANSFER_PROTOCOL.md) |
| 最新：24窗GFS对象在预测起点前已存在于NOAA公共S3档案 | [对象级时间戳审计结果](GFS_PREORIGIN_PUBLIC_ARCHIVE_AUDIT_RESULTS_20261101.md)、[审计协议](GFS_PREORIGIN_PUBLIC_ARCHIVE_AUDIT_PROTOCOL_20261101.md)、[逐对象清单](../results/gfs-preorigin-public-archive-audit-r2/object_timestamps.csv) |
| R6b天气增量按目标国拆分 | [三国差异诊断与预算边界](EXPANDED_ANCHOR_ISSUED_RESULTS.md)、[机器结果和复算代码](evidence/expanded-weather-country-diagnostic-r1/README.md) |
| 年度装机预算如何对齐中国可用字段 | [年度装机跨域增量实验设计](ANNUAL_CAPACITY_CROSS_DOMAIN_NEXT_STEP_20261101.md)、[官方年份与范围审计](PUBLIC_CAPACITY_ANNUAL_AUDIT.md) |
| 2026年11月预测起点可见的最新年度装机 | [EIA-860/DUKES 2025统计年审计与预算增量边界](PUBLIC_CAPACITY_2025_ASOF_NOV2026_AUDIT.md)、[官方原始数据与逐BA表](evidence/public-capacity-2025-asof-nov2026-r1/README.md) |
| 总发电/火电预测能否超过不变外推 | [48套普通源端参考及OOF负结果](DAILY_POWER_SOURCE_RESULTS.md)、[源端协议](DAILY_POWER_SOURCE_PROTOCOL.md) |
| 最新：源校准日历扩充能否可靠缩宽 | [R6c同2400目标小时回放与负结果](EXPANDED_CALIBRATION_BUDGET_RESULTS.md)、[事后开发协议](EXPANDED_CALIBRATION_BUDGET_PROTOCOL.md) |
| 最新：扩展到2400小时，普通锚定能否超过重组 | [R6a无当期预报对照、强基线与局部失败](EXPANDED_ANCHOR_NO_ISSUED_RESULTS.md)、[执行协议](EXPANDED_ANCHOR_TRANSFER_PROTOCOL.md) |
| 最新：同源标签的直接预测是否追平重组 | [R5直接CIF、火电占比、异方差与CQR结果](HOURLY_DIRECT_BASELINES_RESULTS.md)、[预测前协议](HOURLY_DIRECT_BASELINES_PROTOCOL.md)、[精简证据](evidence/hourly-direct-baselines-r5/README.md) |
| 当前：扩大日期与月报变化量强对照 | [48源模型完成、采集进行中及源内负结果](SOURCE_ANCHOR_TRAINING_STATUS.md)、[开发设计](TRANSFER_NEXT_DEVELOPMENT_DESIGN_20261101.md) |
| 最新稳健性更正：日模型冗余特征 | [14个零均值输入清理、HGB收益保留及Ridge敏感性](DAILY_RESOURCE_NUMERICAL_RESULTS.md)、[协议](DAILY_RESOURCE_NUMERICAL_PROTOCOL.md) |
| 最新：源天气配对是否限制了迁移 | [三组A/B/C和35448小时源扩展结果](SOURCE_RESOURCE_EXPANSION_RESULTS.md)、[三组协议](SOURCE_RESOURCE_EXPANSION_PROTOCOL.md)、[精简证据](evidence/resource-response-development-r2-r3/README.md) |
| 日幅度普通强参考是否已足够 | [饱和曲线、约束回归和RF/HGB结果](RESOURCE_CONTROLS_RESULTS.md)、[固定协议](RESOURCE_CONTROLS_PROTOCOL.md) |
| 最新：历史天气参照与日发电幅度 | [600年月、两轨道与10参考结果](RESOURCE_AMPLITUDE_RESULTS.md)、[开发协议](RESOURCE_REFERENCE_DEVELOPMENT_PROTOCOL.md)、[封存证据](evidence/resource-amplitudes-r1/README.md) |
| 最新：强重组内天气增量及日幅度诊断 | [恒等约束、8参考与负结果](COARSE_WEATHER_SHAPE_RESULTS.md)、[开发协议](COARSE_WEATHER_SHAPE_PROTOCOL.md)、[冻结证据](evidence/coarse-weather-shapes-r1/README.md) |
| 新主线首轮实际实验 | [美英小时冷启动结果及限制](CROSS_COUNTRY_PILOT_RESULTS.md)、[执行协议](CROSS_COUNTRY_PILOT_PROTOCOL.md) |
| 粗预算训练与细燃料监督消融 | [时效增强和辅助监督结果](CROSS_COUNTRY_BUDGET_TRAINING_RESULTS.md)、[执行协议](CROSS_COUNTRY_BUDGET_TRAINING_PROTOCOL.md) |
| 粗组成参与解码及受限增量 | [燃料因子/形状/残差对照结果](CROSS_COUNTRY_STRUCTURED_RESULTS.md)、[执行协议](CROSS_COUNTRY_STRUCTURED_PROTOCOL.md) |
| 历史月报适配与概率强基线 | [合法适配、异方差与CQR结果](CROSS_COUNTRY_MONTHLY_ADAPTATION_RESULTS.md)、[执行协议](CROSS_COUNTRY_MONTHLY_ADAPTATION_PROTOCOL.md) |
| 最新：合法月量适配在发电/官方消费标签上的双轨配对 | [结果与晋级判断](DUAL_TRACK_MONTHLY_ADAPTATION_RESULTS_20261101.md)、[冻结协议](DUAL_TRACK_MONTHLY_ADAPTATION_PROTOCOL_20261101.md)、[预测封存和分区指标](../results/dual-track-monthly-adaptation-r1/README.md)、[复现脚本](../scripts/run_dual_track_monthly_adaptation.py) |
| 最新：本地Ridge适配的七BA伪目标复核未达可信覆盖门槛 | [逐BA结果与路线判断](LOCAL_MONTHLY_ADAPTATION_PSEUDOTARGET_RESULTS_20261101.md)、[冻结协议](LOCAL_MONTHLY_ADAPTATION_PSEUDOTARGET_PROTOCOL_20261101.md)、[封存证据](../results/local-monthly-adaptation-pseudotarget-r1/README.md)、[运行脚本](../scripts/run_local_monthly_adaptation_pseudotargets.py) |
| 最新：本地月量Ridge改善组成预测但未传递为可靠小时强度 | [后验拆分与瓶颈判断](LOCAL_MONTHLY_MIX_ACCURACY_DIAGNOSTIC_20261101.md)、[结果证据](../results/local-monthly-mix-accuracy-r1/README.md)、[分析脚本](../scripts/analyze_local_monthly_mix_accuracy.py) |
| 最新：真实月报发行版本的修订敏感性及NESO历史快照缺口 | [EIA发行档案、英国/中国版本边界和NESO补充审计](ACTUAL_MONTHLY_VINTAGE_AUDIT_20261101.md)、[发行版本证据包](evidence/actual-monthly-vintage-audit-r1/README.md) |
| 新增：NESO公开历史燃料接口前瞻归档起点（当前修订快照，非as-issued） | [2026-11-02数据与接口边界](evidence/neso-public-hourly-snapshot-20261102/README.md)、[月度版本审计更新](ACTUAL_MONTHLY_VINTAGE_AUDIT_20261101.md) |
| 新增：US→GB 2026年12月同目标窗口t−2与t−3公开月量前瞻配对（未评分） | [封存预测和结果边界](PROSPECTIVE_GENERATION_US_GB_DEC2026_TMINUS2_RESULTS.md)、[冻结协议](PROSPECTIVE_GENERATION_US_GB_DEC2026_TMINUS2_PROTOCOL.md)、[预测哈希](../results/prospective-generation-us-gb-dec2026-tminus2-r2/README.md) |
| 最新：EPM月报年度修订敏感性双轨回放 | [结果与边界](LOWER48_MONTHLY_VINTAGE_SENSITIVITY_RESULTS_20261102.md)、[冻结协议](LOWER48_MONTHLY_VINTAGE_SENSITIVITY_PROTOCOL_20261102.md)、[r2封存预测与评分](../results/lower48-monthly-vintage-sensitivity-r2/)、[运行脚本](../scripts/run_lower48_monthly_vintage_sensitivity.py) |
| EPM首发/修订逐月双轨失效拆分 | [分析报告](LOWER48_MONTHLY_VINTAGE_MONTHLY_FAILURE_ANALYSIS_20261102.md)、[96个逐月配对结果](../results/lower48-monthly-vintage-sensitivity-r2/monthly_failure_breakdown.csv) |
| 最新：真实EPM首发统计年龄敏感性双轨回放 | [结果与限制](LOWER48_ASISSUED_MONTHLY_AGE_RESULTS_20261102.md)、[冻结协议](LOWER48_ASISSUED_MONTHLY_AGE_PROTOCOL_20261102.md)、[封存结果](../results/lower48-asissued-monthly-age-r1/)、[运行脚本](../scripts/run_lower48_asissued_monthly_age.py) |
| 最新：EIA-923 BA级月量可接小时区域但缺历史as-issued链 | [BA月量对齐结果和迁移边界](EIA923_BA_MONTHLY_ALIGNMENT_AUDIT_20261101.md)、[配对数据证据包](evidence/eia923-ba-mapping-r1/README.md) |
| 最新：EIA-923月报接入US↔GB本地适配，方向性增益仍有MISO欠覆盖 | [适配和区间结果](EIA923_MONTHLY_ADAPTATION_RESULTS_20261101.md)、[冻结协议](EIA923_MONTHLY_ADAPTATION_PROTOCOL_20261101.md)、[运行证据](evidence/eia923-monthly-adaptation-r1/README.md) |
| 最新：EIA-923真实BA月量驱动留区伪目标，源内增益未迁移至GB | [配对结果与研究判断](EIA923_PSEUDOTARGET_MONTHLY_TRANSFER_RESULTS_20261101.md)、[冻结协议](EIA923_PSEUDOTARGET_MONTHLY_TRANSFER_PROTOCOL_20261101.md)、[运行证据包](evidence/eia923-monthly-pseudotarget-r1/README.md) |
| 重要更正：2020–2021 EIA-923旧表WAT水电映射及受影响实验重跑 | [映射更正与修正前后影响](EIA923_2020_2021_FUEL_MAPPING_CORRECTION_20261101.md)、[EIA-923月报适配报告](EIA923_MONTHLY_ADAPTATION_RESULTS_20261101.md) |
| 最新：EIA-923官方BA月统计跨国增量 | [US↔GB双向结果与覆盖判断](EIA923_CROSS_COUNTRY_EXPERIMENT_RESULTS_20261101.md)、[冻结协议](EIA923_CROSS_COUNTRY_EXPERIMENT_PROTOCOL_20261101.md)、[封存证据](evidence/eia923-crosscountry-monthly-r1/README.md) |
| 最新：年度化石热耗预算对官方消费强度的增量（点误差改善但覆盖恶化） | [结果与边界](THERMAL_HEATRATE_OFFICIAL_INCREMENT_RESULTS_20261101.md)、[冻结协议](THERMAL_HEATRATE_OFFICIAL_INCREMENT_PROTOCOL_20261101.md)、[证据包](evidence/thermal-rate-official-increment-r1/README.md) |
| 热耗点预测与区间半径的事后交叉分解 | [交叉表与解释](THERMAL_HEATRATE_OFFICIAL_INCREMENT_RESULTS_20261101.md)、[诊断脚本](../scripts/diagnose_thermal_rate_interval_decomposition.py) |
| 最新：公开年度装机对官方月量的独立增量 | [同任务配对结果与负增量分析](EIA923_CAPACITY_INCREMENT_RESULTS_20261101.md)、[冻结协议](EIA923_CAPACITY_INCREMENT_PROTOCOL_20261101.md)、[封存证据](evidence/eia923-capacity-increment-r1/README.md) |
| 最新：装机到火电因子的源端校准诊断 | [留BA结果及跨国边界](CAPACITY_SOURCE_CALIBRATION_AUDIT_RESULTS_20261101.md)、[冻结协议](CAPACITY_SOURCE_CALIBRATION_AUDIT_PROTOCOL_20261101.md)、[证据包](evidence/capacity-source-calibration-r1/README.md) |
| 最新：源伪目标小时监督能否学习粗组成门控 | [五国结果与论文边界](PSEUDOTARGET_COARSE_CONDITIONAL_STACK_RESULTS_20261101.md)、[冻结协议](PSEUDOTARGET_COARSE_CONDITIONAL_STACK_PROTOCOL_20261101.md)、[证据包](evidence/pseudotarget-coarse-conditional-stack-r1/README.md) |
| 最新：严格五类预算下伪目标条件门控未超过强基线 | [九国结果与论文边界](PSEUDOTARGET_COARSE_CONDITIONAL_STACK_CN5_RESULTS_20261101.md)、[冻结协议](PSEUDOTARGET_COARSE_CONDITIONAL_STACK_CN5_PROTOCOL_20261101.md)、[封存证据](evidence/pseudotarget-coarse-conditional-stack-cn5-r1/README.md) |
| 最新：NL比例重组的份额与内部因子误差分解 | [事后诊断与下一步机制](NL_RATIO_RECOMPOSITION_ERROR_DECOMPOSITION_RESULTS_20261101.md)、[冻结诊断协议](NL_RATIO_RECOMPOSITION_ERROR_DECOMPOSITION_PROTOCOL_20261101.md)、[逐小时证据](evidence/nl-recomposition-error-r1/README.md) |
| 最新：年度装机增量与因子失效主线综合判断 | [跨域证据综合与下一步](CROSS_DOMAIN_CAPACITY_AND_ERROR_SYNTHESIS_20261101.md)、[五国火电因子分解证据](evidence/factor-head-transfer-decomposition-r1/README.md) |
| 最新：GB源区扩展到NESO 14区后对GB→US的影响 | [结果与风险覆盖判断](NESO_REGIONAL_SOURCE_TRANSFER_RESULTS_20261101.md)、[冻结协议](NESO_REGIONAL_SOURCE_TRANSFER_PROTOCOL_20261101.md)、[原始数据与预测证据](evidence/neso-regional-source-transfer-r2/README.md) |
| 最新：伪目标学习热电因子响应收缩 | [五国结果与创新边界](PSEUDOTARGET_FACTOR_RESPONSE_SHRINKAGE_RESULTS_20261101.md)、[冻结协议](PSEUDOTARGET_FACTOR_RESPONSE_SHRINKAGE_PROTOCOL_20261101.md)、[运行证据](evidence/pseudotarget-factor-response-shrinkage-r1/README.md) |
| 第三个国家数据接入 | [德国SMARD原始发电采集与边界](SMARD_THIRD_COUNTRY_ACQUISITION.md) |
| 源域地理留出选型 | [国内收益与跨国失效结果](CROSS_COUNTRY_SOURCE_HOLDOUT_RESULTS.md)、[冻结协议](CROSS_COUNTRY_SOURCE_HOLDOUT_PROTOCOL.md) |
| 三国整国留出及历史条件化 | [单源/联合源与火电因子对照](THREE_COUNTRY_TRANSFER_RESULTS.md)、[冻结协议](THREE_COUNTRY_TRANSFER_PROTOCOL.md) |
| 中国公开输入无标签推理 | [上海接口、时间凭证与外推诊断](SHANGHAI_PUBLIC_INFERENCE_RESULTS.md) |
| 最新：中国31省实际月报无标签批量推理 | [31省接入结果、源支持范围与限制](CHINA_ALL_PROVINCES_PUBLIC_INFERENCE_RESULTS_20261101.md)、[逐省输出与哈希证据](evidence/china-province-public-inference-r2/README.md) |
| 最新：中国31省公开月报在预测起点前可见性审计 | [NBS时间、字段闭合、上海导出对照与限制](CHINA_PROSPECTIVE_NBS_ORIGIN_AUDIT_20261101.md)、[逐省快照及复现产物](../results/china-nbs-prospective-origin-audit-r1/) |
| 最新：高火电源域扩展改变中国输入支持但产生负迁移 | [塞尔维亚六国留出、区间代价与新目标失败](ENERGY_CHARTS_SERBIA_EXPANSION_RESULTS_20261101.md)、[冻结协议](ENERGY_CHARTS_SERBIA_EXPANSION_PROTOCOL_20261101.md)、[预测及范围诊断证据](evidence/energy-charts-serbia-expansion-r2/README.md) |
| 最新：捷克完整小时数据源域扩展 | [六国留出及原五目标配对结果](ENERGY_CHARTS_CZECH_EXPANSION_RESULTS_20261101.md)、[冻结协议](ENERGY_CHARTS_CZECH_EXPANSION_PROTOCOL_20261101.md)、[修正后证据包](evidence/energy-charts-czech-expansion-r4/README.md) |
| 最新：捷克来源内容与校准增量拆分 | [CZ、等量PX伪来源和强基线结果](CZECH_SOURCE_CONTENT_DECOMPOSITION_RESULTS_20261101.md)、[冻结协议](CZECH_SOURCE_CONTENT_DECOMPOSITION_PROTOCOL_20261101.md)、[支持版本r4封存证据](evidence/czech-source-content-decomposition-r4/README.md) |
| CZ/PX来源内容敏感性复核 | [五个预先指定PX种子结果](CZECH_PSEUDO_SOURCE_SEED_SENSITIVITY_RESULTS_20261101.md)、[五种子证据包](evidence/czech-source-content-decomposition-px-seeds-r1/README.md) |
| EIA真实月报发布版BA接入审计 | [2026年8月发行版对齐结果](LIVE_EIA923_BA_VINTAGE_AUDIT_RESULTS_20261101.md)、[冻结协议](LIVE_EIA923_BA_VINTAGE_AUDIT_PROTOCOL_20261101.md)、[原始数据与证据包](evidence/eia923-live-monthly-2026-08-r1/README.md) |
| 同预算供体支持扩展 | [无核供体与有核控制结果](SOURCE_SUPPORT_EXPANSION_RESULTS.md)、[冻结协议](SOURCE_SUPPORT_EXPANSION_PROTOCOL.md) |
| 供体校准恢复及区间代价 | [恢复结果与时间更正](SOURCE_CALIBRATION_RECOVERY_RESULTS.md)、[执行规则](SOURCE_CALIBRATION_RECOVERY_PROTOCOL.md) |
| 新增公开装机信息预算 | [跨国信息增量及简单锚对照](PUBLIC_CAPACITY_TRANSFER_RESULTS.md)、[静态代理协议及年度限制](PUBLIC_CAPACITY_TRANSFER_PROTOCOL.md) |
| 装机条件下的概率强基线 | [异方差、CQR及最差地区结果](CAPACITY_CONDITIONAL_RISK_RESULTS.md)、[执行协议](CAPACITY_CONDITIONAL_RISK_PROTOCOL.md) |
| 高煤源支持与剩余误差定位 | [同预算供体、oracle及轨迹主线](CAPACITY_SOURCE_SUPPORT_RESULTS.md)、[执行协议](CAPACITY_SOURCE_SUPPORT_PROTOCOL.md) |
| 跨国粗预算联合轨迹强参考 | [联合/独立、均值与analog完整结果](COARSE_TRAJECTORY_RESULTS.md)、[执行协议](COARSE_TRAJECTORY_PROTOCOL.md) |
| 组成变化与小时形状配对消融 | [配对无独立增量结果](COARSE_RESPONSE_RESULTS.md)、[执行协议](COARSE_RESPONSE_PROTOCOL.md) |
| 合法预报附加预算可行性 | [真实运行、预报时距及索引审计](LEGAL_WEATHER_ARCHIVE_AUDIT.md) |
| 原生预报实取与起点合同 | [24h窗口、紧凑缓存及独立核验](GFS_NATIVE_PILOT_RESULTS.md) |
| 跨国天气小样本信息价值 | [完整负结果与失效归因](WEATHER_TRANSFER_SEASONAL_RESULTS.md)、[冻结协议](WEATHER_TRANSFER_SEASONAL_PROTOCOL.md)、[执行状态](WEATHER_TRANSFER_SEASONAL_STATUS.md) |
| 无小时标签迁移创新边界 | [近期CIF／风电及聚合监督文献核对](COLD_START_NOVELTY_CHECK_20261031.md) |
| 年度装机版本和范围 | [官方年份、修订及地域审计](PUBLIC_CAPACITY_ANNUAL_AUDIT.md) |
| 用户真实信息预算 | [NBS月字段边界](CHINA_MONTHLY_GENERATION_FIELD_AUDIT_V51.md)、[上海附件核对](V409_SHANGHAI_UPLOAD_RECONCILIATION.md) |
| 官方31省面板与发布时间 | [V392官方数据审计](V392_NBS_OFFICIAL_LONG_PANEL_AUDIT.md)、[V394前向状态结果](V394_ALL_PROVINCE_RELEASE_LAG_STATE_RESULTS.md) |
| 发电侧与消费侧边界 | [V401上海电力平衡](V401_SHANGHAI_ESTIMAND_BOUNDARY.md)、[V402高频标签来源](V402_HOURLY_LABEL_PROVENANCE_AND_BENCHMARK_GATE.md) |
| 最新：中国双轨目标的标签可用性与下一步晋级门槛 | [发电/消费侧分轨、参考标签等级和研究边界](CHINA_DUAL_TRACK_LABEL_READINESS_20261101.md) |
| 最新：中国消费侧年度省际OD流量数据审计 | [2023年度TWh矩阵、可用预算与小时验证限制](CHINA_CONSUMPTION_TRACK_ANNUAL_FLOW_DATA_AUDIT_20261101.md) |
| 最新：双轨小时预测输入可得性审计 | [中国电力市场披露规则、NESO/EIA历史预测限制与M+F/C1+Fflow门槛](DUAL_TRACK_FORECAST_INPUT_AVAILABILITY_AUDIT_20261102.md) |
| 最新：中国小时预测曲线公众访问预检 | [上海/浙江官方入口、权限层级和M+F/C1+Fflow纳入条件](CHINA_PUBLIC_FORECAST_ACCESS_PREFLIGHT_20261102.md)、[双轨创新路线决策更新](DUAL_TRACK_INNOVATION_DECISION_20261102.md) |
| 最新：公众数据预算下的创新边界和下一项实验 | [源形状集合迁移设计、LP信息边界和停止条件](PUBLIC_ONLY_NEXT_RESEARCH_DECISION_20261102.md) |
| 最新：严格CN5源子集分歧无法以可接受区间效率修复最差国覆盖 | [冻结协议](CN5_SOURCE_SUBSET_DISPERSION_PROTOCOL_20261102.md)、[九国逐国结果](CN5_SOURCE_SUBSET_DISPERSION_RESULTS_20261102.md)、[252个源模型子集封存](../results/cn5-source-subset-dispersion-r1/) |
| 最新：NESO风电形状在2025时间留出有小幅信息增益但未过晋级门槛 | [结果与停止决定](NESO_WIND_FORECAST_SHAPE_2025_TEMPORAL_RESULTS_20261102.md)、[冻结协议](NESO_WIND_FORECAST_SHAPE_2025_TEMPORAL_PROTOCOL_20261102.md)、[封存和评分证据](evidence/neso-wind-forecast-shape-2025-r1/README.md) |
| NESO已签发风光形状在2024开发回放的增量与停止决定 | [结果与独立评分复算](NESO_AS_ISSUED_VRE_SHAPE_TRANSFER_RESULTS_20261102.md)、[冻结协议](NESO_AS_ISSUED_VRE_SHAPE_TRANSFER_PROTOCOL_20261102.md)、[预测与哈希证据](evidence/neso-as-issued-vre-shape-transfer-r1/README.md) |
| 最新：US→Denmark粗月预算小时迁移压力测试，四法源区间均严重欠覆盖 | [结果与边界](DENMARK_HOURLY_TRANSFER_STRESS_RESULTS_20261102.md)、[冻结协议](DENMARK_HOURLY_TRANSFER_STRESS_PROTOCOL_20261102.md)、[预测封存和机器结果](evidence/denmark-hourly-transfer-stress-r1/README.md)、[运行脚本](../scripts/run_denmark_transfer_stress.py) |
| 最新：NESO 2024小时风光预报历史签发版本通过信息入口预检 | [362天/17,376半小时、固定提前量与预报—实况诊断](evidence/neso-embedded-forecast-vintage-r1/README.md)、[双轨信息预算与下一步](DUAL_TRACK_FORECAST_INPUT_AVAILABILITY_AUDIT_20261102.md) |
| 固定宽度预算下的源风险月度分配（负结果） | [冻结协议](SOURCE_RISK_WIDTH_ALLOCATION_PROTOCOL_20261102.md)、[结果与停止决定](SOURCE_RISK_WIDTH_ALLOCATION_RESULTS_20261102.md)、[机器证据说明](evidence/source-risk-width-allocation-r2/README.md) |
| 前一年同月负荷形状增量（负结果） | [冻结协议](CN5_LAGGED_LOAD_SHAPE_TRANSFER_PROTOCOL_20261102.md)、[九国结果与停止决定](CN5_LAGGED_LOAD_SHAPE_TRANSFER_RESULTS_20261102.md)、[机器证据说明](evidence/cn5-lagged-load-shape-r3/README.md) |
| 最新：严格五类伪目标下粗Ridge未复现跨域增益 | [九国结果与双轨判断](CN5_LATEST_COARSE_RIDGE_PSEUDOTARGET_RESULTS_20261102.md)、[冻结协议](CN5_LATEST_COARSE_RIDGE_PSEUDOTARGET_PROTOCOL_20261102.md)、[机器证据](evidence/cn5-latest-ridge-loco-r1/README.md)、[运行脚本](../scripts/run_cn5_latest_ridge_loco.py) |
| 最新：燃料组成头/火电因子头的源端嵌套混合未超过单头强基线 | [九国外层结果与停止决定](CN5_FUEL_FACTOR_HEAD_BLEND_RESULTS_20261102.md)、[冻结协议](CN5_FUEL_FACTOR_HEAD_BLEND_PROTOCOL_20261102.md)、[运行脚本](../scripts/run_cn5_fuel_factor_head_blend.py)、[封存证据说明](evidence/cn5-fuel-factor-head-blend-r1/README.md) |
| 最新：按目标五类组成路由火电因子头/燃料组成头未通过源内双重留出 | [源内及2024已看目标结果](CN5_FUEL_FACTOR_HEAD_PROFILE_GATE_RESULTS_20261102.md)、[冻结协议](CN5_FUEL_FACTOR_HEAD_PROFILE_GATE_PROTOCOL_20261102.md)、[分析脚本](../scripts/analyze_cn5_fuel_factor_head_profile_gate.py)、[机器输出](../results/cn5-fuel-factor-head-profile-gate-r2/) |
| 最新：预测头分歧不能可靠排序高风险日，停止选择性拒绝分支 | [源端双重留国风险排序诊断](CN5_DISAGREEMENT_SELECTIVE_RISK_RESULTS_20261102.md)、[分析脚本](../scripts/analyze_cn5_disagreement_selective_risk.py)、[机器结果](../results/cn5-disagreement-selective-risk-r1/) |
| 最新：月度目标水平的Oracle纠错上限与源伪目标偏差迁移失败 | [水平—形状Oracle分解](CN5_LEVEL_SHAPE_ORACLE_RESULTS_20261102.md)、[月均Ridge校正协议与结果](CN5_SOURCE_MONTHLY_BIAS_META_PROTOCOL_20261102.md)、[结果分析](CN5_SOURCE_MONTHLY_BIAS_META_RESULTS_20261102.md)、[运行脚本](../scripts/run_cn5_source_monthly_bias_meta.py)、[有效机器结果r4](../results/cn5-source-monthly-bias-meta-r4/)、[跟踪哈希和结果快照](evidence/cn5-source-monthly-bias-meta-r4/README.md) |
| 最新：冻结燃料组成响应头在葡萄牙新小时域获得正向点预测复核，但与因子历史头持平 | [结果与边界](CN5_FUEL_RESOLVED_PT_CONFIRMATION_RESULTS_20261102.md)、[预测前协议](CN5_FUEL_RESOLVED_PT_CONFIRMATION_PROTOCOL_20261102.md)、[哈希证据](evidence/cn5-fuel-resolved-pt-confirmation-r3/README.md) |
| 部分识别求解器 | [V108方法与门槛](V108_SHARP_SET_METHOD_AND_PAPER_GATE.md)、[V109独立LP核验](V109_INDEPENDENT_LP_SHARP_SET_AUDIT_RESULTS.md) |
| 最小阈值侧信息 | [V97匹配容量对照](V97_SIDE_VS_DETAIL_SAME_CAPACITY_RESULTS.md)、[V98测量误差](V98_SIDE_ENERGY_MEASUREMENT_ERROR_RESULTS.md)、[V408外区复测](V408_THRESHOLD_SIDE_REGION_HOLDOUT_RESULTS.md) |
| 小时预测迁移失败 | [V395–V400状态](V395_V399_TRANSFER_BREAKTHROUGH_STATUS.md)、[V200组成形状校准](V200_MIX_CONDITIONED_INTERVAL_PROBE_RESULTS.md) |
| 年度官方因子预测 | [V404可重建结果](V404_NESTED_REGION_CONFORMAL_MIX_INTERVALS.md)、[V406收缩候选](V406_NESTED_PHYSICAL_ANCHOR_SHRINKAGE_RESULTS.md)、[V407失效诊断](V407_TEMPORAL_FORWARD_SHRINKAGE_DIAGNOSTIC.md) |
| 模拟情景复测及更正 | [V410结果](V410_CHINA_SIMULATED_CIF_SCENARIO_STRESS_RESULTS.md)、[V410–V411来源更正](V410_V411_PROVENANCE_CORRECTION.md) |
| 历史重构/三关 | [重构记录](REFACTOR.md)、[三关结果](THREE_GATES_RESULTS.md)、[技术背景](RESEARCH.md) |

完整检索： [文档目录CSV](catalog/documents.csv)、[脚本目录CSV](catalog/scripts.csv)、[本地产物目录库存](catalog/local-artifacts.csv)。Git 内的 [结果快照](evidence/research-snapshot-20261031/README.md) 提供小型摘要及哈希；原文中 `../results/` 链接在远程未必可打开，应按快照的原始相对路径查找。

证据使用顺序为：明确的更正/撤回说明 → 当前可重建结果 → 原始协议及输入/输出记录 → 早期探索。源内复测、同源代理包含性、模拟压力测试和独立实测确认分开陈述。
