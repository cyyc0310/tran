# 研究文档入口

先读 [四日成果报告](reports/2026-10-28_31.md) 了解目前已知结论，再查具体协议和结果。完整逐步日志 [BREAKTHROUGH_SEARCH](BREAKTHROUGH_SEARCH.md) 是过程记录，不宜单独作为最终论文结论。

| 问题 | 优先阅读 |
|---|---|
| 用户真实信息预算 | [NBS月字段边界](CHINA_MONTHLY_GENERATION_FIELD_AUDIT_V51.md)、[上海附件核对](V409_SHANGHAI_UPLOAD_RECONCILIATION.md) |
| 官方31省面板与发布时间 | [V392官方数据审计](V392_NBS_OFFICIAL_LONG_PANEL_AUDIT.md)、[V394前向状态结果](V394_ALL_PROVINCE_RELEASE_LAG_STATE_RESULTS.md) |
| 发电侧与消费侧边界 | [V401上海电力平衡](V401_SHANGHAI_ESTIMAND_BOUNDARY.md)、[V402高频标签来源](V402_HOURLY_LABEL_PROVENANCE_AND_BENCHMARK_GATE.md) |
| 部分识别求解器 | [V108方法与门槛](V108_SHARP_SET_METHOD_AND_PAPER_GATE.md)、[V109独立LP核验](V109_INDEPENDENT_LP_SHARP_SET_AUDIT_RESULTS.md) |
| 最小阈值侧信息 | [V97匹配容量对照](V97_SIDE_VS_DETAIL_SAME_CAPACITY_RESULTS.md)、[V98测量误差](V98_SIDE_ENERGY_MEASUREMENT_ERROR_RESULTS.md)、[V408外区复测](V408_THRESHOLD_SIDE_REGION_HOLDOUT_RESULTS.md) |
| 小时预测迁移失败 | [V395–V400状态](V395_V399_TRANSFER_BREAKTHROUGH_STATUS.md)、[V200组成形状校准](V200_MIX_CONDITIONED_INTERVAL_PROBE_RESULTS.md) |
| 年度官方因子预测 | [V404可重建结果](V404_NESTED_REGION_CONFORMAL_MIX_INTERVALS.md)、[V406收缩候选](V406_NESTED_PHYSICAL_ANCHOR_SHRINKAGE_RESULTS.md)、[V407失效诊断](V407_TEMPORAL_FORWARD_SHRINKAGE_DIAGNOSTIC.md) |
| 模拟情景复测及更正 | [V410结果](V410_CHINA_SIMULATED_CIF_SCENARIO_STRESS_RESULTS.md)、[V410–V411来源更正](V410_V411_PROVENANCE_CORRECTION.md) |
| 历史重构/三关 | [重构记录](REFACTOR.md)、[三关结果](THREE_GATES_RESULTS.md)、[技术背景](RESEARCH.md) |

完整检索： [文档目录CSV](catalog/documents.csv)、[脚本目录CSV](catalog/scripts.csv)、[本地产物目录库存](catalog/local-artifacts.csv)。Git 内的 [结果快照](evidence/research-snapshot-20261031/README.md) 提供小型摘要及哈希；原文中 `../results/` 链接在远程未必可打开，应按快照的原始相对路径查找。

证据使用顺序为：明确的更正/撤回说明 → 当前可重建结果 → 原始协议及输入/输出记录 → 早期探索。源内复测、同源代理包含性、模拟压力测试和独立实测确认分开陈述。
