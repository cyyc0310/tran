# 粗组成专家门控支持距离诊断

报告：[事后支持距离诊断](../../COARSE_EXPERT_SUPPORT_DIAGNOSTIC_20261101.md)。原始预测与协议见[粗组成条件专家r5证据](../coarse-conditional-experts-r5/README.md)。

此证据包只保存对既有封存预测的后验分层汇总，不包含新模型输出。`diagnostic_manifest.json`记录来源代码、Energy-Charts输入、r5预测及门控原型文件的SHA-256。目标标签已在早前外层评分中读取；所有基于目标标签选出的alpha明确标为事后上界，不是可部署参数。

文件：

- `within_country_support_quartiles.csv`：逐国、逐目标国自身距离分位误差。
- `macro_support_quartiles.csv`：各分位的等国宏平均。
- `country_overall.csv`：逐国整体门控与比例基线MAE。
- `global_alpha_hindsight.csv`：用目标标签扫描固定全局混合比例。
- `support_alpha_hindsight_ceiling.csv`：用目标标签选择分位混合比例的事后上界。
- `support_alpha_hindsight_ceiling.json`：固定逐小时聚合口径的汇总数字。
