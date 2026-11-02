# R6b完整天气对照

2026-11-01，固定24窗口、已见三国2023 H2开发集。648000预测、234校准、72成员、144组件。相同目标小时对照无天气分支与天气HGB；静态年度装机和仅月报分开。不是未知国家确认。

`predictions.csv.gz`保留所有预测；`member_component_arrays.tar.gz`保留全部分布成员和小时组件；`independent_review.json`以及同目录审计表重建了源校准、每小时预测和分层指标。天气全窗口`eligible_for_issued_public=false`、`available_at=null`，未验证历史公开签发时间；不要将天气增量写作已认证可部署预报或中国效果。

完整重算依赖项目中由`input_code_seal.json`绑定的R5/R6a模型、源标签、气候参考和紧凑天气窗口。此证据包不是独立的原始数据发行。输入与预测封存在同目录保留。研究结果见[报告](../../EXPANDED_ANCHOR_ISSUED_RESULTS.md)。`manifest.json`列出包内其他文件的SHA256。
