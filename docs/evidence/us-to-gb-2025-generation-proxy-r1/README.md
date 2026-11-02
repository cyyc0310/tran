# US→GB 2025发电侧代理证据包

最终有效运行：[机器预测与评分](../../../results/us-to-gb-2025-generation-proxy-r2/)

总结：[结果、跨年比较和边界](../../US_TO_GB_2025_GENERATION_PROXY_RESULTS_20261102.md)

冻结输入/模型协议：[协议文件](../../US_TO_GB_2025_GENERATION_PROXY_PROTOCOL_20261102.md)

运行器：[代码](../../../scripts/run_us_to_gb_2025_generation_proxy.py)

2025 NESO目标有8,760小时。全部目标点预测及源校准参数先封存，再计算目标固定因子标签和评分。`startup_hashes.json`记录七个美国源文件、NESO历史燃料文件、协议、runner和共享模型模块SHA-256；`prediction_seal.json`记录预测及源校准哈希。全部哈希经独立复算匹配。

该证据是已查看英国系统的时间外开发，不是未见域确认；NESO为当前修订历史快照，不是as-issued版本。标签是统一固定因子生成的发电代理，不是实测或官方CIF。
