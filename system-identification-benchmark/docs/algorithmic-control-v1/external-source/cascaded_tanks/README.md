# Cascaded Tanks 原始数据

该目录保存 4TU.ResearchData 的官方数据 ZIP 原件，来源为 Schoukens 与 Noël 的级联水箱实体系统基准。数据许可为 **CC BY-SA 4.0**；使用或再发布时应保留作者、DOI、许可说明和对原件的修改说明。

- 数据 DOI：<https://doi.org/10.4121/12960104>
- 官方基准说明：<https://www.nonlinearbenchmark.org/benchmarks/cascaded-tanks>
- 归档文件：[`CascadedTanksFiles.zip`](CascadedTanksFiles.zip)
- 来源、数据合同、ZIP/内文件哈希和字段审计：[`acquisition_manifest.json`](acquisition_manifest.json)、[准入审计](../../CASCADED_TANKS_SOURCE_ADMISSION_AUDIT_20261103.md)

原始 ZIP 内含 CSV、MAT、系统说明 PDF 和设备媒体。官方 PDF 说明泵输入是电压、输出来自未校准的电容式水位传感器，并讨论溢流和输入相关过程扰动；没有给出传感器到实际高度的校准或设备安全上限。该数据已进入单独的稀疏输入预测扩展，但结果是公开测试轨迹上的开发证据，不与其他系统合并排名；见[扩展结果报告](../../CASCADED_TANKS_SPARSE_INPUT_RESULTS_20261103.md)。
