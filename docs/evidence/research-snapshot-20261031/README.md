# 2026-10-31 紧凑结果快照

本目录随 Git 提交，保存本地 `results/` 中577份小型摘要/宏观CSV的原始字节，共6,627,193字节。每个文件的原始路径、复制路径、字节数和SHA-256在 [manifest.json](manifest.json) 中。`results/` 子目录按原始实验目录命名，例如 [V408摘要](results/v408-threshold-side-region-holdout-r1/summary.json)、[V410情景指标](results/v410-china-simulated-hourly-cif-stress-r1/summary_by_scenario_method.csv)。

该快照用于查阅和备份。包含旧运行、失败、被替代或不合格结果，不能按目录存在或 `status=complete` 推断科学主张成立。当前判定先看 [四日成果报告](../../reports/2026-10-28_31.md) 与各版本更正。尤其 [V410–V411来源更正](../../V410_V411_PROVENANCE_CORRECTION.md) 说明同一情景标签此前已被查看。

完整本地结果目录库存见 [local-artifacts.csv](../../catalog/local-artifacts.csv)。此次扫描784个结果子目录、约94.0 GB（十进制）；大预测表、NPZ、模型权重、第三方原始数据、外部文献未复制。仅有摘要不能完整重跑历史实验；还需对应原始输入及历史源码版本。一些原收据记录本机绝对路径，保留原文是为保证文件哈希，而非保证别的计算机可直接运行。

生成工具：[maintenance_snapshot.py](../../../scripts/maintenance_snapshot.py)。源 `results/` 没有移动或修改。文档目录：[documents.csv](../../catalog/documents.csv)；脚本目录：[scripts.csv](../../catalog/scripts.csv)。
