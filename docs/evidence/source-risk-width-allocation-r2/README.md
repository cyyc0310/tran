# Source-risk width allocation r2

冻结实验协议：[SOURCE_RISK_WIDTH_ALLOCATION_PROTOCOL_20261102.md](../../SOURCE_RISK_WIDTH_ALLOCATION_PROTOCOL_20261102.md)。结果分析：[SOURCE_RISK_WIDTH_ALLOCATION_RESULTS_20261102.md](../../SOURCE_RISK_WIDTH_ALLOCATION_RESULTS_20261102.md)。

机器结果保存在仓库工作目录 `results/source-risk-width-allocation-r2/`，包括逐月逐国覆盖率和Winkler型区间分数、国别汇总、宏观摘要与输入哈希。入口脚本为 `scripts/run_source_risk_width_allocation.py`。小时预测采用已封存的Energy-Charts Czech expansion r4预测文件，并在运行时校验SHA和时间戳对齐。风险概率使用嵌套留国r2结果。六个目标国均已看过，结果不是盲确认。

主要结果为负：固定12月平均宽度后，完整风险Logistic对比例重组区间分数恶化0.41%，宏覆盖率下降0.25个百分点；对火电因子历史头区间分数恶化0.58%，覆盖率仅增加0.08个百分点。细节和停止结论见结果文档。
