# 五国火电因子迁移分解 r1 证据

本证据包只解释已经评分的五国2024 LOCO预测，没有拟合新模型或选择新参数。冻结设计见 [`FACTOR_HEAD_TRANSFER_DECOMPOSITION_PROTOCOL_20261101.md`](../../FACTOR_HEAD_TRANSFER_DECOMPOSITION_PROTOCOL_20261101.md)。

## 文件

- `country_factor_decomposition.csv`：按目标国家等月汇总的预测和火电因子误差。
- `monthly_factor_decomposition.csv`：逐目标国家、逐月指标。
- `hourly_factor_decomposition.csv.gz`：逐小时预测复现和oracle分解。此文件含目标标签的事后诊断值，只可用于已封存开发折分析。
- `summary.json`：宏平均和运行规模。
- `reconstruction_seal.json`：五个目标预测文件的SHA-256与行数。
- `post_run_verification.json`：运行结束时的重建、代码和输入核验结果。
- `input_hashes.json`、`startup_code_hashes.json`、`input_audit.json`：来源输入、运行代码与数据审计。

## 核验结果

五个外层国家共43,918小时；12个输入文件和6个代码/协议文件的哈希均无差异。四个重算基线的所有目标点预测逐行与封存输出一致。`post_run_verification.json`记录 `missing_outputs=[]`、无输入/代码哈希差异以及 `sealed_prediction_reconstruction_match=true`。

结果是已看国家上的固定因子发电代理回测，不是中国实测CIF精度或未知国家确认。Oracle列使用了目标小时真值，不能作为部署输入、候选模型或可加总的收益分解。
