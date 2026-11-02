# 消费侧滞后粗月报风险证据

实验定义与结果见[协议](../../LAGGED_COARSE_RISK_TRANSFER_PROTOCOL_20261101.md)和[报告](../../LAGGED_COARSE_RISK_TRANSFER_RESULTS_20261101.md)。此目录只含消费侧标签结果。净受电扩展是独立的额外信息预算，不代表中国已有同口径公开省级月受电字段。

输入为EIA七BA最终修订工作簿，目标特征取`t−2`五类月电量；扩展另加`t−2`月净受电占消费电量比例。因交换字段缺失，在共同支持集上进行配对比较：80个BA-month、2,408个完整本地日。训练2019–2023、来源BA留出校准2024、时间外测试2025。不是历史as-issued版本回放。

| 文件 | SHA-256 |
|---|---|
| `consumed_by_ba_metrics.csv` | `6169e30e08fbe80f34d473543d7e4c65be214903fc39543216a78c5185533c7b` |
| `consumed_macro_metrics.csv` | `eeb39c3f25e727913b0dcd7094d577d49525b06015ff9c60a3843b2be4dd964f` |
| `consumed_monthly_risk_predictions.csv` | `00c4b21d9e682c17ee85322f40914b310ed8920ce74563bd5eceafbe73785e7c` |
| `provenance.json` | `3c6d439d2327a551263186006050f0b785e09c46313ff22afe74eac4b38f6975` |
