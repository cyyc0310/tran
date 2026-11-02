# 发电侧滞后粗月报风险证据

实验定义与结果见[协议](../../LAGGED_COARSE_RISK_TRANSFER_PROTOCOL_20261101.md)和[报告](../../LAGGED_COARSE_RISK_TRANSFER_RESULTS_20261101.md)。此目录只含发电侧标签结果。风险标签绕目标月oracle加权均值计算，属尾风险分解，不是端到端预测成绩。

输入为EIA七BA最终修订工作簿，目标特征取`t−2`五类月电量；训练2019–2023、来源BA留出校准2024、时间外测试2025。不是历史as-issued版本回放。

| 文件 | SHA-256 |
|---|---|
| `generated_by_ba_metrics.csv` | `04ed127437ced08bcd6c60d46d4308d73f682b74b89d1d443b7e56141438f044` |
| `generated_macro_metrics.csv` | `82b790f8e857a2e96dda0f7673450e4ed941dd62a36cfd4f1b717fcf872dcc6f` |
| `generated_monthly_risk_predictions.csv` | `18c30b88be32e083e04fe1210fe4fa8710f24b36930124ed8fa464737e0e73f7` |
| `provenance.json` | `3c6d439d2327a551263186006050f0b785e09c46313ff22afe74eac4b38f6975` |
