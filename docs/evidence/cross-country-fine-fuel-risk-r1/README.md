# US→GB 源区留BA风险半径诊断

该证据包对应[双向结果和风险诊断](../../CROSS_COUNTRY_FINE_FUEL_BIDIRECTIONAL_RESULTS_20261101.md)及[冻结诊断协议](../../CROSS_COUNTRY_FINE_FUEL_RISK_CALIBRATION_PROTOCOL_20261101.md)。美国七个BA作留一区伪目标，拟合90%绝对残差半径后应用至同一已封存的GB开发预测；这是一项事后开发诊断，不提供跨国覆盖保证。

| 文件 | 内容 |
|---|---|
| `source_lobo_residuals.csv.gz` | 各BA留出折的源伪目标残差 |
| `target_gb_interval_metrics.csv` | GB目标区原始与留BA半径的覆盖/宽度指标 |
| `provenance.json` | 运行合同及前序双向预测来源 |

对应运行目录：`results/cross-country-fine-fuel-risk-r1/`。证据文件与运行输出逐文件SHA-256相同：

```text
51cfebcb98fd0a4ed5c03f5efa27b246eb5485ed158181bbff5b89c8cdc83c7b  provenance.json
3f1babd11efd094d847492418bf2b08f2ce30ce8973e1695190850277c3fa1ba  source_lobo_residuals.csv.gz
94729184f0ec61f1f8c058c2f63912f2e94228f921dc7410ef33155622d80f5c  target_gb_interval_metrics.csv
```
