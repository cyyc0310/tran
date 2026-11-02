# US↔GB 细分燃料双向开发结果

该证据包对应[双向实验报告](../../CROSS_COUNTRY_FINE_FUEL_BIDIRECTIONAL_RESULTS_20261101.md)与[预先冻结协议](../../CROSS_COUNTRY_FINE_FUEL_BIDIRECTIONAL_PROTOCOL_20261101.md)。目标小时来自2024年2、4、6月共同队列；结果是固定EF发电侧代理，细分目标份额由小时量回算，不是中国五类M预算、官方消费侧CIF或前瞻as-issued回测。

| 文件 | 内容 |
|---|---|
| `hourly_predictions.csv.gz` | 两方向、各方法逐小时预测与区间 |
| `region_month_metrics.csv` | BA/月粒度配对指标 |
| `macro_metrics.csv` | 方法和方向的宏观汇总指标 |
| `provenance.json` | 数据/模型参数、运行合同与来源记录 |

对应运行目录：`results/cross-country-fine-fuel-bidirectional-r1/`。证据文件与运行输出逐文件SHA-256相同：

```text
69bf355392b0d34f3b9e22782619cad40bae0de06dd6d365cd42e425a68f64b8  hourly_predictions.csv.gz
872065e2e990801a6ac5cc281dbb8fc18184a2df7bb64b4edcf808cacc623193  macro_metrics.csv
554b35a31051f77c8aaa3bc545099dc04ea93bf2ba2b74db8a263160ba038c30  provenance.json
71c79fa27106d4fbb04115d889177f26201f44e29169ac45feda0c4c41e00e22  region_month_metrics.csv
```
