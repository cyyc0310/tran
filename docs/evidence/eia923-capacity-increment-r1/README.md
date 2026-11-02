# EIA-923月统计＋年度煤气装机增量证据包

US↔GB已查看地区的配对开发实验。此处将年度煤／气装机作为独立信息预算，叠加到上一轮EIA-923官方BA月度粗组成模型。

## 内容

- `common_country_metrics.csv`：共同2024年2、4、6月国家/地区宏平均。
- `common_region_metrics.csv`：逐BA和GB区域指标。
- `monthly_metrics.csv`：逐月、逐方法、逐区间指标。
- `prediction_seal.json`：112份预测文件SHA-256。
- `input_hashes.json`：12个输入文件哈希，包括年度容量表、EIA-923 2022—2024 ZIP和8份小时数据。
- `startup_code_hashes.json`：运行脚本、依赖解析/预测模块和冻结协议的哈希。
- `post_run_verification.json`：封存、输入、代码哈希与前一轮M臂一致性检查。
- `capacity_ef.json`：按2020年度煤/气装机份额换算的逐地区因子。
- `input_audit.json`：逐地区小时源文件覆盖摘要。
- `environment.json`：NumPy、pandas、scikit-learn版本。

完整预测仍保留在[`results/eia923-capacity-increment-r1/`](../../../results/eia923-capacity-increment-r1/)，本证据目录提供指标与校验清单。

## 复现

从仓库根目录执行：

```bash
PYTHONPATH=src:scripts python scripts/run_eia923_capacity_increment.py \
  --output results/eia923-capacity-increment-r1 \
  --capacity-json data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json \
  --eia923-2022 docs/evidence/eia923-ba-mapping-r1/f923_2022.zip \
  --eia923-2023 docs/evidence/eia923-ba-mapping-r1/f923_2023.zip \
  --eia923-2024 docs/evidence/eia923-ba-mapping-r1/f923_2024.zip
```

这是最终版统计的滞后模拟，不是历史as-issued回放；也不是新国家确认。
