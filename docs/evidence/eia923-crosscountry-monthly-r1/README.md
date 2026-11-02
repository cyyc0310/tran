# EIA-923跨国月统计增量证据包

US↔GB开发实验。EIA-923历史最终版按BA聚合，作为五类粗月输入替换美国侧小时回算月统计；GB维持原小时回算月输入。冻结协议、评价切分和研究结论见[实验协议](../../EIA923_CROSS_COUNTRY_EXPERIMENT_PROTOCOL_20261101.md)及[结果报告](../../EIA923_CROSS_COUNTRY_EXPERIMENT_RESULTS_20261101.md)。

## 内容

- `common_country_metrics.csv`：共同2024年2、4、6月的地区/国家宏平均指标。
- `common_region_metrics.csv`：逐BA及GB区域指标。
- `monthly_metrics.csv`：逐地区、方法、区间方法、目标月份的完整指标。
- `prediction_seal.json`：112个封存的预测文件名及SHA-256。
- `input_hashes.json`：EIA-923三个年度ZIP和8份区域小时源文件哈希。
- `startup_code_hashes.json`：运行脚本、解析器、研究模块和冻结协议的启动哈希。
- `post_run_verification.json`：运行后预测、输入、代码哈希核验摘要。
- `environment.json`：NumPy、pandas、scikit-learn版本。
- `hourly_reconstructed_table_availability.json`、`eia923_us_table_availability.json`：五类月表覆盖范围。

逐月目标预测和拟合风险模型保留在本地[`results/eia923-crosscountry-monthly-r1/`](../../../results/eia923-crosscountry-monthly-r1/)，不在证据快照复制；快照中的指标与封存清单可以用于发现结果变更。所有目标标签只在预测输出写完并封存后用于评分。

## 复现

从仓库根目录运行：

```bash
PYTHONPATH=src:scripts python scripts/run_eia923_crosscountry_monthly.py \
  --output results/eia923-crosscountry-monthly-r1 \
  --eia923-2022 docs/evidence/eia923-ba-mapping-r1/f923_2022.zip \
  --eia923-2023 docs/evidence/eia923-ba-mapping-r1/f923_2023.zip \
  --eia923-2024 docs/evidence/eia923-ba-mapping-r1/f923_2024.zip
```

该实验使用EIA-923历史最终值模拟滞后可见，不是历史as-issued回放；也不属于未查看国家确认。
