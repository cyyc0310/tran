# 年度煤／气装机信息预算 r1 证据

实验协议：[`OFFICIAL_ANNUAL_COAL_GAS_CAPACITY_PROTOCOL_20261101.md`](../../OFFICIAL_ANNUAL_COAL_GAS_CAPACITY_PROTOCOL_20261101.md)

重跑命令（仓库根目录）：

```bash
PYTHONPATH=src .venv-research-py312-20261101/bin/python scripts/prepare_official_annual_capacity_budget.py
PYTHONPATH=src .venv-research-py312-20261101/bin/python scripts/run_three_country.py \
  --capacity-table data/capacity-public/annual-2021/annual-coal-gas-budget-2020.json \
  --output results/annual-coal-gas-capacity-r1
```

`results/`下的原始运行目录含全部逐小时预测文件与模型包。本证据快照保存了所有预测 CSV、封存清单、逐月/地区/国家指标、输入审计、启动代码哈希、来源容量表及运行所需的相关源码。原始官方PDF、工作簿和美国机组ZIP继续留在本地数据目录，SHA在来源审计和启动哈希中；不复制大原件。

容量预算固定为2020统计年。美国按EIA-860 OP机组主燃料与登记BA聚合；德国与英国取官方年度表煤/气数据，英国值为UK合计，包含Northern Ireland，而小时目标是GB。德国报告及英国工作簿当前文件晚于最初网页发布而修订，无法认证旧字节；结果是年度信息预算的回测近似。标签是小时发电构成乘固定EF的代理，不是官方消费侧强度或现场测量。容量是软预测输入；任何预测都不把年装机当作实际可用出力。

运行时未读新国家，也未用目标成绩选方法。`prediction_seal.json`中的哈希可核验所有预测文件完整性。M基线与先前三国月报实验逐项同值。
