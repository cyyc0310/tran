# EIA-923官方月量下的历史月报适配证据

该证据包比较真实EIA-923 BA月统计与小时回算月组成，接入现有七种本地/源端月量预测器和三种风险基线。研究解释见[结果报告](../../EIA923_MONTHLY_ADAPTATION_RESULTS_20261101.md)，运行合同见[冻结协议](../../EIA923_MONTHLY_ADAPTATION_PROTOCOL_20261101.md)。

## 结果文件

- `results-r3/`是修复2020–2021旧代码`WAT`水电映射后的权威重跑；包含112个封存预测文件、哈希、全月和共同月逐区/逐国指标、EIA-923映射份额及运行状态。
- `results-r2/`是映射修复前、但代码精简后的留档；`results/`是首次运行留档。二者均仅供追溯，引用修正后结果时以`results-r3/`为准。
- 2020/2021官方输入档案和下载元数据在本证据包`../eia923-monthly-pseudotarget-r1/input/`。
- 2022–2024档案与先前BA映射审计共用，位于`../eia923-ba-mapping-r1/`。

## 运行

在仓库根目录执行：

```sh
PYTHONPATH=src python scripts/run_eia923_monthly_adaptation.py \
  --output docs/evidence/eia923-monthly-adaptation-r1/results-r3
```

数据是年度最终修订版本。精确`t−2`只表示滞后索引模拟，不认证历史发行时点可用性。
