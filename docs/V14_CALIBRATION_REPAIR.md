# V14 校准响应队列修复说明

2026-10-24，Sol。V14是V13空响应队列的运行时修复，不是根据目标效果追加的模型搜索。科学问题和原始文献沿用 [V13设计审查](V13_DESIGN_AND_LITERATURE.md)，执行合同见 [V14协议](WEATHER_COARSE_V14_PROTOCOL.md)。

V13第一轮源预测、评分和选择已完成并查看。V13第二轮在构建ISNE的2024H2风险响应时，合法月监督过滤后没有任何日，`np.stack([])`抛出`ValueError: need at least one array to stack`。第二轮尚未完成源预测/评分；该失败发生在任何V13目标预测或目标评分之前。没有可用的V13第二轮性能结论。V13代码、协议、首轮源结果和失败输出均保留，不覆盖、不修补其原字节。

## 元数据核查

只检查冻结源月表`results/heldout-2026h1-data/monthly.csv`的地区、`period_start/period_end/available_at`，不读取新地区小时标签。限定月起点在`[2024-07-01,2025-01-01)`，并要求`period_end <= 2025-01-01`且`available_at <= 2025-01-01`。结果为以下**28条不同地区月响应记录**，不是28个完整小时地区日，也不是28个独立日样本：

| 地区 | 已发布H2月数 | 对应月份 |
| --- | ---: | --- |
| BPAT | 5 | 7、8、9、10、11月 |
| CISO | 5 | 7、8、9、10、11月 |
| ERCO | 3 | 7、8、11月 |
| FPL | 5 | 7、8、9、10、11月 |
| ISNE | 0 | 无 |
| MISO | 5 | 7、8、9、10、11月 |
| NYIS | 5 | 7、8、9、10、11月 |

月表的发布时点来自冻结`available_at`，不能以小时遥测lag7替代。12月月响应尚不满足固定截止时间；ISNE在该H2范围没有已批准月行。此修复不补造ISNE响应，不放宽截止日期，不填补ERCO缺失月份。完整风险日队列还须通过既有完整24小时/特征及标签检查，实际日数在预测审计中报告。

## 唯一方法执行变更

预检由月表元数据派生`calibration_sources=BPAT/CISO/ERCO/FPL/MISO/NYIS`，在V14两轮冻结文件中保存，并在验证时重新核对。新风险响应池为原外层允许源集与该名单的交集：留出这六区之一时为五区，留出ISNE或预测既有目标时为六区。所有新global/hetero/location_scale/QRF/level/physical家族共用完全相同的合法响应日。

ISNE继续属于全部七源proper、donor和门控训练；作为外层被评估源区时仍完整保留，最差地区覆盖门槛继续包含ISNE。只对没有合法H2月监督的风险响应跳过ISNE，不能把它从源评价中删除。风险模型、位置/尺度折外及深层排除必须围绕实际有响应的地区迭代，不能对不存在的响应数组执行堆叠；proper/gate的源集合权限保持原定义。

R1仍为六分支，R2仍为22分支；天气动作、模型参数、共同特征、128情景、倍率网格、source-only选择、配对区间、实用/机制门槛均不变。V14首轮重新生成的全部预测数组和分数及源选结果，应与V13首轮完全相同，允许版本/路径/运行时间等元数据不同。核实这一身份后才使用V14源选policy进入第二轮；不直接用本次故障改变policy。

V13首轮源分数已经被查看，因此V14不被称为一次新的独立源确认。该修复在任何目标预测/评分前提出，依据缺失月响应元数据和运行异常，而非目标表现。完成V14两轮源选择、封存两轮全部目标预测后才评分；未查看地区继续封存，创新与新地区自动触发标志仍为false。

## 修复前证据快照

以下是本次只读核查时的SHA-256；后续完整封存清单由运行者建立。

| 路径 | SHA-256 |
| --- | --- |
| results/heldout-2026h1-data/monthly.csv | `77cb3b47d079094f13d3437f19b0315ba09ac50305981d69fd4f9991a7f07ee5` |
| scripts/weather_coarse_v13.py | `6b0bd24d9cf64629d7be6841ca68221e35987ec439cb226c216b594eafd56285` |
| scripts/run_weather_coarse_v13.py | `3b50797e8b9dd555fedade5e76b2edb19005838fa5088514fb884e98e2cee706` |
| scripts/analyze_weather_coarse_v13.py | `e53dd46c5b2dfcba13d9b08f14e7f3ac2ab26e6373d48cd4019a1e5a8e443505` |
| docs/WEATHER_COARSE_V13_PROTOCOL.md | `759769e5161f1e0a4b7896881e133ff5979f4840eb604130820bb640d689e25f` |
| results/weather-coarse-v13-round1/source_completed.json | `1e65eef9e5be1ee28d1252a7c72bcf8c92ebf8cb4e635b967a820a0dce3634c9` |
| results/weather-coarse-v13-round1/source_choices.json | `5c766031e225bc8474530b0d70bc6c9dca8f4ccd07796c8feacbd65cd9393224` |
| results/weather-coarse-v13-round2/prefit_freeze.json | `44bdcef8f77f82efc029562282b4ae944997015a6765cdeac8a7ec22e85d2e35` |
| results/weather-coarse-v13-verification/round2-source.log | `b874c046d78cfff5295b3565144569acafbb0a003cd8659bb779d726f9e929e4` |
