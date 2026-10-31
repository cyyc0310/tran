# V153：小时相位取得稳定开发增量，边际重组没有独立收益

本轮恢复936地区月的UTC时间，全部677,928行发电数组逐行复现V144。六种小时分布×两预算×两源划分×两校准完成1,728条目标月份评分；整月条件池逐项复现V151的288条均值/CRPS/端点/覆盖记录。期望燃料边际核验295,560次，最大误差1.11e-15，缺单元回退0。所有36个2025目标月均已用于开发，未打开新地区。

## 稳定收益来自小时条件化

相对V151整月条件分布，普通UTC小时条件分布在两预算、两源划分下均达到宏小时覆盖≥90%、最差地区≥85%，CRPS下降超过5%，宽度下降，Winkler下降。这里仅完成相对月内恒定CDF的开发筛查；尚未超过同预算小时条件回归等强基线，不等于整个原始创新gate通过。

| seed | 预算 | CRPS变化 | 宽度变化 | CRPS差异 | 配对月份描述95%范围 |
|---:|---|---:|---:|---:|---:|
| 41 | asof_30d | -11.77% | -4.78% | -6.339 | [-7.644, -5.087] |
| 41 | same_month | -17.84% | -9.84% | -8.099 | [-9.745, -6.369] |
| 83 | asof_30d | -11.34% | -6.98% | -6.128 | [-7.353, -4.912] |
| 83 | same_month | -17.31% | -10.51% | -7.870 | [-9.492, -6.223] |

配对区间以12个日历月同步重采样三个地区，1,000次、seed20261028，为已查看开发数据的描述结果。两次源角色划分不是独立地区重复。完整同月分支仍属事后重建；asof30天是模拟发布条件，未验证真实发布快照。

PACE贡献最大：同月普通小时条件池CRPS61.39/62.27，相对原整月条件池82.44/82.54下降约25%；pooled覆盖从81.39%/80.80%提高到87.55%/87.36%，宽度从398.44/396.92降至342.20/342.14。PACE本身仍未达到90%覆盖，不能用宏指标掩盖。LGEE同月CRPS19.98/20.30比原19.87/20.15略差，说明日历强度的跨区迁移并非每区都获益。

## 消融解释

仅小时条件化与“小时+期望边际重组”的指标相同到当前输出精度。完整UTC月各小时占比均匀，当前源样本的相应平均形状归一也没有产生实质调整；边际重组不能作为增量机制。

增加周末后略差或几乎相同，边际重组对CRPS仅产生很小变化。供体小时统一偏移+7小时后，完整同月CRPS约53.43/53.05，显著差于正确小时37.30/37.59；滞后预算也类似。负对照支持收益来自日内相位对齐，不能归因于单纯把数据切成更多组。

## 目标标签oracle的日历信号检查

另作事后机制诊断：目标小时经验分布按UTC小时或小时×周末分组，对比20次同大小随机分组。以下是地区内按小时汇总的自评分；oracle不部署，不作独立确认。分组越细，有限样本自身拟合也会降低自评分，因此必须同时看随机分组控制。

| 地区 | 分组 | 整月oracle CRPS | 日历oracle CRPS | 随机分组平均 |
|---|---|---:|---:|---:|
| LGEE | hour | 18.169 | 17.274 | 17.598 |
| LGEE | hour_weekend | 18.169 | 16.331 | 16.974 |
| PACE | hour | 79.941 | 50.365 | 77.486 |
| PACE | hour_weekend | 79.941 | 47.575 | 74.767 |
| SC | hour | 33.110 | 26.118 | 32.049 |
| SC | hour_weekend | 33.110 | 25.279 | 30.932 |

PACE的日历oracle改善远大于随机分组，支持存在可用的日内结构；LGEE日历信号弱得多。这是下一机制定位依据，不是目标可学性或新模型效果的证明。

## 全部宏小时结果

CRPS/MAE/宽度/Winkler为g/kWh；覆盖是名义90%小时区间的经验比例。所有模型仅用完整月度统计、UTC日历和proper源小时形状。预期边际校正可解释为给q施加归一因子，使按完整计划日历平均的q为1；若构造发电轨迹，M/T这个公共因子在CIF比例中抵消。目标有效小时占比从未用于该归一化。

| seed | 预算 | 方法 | 校准 | CRPS | 覆盖 | 最差覆盖 | 宽度 | Winkler |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 41 | asof_30d | hour_raw | max_region | 47.534 | 100.00% | 99.99% | 617.53 | 617.79 |
| 41 | asof_30d | hour_raw | pooled | 47.534 | 96.09% | 93.09% | 345.90 | 364.50 |
| 41 | asof_30d | hour_reconciled | max_region | 47.534 | 100.00% | 99.99% | 617.53 | 617.79 |
| 41 | asof_30d | hour_reconciled | pooled | 47.534 | 96.09% | 93.09% | 345.90 | 364.50 |
| 41 | asof_30d | hour_weekend_raw | max_region | 47.620 | 100.00% | 99.99% | 615.50 | 615.76 |
| 41 | asof_30d | hour_weekend_raw | pooled | 47.620 | 95.78% | 92.84% | 346.00 | 367.08 |
| 41 | asof_30d | hour_weekend_reconciled | max_region | 47.597 | 100.00% | 99.99% | 615.40 | 615.66 |
| 41 | asof_30d | hour_weekend_reconciled | pooled | 47.597 | 95.79% | 92.87% | 345.94 | 366.90 |
| 41 | asof_30d | month_conditional | max_region | 53.872 | 100.00% | 99.99% | 632.71 | 632.95 |
| 41 | asof_30d | month_conditional | pooled | 53.872 | 94.85% | 92.23% | 363.26 | 393.76 |
| 41 | asof_30d | shift7_reconciled | max_region | 61.452 | 99.92% | 99.78% | 608.56 | 609.27 |
| 41 | asof_30d | shift7_reconciled | pooled | 61.452 | 92.35% | 87.04% | 375.11 | 436.65 |
| 41 | same_month | hour_raw | max_region | 37.297 | 95.83% | 94.96% | 290.91 | 309.20 |
| 41 | same_month | hour_raw | pooled | 37.297 | 90.69% | 87.55% | 237.26 | 291.31 |
| 41 | same_month | hour_reconciled | max_region | 37.297 | 95.83% | 94.96% | 290.91 | 309.20 |
| 41 | same_month | hour_reconciled | pooled | 37.297 | 90.69% | 87.55% | 237.26 | 291.31 |
| 41 | same_month | hour_weekend_raw | max_region | 37.404 | 95.73% | 94.78% | 289.53 | 309.36 |
| 41 | same_month | hour_weekend_raw | pooled | 37.404 | 90.51% | 87.48% | 236.51 | 292.31 |
| 41 | same_month | hour_weekend_reconciled | max_region | 37.397 | 95.73% | 94.78% | 289.61 | 309.34 |
| 41 | same_month | hour_weekend_reconciled | pooled | 37.397 | 90.49% | 87.47% | 236.46 | 292.30 |
| 41 | same_month | month_conditional | max_region | 45.396 | 93.89% | 89.35% | 313.91 | 346.36 |
| 41 | same_month | month_conditional | pooled | 45.396 | 88.83% | 81.39% | 263.15 | 341.53 |
| 41 | same_month | shift7_reconciled | max_region | 53.434 | 95.26% | 88.26% | 390.19 | 429.90 |
| 41 | same_month | shift7_reconciled | pooled | 53.434 | 84.24% | 73.20% | 262.81 | 416.17 |
| 83 | asof_30d | hour_raw | max_region | 47.904 | 99.95% | 99.84% | 527.62 | 528.16 |
| 83 | asof_30d | hour_raw | pooled | 47.904 | 90.94% | 88.86% | 279.07 | 327.38 |
| 83 | asof_30d | hour_reconciled | max_region | 47.904 | 99.95% | 99.84% | 527.62 | 528.16 |
| 83 | asof_30d | hour_reconciled | pooled | 47.904 | 90.94% | 88.86% | 279.07 | 327.38 |
| 83 | asof_30d | hour_weekend_raw | max_region | 47.962 | 99.94% | 99.83% | 525.92 | 526.45 |
| 83 | asof_30d | hour_weekend_raw | pooled | 47.962 | 90.85% | 88.54% | 279.17 | 330.55 |
| 83 | asof_30d | hour_weekend_reconciled | max_region | 47.941 | 99.94% | 99.82% | 525.83 | 526.35 |
| 83 | asof_30d | hour_weekend_reconciled | pooled | 47.941 | 90.83% | 88.54% | 279.03 | 330.29 |
| 83 | asof_30d | month_conditional | max_region | 54.033 | 99.95% | 99.84% | 546.96 | 547.44 |
| 83 | asof_30d | month_conditional | pooled | 54.033 | 88.50% | 82.83% | 300.01 | 377.13 |
| 83 | asof_30d | shift7_reconciled | max_region | 61.236 | 99.73% | 99.41% | 528.70 | 530.28 |
| 83 | asof_30d | shift7_reconciled | pooled | 61.236 | 89.35% | 77.68% | 330.99 | 431.39 |
| 83 | same_month | hour_raw | max_region | 37.589 | 96.24% | 94.74% | 290.37 | 305.95 |
| 83 | same_month | hour_raw | pooled | 37.589 | 91.24% | 87.36% | 235.85 | 285.51 |
| 83 | same_month | hour_reconciled | max_region | 37.589 | 96.24% | 94.74% | 290.37 | 305.95 |
| 83 | same_month | hour_reconciled | pooled | 37.589 | 91.24% | 87.36% | 235.85 | 285.51 |
| 83 | same_month | hour_weekend_raw | max_region | 37.672 | 96.08% | 94.60% | 288.12 | 305.61 |
| 83 | same_month | hour_weekend_raw | pooled | 37.672 | 91.12% | 87.23% | 235.30 | 286.96 |
| 83 | same_month | hour_weekend_reconciled | max_region | 37.667 | 96.09% | 94.63% | 287.81 | 305.40 |
| 83 | same_month | hour_weekend_reconciled | pooled | 37.667 | 91.15% | 87.29% | 235.19 | 286.86 |
| 83 | same_month | month_conditional | max_region | 45.459 | 93.97% | 89.00% | 316.58 | 348.44 |
| 83 | same_month | month_conditional | pooled | 45.459 | 89.01% | 80.80% | 263.56 | 342.31 |
| 83 | same_month | shift7_reconciled | max_region | 53.049 | 96.23% | 91.28% | 403.65 | 432.56 |
| 83 | same_month | shift7_reconciled | pooled | 53.049 | 87.32% | 74.38% | 286.00 | 411.07 |

## 创新边界与下一步

将普通小时条件重组升级为必须战胜的强基线，停止将边际归一化作为主创新。下一轮先补同预算小时条件分位数/树模型及源小时模板，再检验是否能利用源区日历信息强度的差异取得进一步迁移增量。UTC是本轮可复现基线，尚需当地民用/太阳时等合法地理时间对照；不能把UTC基线未解决的问题外推为全部小时方法的上限。

相关原始论文也要求控制立意： [Zhang等，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/41310)的多频局部时间/跨变量模块使用历史多变量窗口，正文问题定义见§Proposed Method；增加小波或时间条件本身已有充分先例。[CarbonCast，BuildSys 2022](https://lass.cs.umass.edu/papers/pdf/buildsys2022-carboncast.pdf)结合分燃料产出预测、历史CIF和天气。两者不能直接按本轮零目标小时历史预算比较；这属于输入预算区别，不表示已验证我们优于这些方法。

当前进展是找到具有稳定实测收益的小时优化空间，尚没有可归因的原创机制或未查看地区验证。中国真实CIF、真实发布版本和原始完整gate继续保留。

依据：[协议](V153_CALENDAR_LAW_PROTOCOL.md)、[主结果](../results/v153-calendar-law-r1/summary.json)、[逐区指标](../results/v153-calendar-law-r1/region_metrics.csv)、[配对差异](../results/v153-calendar-law-audit-r1/paired_month_differences.csv)、[时间对齐收据](../results/v153-calendar-times-r1/summary.json)、[日历oracle诊断](../results/v153-calendar-floor-diagnostic-r1/summary.json)。输出目录均不存在的干净副本中运行：

```bash
/opt/homebrew/bin/python3.12 scripts/v153_prepare_timestamps.py
/opt/homebrew/bin/python3.12 scripts/v153_calendar_law.py
```
